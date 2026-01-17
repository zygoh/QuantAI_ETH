"""
机器学习服务
"""
# StdLib
import asyncio
import gc
import logging
import os
import pickle
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Third-Party
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Local App
from app.core.cache import cache_manager
from app.core.config import settings
from app.core.constants import (
    FEATURE_IMPORTANCE_THRESHOLD_HIGH,
    FEATURE_IMPORTANCE_THRESHOLD_LOW,
    LABEL_PT_SL_CONFIG,
    LABEL_VOLATILITY_DEFAULT,
    LABEL_VOLATILITY_MIN,
    LABEL_VOLATILITY_WINDOW,
    LABEL_WINDOW_CONFIG,
    LGB_BAGGING_FRACTION,
    LGB_BAGGING_FREQ,
    LGB_FEATURE_FRACTION,
    LGB_LEARNING_RATE,
    LGB_MAX_DEPTH,
    LGB_MIN_CHILD_SAMPLES,
    LGB_MIN_SPLIT_GAIN,
    LGB_N_ESTIMATORS,
    LGB_NUM_LEAVES,
    LGB_REG_ALPHA,
    LGB_REG_LAMBDA
)
from app.core.database import postgresql_manager
from app.exchange.exchange_factory import ExchangeFactory
from app.model.base.utils import (
    compute_effective_sample_weights,
    prepare_features_labels
)
from app.model.feature_engineering import feature_engineer

logger = logging.getLogger(__name__)

class MLService:
    """机器学习服务（支持多时间框架独立模型）"""
    
    def __init__(self):
        self.is_running = False
        # 多时间框架模型：{'3m': model, '5m': model, '15m': model}
        self.models = {}
        self.scalers = {}
        self.feature_columns_dict = {}
        
        self.feature_engineer = feature_engineer
        self.model_metrics = {}
        self.training_task = None
        self.is_first_training = True  # 标记是否首次训练（只有首次才写数据库）
        
        # 训练数据源固定使用 Binance（确保数据一致性）
        self.exchange_client = ExchangeFactory.get_current_client()
        logger.info("训练数据源已固定为 Binance（信号系统：仅数据获取）")
        
        # 模型参数
        # LightGBM基础参数（所有时间框架共享）
        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,  # 0: 下跌, 1: 横盘, 2: 上涨
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': LGB_N_ESTIMATORS,
            'num_leaves': LGB_NUM_LEAVES,
            'learning_rate': LGB_LEARNING_RATE,
            'feature_fraction': LGB_FEATURE_FRACTION,
            'bagging_fraction': LGB_BAGGING_FRACTION,
            'bagging_freq': LGB_BAGGING_FREQ,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
            'max_depth': LGB_MAX_DEPTH,
            'min_child_samples': LGB_MIN_CHILD_SAMPLES,
            'reg_alpha': LGB_REG_ALPHA,
            'reg_lambda': LGB_REG_LAMBDA,
            'min_split_gain': LGB_MIN_SPLIT_GAIN,
            'is_unbalance': True  # 自动处理不平衡类别
        }

        # ✅ 差异化配置：防止过拟合的保守策略（仅3m/5m/15m）
        self.lgb_params_by_timeframe = {
            '3m': {
                'num_leaves': 110,       # 样本充足，保持较高复杂度
                'min_child_samples': 45,
                'max_depth': 8,
                'reg_alpha': 0.4,
                'reg_lambda': 0.4
            },
            '5m': {
                'num_leaves': 110,       # 样本充足，保持较高复杂度
                'min_child_samples': 45,
                'max_depth': 8,
                'reg_alpha': 0.4,
                'reg_lambda': 0.4
            },
            '15m': {
                'num_leaves': 110,       # 样本充足(33k+)，保持较高复杂度
                'min_child_samples': 45,
                'max_depth': 8,
                'reg_alpha': 0.4,
                'reg_lambda': 0.4
            }
        }
        
        # GPU配置
        if settings.USE_GPU:
            self.lgb_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })

        # 模型文件路径（每个时间框架独立）
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _compute_effective_sample_weights(self, y: pd.Series, timeframe: str) -> np.ndarray:
        """使用有效样本数计算样本权重（使用模块函数）"""
        return compute_effective_sample_weights(y, timeframe)
    
    def _get_model_paths(self, timeframe: str) -> Dict[str, str]:
        """获取指定时间框架的模型文件路径"""
        # 🔧 修复：处理SYMBOL中的/字符（如"ETH/USDT"），替换为_避免路径问题
        # 必须与ensemble_ml_service中的逻辑保持一致
        safe_symbol = settings.SYMBOL.replace('/', '_')
        return {
            'model': os.path.join(self.model_dir, f"{safe_symbol}_{timeframe}_model.pkl"),
            'scaler': os.path.join(self.model_dir, f"{safe_symbol}_{timeframe}_scaler.pkl"),
            'features': os.path.join(self.model_dir, f"{safe_symbol}_{timeframe}_features.pkl")
        }
    
    async def start(self):
        """启动机器学习服务"""
        try:
            logger.info("启动机器学习服务...")
            
            # 加载已有模型
            await self._load_model()
            
            # 注意：训练任务已由 scheduler 统一管理（每天00:01执行）
            # 不再在此处启动独立的训练循环
            
            self.is_running = True
            logger.info("机器学习服务启动完成（训练由scheduler管理）")
            
        except Exception as e:
            logger.error(f"启动机器学习服务失败: {e}")
            raise
    
    async def stop(self):
        """停止机器学习服务"""
        try:
            logger.info("停止机器学习服务...")
            
            self.is_running = False
            
            # 取消自动训练任务
            if self.training_task:
                self.training_task.cancel()
                try:
                    await self.training_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("机器学习服务已停止")
            
        except Exception as e:
            logger.error(f"停止机器学习服务失败: {e}")
    
    async def train_model(self, force_retrain: bool = False) -> Dict[str, Any]:
        """训练模型（为每个时间框架训练独立模型）"""
        try:
            logger.info("🚀 开始多时间框架模型训练...")
            logger.info(f"GPU配置: USE_GPU={settings.USE_GPU}")
            logger.info(f"时间框架: {settings.TIMEFRAMES}")
            
            all_metrics = {}
            all_training_data = []  # 收集所有训练数据
            
            # 为每个时间框架训练独立模型
            for timeframe in settings.TIMEFRAMES:
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 训练 {timeframe} 时间框架模型...")
                logger.info(f"{'='*60}")
                
                try:
                    # 训练单个时间框架（返回metrics和training_data）
                    metrics, training_data = await self._train_single_timeframe(timeframe)
                    all_metrics[timeframe] = metrics
                    all_training_data.append(training_data)
                    logger.info(f"✅ {timeframe} 模型训练完成 - 准确率: {metrics['accuracy']:.4f}")
                except Exception as e:
                    logger.error(f"❌ {timeframe} 模型训练失败: {e}")
                    logger.error(f"详细错误: {traceback.format_exc()}")
                    all_metrics[timeframe] = {'success': False, 'error': str(e), 'accuracy': 0.0, 'training_time': 0.0}
            
            # 保存模型（即使有个别时间框架失败也保存成功的）
            await self._save_model()
            
            # 🔥 禁用首次训练数据写入（节省2分钟）
            # 原因：
            # 1. 数据库数据仅用于前端展示，不影响预测
            # 2. WebSocket缓冲区已有60天数据，足够预测
            # 3. 实时WebSocket数据会持续写入数据库
            # 4. 节省144秒启动时间，提升用户体验
            # if self.is_first_training and all_training_data:
            #     try:
            #         await self._save_training_data_to_db(all_training_data)
            #         logger.info("💡 首次训练完成，后续训练将不再写入历史数据")
            #     except Exception as e:
            #         logger.warning(f"保存训练数据到数据库失败（不影响训练）: {e}")
            #     finally:
            #         self.is_first_training = False
            
            logger.info("💡 首次训练完成（历史数据已禁用写入，仅保留实时WebSocket数据）")
            self.is_first_training = False
            
            # 汇总所有模型指标
            successful_metrics = [m for m in all_metrics.values() if 'accuracy' in m and not np.isnan(m['accuracy'])]
            avg_accuracy = np.mean([m['accuracy'] for m in successful_metrics]) if successful_metrics else 0.0
            total_training_time = sum([m.get('training_time', 0) for m in all_metrics.values()])
            
            self.model_metrics = {
                'timeframe_metrics': all_metrics,
                'average_accuracy': avg_accuracy,
                'accuracy': avg_accuracy,  # 兼容旧代码
                'training_time': total_training_time,
                'version': '2.0',
                'training_date': datetime.now().isoformat()
            }
            
            # 缓存模型指标（不过期）
            await cache_manager.set_model_metrics(settings.SYMBOL, self.model_metrics, expire=None)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🎉 多时间框架模型训练完成")
            logger.info(f"成功训练: {len(successful_metrics)}/{len(settings.TIMEFRAMES)} 个时间框架")
            logger.info(f"平均准确率: {avg_accuracy:.4f}")
            logger.info(f"总训练时间: {total_training_time:.2f}秒")
            logger.info(f"{'='*60}\n")
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return {}
    
    async def _train_single_timeframe(self, timeframe: str) -> tuple:
        """训练单个时间框架的模型
        
        Returns:
            tuple: (metrics, training_data_with_timeframe) 
        """
        try:
            start_time = time.time()
            
            # 获取该时间框架的训练数据
            train_data = await self._prepare_training_data_for_timeframe(timeframe)
            
            if train_data.empty:
                raise Exception(f"{timeframe} 训练数据为空")
            
            
            # 保存原始训练数据（用于后续写入数据库）
            train_data_with_timeframe = train_data.copy()
            train_data_with_timeframe['timeframe'] = timeframe
            
            # 特征工程
            train_data = self.feature_engineer.create_features(train_data)
            
            if train_data.empty:
                raise Exception(f"{timeframe} 特征工程后数据为空")
            
            # 创建标签（传入timeframe使用差异化阈值）
            train_data = self._create_labels(train_data, timeframe=timeframe)
            
            # 准备训练数据
            X, y = self._prepare_features_labels(train_data, timeframe)
            
            if len(X) == 0:
                raise Exception(f"{timeframe} 特征数据为空")
            
            # 数据预处理
            X_scaled = self._scale_features(X, timeframe=timeframe, fit=True)
            
            # 时间序列分割（前80%训练，后20%验证）
            split_idx = int(len(X_scaled) * 0.8)
            X_train = X_scaled[:split_idx]
            X_val = X_scaled[split_idx:]
            y_train = y.iloc[:split_idx]
            y_val = y.iloc[split_idx:]
            
            logger.info(f"📊 {timeframe} 时间序列分割: 训练{len(X_train)}条, 验证{len(X_val)}条")
            
            # 训练模型（传入timeframe以使用差异化参数）
            model = self._train_lightgbm(X_train, y_train, X_val, y_val, timeframe=timeframe)
            self.models[timeframe] = model
            
            # 评估模型
            metrics = self._evaluate_model_for_timeframe(X_val, y_val, timeframe)
            
            training_time = time.time() - start_time
            metrics['training_time'] = training_time
            metrics['timeframe'] = timeframe
            
            logger.info(f"⏱️ {timeframe} 训练耗时: {training_time:.2f}秒")
            
            # 返回metrics和训练数据
            return metrics, train_data_with_timeframe
            
        except Exception as e:
            logger.error(f"{timeframe} 单时间框架训练失败: {e}")
            raise
    
    async def predict(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """模型预测（需指定时间框架）"""
        try:
            
            # 检查该时间框架的模型是否加载
            if timeframe not in self.models or self.models[timeframe] is None:
                logger.warning(f"⚠️ {timeframe} 模型未加载，尝试加载模型")
                await self._load_model()
                
                if timeframe not in self.models or self.models[timeframe] is None:
                    raise Exception(f"{timeframe} 模型不可用")
            
            # 特征工程 (移至线程池运行，避免阻塞事件循环导致的死锁)
            logger.debug(f"📊 {timeframe} 特征工程...")
            loop = asyncio.get_running_loop()
            processed_data = await loop.run_in_executor(
                None,
                lambda: self.feature_engineer.create_features(data.copy(), loop=loop)
            )
            
            if processed_data.empty:
                raise Exception("特征工程后数据为空")
            
            
            # 获取最新一行数据（使用该时间框架的特征列）
            feature_columns = self.feature_columns_dict.get(timeframe, [])
            if not feature_columns:
                raise Exception(f"{timeframe} 特征列未找到")
            
            latest_data = processed_data.iloc[-1:][feature_columns]
            
            # ✅ 调试日志：验证输入数据
            if 'close' in processed_data.columns:
                last_3_closes = processed_data['close'].tail(3).tolist()
            
            
            # ✅ 记录关键特征值（用于诊断）
            if 'price_change' in latest_data.columns:
                price_change = latest_data['price_change'].iloc[0]
            
            # 数据预处理（使用该时间框架的scaler）
            X_scaled = self._scale_features(latest_data, timeframe=timeframe, fit=False)
            
            # 预测（使用该时间框架的模型）
            model = self.models[timeframe]
            probabilities = model.predict_proba(X_scaled)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # 转换预测结果
            signal_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
            signal_type = signal_map[prediction]
            
            # 简洁记录预测结果（使用图标+中文）
            logger.info(f"🎯 {timeframe} 预测: {format_signal_type(signal_type)} (置信度={confidence:.4f}, 概率: 📉{probabilities[0]:.2f} ⏸️{probabilities[1]:.2f} 📈{probabilities[2]:.2f})")
            
            result = {
                'signal_type': signal_type,
                'confidence': float(confidence),
                'probabilities': {
                    'short': float(probabilities[0]),
                    'hold': float(probabilities[1]),
                    'long': float(probabilities[2])
                },
                'timestamp': datetime.now(),
                'model_version': self.model_metrics.get('version', '1.0')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {timeframe} 模型预测失败: {e}", exc_info=True)
            return {}
    
    async def _prepare_training_data_for_timeframe(self, timeframe: str) -> pd.DataFrame:
        """为单个时间框架准备训练数据（差异化训练天数）"""
        try:
            symbol = settings.SYMBOL
            
            # 🔑 超短线训练天数配置：确保足够的高频样本
            training_days_config = {
                '3m': 120,   # 超短期：120天（57,600条）
                '5m': 120,   # 主时间框架：120天（34,560条）
                '15m': 120   # 趋势确认：120天（11,520条）
            }
            training_days = training_days_config.get(timeframe, 120)
            
            # 时间周期对应的分钟数
            interval_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
                '12h': 720, '1d': 1440
            }
            
            # 根据时间周期计算需要的K线数量
            minutes = interval_minutes.get(timeframe, 60)
            required_klines = int((training_days * 24 * 60) / minutes)
            
            logger.info(f"📥 获取 {timeframe} 数据: {required_klines}条K线 ({training_days}天)")
            
            # ✅ 统一使用分页方法（自动处理超过1500的情况，支持多交易所）
            all_klines = self.exchange_client.get_klines_paginated(
                symbol=symbol,
                interval=timeframe,
                limit=required_klines,
                rate_limit_delay=0.1
            )
            
            # 转换为DataFrame（不依赖reverse，直接用时间戳排序）
            df = pd.DataFrame(all_klines)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 🔑 关键：依赖时间戳排序，而不是假设API返回顺序
                df = df.sort_values('timestamp', ascending=True)  # 明确指定升序（旧→新）
                
                # ✅ 去重（防止批次边界重复）
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # 🔥 过滤未完成的K线（volume=0表示K线未完成）
                rows_before_filter = len(df)
                if 'volume' in df.columns:
                    df = df[df['volume'] > 0]
                    filtered_count = rows_before_filter - len(df)
                    if filtered_count > 0:
                        logger.warning(f"⚠️ 过滤掉{filtered_count}条未完成K线（volume=0）")
                
                # 设置索引
                df = df.set_index('timestamp')
                
                logger.info(f"✅ {timeframe} 数据获取成功: {len(df)}条（已过滤未完成K线）")
            else:
                logger.warning(f"⚠️ {timeframe} 数据为空")
            
            return df
            
        except Exception as e:
            logger.error(f"为{timeframe}准备训练数据失败: {e}")
            return pd.DataFrame()
    
    
    async def _save_training_data_to_db(self, all_data: list):
        """将训练数据保存到数据库（供前端展示）
        
        优化：合并所有时间框架，一次性批量写入，减少数据库连接压力
        
        Args:
            all_data: List of DataFrames with 'timeframe' column
        """
        try:
            logger.info("📥 开始将训练数据写入数据库...")
            
            # 🔥 优化：先收集所有数据，然后一次性写入
            all_klines = []
            
            for df in all_data:
                if df is None or df.empty:
                    logger.warning("跳过空DataFrame")
                    continue
                
                # 检查timeframe列是否存在
                if 'timeframe' not in df.columns:
                    logger.warning(f"DataFrame缺少timeframe列，跳过: {df.columns.tolist()}")
                    continue
                
                # 获取时间框架
                timeframe = df['timeframe'].iloc[0]
                logger.info(f"  处理 {timeframe} 数据...")
                
                # 移除timeframe列，准备写入
                df_to_save = df.drop('timeframe', axis=1).copy()
                
                # 转换为字典列表（批量处理）
                for idx, row in df_to_save.iterrows():
                    try:
                        kline = {
                            'symbol': settings.SYMBOL,
                            'interval': timeframe,
                            'timestamp': int(idx.timestamp() * 1000),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                            'close_time': int(idx.timestamp() * 1000),
                            'quote_volume': float(row.get('quote_volume', 0)),
                            'trades': int(row.get('trades', 0)),
                            'taker_buy_base_volume': float(row.get('taker_buy_base_volume', 0)),
                            'taker_buy_quote_volume': float(row.get('taker_buy_quote_volume', 0))
                        }
                        all_klines.append(kline)
                    except Exception as e:
                        logger.warning(f"跳过无效行: {e}")
                        continue
                
                logger.info(f"  ✓ {timeframe} 准备完成: {len(df_to_save)}条")
            
            # 🔥 一次性写入所有数据（减少连接池压力）
            if all_klines:
                logger.info(f"📊 开始一次性写入所有数据: {len(all_klines)}条...")
                try:
                    await postgresql_manager.write_kline_data(all_klines)
                    logger.info(f"✅ 训练数据写入数据库完成: 总计{len(all_klines)}条")
                except Exception as e:
                    logger.error(f"  ✗ 数据库写入失败: {e}")
                    raise
            else:
                logger.warning("没有有效数据可写入")
            
        except Exception as e:
            logger.error(f"训练数据写入数据库失败: {e}")
            
            return df
            
    def _create_labels(self, df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
        """
        创建标签：使用三重障碍法 (Triple Barrier Method)
        
        不再仅仅预测下一根K线，而是预测未来一段时间内价格路径是先触及止盈线(Top)还是止损线(Bottom)。
        如果在时间窗口内均未触及，则标记为HOLD。
        
        Args:
            df: K线数据
            timeframe: 时间框架
            
        Returns:
            df: 包含 'label' 列的DataFrame (0: SHORT, 1: HOLD, 2: LONG)
        """
        try:
            # 1. 参数配置 (根据时间框架调整)
            # 时间窗口 (预测未来多少根K线)
            window = LABEL_WINDOW_CONFIG.get(timeframe, 20)
            pt_mult, sl_mult = LABEL_PT_SL_CONFIG.get(timeframe, (2.0, 1.5))
            
            # 2. 计算波动率 (使用ATR或Rolling Std)
            # 这里简单使用Rolling Std of Returns (Close-to-Close)
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=LABEL_VOLATILITY_WINDOW).std()
            
            # 处理NaN波动率 (使用均值填充)
            vol_mean = volatility.mean() if not volatility.isna().all() else LABEL_VOLATILITY_DEFAULT
            volatility = volatility.fillna(vol_mean)
            
            # 确保波动率有一个下限，防止死市时的极小阈值
            volatility = volatility.clip(lower=LABEL_VOLATILITY_MIN)
            
            # 3. 向量化计算三重障碍
            # 这种方法避免了慢速Python循环
            
            close_prices = df['close'].values
            n_samples = len(df)
            
            # 初始化标签为 HOLD (1)
            labels = np.ones(n_samples, dtype=int)
            
            # 为了提高效率，我们只在一个合理的lookahead窗口内检查
            # 构建未来价格矩阵: shape (n_samples, window)
            # future_prices[i, j] = price at time i + j + 1
            future_prices = np.full((n_samples, window), np.nan)
            
            for i in range(1, window + 1):
                # shift(-i) 将未来的数据以前移
                future_prices[:, i-1] = df['close'].shift(-i).values
                
            # 计算相对于当前价格的收益率矩阵
            # returns_matrix[i, j] = (price[i+j+1] - price[i]) / price[i]
            current_prices = close_prices.reshape(-1, 1)
            returns_matrix = (future_prices - current_prices) / current_prices
            
            # 动态阈值矩阵 (n_samples, 1)
            vol_array = volatility.values.reshape(-1, 1)
            upper_thresholds = vol_array * pt_mult
            lower_thresholds = -vol_array * sl_mult
            
            # 识别触碰 (Boolean Matrices)
            # 触碰上界
            hit_upper = returns_matrix > upper_thresholds
            # 触碰下界
            hit_lower = returns_matrix < lower_thresholds
            
            # 找到每次触碰的时间点 (argmax返回第一个True的索引，全False返回0)
            any_hit_upper = hit_upper.any(axis=1)
            any_hit_lower = hit_lower.any(axis=1)
            
            first_upper_idx = np.argmax(hit_upper, axis=1)
            first_lower_idx = np.argmax(hit_lower, axis=1)
            
            # 逻辑判定
            # 1. 既没碰上也没碰下 -> HOLD (已初始化为1)
            
            # 2. 只碰上 -> LONG (2)
            mask_only_upper = any_hit_upper & (~any_hit_lower)
            labels[mask_only_upper] = 2
            
            # 3. 只碰下 -> SHORT (0)
            mask_only_lower = (~any_hit_upper) & any_hit_lower
            labels[mask_only_lower] = 0
            
            # 4. 都碰了 -> 看谁先碰到
            mask_both = any_hit_upper & any_hit_lower
            
            # 如果 upper_idx < lower_idx -> 先涨 -> LONG
            mask_upper_first = mask_both & (first_upper_idx < first_lower_idx)
            labels[mask_upper_first] = 2
            
            # 如果 lower_idx < upper_idx -> 先跌 -> SHORT
            mask_lower_first = mask_both & (first_lower_idx < first_upper_idx)
            labels[mask_lower_first] = 0
            
            df['label'] = labels
            
            # 清理：最后 window 行的数据无效，因为看不了未来（这些行全是HOLD，最好去掉以免误导）
            # 或者保留它们但明确知道它们是HOLD。通常训练时去掉。
            # 这里选择保留前n_samples-window行
            if len(df) > window:
                df = df.iloc[:-window]
            
            # ========================================
            # 标签分布统计与质量检查
            # ========================================
            label_counts = df['label'].value_counts().sort_index()
            total = len(df)
            
            short_count = label_counts.get(0, 0)
            hold_count = label_counts.get(1, 0)
            long_count = label_counts.get(2, 0)
            
            short_pct = short_count / total * 100
            hold_pct = hold_count / total * 100
            long_pct = long_count / total * 100
            
            logger.info(f"📊 {timeframe} 三重障碍标签分布 (Window={window}, PT={pt_mult}x, SL={sl_mult}x):")
            logger.info(f"  SHORT (0): {short_count:5d}条 ({short_pct:5.1f}%)")
            logger.info(f"  HOLD  (1): {hold_count:5d}条 ({hold_pct:5.1f}%)")
            logger.info(f"  LONG  (2): {long_count:5d}条 ({long_pct:5.1f}%)")
            
            # 质量检查
            if hold_pct > 60:
                logger.warning(f"⚠️ {timeframe} HOLD占比过高 ({hold_pct:.1f}%)，建议降低障碍系数")
            elif hold_pct < 20:
                logger.warning(f"⚠️ {timeframe} HOLD占比过低 ({hold_pct:.1f}%)，建议增加障碍系数或窗口")

            return df
            
        except Exception as e:
            logger.error(f"创建标签失败: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def _prepare_features_labels(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备特征和标签（使用模块函数）"""
        try:
            # 检查label列是否存在
            if 'label' not in df.columns:
                logger.error(f"{timeframe} DataFrame中缺少'label'列，无法准备训练数据")
                raise ValueError(f"{timeframe} DataFrame中缺少'label'列")
            
            exclude_cols = [
                'timestamp', 'datetime', 'open', 'high', 'low', 'close', 
                'volume', 'quote_volume', 'label', 'next_return'
            ]
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            y = df['label'].copy()
            
            if timeframe not in self.feature_columns_dict or not self.feature_columns_dict[timeframe]:
                selected_features = self._select_features_intelligent(
                    df[feature_cols], 
                    y, 
                    timeframe
                )
                self.feature_columns_dict[timeframe] = selected_features
                logger.info(f"{timeframe} 特征选择完成: {len(selected_features)}/{len(feature_cols)} 个特征")
            
            feature_columns = self.feature_columns_dict[timeframe]
            
            if not feature_columns or len(feature_columns) == 0:
                logger.error(f"{timeframe} 特征列为空，无法继续训练")
                raise Exception(f"{timeframe} 特征选择失败：没有可用特征")
            
            # 创建包含特征列和label列的DataFrame，供prepare_features_labels使用
            X_with_label = df[feature_columns + ['label']].copy()
            
            if 'index' in X_with_label.columns:
                X_with_label = X_with_label.drop(columns=['index'])
                logger.warning(f"{timeframe} 训练数据中移除了'index'列")
            
            return prepare_features_labels(X_with_label, feature_columns)
        except Exception as e:
            logger.error(f"准备特征和标签失败: {e}", exc_info=True)
            return pd.DataFrame(), pd.Series()
    
    def _select_features_intelligent(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        timeframe: str
    ) -> list:
        """
        智能特征选择（两阶段 + 动态预算）
        
        阶段1: Filter过滤零增益特征
        阶段2: 嵌入式选择 + 动态预算
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            timeframe: 时间框架
        
        Returns:
            选中的特征列表
        """
        try:
            n_samples = len(X)
            n_feats = len(X.columns)
            ratio = n_samples / n_feats if n_feats > 0 else 0
            
            # 1. 动态预算计算（根据样本/特征比）
            # 不同时间框架的最少样本数/特征系数
            # 🎯 调整策略：允许更多特征以达到50%准确率目标
            ratio_map = {
                '3m': 120,   # 允许更多特征
                '5m': 120,   # 允许更多特征
                '15m': 120   # 150→120，允许更多特征（34360/120=286个，取150）
            }
            k = ratio_map.get(timeframe, 100)
            # 🆕 Kim建议2: 保底8个特征，封顶150
            budget = max(8, min(int(n_samples / k), 150))
            
            logger.info(f"📊 {timeframe} 样本/特征比={ratio:.1f}, 动态预算={budget}个特征")
            
            # 2. 阶段①：Filter过滤零增益特征
            logger.info(f"🔍 阶段1: Filter零增益特征...")
            lgb_filter = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1  # 静默模式
            )
            lgb_filter.fit(X, y)
            
            # 获取特征重要性
            imp = lgb_filter.feature_importances_
            # 🆕 Kim建议1: 使用均值阈值，过滤噪音特征（如1e-6）
            imp_threshold = imp.mean() * FEATURE_IMPORTANCE_THRESHOLD_HIGH
            stage1_mask = imp > imp_threshold
            stage1_cols = X.columns[stage1_mask].tolist()
            
            filtered_count = n_feats - len(stage1_cols)
            logger.info(f"✅ 过滤了{filtered_count}个低重要性特征(<{imp_threshold:.6f}), 剩余{len(stage1_cols)}个")
            
            # 🔧 修复：如果过滤后特征数为0，使用更宽松的阈值
            if len(stage1_cols) == 0:
                logger.warning(f"⚠️ {timeframe} 特征选择后剩余0个特征，使用更宽松的阈值（均值的1%）")
                imp_threshold = imp.mean() * FEATURE_IMPORTANCE_THRESHOLD_LOW
                stage1_mask = imp > imp_threshold
                stage1_cols = X.columns[stage1_mask].tolist()
                filtered_count = n_feats - len(stage1_cols)
                logger.info(f"✅ 宽松过滤后剩余{len(stage1_cols)}个特征")
                
                # 如果仍然为0，使用最低阈值（保留至少前N个特征）
                if len(stage1_cols) == 0:
                    logger.warning(f"⚠️ {timeframe} 宽松过滤后仍为0，保留重要性最高的{min(budget, n_feats)}个特征")
                    # 按重要性排序，取前budget个
                    imp_sorted = sorted(enumerate(imp), key=lambda x: x[1], reverse=True)
                    top_indices = [idx for idx, _ in imp_sorted[:min(budget, n_feats)]]
                    stage1_cols = [X.columns[i] for i in top_indices]
                    logger.info(f"✅ 保留重要性最高的{len(stage1_cols)}个特征")
            
            # 🆕 Kim建议4: 释放内存
            del lgb_filter
            gc.collect()
            
            # 如果过滤后特征数已经<=预算，直接返回
            if len(stage1_cols) <= budget:
                logger.info(f"✅ {timeframe} 特征数已满足预算，跳过阶段2")
                return stage1_cols
            
            # 3. 阶段②：嵌入式选择（基于预算）
            logger.info(f"🔍 阶段2: 嵌入式选择Top {budget}...")
            selector = SelectFromModel(
                lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                max_features=budget,
                threshold=-np.inf,  # 只受max_features限制
                importance_getter='auto',
                prefit=False  # 🆕 Kim建议3: 显式声明，保持可控
            )
            
            selector.fit(X[stage1_cols], y)
            stage2_mask = selector.get_support()
            selected_cols = [stage1_cols[i] for i, selected in enumerate(stage2_mask) if selected]
            
            # 🆕 Kim建议4: 释放内存
            del selector
            gc.collect()
            
            # 计算最终样本/特征比
            final_ratio = n_samples / len(selected_cols) if len(selected_cols) > 0 else 0
            
            logger.info(f"✅ {timeframe} 两阶段特征选择完成:")
            logger.info(f"   原始: {n_feats}个 → Filter: {len(stage1_cols)}个 → 最终: {len(selected_cols)}个")
            logger.info(f"   样本数: {n_samples}, 样本/特征比: {final_ratio:.1f}")
            
            # 🆕 Kim建议5: 过拟合警戒线
            if final_ratio < 50:
                logger.warning(f"⚠️ {timeframe} 样本/特征比 <50，建议调大k值或增加外部正则化")
            elif final_ratio < 100:
                logger.warning(f"⚠️ {timeframe} 样本/特征比 <100，注意监控过拟合")
            
            return selected_cols
            
        except Exception as e:
            logger.error(f"智能特征选择失败: {e}")
            logger.error(traceback.format_exc())
            
            # 降级方案：使用简单的top_n选择
            logger.warning(f"⚠️ 降级到简单特征选择...")
            top_n = {'3m': 100, '5m': 100, '15m': 100}.get(timeframe, 80)
            feature_importance = self.feature_engineer.get_feature_importance(X)
            selected = list(feature_importance.keys())[:top_n]
            return selected
    
    def _scale_features(self, X: pd.DataFrame, timeframe: str, fit: bool = False) -> np.ndarray:
        """特征缩放（多时间框架独立Scaler）
        
        Args:
            X: 特征DataFrame
            timeframe: 时间框架（必需）
            fit: 是否拟合新的scaler
            
        Returns:
            缩放后的特征数组
        """
        try:
            # 🔥 第一步：检查并处理无穷大值（Critical Fix）
            inf_count = 0
            nan_count = 0
            large_count = 0
            
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # 检查inf
                    inf_mask = np.isinf(X[col])
                    if inf_mask.any():
                        inf_count += inf_mask.sum()
                        # 用该列的最大有限值替换正无穷，最小有限值替换负无穷
                        finite_values = X.loc[~inf_mask, col]
                        if len(finite_values) > 0:
                            X.loc[X[col] == np.inf, col] = finite_values.max()
                            X.loc[X[col] == -np.inf, col] = finite_values.min()
                        else:
                            X.loc[inf_mask, col] = 0  # 如果没有有限值，用0填充
                    
                    # 检查过大值（可能导致缩放时溢出）
                    large_value_threshold = 1e15
                    large_mask = np.abs(X[col]) > large_value_threshold
                    if large_mask.any():
                        large_count += large_mask.sum()
                        # 限制在阈值范围内
                        X.loc[large_mask & (X[col] > 0), col] = large_value_threshold
                        X.loc[large_mask & (X[col] < 0), col] = -large_value_threshold
                    
                    # 检查NaN
                    nan_mask = X[col].isna()
                    if nan_mask.any():
                        nan_count += nan_mask.sum()
                        X.loc[nan_mask, col] = X[col].median()  # 用中位数填充NaN
            
            if inf_count > 0:
                logger.warning(f"⚠️ 特征缩放前处理了{inf_count}个无穷大值（inf）")
            if large_count > 0:
                logger.warning(f"⚠️ 特征缩放前处理了{large_count}个过大值（>1e15）")
            if nan_count > 0:
                logger.warning(f"⚠️ 特征缩放前处理了{nan_count}个缺失值（NaN）")
            
            # 🔧 修复：检查特征数据是否为空
            if X.empty or len(X.columns) == 0:
                logger.error(f"❌ {timeframe} 特征数据为空，无法进行缩放")
                raise ValueError(f"{timeframe} 特征数据为空，无法进行缩放")
            
            # 每个时间框架独立的scaler
            # 🔧 修复：支持字典结构的scaler（用于Informer-2）
            if fit or timeframe not in self.scalers or self.scalers[timeframe] is None:
                # 创建新的scaler（如果是字典结构，需要创建'traditional'键）
                if isinstance(self.scalers.get(timeframe), dict):
                    # 如果已经是字典，创建新的traditional scaler
                    if self.scalers[timeframe] is None:
                        self.scalers[timeframe] = {}
                    self.scalers[timeframe]['traditional'] = StandardScaler()
                    X_scaled = self.scalers[timeframe]['traditional'].fit_transform(X)
                else:
                    # 直接创建StandardScaler对象
                    self.scalers[timeframe] = StandardScaler()
                    X_scaled = self.scalers[timeframe].fit_transform(X)
            else:
                # 检查scaler是字典还是StandardScaler对象
                scaler = self.scalers[timeframe]
                if isinstance(scaler, dict):
                    # 字典结构：使用'traditional'键的scaler（传统模型）
                    if 'traditional' in scaler:
                        actual_scaler = scaler['traditional']
                        # 🔧 修复：处理scaler期望的无效列（如'index'）
                        X_scaled = self._transform_with_scaler_adapter(actual_scaler, X, timeframe)
                    else:
                        # 如果没有'traditional'键，创建新的scaler
                        logger.warning(f"⚠️ {timeframe} scaler字典中缺少'traditional'键，创建新的scaler")
                        self.scalers[timeframe]['traditional'] = StandardScaler()
                        X_scaled = self.scalers[timeframe]['traditional'].fit_transform(X)
                else:
                    # StandardScaler对象：直接使用
                    X_scaled = self._transform_with_scaler_adapter(scaler, X, timeframe)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"特征缩放失败: {e}", exc_info=True)
            return X.values
    
    def _transform_with_scaler_adapter(self, scaler, X: pd.DataFrame, timeframe: str) -> np.ndarray:
        """
        适配器：处理scaler期望的特征列与输入X不匹配的情况
        
        问题：旧模型训练时scaler期望包含'index'等无效列，但预测时已过滤掉
        解决：检查scaler期望的特征名，移除无效列，确保X只包含有效列
        """
        try:
            # 检查scaler是否有feature_names_in_属性（sklearn 0.24+）
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = list(scaler.feature_names_in_)
                invalid_cols = {'index', 'timestamp', 'date', 'label', 'target'}
                
                # 过滤掉scaler期望的无效列
                valid_expected = [f for f in expected_features if f not in invalid_cols]
                
                # 检查X中是否有scaler期望但无效的列
                missing_valid = [f for f in valid_expected if f not in X.columns]
                if missing_valid:
                    logger.error(f"❌ {timeframe} scaler期望的有效列缺失: {missing_valid[:5]}{'...' if len(missing_valid) > 5 else ''}")
                    raise ValueError(f"特征列不匹配：缺少 {missing_valid[:3]}")
                
                # 确保X只包含scaler期望的有效列，且顺序一致
                X_aligned = X[valid_expected].copy()
                
                # 如果scaler期望的列数多于有效列数（说明有无效列），需要调整
                if len(expected_features) != len(valid_expected):
                    removed = set(expected_features) - set(valid_expected)
                    logger.warning(f"⚠️ {timeframe} scaler适配：移除了{len(removed)}个无效期望列 {removed}")
                
                return scaler.transform(X_aligned)
            else:
                # 旧版本sklearn或没有feature_names_in_，直接transform
                return scaler.transform(X)
        except ValueError as e:
            # 如果仍然失败，尝试重新fit（作为最后手段）
            if "feature names" in str(e).lower() or "number of features" in str(e).lower():
                logger.warning(f"⚠️ {timeframe} scaler特征不匹配，尝试重新fit")
                # 创建新的scaler并fit
                new_scaler = StandardScaler()
                return new_scaler.fit_transform(X)
            raise
    
    # 注：_train_lightgbm() 方法已移至 ensemble_ml_service.py（统一三模型训练代码位置）
    # 原实现已被子类覆盖，此处删除以避免代码冗余
    
    def _evaluate_model_for_timeframe(self, X_val: np.ndarray, y_val: np.ndarray, timeframe: str) -> Dict[str, Any]:
        """评估特定时间框架的模型"""
        try:
            model = self.models.get(timeframe)
            if not model:
                raise Exception(f"{timeframe} 模型不存在")
            
            # 预测
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            # 计算指标
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            # 多分类AUC
            try:
                auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
            except:
                auc = 0.0
            
            # 特征重要性
            feature_columns = self.feature_columns_dict.get(timeframe, [])
            feature_importance = dict(zip(
                feature_columns, 
                model.feature_importances_
            ))
            
            # 按重要性排序，并转换为Python原生float
            feature_importance = {
                k: float(v) for k, v in sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            }
            
            # ✅ 转换所有numpy类型为Python原生类型（防止JSON序列化错误）
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc),
                'feature_importance': feature_importance,
                'timeframe': timeframe,
                'training_time': datetime.now().isoformat(),
                'version': '2.0'  # 多时间框架版本
            }
            
            logger.info(f"📊 {timeframe} 模型评估:")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  精确率: {precision:.4f}")
            logger.info(f"  召回率: {recall:.4f}")
            logger.info(f"  F1分数: {f1:.4f}")
            logger.info(f"  AUC: {auc:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"{timeframe} 模型评估失败: {e}")
            return {}
    
    async def _save_model(self):
        """保存多时间框架模型"""
        try:
            saved_count = 0
            for timeframe in settings.TIMEFRAMES:
                if timeframe in self.models:
                    paths = self._get_model_paths(timeframe)
                    
                    # 保存模型
                    joblib.dump(self.models[timeframe], paths['model'])
                    
                    # 保存缩放器
                    if timeframe in self.scalers:
                        joblib.dump(self.scalers[timeframe], paths['scaler'])
                    
                    # 保存特征列
                    if timeframe in self.feature_columns_dict:
                        with open(paths['features'], 'wb') as f:
                            pickle.dump(self.feature_columns_dict[timeframe], f)
                    
                    saved_count += 1
                    logger.info(f"✅ {timeframe} 模型保存完成")
            
            logger.info(f"🎉 所有模型保存完成 ({saved_count}个时间框架)")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    async def _load_model(self):
        """加载多时间框架模型"""
        try:
            loaded_count = 0
            for timeframe in settings.TIMEFRAMES:
                paths = self._get_model_paths(timeframe)
                
                try:
                    # 加载模型
                    if os.path.exists(paths['model']):
                        self.models[timeframe] = joblib.load(paths['model'])
                        loaded_count += 1
                    
                    # 加载缩放器
                    if os.path.exists(paths['scaler']):
                        self.scalers[timeframe] = joblib.load(paths['scaler'])
                    
                    # 加载特征列
                    if os.path.exists(paths['features']):
                        with open(paths['features'], 'rb') as f:
                            raw_features = pickle.load(f)
                            # 过滤掉无效列（如'index'等非特征列）
                            invalid_cols = {'index', 'timestamp', 'date', 'label', 'target'}
                            cleaned_features = [f for f in raw_features if f not in invalid_cols]
                            if len(cleaned_features) != len(raw_features):
                                removed = set(raw_features) - set(cleaned_features)
                                logger.warning(f"⚠️ {timeframe} 特征列过滤: 移除了无效列 {removed}")
                            self.feature_columns_dict[timeframe] = cleaned_features
                    
                    if timeframe in self.models:
                        feature_count = len(self.feature_columns_dict.get(timeframe, []))
                
                except Exception as e:
                    logger.warning(f"⚠️ {timeframe} 模型加载失败: {e}")
            
            if loaded_count > 0:
                logger.info(f"🎉 模型加载完成 ({loaded_count}/{len(settings.TIMEFRAMES)}个时间框架)")
            else:
                logger.warning("⚠️ 未找到已保存的模型，需要训练")
            
            # 加载模型指标
            cached_metrics = await cache_manager.get_model_metrics(settings.SYMBOL)
            if cached_metrics:
                self.model_metrics = cached_metrics
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    # 注意：自动训练循环已移除，改由 scheduler 统一管理
    # scheduler 会在每天00:01调用 train_model() 方法
    
    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息（多时间框架版本）"""
        try:
            # 统计已加载的模型
            loaded_models = {}
            total_features = 0
            
            for timeframe in settings.TIMEFRAMES:
                is_loaded = timeframe in self.models and self.models[timeframe] is not None
                feature_count = len(self.feature_columns_dict.get(timeframe, []))
                
                loaded_models[timeframe] = {
                    'loaded': is_loaded,
                    'feature_count': feature_count,
                    'model_path': self._get_model_paths(timeframe)['model'] if is_loaded else None
                }
                
                if is_loaded:
                    total_features += feature_count
            
            info = {
                'models_loaded': loaded_models,
                'total_models': len(self.models),
                'expected_models': len(settings.TIMEFRAMES),
                'total_features': total_features,
                'metrics': self.model_metrics,
                'last_training': self.model_metrics.get('training_date', 'Unknown'),
                'version': self.model_metrics.get('version', '2.0')
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {
                'total_models': 0,
                'expected_models': len(settings.TIMEFRAMES),
                'error': str(e)
            }