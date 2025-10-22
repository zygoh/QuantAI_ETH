"""
机器学习服务
"""
import asyncio
import logging
import pickle
import os
import gc
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import joblib

from app.core.config import settings
from app.core.database import postgresql_manager
from app.core.cache import cache_manager
from app.services.feature_engineering import feature_engineer
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

class MLService:
    """机器学习服务（支持多时间框架独立模型）"""
    
    def __init__(self):
        self.is_running = False
        # 多时间框架模型：{'15m': model, '2h': model, '4h': model}
        self.models = {}
        self.scalers = {}
        self.feature_columns_dict = {}
        
        # 🔑 初始化特征工程器（修复：子类需要访问）
        self.feature_engineer = feature_engineer
        self.model_metrics = {}
        self.training_task = None
        self.is_first_training = True  # 标记是否首次训练（只有首次才写数据库）
        
        # 模型参数
        # LightGBM基础参数（所有时间框架共享）
        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,  # 0: 下跌, 1: 横盘, 2: 上涨
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 500,  # ✅ 增加训练轮数（默认100→500）
            'num_leaves': 31,  # 默认值，训练时会根据时间框架调整
            'learning_rate': 0.03,  # ✅ 降低学习率（0.05→0.03），更稳定
            'feature_fraction': 0.85,  # ✅ 特征采样（防止过拟合）
            'bagging_fraction': 0.85,  # ✅ 数据采样（防止过拟合）
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
            'max_depth': 8,  # ✅ 限制树深度（防止过拟合）
            'min_child_samples': 30,  # ✅ 增加叶子节点最小样本（20→30）
            'reg_alpha': 0.3,  # ✅ 增强L1正则化（0.1→0.3）
            'reg_lambda': 0.3,  # ✅ 增强L2正则化（0.1→0.3）
            'min_split_gain': 0.01,  # ✅ 最小分裂增益（防止过度分裂）
            'is_unbalance': True  # 自动处理不平衡类别
        }
        
        # ✅ 差异化配置：根据新数据量调整模型复杂度
        self.lgb_params_by_timeframe = {
            '15m': {
                'num_leaves': 110,       # 95→110，适度增加复杂度
                'min_child_samples': 45,  # 50→45，略微放松
                'max_depth': 8,          # 7→8，增加深度
                'reg_alpha': 0.4,        # 0.5→0.4，略微放松L1正则
                'reg_lambda': 0.4        # 0.5→0.4，略微放松L2正则
            },
            '2h': {
                'num_leaves': 63,        # 数据增加(5,184条)，增加复杂度
                'min_child_samples': 30,  # 每叶子约82条样本
                'max_depth': 9
            },
            '4h': {
                'num_leaves': 47,        # 数据增加(3,456条)，增加复杂度
                'min_child_samples': 30,  # 每叶子约73条样本
                'max_depth': 8
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
    
    def _get_model_paths(self, timeframe: str) -> Dict[str, str]:
        """获取指定时间框架的模型文件路径"""
        return {
            'model': os.path.join(self.model_dir, f"{settings.SYMBOL}_{timeframe}_model.pkl"),
            'scaler': os.path.join(self.model_dir, f"{settings.SYMBOL}_{timeframe}_scaler.pkl"),
            'features': os.path.join(self.model_dir, f"{settings.SYMBOL}_{timeframe}_features.pkl")
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
                    import traceback
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
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return {}
    
    async def _train_single_timeframe(self, timeframe: str) -> tuple:
        """训练单个时间框架的模型
        
        Returns:
            tuple: (metrics, training_data_with_timeframe) 
        """
        try:
            import time
            start_time = time.time()
            
            # 获取该时间框架的训练数据
            train_data = await self._prepare_training_data_for_timeframe(timeframe)
            
            if train_data.empty:
                raise Exception(f"{timeframe} 训练数据为空")
            
            
            # 保存原始训练数据（用于后续写入数据库）
            train_data_with_timeframe = train_data.copy()
            train_data_with_timeframe['timeframe'] = timeframe
            
            # 特征工程
            train_data = feature_engineer.create_features(train_data)
            
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
            
            # 特征工程
            logger.debug(f"📊 {timeframe} 特征工程...")
            processed_data = feature_engineer.create_features(data.copy())
            
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
            from app.utils.helpers import format_signal_type
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
            from app.services.binance_client import binance_client
            
            symbol = settings.SYMBOL
            
            # 差异化训练天数：平衡数据量与时效性
            # 原则：时间框架越长，数据衰减越快，不宜用太多历史数据
            training_days_config = {
                '15m': 360,  # 短期：360天（34,560条）← 短期模式稳定，可以用更多数据
                '2h': 270,   # 中期：270天（3,240条）← 适度减少，平衡时效性
                '4h': 360    # 长期：360天（2,160条）← 大幅减少，避免过时数据
            }
            training_days = training_days_config.get(timeframe, 360)
            
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
            
            # 分批获取数据（每次最多1500条）
            all_klines = []
            max_per_request = 1500
            batches_needed = (required_klines + max_per_request - 1) // max_per_request
            
            end_time = None  # 最新数据
            
            for batch in range(batches_needed):
                batch_limit = min(max_per_request, required_klines - len(all_klines))
                
                if batch_limit <= 0:
                    break
                
                klines = binance_client.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=batch_limit,
                    end_time=end_time
                )
                
                if klines:
                    all_klines.extend(klines)
                    # 设置下一批次的end_time为当前批次最早的时间 - 1ms
                    end_time = klines[0]['timestamp'] - 1
                    await asyncio.sleep(0.1)  # 避免API限流
                else:
                    break
            
            # 转换为DataFrame（不依赖reverse，直接用时间戳排序）
            df = pd.DataFrame(all_klines)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 🔑 关键：依赖时间戳排序，而不是假设API返回顺序
                df = df.sort_values('timestamp', ascending=True)  # 明确指定升序（旧→新）
                
                # ✅ 去重（防止批次边界重复）
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # 设置索引
                df = df.set_index('timestamp')
                
                logger.info(f"✅ {timeframe} 数据获取成功: {len(df)}条")
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
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            raise
    
    def _create_labels(self, df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
        """创建标签（动态阈值，基于ATR波动率自适应调整）
        
        Args:
            df: K线数据
            timeframe: 时间框架（用于差异化阈值配置）
        """
        try:
            # ✅ 修复：只看下一根K线（不是未来5根）
            df['next_return'] = df['close'].shift(-1) / df['close'] - 1
            
            # 🎯 动态阈值：基于ATR波动率自适应调整
            # 计算ATR（14周期）
            if 'atr' not in df.columns or df['atr'].isna().all():
                # 如果没有ATR特征，手动计算
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean()
            else:
                atr = df['atr']
            
            # ATR百分比（相对于价格）
            atr_pct = (atr / df['close']).rolling(window=50).mean()  # 50根K线平均
            
            # 基础阈值配置（中频交易：灵敏策略）
            base_threshold_config = {
                '15m': 0.001,   # 基础±0.1% ✅ 中频交易极度灵敏
                '2h': 0.0035,   # 基础±0.35%
                '4h': 0.005     # 基础±0.5%
            }
            
            base_threshold = base_threshold_config.get(timeframe, 0.010)
            
            # 动态调整系数（基于ATR波动率）
            # 如果波动率高 → 扩大阈值；波动率低 → 缩小阈值
            median_atr_pct = atr_pct.median()
            
            if pd.isna(median_atr_pct) or median_atr_pct == 0:
                # 降级为固定阈值
                up_threshold = base_threshold
                down_threshold = -base_threshold
                logger.info(f"⚠️ ATR计算失败，使用固定阈值: ±{base_threshold*100:.2f}%")
            else:
                # 动态调整：ATR高时放宽阈值，ATR低时收紧阈值
                # 调整范围：0.7x ~ 1.3x
                adjustment = np.clip(median_atr_pct / 0.005, 0.7, 1.3)  # 0.5%为基准
                
                up_threshold = base_threshold * adjustment
                down_threshold = -base_threshold * adjustment
                
                logger.info(f"🎯 {timeframe} 动态阈值: ±{up_threshold*100:.2f}% "
                          f"(基础={base_threshold*100:.2f}%, ATR调整={adjustment:.2f}x, "
                          f"ATR%={median_atr_pct*100:.3f}%)")
            
            # 创建分类标签
            conditions = [
                df['next_return'] <= down_threshold,  # 下跌 → SHORT
                (df['next_return'] > down_threshold) & (df['next_return'] < up_threshold),  # 横盘 → HOLD
                df['next_return'] >= up_threshold     # 上涨 → LONG
            ]
            
            choices = [0, 1, 2]  # 0: SHORT, 1: HOLD, 2: LONG
            df['label'] = np.select(conditions, choices, default=1)
            
            # 移除最后1行（没有next_return）
            df = df[:-1]
            
            # ✅ 详细的标签分布统计（生产环境必须监控）
            label_counts = df['label'].value_counts().sort_index()
            total = len(df)
            
            short_count = label_counts.get(0, 0)
            hold_count = label_counts.get(1, 0)
            long_count = label_counts.get(2, 0)
            
            logger.info(f"📊 {timeframe} 标签分布（阈值: ±{up_threshold*100:.2f}%）:")  # 改为.2f精确显示
            logger.info(f"  SHORT (0): {short_count:5d}条 ({short_count/total*100:5.1f}%)")
            logger.info(f"  HOLD  (1): {hold_count:5d}条 ({hold_count/total*100:5.1f}%)")
            logger.info(f"  LONG  (2): {long_count:5d}条 ({long_count/total*100:5.1f}%)")
            
            # 警告：类别严重不平衡
            if hold_count / total > 0.60:
                logger.warning(f"⚠️ {timeframe} HOLD类别占比过高 ({hold_count/total*100:.1f}%)，建议提高阈值")
            
            if short_count / total < 0.20 or long_count / total < 0.20:
                logger.warning(f"⚠️ {timeframe} LONG/SHORT类别过少，建议降低阈值")
            
            
            return df
            
        except Exception as e:
            logger.error(f"创建标签失败: {e}")
            return df
    
    def _prepare_features_labels(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备特征和标签（多时间框架独立特征）
        
        Args:
            df: 包含label列的DataFrame
            timeframe: 时间框架（必需）
            
        Returns:
            (X, y): 特征DataFrame和标签Series
        """
        try:
            # 排除非特征列
            exclude_cols = [
                'timestamp', 'datetime', 'open', 'high', 'low', 'close', 
                'volume', 'quote_volume', 'label', 'next_return'
            ]
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # 🔑 先提取标签（特征选择需要用到）
            y = df['label'].copy()
            
            # 为每个时间框架选择独立的重要特征（基于模型的两阶段选择）
            if timeframe not in self.feature_columns_dict or not self.feature_columns_dict[timeframe]:
                # 🆕 智能特征选择：基于LightGBM重要性的两阶段选择
                selected_features = self._select_features_intelligent(
                    df[feature_cols], 
                    y, 
                    timeframe
                )
                self.feature_columns_dict[timeframe] = selected_features
                logger.info(f"✅ {timeframe} 特征选择完成: {len(selected_features)}/{len(feature_cols)} 个特征")
            
            feature_columns = self.feature_columns_dict[timeframe]
            
            X = df[feature_columns].copy()
            
            # 移除包含NaN的行
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            logger.info(f"特征数量: {len(feature_columns)}, 样本数量: {len(X)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"准备特征和标签失败: {e}")
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
            from sklearn.feature_selection import SelectFromModel
            
            n_samples = len(X)
            n_feats = len(X.columns)
            ratio = n_samples / n_feats if n_feats > 0 else 0
            
            # 1. 动态预算计算（根据样本/特征比）
            # 不同时间框架的最少样本数/特征系数
            # 🎯 调整策略：允许更多特征以达到50%准确率目标
            ratio_map = {
                '15m': 120,  # 150→120，允许更多特征（34360/120=286个，取150）
                '2h': 80,    # 从150降低→允许更多特征（3040/80=38个）
                '4h': 50     # 从100降低→允许更多特征（1960/50=39个）
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
            imp_threshold = imp.mean() * 0.1  # 均值的10%
            stage1_mask = imp > imp_threshold
            stage1_cols = X.columns[stage1_mask].tolist()
            
            filtered_count = n_feats - len(stage1_cols)
            logger.info(f"✅ 过滤了{filtered_count}个低重要性特征(<{imp_threshold:.6f}), 剩余{len(stage1_cols)}个")
            
            # 🆕 Kim建议4: 释放内存
            import gc
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
            import traceback
            logger.error(traceback.format_exc())
            
            # 降级方案：使用简单的top_n选择
            logger.warning(f"⚠️ 降级到简单特征选择...")
            top_n = {'15m': 100, '2h': 80, '4h': 60}.get(timeframe, 80)
            feature_importance = feature_engineer.get_feature_importance(X)
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
            # 每个时间框架独立的scaler
            if fit or timeframe not in self.scalers or self.scalers[timeframe] is None:
                self.scalers[timeframe] = StandardScaler()
                X_scaled = self.scalers[timeframe].fit_transform(X)
            else:
                X_scaled = self.scalers[timeframe].transform(X)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"特征缩放失败: {e}")
            return X.values
    
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
                            self.feature_columns_dict[timeframe] = pickle.load(f)
                    
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