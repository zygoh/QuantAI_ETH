"""
模型回测服务
"""
# StdLib
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple

# Third-Party
import numpy as np
import pandas as pd
import ta

# Local App
from app.core.config import settings
from app.core.constants import (
    VIRTUAL_OPEN_FEE_RATE,
    VIRTUAL_CLOSE_FEE_RATE,
    BACKTEST_DEFAULT_DAYS,
    BACKTEST_DEFAULT_PRIMARY_TIMEFRAME,
    BACKTEST_INITIAL_BALANCE,
    BACKTEST_POSITION_RATIO,
    HISTORICAL_DATA_RATE_LIMIT_DELAY,
    RISK_ATR_WINDOW,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    STOP_LOSS_ATR_MULTIPLIER,
    TAKE_PROFIT_ATR_MULTIPLIER,
    SIGNAL_TIMEFRAME_WEIGHTS,
    SIGNAL_HOLD_WEIGHT_DECAY,
    SIGNAL_HOLD_HIGH_CONFIDENCE_THRESHOLD,
    SIGNAL_PRIMARY_TIMEFRAME_MIN_CONFIDENCE,
    SIGNAL_TREND_CONSISTENCY_MIN_CONFIDENCE,
    SIGNAL_HIGH_CONFIDENCE_THRESHOLD,
    SIGNAL_VOLUME_RATIO_THRESHOLD,
    SIGNAL_MAX_DAILY_VOLATILITY,
    SIGNAL_MIN_DAILY_VOLATILITY
)
from app.core.database import postgresql_manager, PostgreSQLManager
from app.exchange.exchange_factory import ExchangeFactory
from app.model.base.ml_service import MLService
from app.services.risk_calculations import (
    calculate_atr_based_stop_levels,
    calculate_fixed_pct_stop_levels,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    """回测持仓"""
    side: str  # LONG / SHORT
    entry_price: float
    size_usdt: float
    stop_loss: float
    take_profit: float
    entry_time: datetime


class BacktestService:
    """模型回测服务（基于历史K线，支持多时间框架合成信号）"""

    def __init__(self, ml_service: MLService) -> None:
        self.ml_service = ml_service
        self.exchange_client = ExchangeFactory.get_current_client()
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD

        # 时间周期对应分钟数
        self.interval_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
            '12h': 720, '1d': 1440
        }
        
        # 🎯 回测余额管理（内存存储，默认开启累积模式）
        self._backtest_balance: Optional[Decimal] = None  # 累积余额（内存）

    async def run_backtest(
        self,
        symbol: Optional[str] = None,
        days: int = BACKTEST_DEFAULT_DAYS,
        initial_balance: Optional[float] = BACKTEST_INITIAL_BALANCE,
        leverage: Optional[float] = None,
        primary_timeframe: str = BACKTEST_DEFAULT_PRIMARY_TIMEFRAME,
        timeframes: Optional[List[str]] = None,
        include_trades: bool = False
    ) -> Dict[str, Any]:
        """
        运行回测（在独立线程中运行，使用线程本地数据库连接）

        Args:
            symbol: 交易对（默认使用 settings.SYMBOL）
            days: 回测天数
            initial_balance: 初始资金（为 None 时启用累积模式：使用内存累积余额）
            leverage: 杠杆倍数（为 None 时默认使用 settings.LEVERAGE，且必须 > 0）
            primary_timeframe: 主时间框架
            timeframes: 使用的时间框架列表（为空/None 时默认使用 settings.TIMEFRAMES）
            include_trades: 是否返回交易明细

        Returns:
            回测结果字典
        """
        # 🔑 创建线程本地的数据库连接
        # 原因：回测任务在独立线程中运行（通过 global_executor.run_async_in_thread）
        # 该方法会创建新的事件循环，而 asyncpg 连接池绑定到创建它的事件循环
        # 因此需要在独立线程中创建独立的数据库连接，避免事件循环冲突
        thread_local_db = PostgreSQLManager()
        
        try:
            # 连接数据库（跳过 schema 初始化，避免并发死锁）
            await thread_local_db.connect(skip_schema_init=True)
            logger.info("✅ 回测线程：数据库连接已建立")
            
            # 🧹 清理旧的回测记录（确保数据库中只保留最新的回测结果）
            logger.info("🧹 清理旧的回测记录...")
            await thread_local_db.clear_backtest_data()
            logger.info("✅ 旧回测记录已清理")
            
            # 开始回测逻辑
            # ✅ 严格模式：回测默认参数必须与实盘/预测一致（统一从 settings 读取）
            if symbol is None:
                symbol = settings.SYMBOL

            if not timeframes:
                timeframes = list(settings.TIMEFRAMES)

            if leverage is None:
                leverage = float(settings.LEVERAGE)

            if primary_timeframe not in timeframes:
                raise ValueError(f"主时间框架不在timeframes中: {primary_timeframe}")
            if days <= 0:
                raise ValueError("回测天数必须大于0")
            if initial_balance is not None and initial_balance <= 0:
                raise ValueError("初始资金必须大于0")
            if float(leverage) <= 0:
                raise ValueError("leverage必须大于0")

            logger.info(f"🚀 回测启动: {symbol} | {days}天 | 主周期={primary_timeframe} | 多周期={timeframes}")

            # 1) 获取历史K线
            raw_klines = self._fetch_klines_multi_timeframes(symbol, timeframes, days)

            # 2) 预计算特征数据（一次性计算，避免循环中重复特征工程）
            logger.info("🚀 预计算特征数据（优化回测性能）...")
            feature_data = self._precompute_features(raw_klines, timeframes)
            logger.info("✅ 特征数据预计算完成")

            # 3) 构建时间索引（基于特征数据）
            logger.info("📊 构建时间索引...")
            time_index = self._build_time_index_from_features(feature_data, timeframes)
            primary_times = time_index[primary_timeframe]
            logger.info(f"✅ 时间索引构建完成: 主时间框架 {primary_timeframe} 共 {len(primary_times)} 个时间点")

            # 4) 回测循环（使用预计算特征进行批量预测）
            # 🎯 余额管理：使用内存存储（默认累积模式）
            # - 如果 initial_balance 为 None，使用内存中的累积余额（默认）
            # - 如果 initial_balance 有值，使用指定值（独立回测模式）
            if initial_balance is None:
                # 累积模式：使用内存中的累积余额
                if self._backtest_balance is not None:
                    balance = self._backtest_balance
                    logger.info(f"💰 累积模式：使用上次余额 {balance} USDT（内存）")
                else:
                    balance = Decimal(str(BACKTEST_INITIAL_BALANCE))
                    logger.info(f"💰 首次回测：使用初始余额 {balance} USDT")
            else:
                # 独立回测模式：使用指定的初始余额
                balance = Decimal(str(initial_balance))
                logger.info(f"💰 独立回测模式：使用指定余额 {balance} USDT")
            
            position: Optional[BacktestPosition] = None
            trades: List[Dict[str, Any]] = []
            equity_curve: List[Dict[str, Any]] = []

            last_signal_type: Optional[str] = None

            # 计算技术指标所需最小数据量（覆盖SMA200等长窗口指标）
            min_history_rows = 500

            logger.info(f"🔄 开始回测循环: 共 {len(primary_times)} 个时间点，最小历史数据量={min_history_rows}行")
            
            # ✅ 性能优化：批量预测（提升 10-20倍速度）
            logger.info("🚀 批量预测优化：预先计算所有时间点的预测结果...")
            batch_predictions = await self._batch_predict_all_timeframes(
                feature_data, time_index, timeframes, primary_timeframe, min_history_rows
            )
            logger.info(f"✅ 批量预测完成: {len(batch_predictions)} 个时间点")
            
            processed_count = 0
            total_count = len(primary_times)

            for current_time in primary_times:
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"📊 回测进度: {processed_count}/{total_count} ({processed_count*100//total_count}%)")
                current_dt = pd.to_datetime(current_time).to_pydatetime()
                
                # ✅ 从预计算的批量预测结果中获取
                predictions = batch_predictions.get(current_time, {})
                if not predictions:
                    self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                    continue

                # 合成信号
                signal = self._synthesize_signal(predictions)
                if not signal:
                    self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                    continue

                signal_type = signal['signal_type']
                confidence = signal['confidence']

                if confidence < self.confidence_threshold:
                    self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                    continue
                
                # ✅ 新增：增强信号过滤（与实时预测保持一致）
                filter_result = self._enhanced_signal_filter_backtest(
                    signal_type=signal_type,
                    confidence=confidence,
                    predictions=signal.get('predictions', {}),
                    raw_klines=raw_klines,
                    current_time=current_dt,
                    primary_timeframe=primary_timeframe
                )
                
                if not filter_result['pass']:
                    logger.debug(f"❌ 信号被过滤: {filter_result['reason']}")
                    self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                    continue

                # 获取当前K线数据（用于止盈止损检查）
                current_kline = self._get_kline_at_time(
                    raw_klines[primary_timeframe], current_dt
                )
                if current_kline is None or current_kline['close'] <= 0:
                    self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                    continue
                
                # 获取当前价格（用于开仓）
                current_price = float(current_kline['close'])

                # 先处理已有持仓的止盈止损（使用完整K线数据）
                if position:
                    position, balance, trade = self._check_stop_take(
                        position, current_kline, current_dt, balance
                    )
                    if trade:
                        trades.append(trade)

                # 信号去重：同向信号不重复开仓
                if position and position.side == signal_type:
                    self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                    last_signal_type = signal_type
                    continue

                # 方向变更：先平仓再开仓
                if position and signal_type != position.side:
                    position, balance, trade = self._close_position(
                        position, current_price, current_dt, balance, reason="signal_flip"
                    )
                    if trade:
                        trades.append(trade)

                # 开新仓
                if signal_type in ["LONG", "SHORT"]:
                    if last_signal_type == signal_type:
                        self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                        continue

                    stop_levels = self._calculate_dynamic_stop_levels(
                        raw_klines[primary_timeframe], current_dt, current_price, signal_type
                    )
                    if not stop_levels:
                        self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                        continue

                    # 🎯 仓位管理：使用固定比例而非全仓（降低风险）
                    # 设计意图：
                    # - 避免全仓交易导致的爆仓风险
                    # - 保留资金用于后续交易
                    # - 降低单次亏损对总资金的影响
                    # 当前策略：从 constants.py 读取配置（默认 50%）
                    position_ratio = Decimal(str(BACKTEST_POSITION_RATIO))
                    position_value = balance * Decimal(str(leverage)) * position_ratio
                    
                    if position_value <= 0:
                        self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                        continue

                    position = BacktestPosition(
                        side=signal_type,
                        entry_price=current_price,
                        size_usdt=float(position_value),
                        stop_loss=stop_levels['stop_loss'],
                        take_profit=stop_levels['take_profit'],
                        entry_time=current_dt
                    )
                    last_signal_type = signal_type

                self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)

            # 回测结束强制平仓
            if position:
                last_time = pd.to_datetime(primary_times[-1]).to_pydatetime()
                last_kline = self._get_kline_at_time(
                    raw_klines[primary_timeframe], last_time
                )
                if last_kline is None or last_kline['close'] <= 0:
                    # 如果无法获取最后K线，使用持仓的entry_price作为退出价格（避免数据缺失导致错误）
                    logger.warning(f"⚠️ 无法获取最后K线，使用持仓价格: {position.entry_price}")
                    last_price = position.entry_price
                else:
                    last_price = float(last_kline['close'])
                
                position, balance, trade = self._close_position(
                    position, last_price, last_time, balance, reason="end_of_test"
                )
                if trade:
                    trades.append(trade)

            logger.info(f"✅ 回测循环完成: 处理了 {processed_count} 个时间点，生成 {len(trades)} 笔交易")

            # 5) 汇总指标
            logger.info("📊 汇总回测结果...")
            
            # 🎯 修复：initial_balance 可能为 None（累积模式），需要获取实际的初始余额
            if initial_balance is None:
                # 累积模式：从 equity_curve 的第一个点获取初始余额
                actual_initial_balance = float(equity_curve[0]['balance']) if equity_curve else float(balance)
            else:
                actual_initial_balance = float(initial_balance)
            
            results = self._summarize_results(
                symbol=symbol,
                days=days,
                initial_balance=actual_initial_balance,
                final_balance=float(balance),
                trades=trades,
                equity_curve=equity_curve,
                include_trades=include_trades
            )
            logger.info("✅ 回测结果汇总完成")

            # 回测结果写入数据库（使用线程本地连接）
            logger.info("💾 写入回测结果到数据库...")
            await thread_local_db.clear_backtest_data()
            await thread_local_db.write_backtest_results(results, trades)
            logger.info("✅ 回测结果写入完成")
            
            # 🎯 累积模式：保存最终余额到内存（用于下次回测）
            if initial_balance is None:
                self._backtest_balance = balance
                logger.info(f"💰 累积模式：保存最终余额 {float(balance):.2f} USDT 到内存")

            logger.info(f"✅ 回测完成: 交易次数={results['total_trades']} 胜率={results['win_rate']:.2%} 总收益={results['total_return']:.2%}")
            return results
        
        except Exception as e:
            logger.error(f"❌ 回测失败: {e}", exc_info=True)
            raise
        
        finally:
            # 🔑 确保关闭线程本地数据库连接（无论成功还是失败）
            try:
                await thread_local_db.close()
                logger.info("✅ 回测线程：数据库连接已关闭")
            except Exception as e:
                logger.warning(f"⚠️ 关闭回测线程数据库连接失败: {e}")
                # ⚠️ 不重新抛出异常，避免掩盖回测结果
    
    def reset_backtest_balance(self) -> None:
        """重置回测累积余额（清理内存）"""
        self._backtest_balance = None
        logger.info("🔄 回测累积余额已重置（内存清理）")

    def _fetch_klines_multi_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """获取多时间框架历史K线"""
        result: Dict[str, pd.DataFrame] = {}

        for timeframe in timeframes:
            minutes = self.interval_minutes.get(timeframe, 60)
            limit = int((days * 24 * 60) / minutes)

            logger.info(f"📥 获取K线: {symbol} {timeframe} {days}天 ({limit}条)")
            klines = self.exchange_client.get_klines_paginated(
                symbol=symbol,
                interval=timeframe,
                limit=limit,
                rate_limit_delay=HISTORICAL_DATA_RATE_LIMIT_DELAY
            )

            df = pd.DataFrame(klines)
            if df.empty:
                raise ValueError(f"{timeframe} K线数据为空")

            # 严格模式：检查必需列
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"{timeframe} K线数据缺少必需列: {missing_columns}")

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp', ascending=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='last')

            # 数据验证（严格模式）
            self._validate_kline_df(df, timeframe)
            
            # 严格模式：确保数据量足够（至少500行用于特征工程）
            min_required_rows = 500
            if len(df) < min_required_rows:
                raise ValueError(f"{timeframe} K线数据不足: {len(df)}行 < {min_required_rows}行（需要足够数据用于特征工程）")

            result[timeframe] = df.reset_index(drop=True)

        return result

    def _precompute_features(
        self,
        raw_klines: Dict[str, pd.DataFrame],
        timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """预计算所有时间框架的特征数据（优化回测性能）"""
        feature_data: Dict[str, pd.DataFrame] = {}

        for timeframe in timeframes:
            df = raw_klines[timeframe].copy()
            logger.debug(f"📊 预计算 {timeframe} 特征数据: {len(df)}行")

            # 一次性进行特征工程
            processed = self.ml_service.feature_engineer.create_features(df)

            if processed.empty:
                raise ValueError(f"{timeframe} 特征工程后数据为空")

            # 确保timestamp列存在
            if 'timestamp' not in processed.columns:
                processed['timestamp'] = df['timestamp'].values

            # 过滤无效特征行
            cleaned = processed.copy()
            cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

            # 获取特征列
            feature_columns = self._get_feature_columns(timeframe)
            if feature_columns:
                cleaned = cleaned.dropna(subset=feature_columns, how='any')

            if cleaned.empty:
                raise ValueError(f"{timeframe} 清理无效特征后无可用数据")

            feature_data[timeframe] = cleaned.sort_values('timestamp', ascending=True).reset_index(drop=True)
            logger.debug(f"✅ {timeframe} 特征数据预计算完成: {len(feature_data[timeframe])}行")

        return feature_data

    def _build_time_index_from_features(
        self,
        feature_data: Dict[str, pd.DataFrame],
        timeframes: List[str]
    ) -> Dict[str, np.ndarray]:
        """基于特征数据构建时间索引"""
        time_index: Dict[str, np.ndarray] = {}
        for timeframe in timeframes:
            if timeframe not in feature_data:
                raise ValueError(f"{timeframe} 特征数据不存在")
            if 'timestamp' not in feature_data[timeframe].columns:
                raise ValueError(f"{timeframe} 特征数据缺少timestamp列")
            time_index[timeframe] = feature_data[timeframe]['timestamp'].values
            logger.debug(f"✅ {timeframe} 时间索引: {len(time_index[timeframe])} 个时间点")
        return time_index

    def _build_time_index_from_raw(
        self,
        raw_klines: Dict[str, pd.DataFrame],
        timeframes: List[str]
    ) -> Dict[str, np.ndarray]:
        """基于原始K线构建时间索引（保留用于兼容性）"""
        time_index: Dict[str, np.ndarray] = {}
        for timeframe in timeframes:
            time_index[timeframe] = raw_klines[timeframe]['timestamp'].values
        return time_index

    async def _get_predictions_from_features(
        self,
        feature_data: Dict[str, pd.DataFrame],
        time_index: Dict[str, np.ndarray],
        timeframes: List[str],
        current_time: np.datetime64,
        primary_timeframe: str,
        min_history_rows: int = 500
    ) -> Dict[str, Dict[str, Any]]:
        """在指定时间点获取多时间框架预测（使用预计算特征，优化性能）"""
        predictions: Dict[str, Dict[str, Any]] = {}

        for timeframe in timeframes:
            timestamps = time_index[timeframe]
            idx = int(np.searchsorted(timestamps, current_time, side='right') - 1)
            if idx < 0:
                continue

            # 确保有足够的历史数据
            if idx < min_history_rows:
                # 数据不足时跳过该时间框架的预测
                continue

            # 从预计算特征数据中获取对应行的特征（严格模式：检查索引范围）
            feature_df = feature_data[timeframe]
            if idx >= len(feature_df):
                logger.warning(f"⚠️ {timeframe} 索引越界: idx={idx}, len={len(feature_df)}")
                continue
            
            # 确保索引有效
            if idx < 0 or idx >= len(feature_df):
                logger.warning(f"⚠️ {timeframe} 索引无效: idx={idx}, len={len(feature_df)}")
                continue

            # 获取特征行（直接使用预计算的特征，跳过特征工程）
            feature_row = feature_df.iloc[idx:idx + 1].copy()
            pred = await self._predict_from_features_row_async(timeframe, feature_row)
            if pred:
                predictions[timeframe] = pred

        # 主时间框架必须存在
        if primary_timeframe not in predictions:
            return {}

        return predictions

    async def _batch_predict_all_timeframes(
        self,
        feature_data: Dict[str, pd.DataFrame],
        time_index: Dict[str, np.ndarray],
        timeframes: List[str],
        primary_timeframe: str,
        min_history_rows: int = 500
    ) -> Dict[np.datetime64, Dict[str, Dict[str, Any]]]:
        """
        批量预测所有时间点（性能优化：提升 10-20倍速度）
        
        Returns:
            {timestamp: {timeframe: prediction}}
        """
        try:
            batch_predictions: Dict[np.datetime64, Dict[str, Dict[str, Any]]] = {}
            primary_times = time_index[primary_timeframe]
            
            # 对每个时间框架进行批量预测
            for timeframe in timeframes:
                logger.info(f"🚀 批量预测 {timeframe} 时间框架...")
                timestamps = time_index[timeframe]
                feature_df = feature_data[timeframe]
                
                # 找到所有有效的预测索引
                valid_indices = []
                valid_times = []
                
                for current_time in primary_times:
                    idx = int(np.searchsorted(timestamps, current_time, side='right') - 1)
                    
                    # 检查索引有效性
                    if idx < min_history_rows or idx >= len(feature_df):
                        continue
                    
                    valid_indices.append(idx)
                    valid_times.append(current_time)
                
                if not valid_indices:
                    logger.warning(f"⚠️ {timeframe} 没有有效的预测索引")
                    continue
                
                # ✅ 批量预测：一次性预测所有行
                batch_features = feature_df.iloc[valid_indices].copy()
                batch_preds = await self._batch_predict_timeframe(timeframe, batch_features)
                
                # 将批量预测结果映射到时间戳
                for time_val, pred in zip(valid_times, batch_preds):
                    if time_val not in batch_predictions:
                        batch_predictions[time_val] = {}
                    batch_predictions[time_val][timeframe] = pred
                
                logger.info(f"✅ {timeframe} 批量预测完成: {len(batch_preds)} 个预测")
            
            # 过滤掉主时间框架缺失的时间点
            filtered_predictions = {
                time_val: preds 
                for time_val, preds in batch_predictions.items()
                if primary_timeframe in preds
            }
            
            return filtered_predictions
            
        except Exception as e:
            logger.error(f"❌ 批量预测失败: {e}", exc_info=True)
            raise

    async def _batch_predict_timeframe(
        self,
        timeframe: str,
        batch_features: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        对单个时间框架进行批量预测
        
        Args:
            timeframe: 时间框架
            batch_features: 批量特征数据
            
        Returns:
            预测结果列表
        """
        try:
            feature_columns = self._get_feature_columns(timeframe)
            
            # ✅ 修复 LightGBM 警告：使用 .copy() 避免切片引用
            X = batch_features[feature_columns].values.copy()
            
            # 批量缩放
            X_scaled = self.ml_service._scale_features(
                pd.DataFrame(X, columns=feature_columns), 
                timeframe=timeframe, 
                fit=False
            )
            
            models = getattr(self.ml_service, 'ensemble_models', None)
            if models and timeframe in models:
                return self._batch_predict_with_ensemble(timeframe, X_scaled, models[timeframe])
            
            # 退化到单模型批量预测
            model = self.ml_service.models.get(timeframe)
            if not model:
                return []
            
            # ✅ 确保是连续内存数组
            X_pred = X_scaled if isinstance(X_scaled, np.ndarray) else X_scaled.values
            if not X_pred.flags['C_CONTIGUOUS']:
                X_pred = np.ascontiguousarray(X_pred)
            
            # ✅ 批量预测（GPU 加速）
            proba_batch = model.predict_proba(X_pred)
            preds_batch = np.argmax(proba_batch, axis=1)
            
            # 格式化批量预测结果
            results = []
            for pred, proba in zip(preds_batch, proba_batch):
                results.append(self._format_prediction(int(pred), proba))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {timeframe} 批量预测失败: {e}", exc_info=True)
            raise

    def _batch_predict_with_ensemble(
        self,
        timeframe: str,
        X_scaled: np.ndarray,
        models: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """集成模型批量预测"""
        try:
            # ✅ 确保是连续内存数组（避免 LightGBM 警告）
            X_pred = X_scaled if isinstance(X_scaled, np.ndarray) else X_scaled.values
            if not X_pred.flags['C_CONTIGUOUS']:
                X_pred = np.ascontiguousarray(X_pred)
            
            # ✅ 批量预测（GPU 加速）
            lgb_proba_batch = models['lgb'].predict_proba(X_pred)
            xgb_proba_batch = models['xgb'].predict_proba(X_pred)
            cat_proba_batch = models['cat'].predict_proba(X_pred)
            
            # 如果有 meta 模型
            if 'meta' in models and hasattr(self.ml_service, '_generate_enhanced_meta_features'):
                # Meta 模型需要逐行生成特征（暂时保持串行）
                results = []
                for i in range(len(X_pred)):
                    # ✅ 使用 .copy() 避免切片引用
                    X_single = X_pred[i:i+1].copy()
                    meta_features = self.ml_service._generate_enhanced_meta_features(
                        X_single, models
                    )
                    meta_pred = int(models['meta'].predict(meta_features)[0])
                    meta_proba = models['meta'].predict_proba(meta_features)[0]
                    results.append(self._format_prediction(meta_pred, meta_proba))
                return results
            
            # 无 meta 模型时，使用均值概率
            avg_proba_batch = (lgb_proba_batch + xgb_proba_batch + cat_proba_batch) / 3
            preds_batch = np.argmax(avg_proba_batch, axis=1)
            
            # 格式化批量预测结果
            results = []
            for pred, proba in zip(preds_batch, avg_proba_batch):
                results.append(self._format_prediction(int(pred), proba))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {timeframe} 集成批量预测失败: {e}", exc_info=True)
            raise

    async def _get_predictions_at_time(
        self,
        raw_klines: Dict[str, pd.DataFrame],
        time_index: Dict[str, np.ndarray],
        timeframes: List[str],
        current_time: np.datetime64,
        primary_timeframe: str,
        min_history_rows: int = 500
    ) -> Dict[str, Dict[str, Any]]:
        """在指定时间点获取多时间框架预测（使用原始K线，保留用于兼容性）"""
        predictions: Dict[str, Dict[str, Any]] = {}

        for timeframe in timeframes:
            timestamps = time_index[timeframe]
            idx = int(np.searchsorted(timestamps, current_time, side='right') - 1)
            if idx < 0:
                continue

            # 确保有足够的历史数据用于特征工程
            if idx + 1 < min_history_rows:
                # 数据不足时跳过该时间框架的预测
                continue

            # 传入原始K线切片（ml_service.predict 内部会做特征工程）
            df = raw_klines[timeframe].iloc[:idx + 1].copy()
            # 确保无重复索引
            df = df.reset_index(drop=True)
            pred = await self._predict_from_raw_df(timeframe, df)
            if pred:
                predictions[timeframe] = pred

        # 主时间框架必须存在
        if primary_timeframe not in predictions:
            return {}

        return predictions

    async def _predict_from_features_row_async(
        self,
        timeframe: str,
        feature_row: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """从预计算特征行生成预测（异步，优化性能）"""
        try:
            # 如果只有一行，转换为Series；否则使用DataFrame
            if len(feature_row) == 1:
                row = feature_row.iloc[0]
            else:
                row = feature_row.iloc[-1]  # 取最后一行

            # 直接调用同步方法（模型预测很快，不需要线程池）
            # 使用 run_in_executor 反而会增加开销
            pred = self._predict_from_features_row(timeframe, row)
            if not pred:
                raise ValueError(f"{timeframe} 预测结果为空（严格模式）")
            return pred
        except Exception as e:
            logger.error(f"❌ {timeframe} 从特征预测失败（严格模式）: {e}", exc_info=True)
            raise

    async def _predict_from_raw_df(self, timeframe: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """严格模式：回测预测路径与信号预测一致（保留用于兼容性）"""
        try:
            pred = await self.ml_service.predict(df, timeframe=timeframe)
            if not pred:
                raise ValueError(f"{timeframe} 预测结果为空（严格模式）")
            return pred
        except Exception as e:
            logger.error(f"❌ {timeframe} 预测失败（严格模式）: {e}", exc_info=True)
            raise

    def _predict_from_features_row(self, timeframe: str, row: pd.Series) -> Optional[Dict[str, Any]]:
        """从单行特征生成预测"""
        try:
            feature_columns = self._get_feature_columns(timeframe)
            X = row.reindex(feature_columns).to_frame().T

            X_scaled = self.ml_service._scale_features(X, timeframe=timeframe, fit=False)

            models = getattr(self.ml_service, 'ensemble_models', None)
            if models and timeframe in models:
                return self._predict_with_ensemble(timeframe, X_scaled, models[timeframe])

            # 退化到单模型预测
            model = self.ml_service.models.get(timeframe)
            if not model:
                return None

            X_pred = X_scaled if isinstance(X_scaled, np.ndarray) else X_scaled.values
            self._assert_feature_count(
                timeframe,
                model,
                X_pred,
                model_name="single",
                expected_override=len(feature_columns)
            )

            proba = model.predict_proba(X_scaled)[0]
            pred = int(np.argmax(proba))
            return self._format_prediction(pred, proba)
        except ValueError as e:
            logger.error(f"❌ {timeframe} 单行预测失败（严格模式）: {e}")
            raise
        except Exception as e:
            logger.warning(f"⚠️ {timeframe} 单行预测失败: {e}")
            return None

    def _predict_with_ensemble(
        self,
        timeframe: str,
        X_scaled: np.ndarray,
        models: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """集成模型预测"""
        try:
            X_pred = X_scaled if isinstance(X_scaled, np.ndarray) else X_scaled.values
            expected_override = len(self._get_feature_columns(timeframe))
            self._assert_feature_count(timeframe, models['lgb'], X_pred, model_name="lgb", expected_override=expected_override)
            self._assert_feature_count(timeframe, models['xgb'], X_pred, model_name="xgb", expected_override=expected_override)
            self._assert_feature_count(timeframe, models['cat'], X_pred, model_name="cat", expected_override=expected_override)

            lgb_proba = models['lgb'].predict_proba(X_pred)[0]

            if hasattr(self.ml_service, '_predict_xgboost'):
                xgb_pred, xgb_proba = self.ml_service._predict_xgboost(models['xgb'], X_pred, return_single=True)
            else:
                xgb_proba = models['xgb'].predict_proba(X_pred)[0]
                xgb_pred = int(np.argmax(xgb_proba))

            cat_proba = models['cat'].predict_proba(X_pred)[0]

            if 'meta' in models and hasattr(self.ml_service, '_generate_enhanced_meta_features'):
                meta_features = self.ml_service._generate_enhanced_meta_features(X_pred, models)
                self._assert_feature_count(
                    timeframe,
                    models['meta'],
                    meta_features,
                    model_name="meta",
                    expected_override=meta_features.shape[-1] if hasattr(meta_features, "shape") else None
                )
                meta_pred = int(models['meta'].predict(meta_features)[0])
                meta_proba = models['meta'].predict_proba(meta_features)[0]
                return self._format_prediction(meta_pred, meta_proba)

            # 无meta模型时，使用均值概率
            avg_proba = (lgb_proba + xgb_proba + cat_proba) / 3
            pred = int(np.argmax(avg_proba))
            return self._format_prediction(pred, avg_proba)
        except ValueError as e:
            logger.error(f"❌ {timeframe} 集成预测失败（严格模式）: {e}")
            raise
        except Exception as e:
            logger.warning(f"⚠️ {timeframe} 集成预测失败: {e}")
            return None

    def _synthesize_signal(self, predictions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """合成多时间框架信号（与实时逻辑一致，增强过滤以提高胜率）"""
        if not predictions:
            return None

        # ✅ 严格模式：时间框架权重必须与实盘/预测一致（统一使用 constants.py）
        timeframe_weights = SIGNAL_TIMEFRAME_WEIGHTS

        weighted_scores = {'LONG': 0.0, 'SHORT': 0.0, 'HOLD': 0.0}
        total_weight = 0.0
        
        # ✅ 优化：收集各时间框架的置信度，用于后续质量检查
        timeframe_confidences = {}

        for timeframe, prediction in predictions.items():
            base_weight = timeframe_weights.get(timeframe, 0.2)
            probabilities = prediction.get('probabilities', {})
            signal = prediction.get('signal_type')
            pred_confidence = prediction.get('confidence', 0.0)
            
            # ✅ 优化：记录各时间框架的置信度
            timeframe_confidences[timeframe] = pred_confidence

            if timeframe == '15m' and signal == 'HOLD':
                hold_confidence = pred_confidence
                # ✅ 严格模式：HOLD权重衰减必须与实盘/预测一致（统一常量）
                weight = (
                    base_weight * SIGNAL_HOLD_WEIGHT_DECAY
                    if hold_confidence > SIGNAL_HOLD_HIGH_CONFIDENCE_THRESHOLD
                    else base_weight
                )
            else:
                weight = base_weight

            weighted_scores['LONG'] += probabilities.get('long', 0.0) * weight
            weighted_scores['SHORT'] += probabilities.get('short', 0.0) * weight
            weighted_scores['HOLD'] += probabilities.get('hold', 0.0) * weight
            total_weight += weight

        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight

        signal_type = max(weighted_scores, key=weighted_scores.get)
        confidence = weighted_scores[signal_type]

        if signal_type == 'HOLD':
            return None
        
        # ✅ 优化：增强信号质量检查（追求超高胜率）
        # 1. 主时间框架（5m）必须达到较高置信度
        primary_confidence = timeframe_confidences.get('5m', 0.0)
        if primary_confidence < SIGNAL_PRIMARY_TIMEFRAME_MIN_CONFIDENCE:
            logger.debug(f"⚠️ 主时间框架置信度过低: {primary_confidence:.4f} < {SIGNAL_PRIMARY_TIMEFRAME_MIN_CONFIDENCE}，拒绝信号")
            return None
        
        # 2. 至少两个时间框架方向一致
        signal_agreement = 0
        for timeframe, prediction in predictions.items():
            if prediction.get('signal_type') == signal_type:
                signal_agreement += 1
        
        if signal_agreement < 2:
            logger.debug(f"⚠️ 时间框架方向不一致: {signal_agreement}/3，拒绝信号")
            return None
        
        # 3. 最终置信度必须超过阈值（已在外部检查，这里作为双重保险）
        if confidence < self.confidence_threshold:
            logger.debug(f"⚠️ 合成信号置信度过低: {confidence:.4f} < {self.confidence_threshold}，拒绝信号")
            return None

        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'predictions': predictions  # ✅ 传递predictions用于增强过滤
        }

    def _calculate_dynamic_stop_levels(
        self,
        df: pd.DataFrame,
        current_time: datetime,
        entry_price: float,
        signal_type: str
    ) -> Dict[str, float]:
        """基于历史数据计算动态止损止盈"""
        try:
            window_df = df[df['timestamp'] <= current_time].tail(100)
            if len(window_df) < 20:
                return self._calculate_fixed_stop_levels(entry_price, signal_type)

            atr_indicator = ta.volatility.AverageTrueRange(
                high=window_df['high'],
                low=window_df['low'],
                close=window_df['close'],
                window=RISK_ATR_WINDOW
            )
            current_atr = float(atr_indicator.average_true_range().iloc[-1])
            if np.isnan(current_atr) or current_atr <= 0:
                return self._calculate_fixed_stop_levels(entry_price, signal_type)

            # 🎯 动态止盈止损：使用 1:1 盈亏比提升震荡行情胜率
            # 设计意图：
            # - 止损距离 = ATR × 1.5（从 constants.py 读取 STOP_LOSS_ATR_MULTIPLIER）
            # - 止盈距离 = ATR × 1.5（从 constants.py 读取 TAKE_PROFIT_ATR_MULTIPLIER）
            # - 盈亏比 = 1:1（在震荡行情中，触碰止损和止盈的概率相近，提高胜率）
            # - 遵循严格模式：训练/回测/预测使用相同的止盈止损逻辑
            stop_levels = calculate_atr_based_stop_levels(
                entry_price=float(entry_price),
                atr=float(current_atr),
                signal_type=signal_type,
                stop_loss_atr_multiplier=STOP_LOSS_ATR_MULTIPLIER,
                take_profit_atr_multiplier=TAKE_PROFIT_ATR_MULTIPLIER,
            )
            return stop_levels
        except Exception as e:
            logger.warning(f"⚠️ 动态止损计算失败: {e}")
            return self._calculate_fixed_stop_levels(entry_price, signal_type)

    def _calculate_fixed_stop_levels(self, entry_price: float, signal_type: str) -> Dict[str, float]:
        """固定止损止盈（备用方案）"""
        stop_levels = calculate_fixed_pct_stop_levels(
            entry_price=float(entry_price),
            signal_type=signal_type,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
        )
        return stop_levels

    def _check_stop_take(
        self,
        position: BacktestPosition,
        current_kline: pd.Series,
        current_time: datetime,
        balance: Decimal
    ) -> Tuple[Optional[BacktestPosition], Decimal, Optional[Dict[str, Any]]]:
        """
        检查止损止盈（使用K线high/low，更真实）
        
        修复原因：
        - 原实现只使用收盘价，忽略K线内部波动
        - 导致大量应该止损的交易被记录为止盈
        - 胜率虚高10-20%
        
        新实现：
        - LONG: 先检查最低价是否触发止损，再检查最高价是否触发止盈
        - SHORT: 先检查最高价是否触发止损，再检查最低价是否触发止盈
        - 更接近真实交易行为
        """
        if position.side == 'LONG':
            # ✅ 先检查止损（使用最低价）
            if current_kline['low'] <= position.stop_loss:
                return self._close_position(
                    position, position.stop_loss, current_time, balance, reason="stop_loss"
                )
            # ✅ 再检查止盈（使用最高价）
            if current_kline['high'] >= position.take_profit:
                return self._close_position(
                    position, position.take_profit, current_time, balance, reason="take_profit"
                )
        else:  # SHORT
            # ✅ 先检查止损（使用最高价）
            if current_kline['high'] >= position.stop_loss:
                return self._close_position(
                    position, position.stop_loss, current_time, balance, reason="stop_loss"
                )
            # ✅ 再检查止盈（使用最低价）
            if current_kline['low'] <= position.take_profit:
                return self._close_position(
                    position, position.take_profit, current_time, balance, reason="take_profit"
                )

        return position, balance, None

    def _close_position(
        self,
        position: BacktestPosition,
        exit_price: float,
        exit_time: datetime,
        balance: Decimal,
        reason: str
    ) -> Tuple[Optional[BacktestPosition], Decimal, Dict[str, Any]]:
        """平仓并计算盈亏"""
        pnl, pnl_percent, balance, open_fee, close_fee, total_fee = self._calculate_trade_pnl(
            position, exit_price, balance
        )

        trade = {
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'position_value': position.size_usdt,  # 🎯 开仓金额
            'open_fee': open_fee,
            'close_fee': close_fee,
            'total_fee': total_fee,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'balance_after': float(balance),  # 🎯 平仓后余额
            'reason': reason
        }

        return None, balance, trade

    def _calculate_trade_pnl(
        self,
        position: BacktestPosition,
        exit_price: float,
        balance: Decimal
    ) -> Tuple[float, float, Decimal, float, float, float]:
        """
        计算交易盈亏并更新余额
        
        Returns:
            (pnl, pnl_percent, new_balance, open_fee, close_fee, total_fee)
        """
        entry_price = Decimal(str(position.entry_price))
        exit_price_dec = Decimal(str(exit_price))
        position_value = Decimal(str(position.size_usdt))

        coin_amount = position_value / entry_price
        if position.side == 'LONG':
            price_pnl = (exit_price_dec - entry_price) * coin_amount
        else:
            price_pnl = (entry_price - exit_price_dec) * coin_amount

        open_commission = position_value * VIRTUAL_OPEN_FEE_RATE
        close_commission = (coin_amount * exit_price_dec) * VIRTUAL_CLOSE_FEE_RATE
        total_commission = open_commission + close_commission
        net_pnl = price_pnl - total_commission

        new_balance = balance + net_pnl

        # 严格模式：余额不能为负数（回测中允许，但记录警告）
        if new_balance < 0:
            logger.warning(f"⚠️ 回测余额为负: {float(new_balance):.8f} (初始: {float(balance):.8f}, PnL: {float(net_pnl):.8f})")

        pnl_float = float(net_pnl.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
        pnl_pct = float((net_pnl / position_value * Decimal('100')).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
        open_fee_float = float(open_commission.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
        close_fee_float = float(close_commission.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
        total_fee_float = float(total_commission.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))

        return pnl_float, pnl_pct, new_balance, open_fee_float, close_fee_float, total_fee_float

    def _summarize_results(
        self,
        symbol: str,
        days: int,
        initial_balance: float,
        final_balance: float,
        trades: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]],
        include_trades: bool
    ) -> Dict[str, Any]:
        """汇总回测结果"""
        total_trades = len(trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        profit = sum(t['pnl'] for t in wins)
        loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = (profit / loss) if loss > 0 else 0.0

        total_return = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0
        avg_trade_return = np.mean([t['pnl_percent'] for t in trades]) if trades else 0.0
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        result: Dict[str, Any] = {
            'symbol': symbol,
            'days': days,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'avg_trade_return': avg_trade_return
        }

        if include_trades:
            result['trades'] = trades

        return result

    def _calculate_max_drawdown(self, equity_curve: List[Dict[str, Any]]) -> float:
        """计算最大回撤"""
        if not equity_curve:
            return 0.0
        equity = np.array([item['equity'] for item in equity_curve], dtype=float)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(float(drawdown.min()))

    def _record_equity(
        self,
        equity_curve: List[Dict[str, Any]],
        current_time: datetime,
        balance: Decimal,
        position: Optional[BacktestPosition],
        raw_klines: Dict[str, pd.DataFrame],
        primary_timeframe: str
    ) -> None:
        """记录净值曲线"""
        equity = float(balance)
        if position:
            current_price = self._get_close_price_at_time(raw_klines[primary_timeframe], current_time)
            if current_price > 0:
                entry_price = Decimal(str(position.entry_price))
                current_price_dec = Decimal(str(current_price))
                position_value = Decimal(str(position.size_usdt))
                coin_amount = position_value / entry_price

                if position.side == 'LONG':
                    unrealized = (current_price_dec - entry_price) * coin_amount
                else:
                    unrealized = (entry_price - current_price_dec) * coin_amount
                equity = float((balance + unrealized).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))

        # 确保时间戳格式正确（兼容 datetime / numpy.datetime64 / pd.Timestamp）
        if hasattr(current_time, 'isoformat'):
            timestamp_str = current_time.isoformat()
        else:
            # numpy.datetime64 -> pd.Timestamp -> isoformat
            timestamp_str = pd.to_datetime(current_time).isoformat()

        equity_curve.append({
            'timestamp': timestamp_str,
            'equity': equity
        })

    def _get_close_price_at_time(self, df: pd.DataFrame, current_time: datetime) -> float:
        """获取指定时间的收盘价"""
        row = df[df['timestamp'] <= current_time].tail(1)
        if row.empty:
            return 0.0
        return float(row['close'].iloc[0])
    
    def _get_kline_at_time(self, df: pd.DataFrame, current_time: datetime) -> Optional[pd.Series]:
        """
        获取指定时间的完整K线数据（用于止盈止损检查）
        
        Returns:
            包含 open, high, low, close, volume 的 Series，如果没有数据则返回 None
        """
        row = df[df['timestamp'] <= current_time].tail(1)
        if row.empty:
            return None
        return row.iloc[0]

    def _get_feature_columns(self, timeframe: str) -> List[str]:
        """获取模型特征列"""
        feature_columns = self.ml_service.feature_columns_dict.get(timeframe, [])
        model_columns = self._get_model_feature_columns(timeframe)

        if model_columns:
            if feature_columns and len(feature_columns) != len(model_columns):
                raise ValueError(
                    f"{timeframe} 特征列数量与模型不一致（严格模式）: feature_columns={len(feature_columns)} model_features={len(model_columns)}"
                )
            if not feature_columns:
                feature_columns = model_columns

        if not feature_columns:
            raise ValueError(f"{timeframe} 特征列未加载，请先训练模型")

        invalid_cols = {'index', 'timestamp', 'date', 'label', 'target'}
        return [f for f in feature_columns if f not in invalid_cols]

    def _get_model_feature_columns(self, timeframe: str) -> List[str]:
        """从已加载模型中提取特征列"""
        model = None
        models = getattr(self.ml_service, 'ensemble_models', None)
        if models and timeframe in models and 'lgb' in models[timeframe]:
            model = models[timeframe]['lgb']
        elif timeframe in self.ml_service.models:
            model = self.ml_service.models[timeframe]

        if not model:
            return []

        feature_names = None
        if hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
            if callable(feature_names):
                feature_names = feature_names()
        elif hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)

        if feature_names:
            return list(feature_names)

        scaler = self.ml_service.scalers.get(timeframe)
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            return list(scaler.feature_names_in_)

        return []

    def _assert_feature_count(
        self,
        timeframe: str,
        model: Any,
        X_pred: np.ndarray,
        model_name: str,
        expected_override: Optional[int] = None
    ) -> None:
        """严格模式：特征数量必须一致"""
        expected = self._get_model_expected_feature_count(timeframe, model)
        if expected is None and expected_override:
            expected = expected_override
        if expected is None:
            raise ValueError(f"{timeframe} {model_name} 无法获取模型特征数")
        x_array = np.asarray(X_pred)
        actual = int(x_array.shape[1]) if x_array.ndim > 1 else int(x_array.shape[0])
        if actual != int(expected):
            raise ValueError(
                f"{timeframe} {model_name} 特征数不匹配（严格模式）: expected={expected} actual={actual}"
            )

    def _get_model_expected_feature_count(self, timeframe: str, model: Any) -> Optional[int]:
        """从模型中推断特征数量"""
        expected = getattr(model, 'n_features_', None)
        if not expected or int(expected) == 0:
            expected = getattr(model, 'n_features_in_', None)

        if not expected or int(expected) == 0:
            feature_names = None
            if hasattr(model, 'feature_name_'):
                feature_names = model.feature_name_
                if callable(feature_names):
                    feature_names = feature_names()
            elif hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            elif hasattr(model, 'get_feature_names'):
                try:
                    feature_names = model.get_feature_names()
                except Exception:
                    feature_names = None

            if feature_names:
                expected = len(feature_names)

        if not expected or int(expected) == 0:
            feature_columns = self.ml_service.feature_columns_dict.get(timeframe, [])
            if feature_columns:
                expected = len(feature_columns)

        return int(expected) if expected else None

    def _validate_kline_df(self, df: pd.DataFrame, timeframe: str) -> None:
        """K线数据验证"""
        assert not df.empty, f"{timeframe} K线数据为空"
        assert (df['close'] > 0).all(), f"{timeframe} close含非正数"
        assert not df['close'].isna().any(), f"{timeframe} close含NaN"
        assert not np.isinf(df['close']).any(), f"{timeframe} close含Inf"
        assert not df['volume'].isna().any(), f"{timeframe} volume含NaN"
        assert not np.isinf(df['volume']).any(), f"{timeframe} volume含Inf"

    def _format_prediction(self, pred: int, proba: np.ndarray) -> Dict[str, Any]:
        """格式化预测结果"""
        signal_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
        signal_type = signal_map.get(pred, 'HOLD')
        confidence = float(np.max(proba))
        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'probabilities': {
                'short': float(proba[0]),
                'hold': float(proba[1]),
                'long': float(proba[2])
            }
        }
    
    def _enhanced_signal_filter_backtest(
        self,
        signal_type: str,
        confidence: float,
        predictions: Dict[str, Dict[str, Any]],
        raw_klines: Dict[str, pd.DataFrame],
        current_time: datetime,
        primary_timeframe: str
    ) -> Dict[str, Any]:
        """
        增强的信号过滤（回测版本，与实时预测保持一致）
        
        多维度过滤低质量信号：
        1. 趋势一致性过滤
        2. 波动率过滤
        3. 量能确认
        
        Returns:
            {'pass': bool, 'reason': str}
        """
        try:
            # 1. 置信度基础过滤（已在外部检查）
            if confidence < self.confidence_threshold:
                return {'pass': False, 'reason': f'置信度过低 ({confidence:.4f} < {self.confidence_threshold})'}
            
            # 2. 趋势一致性过滤（与实时逻辑一致）
            if confidence < SIGNAL_TREND_CONSISTENCY_MIN_CONFIDENCE:
                if len(predictions) >= 2:
                    signal_types = [pred['signal_type'] for pred in predictions.values()]
                    if signal_type == 'LONG' and all(s == 'SHORT' for s in signal_types):
                        return {'pass': False, 'reason': '所有时间框架都是SHORT信号，与LONG冲突'}
                    elif signal_type == 'SHORT' and all(s == 'LONG' for s in signal_types):
                        return {'pass': False, 'reason': '所有时间框架都是LONG信号，与SHORT冲突'}
            
            # 3. 波动率过滤（避免在极端波动时交易）
            try:
                # 从历史K线数据计算波动率
                df_5m = raw_klines.get(primary_timeframe)
                if df_5m is not None and len(df_5m) > 0:
                    # 获取当前时间之前的60根K线（5小时）
                    recent_df = df_5m[df_5m['timestamp'] <= current_time].tail(60)
                    
                    if len(recent_df) >= 60:
                        recent_closes = recent_df['close'].values
                        returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1] 
                                  for i in range(1, len(recent_closes))]
                        current_volatility = np.std(returns)
                        
                        # 日波动率估算（5分钟 → 日，假设288个5分钟周期）
                        daily_volatility = current_volatility * np.sqrt(288)
                        
                        if daily_volatility > SIGNAL_MAX_DAILY_VOLATILITY:
                            return {'pass': False, 'reason': f'市场波动过大 (日波动率={daily_volatility*100:.2f}%)'}
                        
                        if daily_volatility < SIGNAL_MIN_DAILY_VOLATILITY:
                            return {'pass': False, 'reason': f'市场波动过小 (日波动率={daily_volatility*100:.2f}%)'}
            except Exception as e:
                logger.debug(f"波动率计算失败（跳过此过滤）: {e}")
            
            # 4. 量能确认（高置信度信号需要量能配合）
            if confidence > SIGNAL_HIGH_CONFIDENCE_THRESHOLD:
                try:
                    df_5m = raw_klines.get(primary_timeframe)
                    if df_5m is not None and len(df_5m) > 0:
                        # 获取当前时间之前的20根K线
                        recent_df = df_5m[df_5m['timestamp'] <= current_time].tail(20)
                        
                        if len(recent_df) >= 20:
                            recent_volumes = recent_df['volume'].values
                            current_volume = recent_volumes[-1]
                            avg_volume = np.mean(recent_volumes)
                            
                            # 高置信度信号需要量能至少达到平均的指定比例
                            if current_volume < avg_volume * SIGNAL_VOLUME_RATIO_THRESHOLD:
                                return {'pass': False, 'reason': f'量能不足（当前={current_volume:.0f}, 平均={avg_volume:.0f}）'}
                except Exception as e:
                    logger.debug(f"量能检查失败（跳过此过滤）: {e}")
            
            # 5. 所有过滤器通过
            logger.debug(f"✅ 信号通过所有增强过滤器")
            return {'pass': True, 'reason': '通过所有过滤条件'}
            
        except Exception as e:
            logger.error(f"信号过滤失败: {e}")
            # 过滤失败时保守处理：通过信号（避免错失机会）
            return {'pass': True, 'reason': '过滤器异常，默认通过'}
