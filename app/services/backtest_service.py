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
    BACKTEST_DEFAULT_LEVERAGE,
    BACKTEST_DEFAULT_PRIMARY_TIMEFRAME,
    BACKTEST_DEFAULT_SYMBOL,
    BACKTEST_DEFAULT_TIMEFRAMES,
    BACKTEST_INITIAL_BALANCE,
    HISTORICAL_DATA_RATE_LIMIT_DELAY,
    RISK_ATR_WINDOW,
    STOP_LOSS_PCT_FALLBACK,
    TAKE_PROFIT_PCT_FALLBACK
)
from app.core.database import postgresql_manager
from app.exchange.exchange_factory import ExchangeFactory
from app.model.base.ml_service import MLService

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

    async def run_backtest(
        self,
        symbol: str = BACKTEST_DEFAULT_SYMBOL,
        days: int = BACKTEST_DEFAULT_DAYS,
        initial_balance: float = BACKTEST_INITIAL_BALANCE,
        leverage: float = BACKTEST_DEFAULT_LEVERAGE,
        primary_timeframe: str = BACKTEST_DEFAULT_PRIMARY_TIMEFRAME,
        timeframes: Optional[List[str]] = None,
        include_trades: bool = False
    ) -> Dict[str, Any]:
        """
        运行回测

        Args:
            symbol: 交易对
            days: 回测天数
            initial_balance: 初始资金
            leverage: 杠杆倍数（>0）
            primary_timeframe: 主时间框架
            timeframes: 使用的时间框架列表（默认使用settings.TIMEFRAMES）
            include_trades: 是否返回交易明细

        Returns:
            回测结果字典
        """
        try:
            if not timeframes:
                timeframes = BACKTEST_DEFAULT_TIMEFRAMES

            if primary_timeframe not in timeframes:
                raise ValueError(f"主时间框架不在timeframes中: {primary_timeframe}")
            if days <= 0:
                raise ValueError("回测天数必须大于0")
            if initial_balance <= 0:
                raise ValueError("初始资金必须大于0")
            if leverage <= 0:
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
            balance = Decimal(str(initial_balance))
            position: Optional[BacktestPosition] = None
            trades: List[Dict[str, Any]] = []
            equity_curve: List[Dict[str, Any]] = []

            last_signal_type: Optional[str] = None

            # 计算技术指标所需最小数据量（覆盖SMA200等长窗口指标）
            min_history_rows = 500

            logger.info(f"🔄 开始回测循环: 共 {len(primary_times)} 个时间点，最小历史数据量={min_history_rows}行")
            processed_count = 0
            total_count = len(primary_times)

            for current_time in primary_times:
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"📊 回测进度: {processed_count}/{total_count} ({processed_count*100//total_count}%)")
                current_dt = pd.to_datetime(current_time).to_pydatetime()
                # 获取多时间框架预测（使用预计算特征）
                predictions = await self._get_predictions_from_features(
                    feature_data, time_index, timeframes, current_time, primary_timeframe, min_history_rows
                )
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

                # 获取当前价格
                current_price = self._get_close_price_at_time(
                    raw_klines[primary_timeframe], current_dt
                )
                if current_price <= 0:
                    self._record_equity(equity_curve, current_dt, balance, position, raw_klines, primary_timeframe)
                    continue

                # 先处理已有持仓的止盈止损
                if position:
                    position, balance, trade = self._check_stop_take(
                        position, current_price, current_dt, balance
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

                    position_value = balance * Decimal(str(leverage))
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
                last_price = self._get_close_price_at_time(
                    raw_klines[primary_timeframe], last_time
                )
                if last_price <= 0:
                    # 如果无法获取最后价格，使用持仓的entry_price作为退出价格（避免数据缺失导致错误）
                    logger.warning(f"⚠️ 无法获取最后价格，使用持仓价格: {position.entry_price}")
                    last_price = position.entry_price
                position, balance, trade = self._close_position(
                    position, last_price, last_time, balance, reason="end_of_test"
                )
                if trade:
                    trades.append(trade)

            logger.info(f"✅ 回测循环完成: 处理了 {processed_count} 个时间点，生成 {len(trades)} 笔交易")

            # 5) 汇总指标
            logger.info("📊 汇总回测结果...")
            results = self._summarize_results(
                symbol=symbol,
                days=days,
                initial_balance=float(initial_balance),
                final_balance=float(balance),
                trades=trades,
                equity_curve=equity_curve,
                include_trades=include_trades
            )
            logger.info("✅ 回测结果汇总完成")

            # 回测结果写入数据库（先清空历史）
            logger.info("💾 写入回测结果到数据库...")
            await postgresql_manager.clear_backtest_data()
            await postgresql_manager.write_backtest_results(results, trades)
            logger.info("✅ 回测结果写入完成")

            logger.info(f"✅ 回测完成: 交易次数={results['total_trades']} 胜率={results['win_rate']:.2%} 总收益={results['total_return']:.2%}")
            return results

        except Exception as e:
            logger.error(f"❌ 回测失败: {e}", exc_info=True)
            raise

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

        timeframe_weights = {
            '3m': 0.15,
            '5m': 0.70,
            '15m': 0.15
        }

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
                weight = base_weight * 0.5 if hold_confidence > 0.65 else base_weight
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
        if primary_confidence < 0.50:
            logger.debug(f"⚠️ 主时间框架置信度过低: {primary_confidence:.4f} < 0.50，拒绝信号")
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
            'confidence': confidence
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

            # ✅ 优化：调整止盈止损比例，提高胜率（追求超高胜率）
            # 原比例：止损 ATR*1.2，止盈 ATR*3.6（盈亏比3:1）
            # 新比例：止损 ATR*1.5（放宽止损，减少被止损概率），止盈 ATR*3.0（适度降低止盈，提高达成率）
            # 盈亏比仍为 2:1，但胜率会提高
            if signal_type == 'LONG':
                stop_loss = entry_price - (current_atr * 1.5)  # 放宽止损
                take_profit = entry_price + (current_atr * 3.0)  # 适度降低止盈
            else:
                stop_loss = entry_price + (current_atr * 1.5)  # 放宽止损
                take_profit = entry_price - (current_atr * 3.0)  # 适度降低止盈

            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        except Exception as e:
            logger.warning(f"⚠️ 动态止损计算失败: {e}")
            return self._calculate_fixed_stop_levels(entry_price, signal_type)

    def _calculate_fixed_stop_levels(self, entry_price: float, signal_type: str) -> Dict[str, float]:
        """固定止损止盈（备用方案）"""
        stop_loss_pct = STOP_LOSS_PCT_FALLBACK
        take_profit_pct = TAKE_PROFIT_PCT_FALLBACK

        if signal_type == 'LONG':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _check_stop_take(
        self,
        position: BacktestPosition,
        current_price: float,
        current_time: datetime,
        balance: Decimal
    ) -> Tuple[Optional[BacktestPosition], Decimal, Optional[Dict[str, Any]]]:
        """检查止损止盈"""
        if position.side == 'LONG':
            if current_price <= position.stop_loss:
                return self._close_position(position, current_price, current_time, balance, reason="stop_loss")
            if current_price >= position.take_profit:
                return self._close_position(position, current_price, current_time, balance, reason="take_profit")
        else:
            if current_price >= position.stop_loss:
                return self._close_position(position, current_price, current_time, balance, reason="stop_loss")
            if current_price <= position.take_profit:
                return self._close_position(position, current_price, current_time, balance, reason="take_profit")

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
        pnl, pnl_percent, balance = self._calculate_trade_pnl(
            position, exit_price, balance
        )

        trade = {
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'reason': reason
        }

        return None, balance, trade

    def _calculate_trade_pnl(
        self,
        position: BacktestPosition,
        exit_price: float,
        balance: Decimal
    ) -> Tuple[float, float, Decimal]:
        """计算交易盈亏并更新余额"""
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
        net_pnl = price_pnl - open_commission - close_commission

        new_balance = balance + net_pnl

        # 严格模式：余额不能为负数（回测中允许，但记录警告）
        if new_balance < 0:
            logger.warning(f"⚠️ 回测余额为负: {float(new_balance):.8f} (初始: {float(balance):.8f}, PnL: {float(net_pnl):.8f})")

        pnl_float = float(net_pnl.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
        pnl_pct = float((net_pnl / position_value * Decimal('100')).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

        return pnl_float, pnl_pct, new_balance

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
