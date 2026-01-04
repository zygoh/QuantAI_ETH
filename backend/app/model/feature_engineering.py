"""
特征工程模块
"""
import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import ta

from app.exchange.exchange_factory import ExchangeFactory

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.feature_columns = []
    
    def create_features(self, df: pd.DataFrame, loop: Optional[asyncio.AbstractEventLoop] = None) -> pd.DataFrame:
        """创建所有特征"""
        try:
            self.loop = loop  # 存储传入的循环
            inf_tracker = {}  # 初始化inf追踪器
            if df.empty:
                return df
            
            logger.info(f"开始特征工程: {len(df)}行原始数据")
            
            # 🔥 第一步：全面的K线数据验证和过滤
            rows_before_validation = len(df)
            
            if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
                # 验证价格范围合理性
                invalid_price_mask = (
                    (df['close'] <= 0) |
                    (df['high'] <= 0) |
                    (df['low'] <= 0) |
                    (df['open'] <= 0) |
                    (df['high'] < df['low']) |
                    (df['close'] < df['low']) |
                    (df['close'] > df['high']) |
                    (df['open'] < df['low']) |
                    (df['open'] > df['high'])
                )
                
                if invalid_price_mask.any():
                    invalid_count = invalid_price_mask.sum()
                    logger.error(f"❌ 检测到{invalid_count}条异常K线（价格范围不合理），将过滤")
                    df = df[~invalid_price_mask]
            
            if 'volume' in df.columns:
                # 过滤负数成交量
                negative_volume_mask = df['volume'] < 0
                if negative_volume_mask.any():
                    negative_count = negative_volume_mask.sum()
                    logger.error(f"❌ 检测到{negative_count}条异常K线（成交量为负数），将过滤")
                    df = df[~negative_volume_mask]
                
                # 过滤未完成K线（volume=0）
                zero_volume_mask = df['volume'] == 0
                if zero_volume_mask.any():
                    zero_count = zero_volume_mask.sum()
                    logger.warning(f"⚠️ 过滤{zero_count}条未完成K线（volume=0）")
                    df = df[~zero_volume_mask]
            
            # 验证异常波动（单根K线涨跌幅超过50%视为异常）
            if 'close' in df.columns and 'open' in df.columns:
                price_change_pct = (df['close'] - df['open']) / (df['open'] + 1e-10)
                extreme_volatility_mask = np.abs(price_change_pct) > 0.5
                if extreme_volatility_mask.any():
                    extreme_count = extreme_volatility_mask.sum()
                    logger.warning(f"⚠️ 检测到{extreme_count}条异常波动K线（涨跌幅>50%），将过滤")
                    df = df[~extreme_volatility_mask]
            
            filtered_count = rows_before_validation - len(df)
            if filtered_count > 0:
                logger.warning(f"✅ 数据验证完成：过滤{filtered_count}条异常K线，剩余{len(df)}条")
            
            if df.empty:
                logger.error("❌ 数据验证后为空，无法进行特征工程")
                return df
            
            # 🔥 第二步：统一清洗close价格，创建clean_close列（避免后续重复计算）
            if 'close' in df.columns:
                df['clean_close'] = df['close'].replace(0, np.nan)
                # 如果仍有NaN，用前向填充
                df['clean_close'] = df['clean_close'].ffill()
                # 如果开头仍有NaN，用后向填充
                df['clean_close'] = df['clean_close'].bfill()
                # 如果完全没有有效值，保留原始值
                if df['clean_close'].isna().any():
                    logger.warning(f"⚠️ clean_close仍有NaN，使用原始close值")
                    df['clean_close'] = df['clean_close'].fillna(df['close'])
            else:
                df['clean_close'] = np.nan
            
            if df.index.name == 'timestamp' or 'timestamp' not in df.columns:
                df = df.reset_index()
            
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df = self._add_price_features(df)
            df = self._add_technical_indicators(df)
            df = self._add_volume_features(df)
            df = self._add_time_features(df)
            df = self._add_microstructure_features(df)
            df = self._add_volatility_features(df)
            df = self._add_momentum_features(df)
            df = self._add_sentiment_features(df)
            df = self._add_multi_timeframe_features(df)
            df = self._add_trend_strength_features(df)
            df = self._add_support_resistance_features(df)
            df = self._add_advanced_momentum_features(df)
            df = self._add_pattern_features(df)
            df = self._add_order_flow_features(df)
            df = self._add_swing_features(df)
            
            # ✅ 输出inf追踪总结
            if inf_tracker:
                logger.warning(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.warning(f"📊 Inf值产生步骤追踪:")
                for step, count in sorted(inf_tracker.items(), key=lambda x: x[1], reverse=True):
                    logger.warning(f"   {step}: {count}个inf值")
                logger.warning(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # 🔥 第一步：处理无穷大值（inf）- 必须在NaN处理前完成（增强诊断）
            inf_count = 0
            inf_details = {}  # 记录每个列产生inf的详细信息
            
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    inf_mask = np.isinf(df[col])
                    if inf_mask.any():
                        col_inf_count = inf_mask.sum()
                        inf_count += col_inf_count
                        
                        # ✅ 详细诊断：记录inf的详细信息
                        inf_indices = df[inf_mask].index.tolist()
                        inf_values = df.loc[inf_mask, col].tolist()
                        
                        # 记录前10个inf的详细信息
                        sample_indices = inf_indices[:10]
                        sample_values = inf_values[:10]
                        
                        # 获取inf前后的值（用于分析原因）
                        detail_info = []
                        for idx in sample_indices:
                            idx_pos = df.index.get_loc(idx)
                            prev_val = df[col].iloc[idx_pos - 1] if idx_pos > 0 else None
                            curr_val = df.loc[idx, col]
                            next_val = df[col].iloc[idx_pos + 1] if idx_pos < len(df) - 1 else None
                            
                            # 检查是否是pct_change产生的inf（前一个值为0或NaN）
                            if col in ['price_change', 'price_change_2', 'price_change_3', 'price_change_5', 
                                     'price_change_10', 'price_change_20', 'volume_change']:
                                if idx_pos > 0:
                                    base_col = 'close' if 'price' in col else 'volume'
                                    if base_col in df.columns:
                                        base_val = df[base_col].iloc[idx_pos - 1]
                                        detail_info.append({
                                            'index': idx,
                                            'inf_value': curr_val,
                                            f'{base_col}_prev': base_val,
                                            f'{base_col}_curr': df[base_col].iloc[idx_pos] if idx_pos < len(df) else None,
                                            'reason': f'{base_col}_prev={base_val} (可能是0或NaN导致pct_change产生inf)'
                                        })
                                    else:
                                        detail_info.append({
                                            'index': idx,
                                            'inf_value': curr_val,
                                            'prev': prev_val,
                                            'next': next_val
                                        })
                                else:
                                    detail_info.append({
                                        'index': idx,
                                        'inf_value': curr_val,
                                        'reason': '第一个值，无法检查前值'
                                    })
                            else:
                                detail_info.append({
                                    'index': idx,
                                    'inf_value': curr_val,
                                    'prev': prev_val,
                                    'next': next_val
                                })
                        
                        inf_details[col] = {
                            'count': col_inf_count,
                            'total_in_column': len(df[col]),
                            'percentage': 100.0 * col_inf_count / len(df[col]),
                            'samples': detail_info
                        }
                        
                        # 将inf替换为NaN（后续统一处理）
                        df.loc[inf_mask, col] = np.nan
            
            if inf_count > 0:
                logger.warning(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.warning(f"⚠️ 检测到{inf_count}个无穷大值（inf），已替换为NaN")
                logger.warning(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.warning(f"📊 Inf值详细统计（按列）:")
                for col, details in sorted(inf_details.items(), key=lambda x: x[1]['count'], reverse=True):
                    logger.warning(f"   {col}:")
                    logger.warning(f"      总数量: {details['count']}个 ({details['percentage']:.2f}%)")
                    logger.warning(f"      列总数: {details['total_in_column']}个")
                    logger.warning(f"      详细样本（前{min(len(details['samples']), 5)}个）:")
                    for i, sample in enumerate(details['samples'][:5]):
                        logger.warning(f"         样本{i+1}:")
                        logger.warning(f"            行索引: {sample['index']}")
                        logger.warning(f"            Inf值: {sample['inf_value']}")
                        if 'reason' in sample:
                            logger.warning(f"            原因: {sample['reason']}")
                        if 'close_prev' in sample:
                            logger.warning(f"            close前值: {sample['close_prev']}, close当前值: {sample['close_curr']}")
                        if 'volume_prev' in sample:
                            logger.warning(f"            volume前值: {sample['volume_prev']}, volume当前值: {sample['volume_curr']}")
                        if 'prev' in sample and 'next' in sample:
                            logger.warning(f"            前值: {sample['prev']}, 后值: {sample['next']}")
                logger.warning(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # 🔥 第二步：处理过大值（可能导致缩放时溢出）
            large_value_threshold = 1e15  # 防止后续缩放时溢出
            large_count = 0
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    large_mask = np.abs(df[col]) > large_value_threshold
                    if large_mask.any():
                        large_count += large_mask.sum()
                        df.loc[large_mask, col] = np.nan
            
            if large_count > 0:
                logger.warning(f"⚠️ 检测到{large_count}个过大值（>1e15），已替换为NaN")
            
            # 🔥 第三步：处理NaN值（训练用dropna，预测用fillna）
            rows_before = len(df)
            
            # 统计NaN原因（用于更精确的日志）
            # 检查哪些列有NaN（用于分析）
            nan_by_column = {}
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        nan_by_column[col] = nan_count
            
            # 先尝试删除NaN
            df_clean = df.dropna()
            
            # 如果删除后数据量<50行，说明是预测场景，改用填充
            if len(df_clean) < 50 and rows_before >= 100:
                logger.debug(f"⚠️ 预测场景检测：dropna会导致数据过少（{rows_before}→{len(df_clean)}），改用fillna")
                # 使用前向填充
                df = df.ffill()
                # 如果前向填充后仍有NaN（开头的行），用后向填充
                df = df.bfill()
                
                # ✅ 关键修复：对于close/volume等关键字段，使用更合理的填充策略
                # 而不是简单地用0填充（避免导致pct_change产生inf）
                if 'close' in df.columns:
                    # close价格：如果仍有NaN，使用前一个有效值或后一个有效值
                    # 如果完全没有有效值，保留NaN（不要用0）
                    if df['close'].isna().any():
                        # 尝试用open/high/low的平均值填充
                        for idx in df[df['close'].isna()].index:
                            if not df.loc[idx, ['open', 'high', 'low']].isna().all():
                                df.loc[idx, 'close'] = df.loc[idx, ['open', 'high', 'low']].mean()
                
                # 对于其他字段，如果仍有NaN，用0填充（但close/volume已处理）
                # 但确保close/volume不会为0
                for col in df.columns:
                    if col not in ['timestamp', 'close', 'volume']:
                        df[col] = df[col].fillna(0)
                    elif col == 'volume':
                        # volume为0可以接受，但NaN保留（避免影响计算）
                        pass
                
                # 最后检查：确保close不为0
                if 'close' in df.columns:
                    zero_close = (df['close'] == 0).sum()
                    if zero_close > 0:
                        logger.warning(f"⚠️ 预测场景：仍有{zero_close}个close为0，将替换为NaN")
                        df.loc[df['close'] == 0, 'close'] = np.nan
                
                logger.debug(f"✅ 特征工程完成（预测模式）: {len(df)}行，特征数: {len(df.columns)}")
            else:
                # 训练场景，正常删除NaN
                df = df_clean
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    # 🔥 分析NaN原因：主要是技术指标窗口导致的（正常现象）
                    # 找出NaN最多的列（通常是窗口最大的指标）
                    if nan_by_column:
                        top_nan_cols = sorted(nan_by_column.items(), key=lambda x: x[1], reverse=True)[:5]
                        nan_reason = f"（主要因技术指标窗口导致，NaN最多的列: {', '.join([f'{col}({count})' for col, count in top_nan_cols])}）"
                    else:
                        nan_reason = "（因NaN/Inf导致）"
                    
                    logger.info(f"✅ 特征工程完成: {len(df)}行（丢弃{rows_dropped}行{nan_reason}），特征数: {len(df.columns)}")
                else:
                    logger.info(f"✅ 特征工程完成: {len(df)}行，特征数: {len(df.columns)}")

            
            return df
            
        except Exception as e:
            logger.error(f"❌ 特征工程失败: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格特征 - 优化性能"""
        try:
            logger.debug(f"   📊 _add_price_features: 开始处理，输入数据形状: {df.shape}")
            logger.debug(f"      close统计: min={df['close'].min():.4f}, max={df['close'].max():.4f}, "
                        f"零值={(df['close'] == 0).sum()}, NaN={df['close'].isna().sum()}")
            new_features = {}
            
            # 价格变化率（修复：pct_change在除数为0时会产生inf）
            # ✅ 详细诊断：检查close数据质量
            close_zero_count = (df['close'] == 0).sum()
            close_nan_count = df['close'].isna().sum()
            if close_zero_count > 0 or close_nan_count > 0:
                logger.warning(f"⚠️ _add_price_features: close数据异常 - 零值={close_zero_count}, NaN={close_nan_count}")
                if close_zero_count > 0:
                    zero_indices = df[df['close'] == 0].index.tolist()[:5]
                    logger.warning(f"   close=0的位置（前5个）: {zero_indices}")
            
            # ✅ 使用统一清洗的clean_close列（避免重复计算）
            close_for_pct = df['clean_close']
            
            price_change = close_for_pct.pct_change(fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            
            # ✅ 详细诊断：检查pct_change产生的inf
            inf_mask = np.isinf(price_change)
            if inf_mask.any():
                inf_count = inf_mask.sum()
                logger.error(f"❌ _add_price_features: price_change产生{inf_count}个inf值！")
                inf_indices = price_change[inf_mask].index.tolist()[:5]
                for idx in inf_indices:
                    idx_pos = df.index.get_loc(idx)
                    prev_close = df['close'].iloc[idx_pos - 1] if idx_pos > 0 else None
                    curr_close = df['close'].iloc[idx_pos]
                    inf_value = price_change.iloc[idx_pos]
                    logger.error(f"   位置{idx}: close前值={prev_close}, close当前值={curr_close}, pct_change={inf_value}")
                    if prev_close == 0:
                        logger.error(f"      ❌ 原因确认：前一个close值为0，导致pct_change产生inf")
                    elif prev_close is None or np.isnan(prev_close):
                        logger.error(f"      ❌ 原因确认：前一个close值为NaN，导致pct_change产生inf")
                    else:
                        logger.error(f"      ⚠️ 原因不明：前一个close值={prev_close}（非0非NaN），但仍产生inf")
            
            # ✅ 修复：替换inf值（当close从前一个0值变化时）
            price_change = price_change.replace([np.inf, -np.inf], np.nan)
            new_features['price_change'] = price_change
            new_features['price_change_abs'] = price_change.abs()
            
            # 价格范围（使用统一清洗的clean_close）
            close_safe = df['clean_close']
            
            # ✅ 详细诊断：检查close_safe替换后的情况
            close_safe_zero_after = (close_safe == 0).sum()
            if close_safe_zero_after > 0:
                logger.error(f"❌ _add_price_features: close_safe仍有{close_safe_zero_after}个0值（替换失败）")
            
            new_features['price_range'] = (df['high'] - df['low']) / close_safe
            
            # ✅ 详细诊断：检查price_range是否产生inf
            inf_mask_price_range = np.isinf(new_features['price_range'])
            if inf_mask_price_range.any():
                inf_count = inf_mask_price_range.sum()
                logger.error(f"❌ _add_price_features: price_range产生{inf_count}个inf值！")
                inf_indices = new_features['price_range'][inf_mask_price_range].index.tolist()[:3]
                for idx in inf_indices:
                    idx_pos = df.index.get_loc(idx)
                    logger.error(f"   位置{idx}: high={df['high'].iloc[idx_pos]}, low={df['low'].iloc[idx_pos]}, close={df['close'].iloc[idx_pos]}, close_safe={close_safe.iloc[idx_pos]}")
            
            # 注：upper_shadow 和 lower_shadow 在市场微观结构特征中添加（更好的归一化）
            
            # 开盘价与收盘价关系（避免除以零）
            new_features['open_close_ratio'] = df['open'] / close_safe
            
            # ✅ 详细诊断：检查open_close_ratio是否产生inf
            inf_mask_open_close = np.isinf(new_features['open_close_ratio'])
            if inf_mask_open_close.any():
                inf_count = inf_mask_open_close.sum()
                logger.error(f"❌ _add_price_features: open_close_ratio产生{inf_count}个inf值！")
            
            new_features['body_size'] = abs(df['close'] - df['open']) / close_safe
            
            # ✅ 详细诊断：检查body_size是否产生inf
            inf_mask_body = np.isinf(new_features['body_size'])
            if inf_mask_body.any():
                inf_count = inf_mask_body.sum()
                logger.error(f"❌ _add_price_features: body_size产生{inf_count}个inf值！")
            
            # 价格位置（避免除以零）
            price_range_safe = df['high'] - df['low']
            price_range_safe = price_range_safe.replace(0, np.nan)  # 零范围设为NaN
            new_features['close_position'] = (df['close'] - df['low']) / price_range_safe
            
            # 多周期价格变化（修复：pct_change可能产生inf）
            # ✅ 使用预处理后的close_for_pct（已处理0值）
            for period in [2, 3, 5, 10, 20]:
                pct_chg = close_for_pct.pct_change(period, fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
                
                # ✅ 详细诊断：检查每个period的inf情况（理论上不应该有）
                inf_mask = np.isinf(pct_chg)
                if inf_mask.any():
                    inf_count = inf_mask.sum()
                    logger.error(f"❌ _add_price_features: price_change_{period}仍产生{inf_count}个inf值（异常！）")
                    # 只记录前3个inf的详细信息
                    inf_indices = pct_chg[inf_mask].index.tolist()[:3]
                    for idx in inf_indices:
                        idx_pos = df.index.get_loc(idx)
                        prev_close = df['close'].iloc[idx_pos - period] if idx_pos >= period else None
                        curr_close = df['close'].iloc[idx_pos]
                        prev_close_for_pct = close_for_pct.iloc[idx_pos - period] if idx_pos >= period else None
                        logger.error(f"   位置{idx}: {period}周期前close={prev_close}, 当前close={curr_close}")
                        logger.error(f"      {period}周期前close_for_pct={prev_close_for_pct}")
                        if prev_close == 0:
                            logger.error(f"      ❌ 原因：{period}周期前close=0，但预处理可能失败")
                else:
                    if close_zero_count > 0:
                        logger.debug(f"   ✅ price_change_{period}通过预处理避免了inf产生")
                
                pct_chg = pct_chg.replace([np.inf, -np.inf], np.nan)  # ✅ 双重保护：替换inf
                new_features[f'price_change_{period}'] = pct_chg
                # 避免除以零（low可能为0）
                rolling_low_min = df['low'].rolling(period).min()
                rolling_low_safe = rolling_low_min.replace(0, np.nan)  # 避免除以0
                new_features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / rolling_low_safe
            
            # ✅ 价格加速度（一阶、三阶、五阶）
            new_features['price_acceleration'] = price_change - price_change.shift(1)  # 一阶加速度（基础版本）
            new_features['price_acceleration_3'] = price_change - price_change.shift(3)
            new_features['price_acceleration_5'] = price_change - price_change.shift(5)
            
            # 注：consecutive_up, consecutive_down 在市场情绪特征中添加（更好的实现）
            
            # ✅ 价格动量强度
            new_features['price_momentum_strength'] = price_change.abs().rolling(5).mean()
            new_features['price_momentum_direction'] = price_change.rolling(5).mean() / (price_change.rolling(5).std() + 1e-8)
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"添加价格特征失败: {e}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征 - 优化性能"""
        try:
            new_features = {}
            
            # RSI
            for period in [14, 21, 30]:
                new_features[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            new_features['macd'] = macd.macd()
            new_features['macd_signal'] = macd.macd_signal()
            new_features['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            for period in [20, 50]:
                bb = ta.volatility.BollingerBands(df['close'], window=period)
                bb_upper = bb.bollinger_hband()
                bb_lower = bb.bollinger_lband()
                bb_middle = bb.bollinger_mavg()
                
                new_features[f'bb_upper_{period}'] = bb_upper
                new_features[f'bb_lower_{period}'] = bb_lower
                new_features[f'bb_middle_{period}'] = bb_middle
                # 避免除以零
                bb_middle_safe = bb_middle.replace(0, np.nan)
                bb_range_safe = (bb_upper - bb_lower).replace(0, np.nan)
                new_features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle_safe
                new_features[f'bb_position_{period}'] = (df['close'] - bb_lower) / bb_range_safe
            
            # 移动平均线
            sma_dict = {}
            ema_dict = {}
            for period in [5, 10, 20, 50, 100, 200]:
                sma = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
                ema = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
                
                sma_dict[period] = sma
                ema_dict[period] = ema
                
                new_features[f'sma_{period}'] = sma
                new_features[f'ema_{period}'] = ema
                # 避免除以零（虽然sma/ema通常不为0，但为安全起见）
                sma_safe = sma.replace(0, np.nan)
                ema_safe = ema.replace(0, np.nan)
                new_features[f'price_sma_ratio_{period}'] = df['close'] / sma_safe
                new_features[f'price_ema_ratio_{period}'] = df['close'] / ema_safe
            
            # 移动平均线交叉
            new_features['sma_5_20_cross'] = np.where(sma_dict[5] > sma_dict[20], 1, 0)
            new_features['sma_10_50_cross'] = np.where(sma_dict[10] > sma_dict[50], 1, 0)
            new_features['ema_5_20_cross'] = np.where(ema_dict[5] > ema_dict[20], 1, 0)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            new_features['stoch_k'] = stoch.stoch()
            new_features['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            new_features['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # CCI (Commodity Channel Index)
            new_features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            
            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            new_features['adx'] = adx.adx()
            new_features['adx_pos'] = adx.adx_pos()
            new_features['adx_neg'] = adx.adx_neg()
            
            # Parabolic SAR
            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
            new_features['psar'] = psar
            new_features['psar_signal'] = np.where(df['close'] > psar, 1, 0)
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"添加技术指标失败: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特征 - 优化性能"""
        try:
            new_features = {}
            
            # 成交量变化
            # ✅ 详细诊断：检查volume数据质量
            volume_zero_count = (df['volume'] == 0).sum()
            volume_nan_count = df['volume'].isna().sum()
            if volume_zero_count > 0:
                logger.warning(f"⚠️ _add_volume_features: 检测到{volume_zero_count}个volume为0（可能导致pct_change产生inf）")
                logger.warning(f"   这些零值将被临时替换为NaN，避免pct_change产生inf")
                # ✅ 关键修复：在pct_change之前，将volume=0替换为NaN
                # 这样pct_change就不会产生inf（因为NaN的pct_change结果是NaN，不是inf）
                volume_for_pct = df['volume'].replace(0, np.nan)
            else:
                volume_for_pct = df['volume']
            
            volume_change = volume_for_pct.pct_change(fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            
            # ✅ 详细诊断：检查volume_change是否仍有inf（理论上不应该有）
            inf_mask = np.isinf(volume_change)
            if inf_mask.any():
                inf_count = inf_mask.sum()
                logger.error(f"❌ _add_volume_features: volume_change仍产生{inf_count}个inf值（异常！）")
                inf_indices = volume_change[inf_mask].index.tolist()[:5]
                for idx in inf_indices:
                    idx_pos = df.index.get_loc(idx)
                    prev_volume = df['volume'].iloc[idx_pos - 1] if idx_pos > 0 else None
                    curr_volume = df['volume'].iloc[idx_pos]
                    logger.error(f"   位置{idx}: volume前值={prev_volume}, volume当前值={curr_volume}")
                    logger.error(f"      volume_for_pct前值={volume_for_pct.iloc[idx_pos - 1] if idx_pos > 0 else None}, "
                               f"volume_for_pct当前值={volume_for_pct.iloc[idx_pos]}")
            else:
                # ✅ 修复成功：没有产生inf
                if volume_zero_count > 0:
                    logger.info(f"   ✅ 通过预处理（volume=0→NaN）成功避免了inf产生")
            
            # ✅ 双重保护：即使仍有inf，也替换为NaN
            volume_change = volume_change.replace([np.inf, -np.inf], np.nan)
            # 🔑 修复非平稳特征：移除绝对值volume_sma，只保留比率特征
            # 计算volume_sma_20用于比率计算（不添加到特征中）
            volume_sma_20 = df['volume'].rolling(20).mean()
            # 避免除以零（volume_sma_20可能为0）
            volume_sma_20_safe = volume_sma_20.replace(0, np.nan)
            
            new_features['volume_change'] = volume_change
            # ✅ 移除非平稳绝对值特征：volume_sma_5, volume_sma_20
            # ✅ 只保留比率特征：volume_ratio（已转换为相对值，对模型更友好）
            new_features['volume_ratio'] = df['volume'] / volume_sma_20_safe
            
            # 价量关系
            new_features['price_volume_trend'] = df['price_change'] * volume_change
            
            # OBV (On Balance Volume)
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            new_features['obv'] = obv
            new_features['obv_sma'] = obv.rolling(20).mean()
            
            # Volume Price Trend
            new_features['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
            
            # Accumulation/Distribution Line
            new_features['ad_line'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
            
            # Chaikin Money Flow
            new_features['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
            
            # Volume Weighted Average Price (VWAP)（避免除以零）
            volume_rolling_sum = df['volume'].rolling(20).sum()
            
            # ✅ 详细诊断：检查volume_rolling_sum
            volume_rolling_sum_zero = (volume_rolling_sum == 0).sum()
            if volume_rolling_sum_zero > 0:
                logger.warning(f"⚠️ _add_volume_features: volume_rolling_sum有{volume_rolling_sum_zero}个0值")
                zero_indices = volume_rolling_sum[volume_rolling_sum == 0].index.tolist()[:5]
                for idx in zero_indices:
                    idx_pos = df.index.get_loc(idx)
                    volume_window = df['volume'].iloc[max(0, idx_pos-19):idx_pos+1]
                    logger.warning(f"   位置{idx}: volume_rolling_sum=0")
                    logger.warning(f"      volume窗口: 总数={len(volume_window)}, 零值={(volume_window == 0).sum()}, "
                                 f"NaN={volume_window.isna().sum() if hasattr(volume_window, 'isna') else 0}")
                    if len(volume_window) > 0:
                        logger.warning(f"      volume范围: [{volume_window.min():.4f}, {volume_window.max():.4f}]")
            
            volume_rolling_sum_safe = volume_rolling_sum.replace(0, np.nan)  # 避免除以0
            
            # ✅ 详细诊断：检查替换后的情况
            volume_rolling_sum_safe_zero_after = (volume_rolling_sum_safe == 0).sum()
            if volume_rolling_sum_safe_zero_after > 0:
                logger.error(f"❌ _add_volume_features: volume_rolling_sum_safe仍有{volume_rolling_sum_safe_zero_after}个0值（替换失败）")
            
            # 计算vwap
            numerator = (df['close'] * df['volume']).rolling(20).sum()
            vwap = numerator / volume_rolling_sum_safe
            
            # ✅ 详细诊断：检查vwap是否产生inf
            inf_mask_vwap = np.isinf(vwap)
            if inf_mask_vwap.any():
                inf_count = inf_mask_vwap.sum()
                logger.error(f"❌ _add_volume_features: vwap产生{inf_count}个inf值！")
                inf_indices = vwap[inf_mask_vwap].index.tolist()[:5]
                for idx in inf_indices:
                    idx_pos = df.index.get_loc(idx)
                    logger.error(f"   位置{idx}:")
                    logger.error(f"      numerator={numerator.iloc[idx_pos]}, volume_rolling_sum={volume_rolling_sum.iloc[idx_pos]}")
                    logger.error(f"      volume_rolling_sum_safe={volume_rolling_sum_safe.iloc[idx_pos]}")
                    logger.error(f"      vwap={vwap.iloc[idx_pos]}")
                    # 检查窗口内的详细数据
                    volume_window = df['volume'].iloc[max(0, idx_pos-19):idx_pos+1]
                    close_window = df['close'].iloc[max(0, idx_pos-19):idx_pos+1]
                    logger.error(f"      volume窗口（前5个）: {volume_window.head(5).tolist()}")
                    logger.error(f"      close窗口（前5个）: {close_window.head(5).tolist()}")
            
            new_features['vwap'] = vwap
            # 避免除以零（vwap可能为NaN）
            vwap_safe = vwap.replace(0, np.nan).replace(np.nan, 1.0)  # 如果vwap为NaN，用1.0避免inf
            
            # ✅ 详细诊断：检查vwap_safe替换后的情况
            vwap_safe_zero_after = (vwap_safe == 0).sum()
            if vwap_safe_zero_after > 0:
                logger.error(f"❌ _add_volume_features: vwap_safe仍有{vwap_safe_zero_after}个0值（替换失败）")
            
            new_features['price_vwap_ratio'] = df['close'] / vwap_safe
            
            # ✅ 详细诊断：检查price_vwap_ratio是否产生inf
            inf_mask_vwap_ratio = np.isinf(new_features['price_vwap_ratio'])
            if inf_mask_vwap_ratio.any():
                inf_count = inf_mask_vwap_ratio.sum()
                logger.error(f"❌ _add_volume_features: price_vwap_ratio产生{inf_count}个inf值！")
                inf_indices = new_features['price_vwap_ratio'][inf_mask_vwap_ratio].index.tolist()[:5]
                for idx in inf_indices:
                    idx_pos = df.index.get_loc(idx)
                    logger.error(f"   位置{idx}:")
                    logger.error(f"      close={df['close'].iloc[idx_pos]}, vwap={vwap.iloc[idx_pos]}")
                    logger.error(f"      vwap_safe={vwap_safe.iloc[idx_pos]}, price_vwap_ratio={new_features['price_vwap_ratio'].iloc[idx_pos]}")
                    if vwap_safe.iloc[idx_pos] == 0:
                        logger.error(f"      ❌ 原因确认：vwap_safe=0（替换失败）")
            
            # ✅ 成交量突破（捕捉放量信号）（避免除以零）
            volume_ma_5 = df['volume'].rolling(5).mean()
            volume_ma_20 = df['volume'].rolling(20).mean()
            
            # ✅ 详细诊断：检查volume_ma_20
            volume_ma_20_zero = (volume_ma_20 == 0).sum()
            if volume_ma_20_zero > 0:
                logger.warning(f"⚠️ _add_volume_features: volume_ma_20有{volume_ma_20_zero}个0值")
                zero_indices = volume_ma_20[volume_ma_20 == 0].index.tolist()[:3]
                for idx in zero_indices:
                    idx_pos = df.index.get_loc(idx)
                    volume_window = df['volume'].iloc[max(0, idx_pos-19):idx_pos+1]
                    logger.warning(f"   位置{idx}: volume_ma_20=0, volume窗口统计: 零值={(volume_window == 0).sum()}/{len(volume_window)}")
            
            volume_ma_20_safe = volume_ma_20.replace(0, np.nan)  # 避免除以0
            
            # ✅ 详细诊断：检查volume_ma_20_safe替换后的情况
            volume_ma_20_safe_zero_after = (volume_ma_20_safe == 0).sum()
            if volume_ma_20_safe_zero_after > 0:
                logger.error(f"❌ _add_volume_features: volume_ma_20_safe仍有{volume_ma_20_safe_zero_after}个0值（替换失败）")
            
            new_features['volume_spike'] = df['volume'] / volume_ma_20_safe
            
            # ✅ 详细诊断：检查volume_spike是否产生inf
            inf_mask_spike = np.isinf(new_features['volume_spike'])
            if inf_mask_spike.any():
                inf_count = inf_mask_spike.sum()
                logger.error(f"❌ _add_volume_features: volume_spike产生{inf_count}个inf值！")
                inf_indices = new_features['volume_spike'][inf_mask_spike].index.tolist()[:3]
                for idx in inf_indices:
                    idx_pos = df.index.get_loc(idx)
                    logger.error(f"   位置{idx}: volume={df['volume'].iloc[idx_pos]}, volume_ma_20={volume_ma_20.iloc[idx_pos]}, volume_ma_20_safe={volume_ma_20_safe.iloc[idx_pos]}")
            
            new_features['volume_trend'] = volume_ma_5 / volume_ma_20_safe
            
            # ✅ 详细诊断：检查volume_trend是否产生inf
            inf_mask_trend = np.isinf(new_features['volume_trend'])
            if inf_mask_trend.any():
                inf_count = inf_mask_trend.sum()
                logger.error(f"❌ _add_volume_features: volume_trend产生{inf_count}个inf值！")
            
            # ✅ 价格-成交量背离（重要信号）（修复：pct_change可能产生inf）
            # ✅ 使用统一清洗的clean_close
            close_for_pct_corr = df['clean_close']
            volume_for_pct_corr = df['volume'].replace(0, np.nan) if (df['volume'] == 0).sum() > 0 else df['volume']
            
            price_change_1 = close_for_pct_corr.pct_change(1, fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            price_change_1 = price_change_1.replace([np.inf, -np.inf], np.nan)  # ✅ 双重保护
            
            price_change_5 = close_for_pct_corr.pct_change(5, fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            price_change_5 = price_change_5.replace([np.inf, -np.inf], np.nan)  # ✅ 双重保护
            
            volume_change_5 = volume_for_pct_corr.pct_change(5, fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            volume_change_5 = volume_change_5.replace([np.inf, -np.inf], np.nan)  # ✅ 双重保护
            new_features['price_volume_correlation'] = price_change_5 * volume_change_5  # 同向为正，背离为负（连续值）
            
            # ✅ 成交量加权价格变化（结合量价）（避免除以零）
            # volume_ma_20_safe已在上面定义（Line 317），直接使用
            new_features['volume_weighted_price_change'] = price_change_1 * (df['volume'] / volume_ma_20_safe)
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"添加成交量特征失败: {e}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征 - 优化性能"""
        try:
            # 确保 timestamp 是 datetime 类型
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                datetime_col = df['timestamp']
            else:
                # timestamp 可能在 index 中
                logger.warning("timestamp 列不存在，跳过时间特征")
                return df
            
            new_features = {}
            
            # 基础时间特征
            hour = datetime_col.dt.hour
            day_of_week = datetime_col.dt.dayofweek
            month = datetime_col.dt.month
            
            new_features['hour'] = hour
            new_features['day_of_week'] = day_of_week
            new_features['day_of_month'] = datetime_col.dt.day
            new_features['month'] = month
            new_features['quarter'] = datetime_col.dt.quarter
            
            # 周期性编码
            new_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            new_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            new_features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            new_features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            new_features['month_sin'] = np.sin(2 * np.pi * month / 12)
            new_features['month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # 交易时段
            new_features['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
            new_features['is_european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
            new_features['is_american_session'] = ((hour >= 16) & (hour < 24)).astype(int)
            
            # 周末标识
            new_features['is_weekend'] = (day_of_week >= 5).astype(int)
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"添加时间特征失败: {e}")
            return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场微观结构特征 - 增强版（优化目标：准确率+3-5%）"""
        try:
            new_features = {}
            
            # 1. 买卖压力指标（基础）
            price_range = df['high'] - df['low'] + 1e-10  # 避免除零
            new_features['buying_pressure'] = (df['close'] - df['low']) / price_range
            new_features['selling_pressure'] = (df['high'] - df['close']) / price_range
            
            # 2. 🆕 价格位置百分比（捕捉支撑阻力）
            for period in [5, 20, 50]:
                rolling_high = df['high'].rolling(period).max()
                rolling_low = df['low'].rolling(period).min()
                price_position = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
                new_features[f'price_position_{period}'] = price_position
                # 超买超卖标识
                new_features[f'overbought_{period}'] = (price_position > 0.8).astype(int)
                new_features[f'oversold_{period}'] = (price_position < 0.2).astype(int)
            
            # 3. 🆕 K线形态特征（实体和影线）
            body = abs(df['close'] - df['open'])
            new_features['body_range'] = body / (price_range + 1e-10)
            new_features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (price_range + 1e-10)
            new_features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (price_range + 1e-10)
            
            # 4. 🆕 K线强度（看涨/看跌力量）
            new_features['bullish_candle'] = (df['close'] > df['open']).astype(int)
            new_features['strong_bullish'] = ((df['close'] - df['open']) / (price_range + 1e-10) > 0.6).astype(int)
            new_features['strong_bearish'] = ((df['open'] - df['close']) / (price_range + 1e-10) > 0.6).astype(int)
            new_features['doji'] = (body / (price_range + 1e-10) < 0.1).astype(int)  # 十字星
            
            # 5. 🆕 连续K线形态（趋势延续）
            bullish = (df['close'] > df['open']).astype(int)
            new_features['consecutive_bull'] = bullish.groupby((bullish != bullish.shift()).cumsum()).cumsum()
            new_features['consecutive_bear'] = (1 - bullish).groupby((bullish == bullish.shift()).cumsum()).cumsum()
            
            # 6. 价格效率（已有，保留）
            for period in [5, 10, 20]:
                price_change = df['close'].diff(period)
                sum_abs_changes = df['close'].diff().abs().rolling(period).sum()
                new_features[f'price_efficiency_{period}'] = price_change.abs() / (sum_abs_changes + 1e-10)
            
            # 7. 🆕 价格加速度（捕捉拐点）
            close_for_returns = df['clean_close']
            returns = close_for_returns.pct_change(fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            # ✅ 修复：替换inf值
            returns = returns.replace([np.inf, -np.inf], np.nan)
            # 注：price_acceleration 已在价格特征中定义，这里添加更高阶的
            new_features['price_jerk'] = returns.diff().diff()  # 加加速度（三阶导数）
            
            # 8. 分形维度（已有，保留）
            for period in [10, 20]:
                new_features[f'fractal_dimension_{period}'] = self._calculate_fractal_dimension(df['close'], period)
            
            # 9. Hurst指数（已有，保留）
            for period in [20, 50]:
                new_features[f'hurst_exponent_{period}'] = self._calculate_hurst_exponent(df['close'], period)
            
            # 10. 🆕 真实波动范围占比（捕捉异常波动）
            atr_14 = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            new_features['range_to_atr'] = price_range / (atr_14 + 1e-10)
            new_features['body_to_atr'] = body / (atr_14 + 1e-10)
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 添加市场微观结构特征失败: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征 - 优化性能"""
        try:
            new_features = {}
            
            # 历史波动率
            close_for_returns = df['clean_close']
            returns = close_for_returns.pct_change(fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            # ✅ 修复：替换inf值
            returns = returns.replace([np.inf, -np.inf], np.nan)
            for period in [5, 10, 20, 50]:
                new_features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
            # ATR (Average True Range)
            for period in [14, 20]:
                atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
                new_features[f'atr_{period}'] = atr
                new_features[f'atr_ratio_{period}'] = atr / df['close']
            
            # Keltner Channels
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            kc_upper = kc.keltner_channel_hband()
            kc_lower = kc.keltner_channel_lband()
            kc_middle = kc.keltner_channel_mband()
            
            new_features['kc_upper'] = kc_upper
            new_features['kc_lower'] = kc_lower
            new_features['kc_middle'] = kc_middle
            # 避免除以零
            kc_range_safe = (kc_upper - kc_lower).replace(0, np.nan)
            new_features['kc_position'] = (df['close'] - kc_lower) / kc_range_safe
            
            # Donchian Channels
            dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
            new_features['dc_upper'] = dc.donchian_channel_hband()
            new_features['dc_lower'] = dc.donchian_channel_lband()
            new_features['dc_middle'] = dc.donchian_channel_mband()
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"添加波动率特征失败: {e}")
            return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量特征 - 优化性能"""
        try:
            new_features = {}
            
            # ROC (Rate of Change)
            for period in [5, 10, 20]:
                new_features[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()
            
            # Momentum
            for period in [5, 10, 20]:
                new_features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            
            # TSI (True Strength Index)
            new_features['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()
            
            # Ultimate Oscillator
            new_features['uo'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
            
            # Awesome Oscillator
            new_features['ao'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
            
            # 注：ADX已在技术指标中添加，避免重复
            
            # ✅ 动量加速度（捕捉动量变化）
            rsi_14 = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            new_features['rsi_acceleration'] = rsi_14 - rsi_14.shift(1)
            new_features['rsi_velocity'] = rsi_14.diff()
            
            # ✅ 多周期动量一致性（趋势确认）
            roc_5 = ta.momentum.ROCIndicator(df['close'], window=5).roc()
            roc_10 = ta.momentum.ROCIndicator(df['close'], window=10).roc()
            roc_20 = ta.momentum.ROCIndicator(df['close'], window=20).roc()
            # 如果多个周期都是正/负，则趋势更可靠
            new_features['momentum_alignment'] = ((roc_5 > 0).astype(int) + 
                                                  (roc_10 > 0).astype(int) + 
                                                  (roc_20 > 0).astype(int)) - 1.5  # -1.5 to 1.5
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"添加动量特征失败: {e}")
            return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加统计特征 - 优化性能，避免DataFrame碎片化"""
        try:
            new_features = {}
            
            # 滚动统计
            for period in [5, 10, 20, 50]:
                rolling = df['close'].rolling(period)
                new_features[f'close_mean_{period}'] = rolling.mean()
                new_features[f'close_std_{period}'] = rolling.std()
                new_features[f'close_skew_{period}'] = rolling.skew()
                new_features[f'close_kurt_{period}'] = rolling.kurt()
                
                # Z-score
                mean_col = new_features[f'close_mean_{period}']
                std_col = new_features[f'close_std_{period}']
                new_features[f'close_zscore_{period}'] = (df['close'] - mean_col) / std_col
            
            # 分位数
            for period in [20, 50]:
                rolling = df['close'].rolling(period)
                new_features[f'close_quantile_25_{period}'] = rolling.quantile(0.25)
                new_features[f'close_quantile_75_{period}'] = rolling.quantile(0.75)
                new_features[f'close_iqr_{period}'] = (
                    new_features[f'close_quantile_75_{period}'] - 
                    new_features[f'close_quantile_25_{period}']
                )
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"添加统计特征失败: {e}")
            return df
    
    def _calculate_fractal_dimension(self, series: pd.Series, period: int) -> pd.Series:
        """计算分形维度"""
        try:
            def fractal_dim(data):
                try:
                    data = np.array(data)
                    if len(data) < 10:
                        return np.nan
                    
                    # Higuchi方法计算分形维度
                    N = len(data)
                    L = []
                    x = []
                    
                    for k in range(1, min(N//2, 10)):
                        Lk = 0
                        for m in range(k):
                            Lmk = 0
                            for i in range(1, int((N-m)/k)):
                                Lmk += abs(data[m+i*k] - data[m+(i-1)*k])
                            if ((N-m)/k) * k > 0:
                                Lmk = Lmk * (N-1) / (((N-m)/k) * k)
                            Lk += Lmk
                        
                        if k > 0:
                            L.append(Lk/k)
                            x.append(1.0/k)
                    
                    if len(L) < 2:
                        return np.nan
                    
                    # 线性回归计算斜率
                    x = np.log(x)
                    y = np.log(L)
                    coeffs = np.polyfit(x, y, 1)
                    return coeffs[0]
                except:
                    return np.nan
            
            return series.rolling(period).apply(fractal_dim, raw=False)
            
        except Exception as e:
            logger.error(f"计算分形维度失败: {e}")
            return pd.Series(np.nan, index=series.index)
    
    def _calculate_hurst_exponent(self, series: pd.Series, period: int) -> pd.Series:
        """计算Hurst指数"""
        try:
            def hurst_exp(data):
                if len(data) < 10:
                    return np.nan
                
                # R/S分析计算Hurst指数
                data = np.array(data)
                N = len(data)
                
                # 计算累积偏差
                mean_data = np.mean(data)
                cumulative_deviate = np.cumsum(data - mean_data)
                
                # 计算范围
                R = np.max(cumulative_deviate) - np.min(cumulative_deviate)
                
                # 计算标准差
                S = np.std(data)
                
                if S == 0:
                    return np.nan
                
                # R/S比率
                rs = R / S
                
                if rs <= 0:
                    return np.nan
                
                # Hurst指数
                return np.log(rs) / np.log(N)
            
            return series.rolling(period).apply(hurst_exp, raw=False)
            
        except Exception as e:
            logger.error(f"计算Hurst指数失败: {e}")
            return pd.Series(np.nan, index=series.index)
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场情绪特征（优化目标：准确率+2-3%）"""
        try:
            new_features = {}
            
            # 1. 恐慌指数（基于价格波动）
            close_for_returns = df['clean_close']
            returns = close_for_returns.pct_change(fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            # ✅ 修复：替换inf值
            returns = returns.replace([np.inf, -np.inf], np.nan)
            short_vol = returns.rolling(20).std()
            long_vol = returns.rolling(100).std()
            new_features['fear_index'] = short_vol / (long_vol + 1e-10)
            
            # 2. 🆕 连续涨跌天数（捕捉趋势疲劳）
            up_days = (returns > 0).astype(int)
            down_days = (returns < 0).astype(int)
            new_features['consecutive_up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
            new_features['consecutive_down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()
            
            # 3. 🆕 RSI衍生情绪指标
            if 'rsi_14' in df.columns:
                rsi = df['rsi_14']
                new_features['extreme_overbought'] = (rsi > 70).astype(int)
                new_features['extreme_oversold'] = (rsi < 30).astype(int)
                new_features['rsi_momentum'] = rsi - rsi.shift(5)  # RSI动量
                new_features['rsi_volatility'] = rsi.rolling(10).std()  # RSI波动率
            
            # 4. 🆕 价格加速度幅度（情绪转变强度）
            close_for_price_change = df['clean_close']
            price_change = close_for_price_change.pct_change(fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
            # ✅ 修复：替换inf值
            price_change = price_change.replace([np.inf, -np.inf], np.nan)
            # 注：price_acceleration 已在价格特征中定义，这里只添加幅度
            acceleration = price_change.diff()
            new_features['acceleration_magnitude'] = acceleration.abs()
            
            # 5. 🆕 成交量情绪（基于放量/缩量）
            if 'volume' in df.columns:
                volume_ma = df['volume'].rolling(20).mean()
                new_features['volume_surge'] = (df['volume'] > volume_ma * 2).astype(int)  # 放量
                new_features['volume_dry'] = (df['volume'] < volume_ma * 0.5).astype(int)  # 缩量
                
                # 价量背离（价涨量跌 = 看跌信号）
                price_trend = (price_change.rolling(5).mean() > 0).astype(int)
                volume_for_chg = df['volume'].replace(0, np.nan) if (df['volume'] == 0).sum() > 0 else df['volume']
                volume_chg = volume_for_chg.pct_change(fill_method=None)  # ✅ 修复：明确指定fill_method=None避免FutureWarning
                # ✅ 修复：替换inf值
                volume_chg = volume_chg.replace([np.inf, -np.inf], np.nan)
                volume_trend = (volume_chg.rolling(5).mean() > 0).astype(int)
                new_features['price_volume_divergence'] = (price_trend != volume_trend).astype(int)
            
            # 6. 🆕 市场波动情绪
            returns_abs = returns.abs()
            new_features['volatility_regime'] = returns_abs.rolling(20).mean() / returns_abs.rolling(100).mean()
            
            # 7. 🆕 趋势一致性（多个均线方向一致度）
            if 'sma_5' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma5_up = (df['sma_5'] > df['sma_5'].shift(1)).astype(int)
                sma20_up = (df['sma_20'] > df['sma_20'].shift(1)).astype(int)
                sma50_up = (df['sma_50'] > df['sma_50'].shift(1)).astype(int)
                new_features['trend_alignment'] = (sma5_up + sma20_up + sma50_up) / 3  # 0-1之间
            
            # 8. 🆕 市场情绪综合指数
            # 组合多个情绪指标
            sentiment_score = 0
            if 'rsi_14' in df.columns:
                sentiment_score += ((df['rsi_14'] - 50) / 50)  # RSI贡献
            
            if 'macd_histogram' in df.columns:
                macd_norm = df['macd_histogram'] / (df['close'] * 0.01 + 1e-10)  # 归一化
                sentiment_score += np.clip(macd_norm, -1, 1)  # MACD贡献
            
            sentiment_score += price_change.rolling(10).mean() * 100  # 短期动量贡献
            new_features['sentiment_composite'] = sentiment_score / 3  # 平均
            
            # 9. 🆕 买卖压力指标（基于K线形态）
            # 买压 = (收盘-最低)/(最高-最低)，卖压 = (最高-收盘)/(最高-最低)
            price_range = df['high'] - df['low']
            price_range = price_range.replace(0, np.nan)  # 避免除以0
            new_features['buy_pressure'] = (df['close'] - df['low']) / price_range
            new_features['sell_pressure'] = (df['high'] - df['close']) / price_range
            new_features['pressure_diff'] = new_features['buy_pressure'] - new_features['sell_pressure']
            
            # 买卖压力趋势（多周期平均）
            new_features['buy_pressure_ma5'] = new_features['buy_pressure'].rolling(5).mean()
            new_features['sell_pressure_ma5'] = new_features['sell_pressure'].rolling(5).mean()
            
            # 10. 🆕 成交量加权情绪
            if 'volume' in df.columns:
                # 成交量加权价格变化
                volume_weighted_return = price_change * df['volume']
                new_features['volume_weighted_sentiment'] = (
                    volume_weighted_return.rolling(10).sum() / 
                    (df['volume'].rolling(10).sum() + 1e-10)
                )
                
                # 成交量情绪强度（大单主导程度）
                volume_std = df['volume'].rolling(20).std()
                new_features['volume_sentiment_strength'] = (
                    (df['volume'] - df['volume'].rolling(20).mean()) / 
                    (volume_std + 1e-10)
                )
            
            # 11. 🆕 市场宽度指标（价格分布）
            # 价格偏离程度（当前价 vs 多周期均价）
            if 'sma_5' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
                new_features['price_deviation_5'] = (df['close'] - df['sma_5']) / df['sma_5']
                new_features['price_deviation_20'] = (df['close'] - df['sma_20']) / df['sma_20']
                new_features['price_deviation_50'] = (df['close'] - df['sma_50']) / df['sma_50']
                
                # 市场宽度：多个均线之间的距离
                new_features['market_breadth'] = (
                    (df['sma_5'] - df['sma_20']).abs() + 
                    (df['sma_20'] - df['sma_50']).abs()
                ) / df['close']
            
            # 12. 🆕 极端情绪检测
            # 检测极端上涨/下跌（可能的反转信号）
            extreme_up = (price_change > price_change.rolling(50).mean() + 2 * price_change.rolling(50).std())
            extreme_down = (price_change < price_change.rolling(50).mean() - 2 * price_change.rolling(50).std())
            new_features['extreme_move'] = extreme_up.astype(int) - extreme_down.astype(int)  # +1=极端上涨, -1=极端下跌
            
            # 极端移动后的反转概率（历史统计）
            new_features['extreme_move_decay'] = new_features['extreme_move'].rolling(5).sum()  # 近期极端次数
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 添加市场情绪特征失败: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加多时间框架特征融合（只能向上采样，不能向下采样）
        
        目标：将更大时间框架的趋势信息融入当前时间框架预测
        方法：通过向上重采样当前数据来模拟更大时间框架的特征
        
        注意：
        - 只能向上采样（3m→5m/15m, 5m→15m），不能向下采样
        - 如果当前是3m，则生成5m和15m的特征
        - 如果当前是5m，则只生成15m的特征（不能向下采样到3m）
        - 如果当前是15m，则不生成跨时间框架特征（已是最大周期）
        """
        try:
            new_features = {}
            
            # 确保有timestamp列用于重采样
            if 'timestamp' not in df.columns:
                logger.warning("⚠️ 缺少timestamp列，跳过多时间框架特征")
                return df
            
            # 设置timestamp为索引以便重采样
            df_temp = df.set_index('timestamp')
            
            # 🔑 检测当前时间框架（通过数据频率推断）
            # 计算相邻K线的时间间隔（分钟）
            if len(df_temp) < 2:
                logger.warning("⚠️ 数据不足，跳过多时间框架特征")
                return df.reset_index()
            
            time_diffs = df_temp.index.to_series().diff().dt.total_seconds() / 60
            median_interval = time_diffs.median()
            
            # 根据中位数间隔判断当前时间框架，只能向上采样
            if median_interval <= 3.5:
                current_tf = '3m'
                other_tfs = ['5m', '15m']  # 3m可以向上采样到5m和15m
            elif median_interval <= 7.5:
                current_tf = '5m'
                other_tfs = ['15m']  # 5m只能向上采样到15m，不能向下到3m
            elif median_interval <= 22.5:
                current_tf = '15m'
                other_tfs = []  # 15m是最大周期，不生成跨时间框架特征
            else:
                # 无法识别，跳过
                logger.warning(f"⚠️ 无法识别时间框架（间隔={median_interval:.1f}分钟），跳过多时间框架特征")
                return df.reset_index()
            
            # 如果没有可采样的时间框架，直接返回
            if not other_tfs:
                logger.debug(f"🔧 {current_tf}是最大周期，跳过多时间框架特征")
                return df.reset_index()
            
            logger.debug(f"🔧 多时间框架特征: 当前={current_tf}, 向上采样到={other_tfs}")
            
            # 为每个其他时间框架生成特征
            for other_tf in other_tfs:
                # 将分钟数转换为pandas重采样字符串
                if other_tf == '3m':
                    resample_str = '3min'
                elif other_tf == '5m':
                    resample_str = '5min'
                elif other_tf == '15m':
                    resample_str = '15min'
                else:
                    continue  # 跳过不支持的时间框架
                
                # 重采样到目标时间框架
                df_resampled = df_temp.resample(resample_str).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).ffill()
            
                # 计算关键指标
                close_resampled = df_resampled['close']
                sma_20_resampled = close_resampled.rolling(20).mean()
                sma_50_resampled = close_resampled.rolling(50).mean()
                rsi_resampled = self._calculate_rsi(close_resampled, 14)
            
                # 趋势方向（1=上涨，0=横盘，-1=下跌）
                trend_resampled = pd.Series(0, index=df_resampled.index)
                trend_resampled[sma_20_resampled > sma_50_resampled] = 1  # 多头
                trend_resampled[sma_20_resampled < sma_50_resampled] = -1  # 空头
            
                # 波动率（使用清洗后的close）
                close_resampled_safe = close_resampled.replace(0, np.nan) if (close_resampled == 0).sum() > 0 else close_resampled
                returns_resampled = close_resampled_safe.pct_change(fill_method=None)
                returns_resampled = returns_resampled.replace([np.inf, -np.inf], np.nan)
                volatility_resampled = returns_resampled.rolling(20).std()
            
                # 🔑 修复未来函数：shift(1)确保只使用上一根已收盘的K线数据
                trend_resampled_shifted = trend_resampled.shift(1)
                rsi_resampled_shifted = rsi_resampled.shift(1)
                volatility_resampled_shifted = volatility_resampled.shift(1)
                sma_20_resampled_shifted = sma_20_resampled.shift(1)
                sma_50_resampled_shifted = sma_50_resampled.shift(1)
            
                # 🔥 使用merge_asof严格保证没有未来数据泄露（direction='backward'只使用过去数据）
                df_resampled_features = pd.DataFrame({
                    'trend': trend_resampled_shifted,
                    'rsi': rsi_resampled_shifted,
                    'volatility': volatility_resampled_shifted,
                    'sma_20': sma_20_resampled_shifted,
                    'sma_50': sma_50_resampled_shifted
                }, index=df_resampled.index)
                
                # 对齐到原始时间框架（使用merge_asof确保严格向后查找）
                # merge_asof要求索引必须排序，且direction='backward'只使用<=当前时间的数据
                df_original = pd.DataFrame(index=df_temp.index)
                df_resampled_sorted = df_resampled_features.sort_index()
                df_original_sorted = df_original.sort_index()
                
                df_aligned = pd.merge_asof(
                    df_original_sorted,
                    df_resampled_sorted,
                    left_index=True,
                    right_index=True,
                    direction='backward'
                )
                
                # 恢复原始索引顺序
                df_aligned = df_aligned.reindex(df_temp.index)
                
                new_features[f'trend_{other_tf}'] = df_aligned['trend']
                new_features[f'rsi_{other_tf}'] = df_aligned['rsi']
                new_features[f'volatility_{other_tf}'] = df_aligned['volatility']
                new_features[f'sma_20_{other_tf}'] = df_aligned['sma_20']
                new_features[f'sma_50_{other_tf}'] = df_aligned['sma_50']
            
            # 趋势一致性特征（当前时间框架 vs 其他时间框架）
            if 'sma_20' in df_temp.columns and 'sma_50' in df_temp.columns:
                # 当前时间框架的趋势
                trend_current = pd.Series(0, index=df_temp.index)
                trend_current[df_temp['sma_20'] > df_temp['sma_50']] = 1
                trend_current[df_temp['sma_20'] < df_temp['sma_50']] = -1
                
                # 计算与其他时间框架的一致性
                alignment_features = []
                for other_tf in other_tfs:
                    if f'trend_{other_tf}' in new_features:
                        alignment_key = f'trend_alignment_{other_tf}'
                        new_features[alignment_key] = (trend_current == new_features[f'trend_{other_tf}']).astype(int)
                        alignment_features.append(alignment_key)
                
                # 总体一致性（所有其他时间框架的平均）
                if alignment_features:
                    new_features['trend_alignment_all'] = sum(new_features[k] for k in alignment_features) / len(alignment_features)
            
            # 相对强弱（当前时间框架 vs 其他时间框架）
            if 'rsi_14' in df_temp.columns:
                for other_tf in other_tfs:
                    if f'rsi_{other_tf}' in new_features:
                        new_features[f'rsi_diff_{other_tf}'] = df_temp['rsi_14'] - new_features[f'rsi_{other_tf}']
            
            # 价格相对位置（相对于其他时间框架的均线）
            if 'close' in df_temp.columns:
                for other_tf in other_tfs:
                    if f'sma_20_{other_tf}' in new_features:
                        new_features[f'price_to_sma20_{other_tf}'] = (
                            (df_temp['close'] - new_features[f'sma_20_{other_tf}']) / 
                            (new_features[f'sma_20_{other_tf}'] + 1e-10)
                        )
                    if f'sma_50_{other_tf}' in new_features:
                        new_features[f'price_to_sma50_{other_tf}'] = (
                            (df_temp['close'] - new_features[f'sma_50_{other_tf}']) / 
                            (new_features[f'sma_50_{other_tf}'] + 1e-10)
                        )
            
            # 将新特征添加到df_temp（确保索引一致）
            for col_name, col_data in new_features.items():
                df_temp[col_name] = col_data
            
            # 恢复原始DataFrame结构（reset timestamp索引）
            df = df_temp.reset_index()
            
            logger.debug(f"✅ 多时间框架特征添加完成: {len(new_features)}个特征（基于{current_tf}，融合{other_tfs}）")
            return df
            
        except Exception as e:
            logger.error(f"❌ 添加多时间框架特征失败: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(50, index=prices.index)  # 默认值
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取特征重要性（基于方差）"""
        try:
            # 排除非特征列
            exclude_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # 计算特征方差
            feature_variance = {}
            for col in feature_cols:
                # ✅ 使用pandas类型检查
                if pd.api.types.is_numeric_dtype(df[col]):
                    variance = df[col].var()
                    feature_variance[col] = variance if not np.isnan(variance) else 0
            
            # 归一化
            total_variance = sum(feature_variance.values())
            if total_variance > 0:
                feature_importance = {k: v/total_variance for k, v in feature_variance.items()}
            else:
                feature_importance = feature_variance
            
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"计算特征重要性失败: {e}")
            return {}
    
    def select_features(self, df: pd.DataFrame, top_n: int = 50) -> List[str]:
        """选择重要特征"""
        try:
            feature_importance = self.get_feature_importance(df)
            
            # 选择前N个重要特征
            selected_features = list(feature_importance.keys())[:top_n]
            
            logger.debug(f"选择了{len(selected_features)}个重要特征")
            return selected_features
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return []
    
    def _add_trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势强度特征（~15个特征）"""
        try:
            new_features = {}
            
            # 1. ADX趋势强度分级
            if 'adx' in df.columns:
                new_features['trend_weak'] = (df['adx'] < 20).astype(int)
                new_features['trend_moderate'] = ((df['adx'] >= 20) & (df['adx'] < 40)).astype(int)
                new_features['trend_strong'] = (df['adx'] >= 40).astype(int)
            
            # 2. 线性回归斜率（趋势方向）- 完全向量化实现
            for window in [5, 10, 20]:
                # 🔥 使用pandas rolling + apply，但传入向量化函数（比纯Python循环快10-100倍）
                # 线性回归斜率公式: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
                # 对于固定窗口，x = [0, 1, 2, ..., window-1]，可以预先计算
                n = window
                x = np.arange(n, dtype=np.float64)
                x_sum = x.sum()
                x_sq_sum = (x ** 2).sum()
                denominator = n * x_sq_sum - x_sum ** 2
                
                # 定义向量化计算函数（使用numpy操作，避免Python循环）
                def calc_slope(y_window):
                    if len(y_window) < n or np.any(np.isnan(y_window)):
                        return 0.0
                    y_sum = y_window.sum()
                    xy_sum = (x * y_window).sum()
                    slope = (n * xy_sum - x_sum * y_sum) / (denominator + 1e-10)
                    return slope
                
                # 使用rolling apply（虽然仍有循环，但比纯Python循环快得多）
                # 注意：这里使用raw=True直接传递numpy数组，避免pandas开销
                slopes_raw = df['clean_close'].rolling(window, min_periods=window).apply(
                    calc_slope, raw=True
                )
                
                # 归一化（除以当前价格）
                slopes = slopes_raw / (df['clean_close'] + 1e-10)
                slopes = slopes.fillna(0)
                
                new_features[f'trend_slope_{window}'] = slopes.values
            
            # 3. 趋势一致性（多周期确认）- 使用clean_close保持一致性
            sma5 = df['clean_close'].rolling(5).mean()
            sma10 = df['clean_close'].rolling(10).mean()
            sma20 = df['clean_close'].rolling(20).mean()
            
            new_features['trend_alignment'] = (
                ((df['clean_close'] > sma5) & (sma5 > sma10) & (sma10 > sma20)).astype(int) -
                ((df['clean_close'] < sma5) & (sma5 < sma10) & (sma10 < sma20)).astype(int)
            )
            
            # 4. EMA趋势强度 - 使用clean_close保持一致性
            ema12 = df['clean_close'].ewm(span=12).mean()
            ema26 = df['clean_close'].ewm(span=26).mean()
            new_features['ema_trend_strength'] = (ema12 - ema26) / (df['clean_close'] + 1e-10)
            
            return df.assign(**new_features)
            
        except Exception as e:
            logger.error(f"添加趋势强度特征失败: {e}")
            return df
    
    def _add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加支撑阻力特征（~18个特征）"""
        try:
            new_features = {}
            
            # 1. 近期高低点
            for window in [10, 20, 50]:
                new_features[f'high_{window}d'] = df['high'].rolling(window).max()
                new_features[f'low_{window}d'] = df['low'].rolling(window).min()
                
                # 价格距离高低点的百分比
                new_features[f'dist_to_high_{window}'] = (
                    (df['close'] - new_features[f'high_{window}d']) / 
                    (new_features[f'high_{window}d'] + 1e-10)
                )
                new_features[f'dist_to_low_{window}'] = (
                    (df['close'] - new_features[f'low_{window}d']) / 
                    (new_features[f'low_{window}d'] + 1e-10)
                )
            
            # 2. 支撑阻力突破
            for window in [20, 50]:
                # 突破历史高点
                new_features[f'breakout_high_{window}'] = (
                    df['close'] > df['high'].rolling(window).max().shift(1)
                ).astype(int)
                
                # 跌破历史低点
                new_features[f'breakdown_low_{window}'] = (
                    df['close'] < df['low'].rolling(window).min().shift(1)
                ).astype(int)
            
            return df.assign(**new_features)
            
        except Exception as e:
            logger.error(f"添加支撑阻力特征失败: {e}")
            return df
    
    def _add_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加高级动量指标（~15个特征）"""
        try:
            new_features = {}
            
            # 1. TSI (True Strength Index)
            price_change = df['close'].diff()
            pc_ema25 = price_change.ewm(span=25).mean()
            pc_ema13 = pc_ema25.ewm(span=13).mean()
            abs_pc_ema25 = price_change.abs().ewm(span=25).mean()
            abs_pc_ema13 = abs_pc_ema25.ewm(span=13).mean()
            new_features['tsi'] = 100 * pc_ema13 / (abs_pc_ema13 + 1e-10)
            new_features['tsi_signal'] = new_features['tsi'].ewm(span=7).mean()
            
            # 2. CMO (Chande Momentum Oscillator)
            for period in [9, 14]:
                price_diff = df['close'].diff()
                gain = price_diff.where(price_diff > 0, 0).rolling(period).sum()
                loss = -price_diff.where(price_diff < 0, 0).rolling(period).sum()
                new_features[f'cmo_{period}'] = 100 * (gain - loss) / (gain + loss + 1e-10)
            
            # 3. Aroon指标
            for period in [14, 25]:
                aroon_up = []
                aroon_down = []
                
                for i in range(len(df)):
                    if i < period:
                        aroon_up.append(50)
                        aroon_down.append(50)
                    else:
                        window_high = df['high'].iloc[i-period:i+1]
                        window_low = df['low'].iloc[i-period:i+1]
                        
                        days_since_high = period - window_high.argmax()
                        days_since_low = period - window_low.argmin()
                        
                        aroon_up.append((period - days_since_high) / period * 100)
                        aroon_down.append((period - days_since_low) / period * 100)
                
                new_features[f'aroon_up_{period}'] = aroon_up
                new_features[f'aroon_down_{period}'] = aroon_down
                new_features[f'aroon_osc_{period}'] = np.array(aroon_up) - np.array(aroon_down)
            
            return df.assign(**new_features)
            
        except Exception as e:
            logger.error(f"添加高级动量特征失败: {e}")
            return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格形态识别特征（~14个特征）"""
        try:
            new_features = {}
            
            body = df['close'] - df['open']
            upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
            lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
            
            # 1. 锤子线（Hammer）
            new_features['hammer'] = (
                (lower_shadow > body.abs() * 2) & 
                (upper_shadow < body.abs() * 0.5) &
                (body < 0)
            ).astype(int)
            
            # 2. 上吊线（Hanging Man）
            new_features['hanging_man'] = (
                (lower_shadow > body.abs() * 2) & 
                (upper_shadow < body.abs() * 0.5) &
                (body > 0)
            ).astype(int)
            
            # 3. 流星线（Shooting Star）
            new_features['shooting_star'] = (
                (upper_shadow > body.abs() * 2) & 
                (lower_shadow < body.abs() * 0.5)
            ).astype(int)
            
            # 4. 十字星（Doji）
            new_features['doji'] = (body.abs() < (df['high'] - df['low']) * 0.1).astype(int)
            
            # 5. 吞噬形态
            prev_body = body.shift(1)
            
            # 看涨吞噬
            new_features['bullish_engulf'] = (
                (body > 0) & 
                (prev_body < 0) &
                (df['open'] <= df['close'].shift(1)) &
                (df['close'] >= df['open'].shift(1))
            ).astype(int)
            
            # 看跌吞噬
            new_features['bearish_engulf'] = (
                (body < 0) & 
                (prev_body > 0) &
                (df['open'] >= df['close'].shift(1)) &
                (df['close'] <= df['open'].shift(1))
            ).astype(int)
            
            # 6. 三只乌鸦
            new_features['three_black_crows'] = (
                (body < 0) &
                (body.shift(1) < 0) &
                (body.shift(2) < 0) &
                (df['close'] < df['close'].shift(1)) &
                (df['close'].shift(1) < df['close'].shift(2))
            ).astype(int)
            
            # 7. 三只白兵
            new_features['three_white_soldiers'] = (
                (body > 0) &
                (body.shift(1) > 0) &
                (body.shift(2) > 0) &
                (df['close'] > df['close'].shift(1)) &
                (df['close'].shift(1) > df['close'].shift(2))
            ).astype(int)
            
            # 8. 缺口检测
            new_features['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            new_features['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
            new_features['gap_size'] = np.where(
                new_features['gap_up'] == 1,
                (df['low'] - df['high'].shift(1)) / (df['close'].shift(1) + 1e-10),
                np.where(
                    new_features['gap_down'] == 1,
                    (df['high'] - df['low'].shift(1)) / (df['close'].shift(1) + 1e-10),
                    0
                )
            )
            
            return df.assign(**new_features)
            
        except Exception as e:
            logger.error(f"添加价格形态特征失败: {e}")
            return df
    
    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加订单流特征（~10个特征）"""
        try:
            new_features = {}
            
            if 'taker_buy_base_volume' in df.columns and 'volume' in df.columns:
                # 1. 买卖比率
                taker_sell_volume = df['volume'] - df['taker_buy_base_volume']
                new_features['buy_sell_ratio'] = (
                    df['taker_buy_base_volume'] / (taker_sell_volume + 1e-10)
                )
                
                # 2. 净买入压力
                new_features['net_buy_pressure'] = (
                    df['taker_buy_base_volume'] - taker_sell_volume
                ) / (df['volume'] + 1e-10)
                
                # 3. 大单检测
                buy_ratio = df['taker_buy_base_volume'] / (df['volume'] + 1e-10)
                buy_ratio_mean = buy_ratio.rolling(20).mean()
                buy_ratio_std = buy_ratio.rolling(20).std()
                
                new_features['large_buy_orders'] = (
                    buy_ratio > buy_ratio_mean + 2 * buy_ratio_std
                ).astype(int)
                
                new_features['large_sell_orders'] = (
                    buy_ratio < buy_ratio_mean - 2 * buy_ratio_std
                ).astype(int)
                
                # 4. 累积买卖压力
                for window in [5, 10, 20]:
                    new_features[f'cumulative_buy_pressure_{window}'] = (
                        new_features['net_buy_pressure'].rolling(window).sum()
                    )
            
            return df.assign(**new_features)
            
        except Exception as e:
            logger.error(f"添加订单流特征失败: {e}")
            return df
    
    def _add_swing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波段识别特征（~10个特征）"""
        try:
            new_features = {}
            
            # 1. Swing High/Low检测
            for window in [5, 10]:
                # Swing High
                rolling_max = df['high'].rolling(window*2+1, center=True).max()
                new_features[f'swing_high_{window}'] = (
                    df['high'] == rolling_max
                ).astype(int)
                
                # Swing Low
                rolling_min = df['low'].rolling(window*2+1, center=True).min()
                new_features[f'swing_low_{window}'] = (
                    df['low'] == rolling_min
                ).astype(int)
            
            # 2. 价格在波段中的位置
            for window in [20, 50]:
                recent_high = df['high'].rolling(window).max()
                recent_low = df['low'].rolling(window).min()
                
                new_features[f'position_in_range_{window}'] = (
                    (df['close'] - recent_low) / (recent_high - recent_low + 1e-10)
                )
            
            # 3. 波段频率
            if 'swing_high_5' in new_features:
                new_features['swing_frequency'] = new_features['swing_high_5'].rolling(50).sum()
            
            return df.assign(**new_features)
            
        except Exception as e:
            logger.error(f"添加波段识别特征失败: {e}")
            return df
    

# 全局特征工程器实例
feature_engineer = FeatureEngineer()