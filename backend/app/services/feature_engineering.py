"""
特征工程模块
"""
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import ta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.feature_columns = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有特征"""
        try:
            if df.empty:
                return df
            
            logger.info(f"🔧 开始特征工程: {len(df)}行原始数据")
            
            # 处理 timestamp：如果是 index，重置为列
            if df.index.name == 'timestamp' or 'timestamp' not in df.columns:
                df = df.reset_index()
            
            # 🔥 确保 timestamp 列是统一的 datetime 类型（避免混合类型导致排序失败）
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 确保数据按时间排序
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 基础价格特征
            df = self._add_price_features(df)
            
            # 技术指标特征
            df = self._add_technical_indicators(df)
            
            # 成交量特征
            df = self._add_volume_features(df)
            
            # 时间特征
            df = self._add_time_features(df)
            
            # 市场微观结构特征
            df = self._add_microstructure_features(df)
            
            # 波动率特征
            df = self._add_volatility_features(df)
            
            # 动量特征
            df = self._add_momentum_features(df)
            
            # 🆕 市场情绪特征
            df = self._add_sentiment_features(df)
            
            # 🆕 多时间框架特征融合
            df = self._add_multi_timeframe_features(df)
            
            # 🆕 高级特征（Phase 2优化）
            df = self._add_trend_strength_features(df)
            df = self._add_support_resistance_features(df)
            df = self._add_advanced_momentum_features(df)
            df = self._add_pattern_features(df)
            df = self._add_order_flow_features(df)
            df = self._add_swing_features(df)
            
            # 🔥 第一步：处理无穷大值（inf）- 必须在NaN处理前完成
            inf_count = 0
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    inf_mask = np.isinf(df[col])
                    if inf_mask.any():
                        inf_count += inf_mask.sum()
                        # 将inf替换为NaN（后续统一处理）
                        df.loc[inf_mask, col] = np.nan
            
            if inf_count > 0:
                logger.warning(f"⚠️ 检测到{inf_count}个无穷大值（inf），已替换为NaN")
            
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
            
            # 先尝试删除NaN
            df_clean = df.dropna()
            
            # 如果删除后数据量<50行，说明是预测场景，改用填充
            if len(df_clean) < 50 and rows_before >= 100:
                logger.debug(f"⚠️ 预测场景检测：dropna会导致数据过少（{rows_before}→{len(df_clean)}），改用fillna")
                # 使用前向填充
                df = df.ffill()
                # 如果前向填充后仍有NaN（开头的行），用后向填充
                df = df.bfill()
                # 如果还有NaN，用0填充
                df = df.fillna(0)
                logger.debug(f"✅ 特征工程完成（预测模式）: {len(df)}行，特征数: {len(df.columns)}")
            else:
                # 训练场景，正常删除NaN
                df = df_clean
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    logger.info(f"✅ 特征工程完成: {len(df)}行（因NaN/Inf丢弃{rows_dropped}行），特征数: {len(df.columns)}")
                else:
                    logger.info(f"✅ 特征工程完成: {len(df)}行，特征数: {len(df.columns)}")

            
            return df
            
        except Exception as e:
            logger.error(f"❌ 特征工程失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格特征 - 优化性能"""
        try:
            new_features = {}
            
            # 价格变化率
            price_change = df['close'].pct_change()
            new_features['price_change'] = price_change
            new_features['price_change_abs'] = price_change.abs()
            
            # 价格范围
            new_features['price_range'] = (df['high'] - df['low']) / df['close']
            # 注：upper_shadow 和 lower_shadow 在市场微观结构特征中添加（更好的归一化）
            
            # 开盘价与收盘价关系
            new_features['open_close_ratio'] = df['open'] / df['close']
            new_features['body_size'] = abs(df['close'] - df['open']) / df['close']
            
            # 价格位置（避免除以零）
            price_range_safe = df['high'] - df['low']
            price_range_safe = price_range_safe.replace(0, np.nan)  # 零范围设为NaN
            new_features['close_position'] = (df['close'] - df['low']) / price_range_safe
            
            # 多周期价格变化
            for period in [2, 3, 5, 10, 20]:
                new_features[f'price_change_{period}'] = df['close'].pct_change(period)
                new_features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
            
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
                new_features[f'price_sma_ratio_{period}'] = df['close'] / sma
                new_features[f'price_ema_ratio_{period}'] = df['close'] / ema
            
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
            volume_change = df['volume'].pct_change()
            volume_sma_5 = df['volume'].rolling(5).mean()
            volume_sma_20 = df['volume'].rolling(20).mean()
            
            new_features['volume_change'] = volume_change
            new_features['volume_sma_5'] = volume_sma_5
            new_features['volume_sma_20'] = volume_sma_20
            new_features['volume_ratio'] = df['volume'] / volume_sma_20
            
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
            
            # Volume Weighted Average Price (VWAP)
            vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            new_features['vwap'] = vwap
            new_features['price_vwap_ratio'] = df['close'] / vwap
            
            # ✅ 成交量突破（捕捉放量信号）
            volume_ma_5 = df['volume'].rolling(5).mean()
            volume_ma_20 = df['volume'].rolling(20).mean()
            new_features['volume_spike'] = df['volume'] / volume_ma_20
            new_features['volume_trend'] = volume_ma_5 / volume_ma_20
            
            # ✅ 价格-成交量背离（重要信号）
            price_change_1 = df['close'].pct_change(1)  # 定义价格变化率
            price_change_5 = df['close'].pct_change(5)
            volume_change_5 = df['volume'].pct_change(5)
            new_features['price_volume_correlation'] = price_change_5 * volume_change_5  # 同向为正，背离为负（连续值）
            
            # ✅ 成交量加权价格变化（结合量价）
            new_features['volume_weighted_price_change'] = price_change_1 * (df['volume'] / volume_ma_20)
            
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
            returns = df['close'].pct_change()
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
            import traceback
            logger.error(traceback.format_exc())
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征 - 优化性能"""
        try:
            new_features = {}
            
            # 历史波动率
            returns = df['close'].pct_change()
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
            returns = df['close'].pct_change()
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
            price_change = df['close'].pct_change()
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
                volume_trend = (df['volume'].pct_change().rolling(5).mean() > 0).astype(int)
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
            import traceback
            logger.error(traceback.format_exc())
            return df
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加多时间框架特征融合（优化目标：准确率+5-7%）
        
        目标：将长周期趋势信息融入短周期预测，减少逆势交易
        方法：通过重采样当前数据来模拟更长周期的特征
        """
        try:
            new_features = {}
            
            # 确保有timestamp列用于重采样
            if 'timestamp' not in df.columns:
                logger.warning("⚠️ 缺少timestamp列，跳过多时间框架特征")
                return df
            
            # 设置timestamp为索引以便重采样
            df_temp = df.set_index('timestamp')
            
            # 1. 模拟1h数据（长周期趋势参考，对3m/5m/15m都有用）
            # 重采样到1h并向前填充
            df_1h = df_temp.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).ffill()
            
            # 计算1h的关键指标
            close_1h = df_1h['close']
            sma_20_1h = close_1h.rolling(20).mean()
            sma_50_1h = close_1h.rolling(50).mean()
            rsi_1h = self._calculate_rsi(close_1h, 14)
            
            # 1h趋势方向（1=上涨，0=横盘，-1=下跌）
            trend_1h = pd.Series(0, index=df_1h.index)
            trend_1h[sma_20_1h > sma_50_1h] = 1  # 多头
            trend_1h[sma_20_1h < sma_50_1h] = -1  # 空头
            
            # 1h波动率
            returns_1h = close_1h.pct_change()
            volatility_1h = returns_1h.rolling(20).std()
            
            # 将1h数据对齐到原始时间框架
            new_features['trend_1h'] = trend_1h.reindex(df_temp.index, method='ffill')
            new_features['rsi_1h'] = rsi_1h.reindex(df_temp.index, method='ffill')
            new_features['volatility_1h'] = volatility_1h.reindex(df_temp.index, method='ffill')
            new_features['sma_20_1h'] = sma_20_1h.reindex(df_temp.index, method='ffill')
            new_features['sma_50_1h'] = sma_50_1h.reindex(df_temp.index, method='ffill')
            
            # 2. 模拟15m数据（中期趋势参考，对3m/5m有用）
            df_15m = df_temp.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).ffill()
            
            close_15m = df_15m['close']
            sma_20_15m = close_15m.rolling(20).mean()
            sma_50_15m = close_15m.rolling(50).mean()
            rsi_15m = self._calculate_rsi(close_15m, 14)
            
            # 15m趋势方向
            trend_15m = pd.Series(0, index=df_15m.index)
            trend_15m[sma_20_15m > sma_50_15m] = 1
            trend_15m[sma_20_15m < sma_50_15m] = -1
            
            # 15m波动率
            returns_15m = close_15m.pct_change()
            volatility_15m = returns_15m.rolling(20).std()
            
            # 将15m数据对齐到原始时间框架
            new_features['trend_15m'] = trend_15m.reindex(df_temp.index, method='ffill')
            new_features['rsi_15m'] = rsi_15m.reindex(df_temp.index, method='ffill')
            new_features['volatility_15m'] = volatility_15m.reindex(df_temp.index, method='ffill')
            new_features['sma_20_15m'] = sma_20_15m.reindex(df_temp.index, method='ffill')
            new_features['sma_50_15m'] = sma_50_15m.reindex(df_temp.index, method='ffill')
            
            # 3. 趋势一致性特征（短中长周期是否一致）
            if 'sma_20' in df_temp.columns and 'sma_50' in df_temp.columns:
                # 当前时间框架的趋势（使用df_temp避免索引不匹配）
                trend_current = pd.Series(0, index=df_temp.index)
                trend_current[df_temp['sma_20'] > df_temp['sma_50']] = 1
                trend_current[df_temp['sma_20'] < df_temp['sma_50']] = -1
                
                # 多时间框架趋势一致性
                new_features['trend_alignment_15m'] = (trend_current == new_features['trend_15m']).astype(int)
                new_features['trend_alignment_1h'] = (trend_current == new_features['trend_1h']).astype(int)
                new_features['trend_alignment_all'] = (
                    (new_features['trend_alignment_15m'] + new_features['trend_alignment_1h']) / 2
                )
            
            # 4. 相对强弱（当前时间框架 vs 更长周期）
            if 'rsi_14' in df_temp.columns:
                new_features['rsi_diff_15m'] = df_temp['rsi_14'] - new_features['rsi_15m']
                new_features['rsi_diff_1h'] = df_temp['rsi_14'] - new_features['rsi_1h']
            
            # 5. 价格相对位置（相对于更长周期均线）
            if 'close' in df_temp.columns:
                new_features['price_to_sma20_15m'] = (df_temp['close'] - new_features['sma_20_15m']) / new_features['sma_20_15m']
                new_features['price_to_sma50_15m'] = (df_temp['close'] - new_features['sma_50_15m']) / new_features['sma_50_15m']
                new_features['price_to_sma20_1h'] = (df_temp['close'] - new_features['sma_20_1h']) / new_features['sma_20_1h']
                new_features['price_to_sma50_1h'] = (df_temp['close'] - new_features['sma_50_1h']) / new_features['sma_50_1h']
            
            # 将新特征添加到df_temp（确保索引一致）
            for col_name, col_data in new_features.items():
                df_temp[col_name] = col_data
            
            # 恢复原始DataFrame结构（reset timestamp索引）
            df = df_temp.reset_index()
            
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 添加多时间框架特征失败: {e}")
            import traceback
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
            
            # 2. 线性回归斜率（趋势方向）
            for window in [5, 10, 20]:
                slopes = []
                for i in range(len(df)):
                    if i < window:
                        slopes.append(0)
                    else:
                        y = df['close'].iloc[i-window:i].values
                        x = np.arange(window)
                        slope = np.polyfit(x, y, 1)[0] / (df['close'].iloc[i] + 1e-10)
                        slopes.append(slope)
                new_features[f'trend_slope_{window}'] = slopes
            
            # 3. 趋势一致性（多周期确认）
            sma5 = df['close'].rolling(5).mean()
            sma10 = df['close'].rolling(10).mean()
            sma20 = df['close'].rolling(20).mean()
            
            new_features['trend_alignment'] = (
                ((df['close'] > sma5) & (sma5 > sma10) & (sma10 > sma20)).astype(int) -
                ((df['close'] < sma5) & (sma5 < sma10) & (sma10 < sma20)).astype(int)
            )
            
            # 4. EMA趋势强度
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            new_features['ema_trend_strength'] = (ema12 - ema26) / (df['close'] + 1e-10)
            
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