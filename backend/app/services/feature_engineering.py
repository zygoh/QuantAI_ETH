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
            
            # 移除包含NaN的行
            df = df.dropna()
            
            logger.info(f"✅ 特征工程完成: {len(df)}行，特征数: {len(df.columns)}")  # 改为INFO级别
            
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
            new_features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            new_features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
            # 开盘价与收盘价关系
            new_features['open_close_ratio'] = df['open'] / df['close']
            new_features['body_size'] = abs(df['close'] - df['open']) / df['close']
            
            # 价格位置
            new_features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # 多周期价格变化
            for period in [2, 3, 5, 10, 20]:
                new_features[f'price_change_{period}'] = df['close'].pct_change(period)
                new_features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
            
            # ✅ 价格加速度（捕捉趋势加速/减速）
            new_features['price_acceleration'] = price_change - price_change.shift(1)
            new_features['price_acceleration_3'] = price_change - price_change.shift(3)
            new_features['price_acceleration_5'] = price_change - price_change.shift(5)
            
            # ✅ 连续涨跌（捕捉趋势延续性）
            new_features['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int).rolling(5).sum()
            new_features['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int).rolling(5).sum()
            
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
                new_features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                new_features[f'bb_position_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
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
            new_features['price_volume_divergence'] = price_change_5 * volume_change_5  # 同向为正，背离为负
            
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
            new_features['price_acceleration'] = returns.diff()
            new_features['price_jerk'] = returns.diff().diff()  # 加加速度
            
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
            
            logger.info(f"✅ 市场微观结构特征已增强：新增 {len(new_features)} 个特征")  # 改为INFO级别
            
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
            new_features['kc_position'] = (df['close'] - kc_lower) / (kc_upper - kc_lower)
            
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
            
            # ✅ 趋势强度指标（ADX）
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            new_features['adx'] = adx.adx()
            new_features['adx_pos'] = adx.adx_pos()
            new_features['adx_neg'] = adx.adx_neg()
            
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
            
            # 4. 🆕 价格加速度（情绪转变）
            price_change = df['close'].pct_change()
            new_features['price_acceleration'] = price_change.diff()
            new_features['acceleration_magnitude'] = new_features['price_acceleration'].abs()
            
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
            
            # 一次性添加所有特征
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            
            logger.info(f"✅ 市场情绪特征已添加：新增 {len(new_features)} 个特征")  # 改为INFO级别
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 添加市场情绪特征失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df
    
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

# 全局特征工程器实例
feature_engineer = FeatureEngineer()