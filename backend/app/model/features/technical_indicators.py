"""
技术指标特征模块
"""
import logging
import pandas as pd
import numpy as np
import ta

logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加技术指标特征"""
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
            # 避免除以零
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

