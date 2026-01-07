"""
波动率特征模块
"""
import logging
import pandas as pd
import numpy as np
import ta

logger = logging.getLogger(__name__)


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加波动率特征"""
    try:
        new_features = {}
        
        # 历史波动率
        close_for_returns = df['clean_close']
        returns = close_for_returns.pct_change(fill_method=None)
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

