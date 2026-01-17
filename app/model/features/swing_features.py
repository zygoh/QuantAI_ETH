"""
波段识别特征模块
"""
import logging
import pandas as pd
import numpy as np

# Local App
from app.core.constants import (
    SWING_CENTER_MULTIPLIER,
    SWING_FREQUENCY_WINDOW,
    SWING_RANGE_EPS,
    SWING_RANGE_WINDOWS,
    SWING_WINDOWS
)
logger = logging.getLogger(__name__)


def add_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加波段识别特征"""
    try:
        new_features = {}
        
        # 1. Swing High/Low检测
        for window in SWING_WINDOWS:
            # Swing High
            rolling_max = df['high'].rolling(window * SWING_CENTER_MULTIPLIER + 1, center=True).max()
            new_features[f'swing_high_{window}'] = (
                df['high'] == rolling_max
            ).astype(int)
            
            # Swing Low
            rolling_min = df['low'].rolling(window * SWING_CENTER_MULTIPLIER + 1, center=True).min()
            new_features[f'swing_low_{window}'] = (
                df['low'] == rolling_min
            ).astype(int)
        
        # 2. 价格在波段中的位置
        for window in SWING_RANGE_WINDOWS:
            recent_high = df['high'].rolling(window).max()
            recent_low = df['low'].rolling(window).min()
            
            new_features[f'position_in_range_{window}'] = (
                (df['close'] - recent_low) / (recent_high - recent_low + SWING_RANGE_EPS)
            )
        
        # 3. 波段频率
        if 'swing_high_5' in new_features:
            new_features['swing_frequency'] = new_features['swing_high_5'].rolling(SWING_FREQUENCY_WINDOW).sum()
        
        return df.assign(**new_features)
        
    except Exception as e:
        logger.error(f"添加波段识别特征失败: {e}")
        return df

