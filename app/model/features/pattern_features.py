"""
价格形态特征模块
"""
import logging
import pandas as pd
import numpy as np

# Local App
from app.core.constants import (
    PATTERN_DOJI_RATIO,
    PATTERN_GAP_EPS,
    PATTERN_SHIFT_1,
    PATTERN_SHIFT_2,
    PATTERN_SHADOW_MULTIPLIER,
    PATTERN_SHADOW_SMALL_RATIO
)
logger = logging.getLogger(__name__)


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加价格形态识别特征"""
    try:
        new_features = {}
        
        body = df['close'] - df['open']
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        
        # 1. 锤子线（Hammer）
        new_features['hammer'] = (
            (lower_shadow > body.abs() * PATTERN_SHADOW_MULTIPLIER) &
            (upper_shadow < body.abs() * PATTERN_SHADOW_SMALL_RATIO) &
            (body < 0)
        ).astype(int)
        
        # 2. 上吊线（Hanging Man）
        new_features['hanging_man'] = (
            (lower_shadow > body.abs() * PATTERN_SHADOW_MULTIPLIER) &
            (upper_shadow < body.abs() * PATTERN_SHADOW_SMALL_RATIO) &
            (body > 0)
        ).astype(int)
        
        # 3. 流星线（Shooting Star）
        new_features['shooting_star'] = (
            (upper_shadow > body.abs() * PATTERN_SHADOW_MULTIPLIER) &
            (lower_shadow < body.abs() * PATTERN_SHADOW_SMALL_RATIO)
        ).astype(int)
        
        # 4. 十字星（Doji）
        new_features['doji'] = (body.abs() < (df['high'] - df['low']) * PATTERN_DOJI_RATIO).astype(int)
        
        # 5. 吞噬形态
        prev_body = body.shift(PATTERN_SHIFT_1)
        
        # 看涨吞噬
        new_features['bullish_engulf'] = (
            (body > 0) & 
            (prev_body < 0) &
            (df['open'] <= df['close'].shift(PATTERN_SHIFT_1)) &
            (df['close'] >= df['open'].shift(PATTERN_SHIFT_1))
        ).astype(int)
        
        # 看跌吞噬
        new_features['bearish_engulf'] = (
            (body < 0) & 
            (prev_body > 0) &
            (df['open'] >= df['close'].shift(PATTERN_SHIFT_1)) &
            (df['close'] <= df['open'].shift(PATTERN_SHIFT_1))
        ).astype(int)
        
        # 6. 三只乌鸦
        new_features['three_black_crows'] = (
            (body < 0) &
            (body.shift(PATTERN_SHIFT_1) < 0) &
            (body.shift(PATTERN_SHIFT_2) < 0) &
            (df['close'] < df['close'].shift(PATTERN_SHIFT_1)) &
            (df['close'].shift(PATTERN_SHIFT_1) < df['close'].shift(PATTERN_SHIFT_2))
        ).astype(int)
        
        # 7. 三只白兵
        new_features['three_white_soldiers'] = (
            (body > 0) &
            (body.shift(PATTERN_SHIFT_1) > 0) &
            (body.shift(PATTERN_SHIFT_2) > 0) &
            (df['close'] > df['close'].shift(PATTERN_SHIFT_1)) &
            (df['close'].shift(PATTERN_SHIFT_1) > df['close'].shift(PATTERN_SHIFT_2))
        ).astype(int)
        
        # 8. 缺口检测
        new_features['gap_up'] = (df['low'] > df['high'].shift(PATTERN_SHIFT_1)).astype(int)
        new_features['gap_down'] = (df['high'] < df['low'].shift(PATTERN_SHIFT_1)).astype(int)
        new_features['gap_size'] = np.where(
            new_features['gap_up'] == 1,
            (df['low'] - df['high'].shift(PATTERN_SHIFT_1)) / (df['close'].shift(PATTERN_SHIFT_1) + PATTERN_GAP_EPS),
            np.where(
                new_features['gap_down'] == 1,
                (df['high'] - df['low'].shift(PATTERN_SHIFT_1)) / (df['close'].shift(PATTERN_SHIFT_1) + PATTERN_GAP_EPS),
                0
            )
        )
        
        return df.assign(**new_features)
        
    except Exception as e:
        logger.error(f"添加价格形态特征失败: {e}")
        return df

