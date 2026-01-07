"""
价格形态特征模块
"""
import logging
import pandas as pd
import numpy as np

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

