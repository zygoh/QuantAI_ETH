"""
价格特征模块
"""
import logging
import pandas as pd
import numpy as np

# Local App
from app.core.constants import (
    FEATURE_STD_EPS,
    PRICE_ACCELERATION_SHIFTS,
    PRICE_CHANGE_PERIODS,
    PRICE_MOMENTUM_WINDOW
)

logger = logging.getLogger(__name__)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加价格特征"""
    try:
        new_features = {}
        
        # 价格变化率
        close_for_pct = df['clean_close']
        price_change = close_for_pct.pct_change(fill_method=None)
        price_change = price_change.replace([np.inf, -np.inf], np.nan)
        new_features['price_change'] = price_change
        new_features['price_change_abs'] = price_change.abs()
        
        close_safe = df['clean_close']
        new_features['price_range'] = (df['high'] - df['low']) / close_safe
        new_features['open_close_ratio'] = df['open'] / close_safe
        new_features['body_size'] = abs(df['close'] - df['open']) / close_safe
        
        # 价格位置（避免除以零）
        price_range_safe = df['high'] - df['low']
        price_range_safe = price_range_safe.replace(0, np.nan)
        new_features['close_position'] = (df['close'] - df['low']) / price_range_safe
        
        # 多周期价格变化
        for period in PRICE_CHANGE_PERIODS:
            pct_chg = close_for_pct.pct_change(period, fill_method=None)
            pct_chg = pct_chg.replace([np.inf, -np.inf], np.nan)
            new_features[f'price_change_{period}'] = pct_chg
            # 避免除以零（low可能为0）
            rolling_low_min = df['low'].rolling(period).min()
            rolling_low_safe = rolling_low_min.replace(0, np.nan)
            new_features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / rolling_low_safe
        
        # 价格加速度（一阶、三阶、五阶）
        new_features['price_acceleration'] = price_change - price_change.shift(PRICE_ACCELERATION_SHIFTS[0])
        new_features['price_acceleration_3'] = price_change - price_change.shift(PRICE_ACCELERATION_SHIFTS[1])
        new_features['price_acceleration_5'] = price_change - price_change.shift(PRICE_ACCELERATION_SHIFTS[2])
        
        # 价格动量强度
        new_features['price_momentum_strength'] = price_change.abs().rolling(PRICE_MOMENTUM_WINDOW).mean()
        new_features['price_momentum_direction'] = (
            price_change.rolling(PRICE_MOMENTUM_WINDOW).mean() /
            (price_change.rolling(PRICE_MOMENTUM_WINDOW).std() + FEATURE_STD_EPS)
        )
        
        # 一次性添加所有特征
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"添加价格特征失败: {e}")
        return df

