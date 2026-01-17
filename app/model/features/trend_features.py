"""
趋势特征模块（包含趋势强度和支撑阻力）
"""
import logging
import pandas as pd
import numpy as np

# Local App
from app.core.constants import (
    BREAKOUT_WINDOWS,
    SUPPORT_RESISTANCE_WINDOWS,
    TREND_ADX_STRONG,
    TREND_ADX_WEAK,
    TREND_BREAKOUT_SHIFT,
    TREND_EMA_FAST,
    TREND_EMA_SLOW,
    TREND_EPS,
    TREND_SLOPE_WINDOWS,
    TREND_SMA_WINDOWS
)
logger = logging.getLogger(__name__)


def add_trend_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加趋势强度特征"""
    try:
        new_features = {}
        
        # 1. ADX趋势强度分级
        if 'adx' in df.columns:
            new_features['trend_weak'] = (df['adx'] < TREND_ADX_WEAK).astype(int)
            new_features['trend_moderate'] = ((df['adx'] >= TREND_ADX_WEAK) & (df['adx'] < TREND_ADX_STRONG)).astype(int)
            new_features['trend_strong'] = (df['adx'] >= TREND_ADX_STRONG).astype(int)
        
        # 2. 线性回归斜率（趋势方向）
        for window in TREND_SLOPE_WINDOWS:
            n = window
            x = np.arange(n, dtype=np.float64)
            x_sum = x.sum()
            x_sq_sum = (x ** 2).sum()
            denominator = n * x_sq_sum - x_sum ** 2
            
            def calc_slope(y_window):
                if len(y_window) < n or np.any(np.isnan(y_window)):
                    return 0.0
                y_sum = y_window.sum()
                xy_sum = (x * y_window).sum()
                slope = (n * xy_sum - x_sum * y_sum) / (denominator + TREND_EPS)
                return slope
            
            slopes_raw = df['clean_close'].rolling(window, min_periods=window).apply(
                calc_slope, raw=True
            )
            
            slopes = slopes_raw / (df['clean_close'] + TREND_EPS)
            slopes = slopes.fillna(0)
            
            new_features[f'trend_slope_{window}'] = slopes.values
        
        # 3. 趋势一致性（多周期确认）
        sma5 = df['clean_close'].rolling(TREND_SMA_WINDOWS[0]).mean()
        sma10 = df['clean_close'].rolling(TREND_SMA_WINDOWS[1]).mean()
        sma20 = df['clean_close'].rolling(TREND_SMA_WINDOWS[2]).mean()
        
        new_features['trend_alignment'] = (
            ((df['clean_close'] > sma5) & (sma5 > sma10) & (sma10 > sma20)).astype(int) -
            ((df['clean_close'] < sma5) & (sma5 < sma10) & (sma10 < sma20)).astype(int)
        )
        
        # 4. EMA趋势强度
        ema12 = df['clean_close'].ewm(span=TREND_EMA_FAST).mean()
        ema26 = df['clean_close'].ewm(span=TREND_EMA_SLOW).mean()
        new_features['ema_trend_strength'] = (ema12 - ema26) / (df['clean_close'] + TREND_EPS)
        
        return df.assign(**new_features)
        
    except Exception as e:
        logger.error(f"添加趋势强度特征失败: {e}")
        return df


def add_support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加支撑阻力特征"""
    try:
        new_features = {}
        
        # 1. 近期高低点
        for window in SUPPORT_RESISTANCE_WINDOWS:
            new_features[f'high_{window}d'] = df['high'].rolling(window).max()
            new_features[f'low_{window}d'] = df['low'].rolling(window).min()
            
            new_features[f'dist_to_high_{window}'] = (
                (df['close'] - new_features[f'high_{window}d']) /
                (new_features[f'high_{window}d'] + TREND_EPS)
            )
            new_features[f'dist_to_low_{window}'] = (
                (df['close'] - new_features[f'low_{window}d']) /
                (new_features[f'low_{window}d'] + TREND_EPS)
            )
        
        # 2. 支撑阻力突破
        for window in BREAKOUT_WINDOWS:
            new_features[f'breakout_high_{window}'] = (
                df['close'] > df['high'].rolling(window).max().shift(TREND_BREAKOUT_SHIFT)
            ).astype(int)
            
            new_features[f'breakdown_low_{window}'] = (
                df['close'] < df['low'].rolling(window).min().shift(TREND_BREAKOUT_SHIFT)
            ).astype(int)
        
        return df.assign(**new_features)
        
    except Exception as e:
        logger.error(f"添加支撑阻力特征失败: {e}")
        return df

