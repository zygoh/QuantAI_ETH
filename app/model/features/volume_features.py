"""
成交量特征模块
"""
import logging
import pandas as pd
import numpy as np
import ta

# Local App
from app.core.constants import (
    PRICE_CHANGE_LONG_WINDOW,
    PRICE_CHANGE_SHORT_WINDOW,
    VOLUME_BREAKOUT_FAST_WINDOW,
    VOLUME_BREAKOUT_SLOW_WINDOW,
    VOLUME_CHANGE_WINDOW,
    VOLUME_OBV_SMA_WINDOW,
    VOLUME_SMA_WINDOW,
    VWAP_WINDOW
)
logger = logging.getLogger(__name__)


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加成交量特征"""
    try:
        new_features = {}
        
        # 成交量变化
        volume_zero_count = (df['volume'] == 0).sum()
        if volume_zero_count > 0:
            volume_for_pct = df['volume'].replace(0, np.nan)
        else:
            volume_for_pct = df['volume']
        
        volume_change = volume_for_pct.pct_change(fill_method=None)
        volume_change = volume_change.replace([np.inf, -np.inf], np.nan)
        
        # 计算volume_sma_20用于比率计算
        volume_sma_20 = df['volume'].rolling(VOLUME_SMA_WINDOW).mean()
        volume_sma_20_safe = volume_sma_20.replace(0, np.nan)
        
        new_features['volume_change'] = volume_change
        new_features['volume_ratio'] = df['volume'] / volume_sma_20_safe
        
        # 价量关系
        new_features['price_volume_trend'] = df['price_change'] * volume_change
        
        # OBV (On Balance Volume)
        obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        new_features['obv'] = obv
        new_features['obv_sma'] = obv.rolling(VOLUME_OBV_SMA_WINDOW).mean()
        
        # Volume Price Trend
        new_features['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        # Accumulation/Distribution Line
        new_features['ad_line'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        
        # Chaikin Money Flow
        new_features['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        
        # Volume Weighted Average Price (VWAP)
        volume_rolling_sum = df['volume'].rolling(VWAP_WINDOW).sum()
        volume_rolling_sum_safe = volume_rolling_sum.replace(0, np.nan)
        
        numerator = (df['close'] * df['volume']).rolling(VWAP_WINDOW).sum()
        vwap = numerator / volume_rolling_sum_safe
        new_features['vwap'] = vwap
        
        vwap_safe = vwap.replace(0, np.nan).replace(np.nan, 1.0)
        new_features['price_vwap_ratio'] = df['close'] / vwap_safe
        
        # 成交量突破
        volume_ma_5 = df['volume'].rolling(VOLUME_BREAKOUT_FAST_WINDOW).mean()
        volume_ma_20 = df['volume'].rolling(VOLUME_BREAKOUT_SLOW_WINDOW).mean()
        volume_ma_20_safe = volume_ma_20.replace(0, np.nan)
        
        new_features['volume_spike'] = df['volume'] / volume_ma_20_safe
        new_features['volume_trend'] = volume_ma_5 / volume_ma_20_safe
        
        # 价格-成交量背离
        close_for_pct_corr = df['clean_close']
        volume_for_pct_corr = df['volume'].replace(0, np.nan) if (df['volume'] == 0).sum() > 0 else df['volume']
        
        price_change_1 = close_for_pct_corr.pct_change(PRICE_CHANGE_SHORT_WINDOW, fill_method=None)
        price_change_1 = price_change_1.replace([np.inf, -np.inf], np.nan)
        
        price_change_5 = close_for_pct_corr.pct_change(PRICE_CHANGE_LONG_WINDOW, fill_method=None)
        price_change_5 = price_change_5.replace([np.inf, -np.inf], np.nan)
        
        volume_change_5 = volume_for_pct_corr.pct_change(VOLUME_CHANGE_WINDOW, fill_method=None)
        volume_change_5 = volume_change_5.replace([np.inf, -np.inf], np.nan)
        new_features['price_volume_correlation'] = price_change_5 * volume_change_5
        
        # 成交量加权价格变化
        new_features['volume_weighted_price_change'] = price_change_1 * (df['volume'] / volume_ma_20_safe)
        
        # 一次性添加所有特征
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"添加成交量特征失败: {e}")
        return df

