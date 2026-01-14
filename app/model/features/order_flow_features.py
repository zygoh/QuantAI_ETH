"""
订单流特征模块
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加订单流特征"""
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

