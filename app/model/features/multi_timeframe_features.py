"""
多时间框架特征模块
"""
# StdLib
import logging
import traceback

# Third-Party
import numpy as np
import pandas as pd

# Local App
from app.model.features.utils import calculate_rsi
from app.core.constants import (
    FEATURE_EPS,
    MTF_INTERVAL_15M_MAX,
    MTF_INTERVAL_3M_MAX,
    MTF_INTERVAL_5M_MAX,
    MTF_RESAMPLE_MAP,
    MTF_RSI_WINDOW,
    MTF_SHIFT_LAG,
    MTF_SMA_FAST_WINDOW,
    MTF_SMA_SLOW_WINDOW,
    MTF_VOLATILITY_WINDOW
)

logger = logging.getLogger(__name__)


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加多时间框架特征融合（只能向上采样，不能向下采样）"""
    try:
        new_features = {}
        
        if 'timestamp' not in df.columns:
            logger.warning("缺少timestamp列，跳过多时间框架特征")
            return df
        
        df_temp = df.set_index('timestamp')
        
        if len(df_temp) < 2:
            logger.warning("数据不足，跳过多时间框架特征")
            return df.reset_index()
        
        time_diffs = df_temp.index.to_series().diff().dt.total_seconds() / 60
        median_interval = time_diffs.median()
        
        if median_interval <= MTF_INTERVAL_3M_MAX:
            current_tf = '3m'
            other_tfs = ['5m', '15m']
        elif median_interval <= MTF_INTERVAL_5M_MAX:
            current_tf = '5m'
            other_tfs = ['15m']
        elif median_interval <= MTF_INTERVAL_15M_MAX:
            current_tf = '15m'
            other_tfs = []
        else:
            logger.warning(f"无法识别时间框架（间隔={median_interval:.1f}分钟），跳过多时间框架特征")
            return df.reset_index()
        
        if not other_tfs:
            logger.debug(f"{current_tf}是最大周期，跳过多时间框架特征")
            return df.reset_index()
        
        logger.debug(f"多时间框架特征: 当前={current_tf}, 向上采样到={other_tfs}")
        
        for other_tf in other_tfs:
            resample_str = MTF_RESAMPLE_MAP.get(other_tf)
            if not resample_str:
                continue
            
            # Pandas 2.x 兼容：'first'/'last' 需使用 lambda
            df_resampled = df_temp.resample(resample_str).agg({
                'open': lambda x: x.iloc[0] if len(x) > 0 else np.nan,
                'high': 'max',
                'low': 'min',
                'close': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
                'volume': 'sum'
            }).ffill()
            
            close_resampled = df_resampled['close']
            sma_20_resampled = close_resampled.rolling(MTF_SMA_FAST_WINDOW).mean()
            sma_50_resampled = close_resampled.rolling(MTF_SMA_SLOW_WINDOW).mean()
            rsi_resampled = calculate_rsi(close_resampled, MTF_RSI_WINDOW)
            
            trend_resampled = pd.Series(0, index=df_resampled.index)
            trend_resampled[sma_20_resampled > sma_50_resampled] = 1
            trend_resampled[sma_20_resampled < sma_50_resampled] = -1
            
            close_resampled_safe = close_resampled.replace(0, np.nan) if (close_resampled == 0).sum() > 0 else close_resampled
            returns_resampled = close_resampled_safe.pct_change(fill_method=None)
            returns_resampled = returns_resampled.replace([np.inf, -np.inf], np.nan)
            volatility_resampled = returns_resampled.rolling(MTF_VOLATILITY_WINDOW).std()
            
            trend_resampled_shifted = trend_resampled.shift(MTF_SHIFT_LAG)
            rsi_resampled_shifted = rsi_resampled.shift(MTF_SHIFT_LAG)
            volatility_resampled_shifted = volatility_resampled.shift(MTF_SHIFT_LAG)
            sma_20_resampled_shifted = sma_20_resampled.shift(MTF_SHIFT_LAG)
            sma_50_resampled_shifted = sma_50_resampled.shift(MTF_SHIFT_LAG)
            
            df_resampled_features = pd.DataFrame({
                'trend': trend_resampled_shifted,
                'rsi': rsi_resampled_shifted,
                'volatility': volatility_resampled_shifted,
                'sma_20': sma_20_resampled_shifted,
                'sma_50': sma_50_resampled_shifted
            }, index=df_resampled.index)
            
            df_original = pd.DataFrame(index=df_temp.index)
            df_resampled_sorted = df_resampled_features.sort_index()
            df_original_sorted = df_original.sort_index()
            
            df_aligned = pd.merge_asof(
                df_original_sorted,
                df_resampled_sorted,
                left_index=True,
                right_index=True,
                direction='backward'
            )
            
            # 避免重复索引问题：使用 loc 而非 reindex
            if not df_temp.index.is_unique:
                df_temp = df_temp[~df_temp.index.duplicated(keep='last')]
            df_aligned = df_aligned.reindex(df_temp.index)
            
            new_features[f'trend_{other_tf}'] = df_aligned['trend']
            new_features[f'rsi_{other_tf}'] = df_aligned['rsi']
            new_features[f'volatility_{other_tf}'] = df_aligned['volatility']
            new_features[f'sma_20_{other_tf}'] = df_aligned['sma_20']
            new_features[f'sma_50_{other_tf}'] = df_aligned['sma_50']
        
        if 'sma_20' in df_temp.columns and 'sma_50' in df_temp.columns:
            trend_current = pd.Series(0, index=df_temp.index)
            trend_current[df_temp['sma_20'] > df_temp['sma_50']] = 1
            trend_current[df_temp['sma_20'] < df_temp['sma_50']] = -1
            
            alignment_features = []
            for other_tf in other_tfs:
                if f'trend_{other_tf}' in new_features:
                    alignment_key = f'trend_alignment_{other_tf}'
                    new_features[alignment_key] = (trend_current == new_features[f'trend_{other_tf}']).astype(int)
                    alignment_features.append(alignment_key)
            
            if alignment_features:
                new_features['trend_alignment_all'] = sum(new_features[k] for k in alignment_features) / len(alignment_features)
        
        if 'rsi_14' in df_temp.columns:
            for other_tf in other_tfs:
                if f'rsi_{other_tf}' in new_features:
                    new_features[f'rsi_diff_{other_tf}'] = df_temp['rsi_14'] - new_features[f'rsi_{other_tf}']
        
        if 'close' in df_temp.columns:
            for other_tf in other_tfs:
                if f'sma_20_{other_tf}' in new_features:
                    new_features[f'price_to_sma20_{other_tf}'] = (
                        (df_temp['close'] - new_features[f'sma_20_{other_tf}']) / 
                        (new_features[f'sma_20_{other_tf}'] + FEATURE_EPS)
                    )
                if f'sma_50_{other_tf}' in new_features:
                    new_features[f'price_to_sma50_{other_tf}'] = (
                        (df_temp['close'] - new_features[f'sma_50_{other_tf}']) / 
                        (new_features[f'sma_50_{other_tf}'] + FEATURE_EPS)
                    )
        
        for col_name, col_data in new_features.items():
            df_temp[col_name] = col_data
        
        df = df_temp.reset_index()
        
        logger.debug(f"多时间框架特征添加完成: {len(new_features)}个特征（基于{current_tf}，融合{other_tfs}）")
        return df
        
    except Exception as e:
        logger.error(f"添加多时间框架特征失败: {e}")
        logger.error(traceback.format_exc())
        return df

