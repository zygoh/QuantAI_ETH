"""
市场微观结构特征模块
"""
# StdLib
import logging
import traceback

# Third-Party
import numpy as np
import pandas as pd
import ta

# Local App
from app.model.features.utils import calculate_fractal_dimension, calculate_hurst_exponent
from app.core.constants import (
    MICROSTRUCTURE_ATR_EPS,
    MICROSTRUCTURE_BODY_STRONG_RATIO,
    MICROSTRUCTURE_DOJI_RATIO,
    MICROSTRUCTURE_EFFICIENCY_PERIODS,
    MICROSTRUCTURE_FRACTAL_PERIODS,
    MICROSTRUCTURE_HURST_PERIODS,
    MICROSTRUCTURE_OVERBOUGHT_THRESHOLD,
    MICROSTRUCTURE_OVERSOLD_THRESHOLD,
    MICROSTRUCTURE_POSITION_WINDOWS,
    MICROSTRUCTURE_PRICE_RANGE_EPS,
    RISK_ATR_WINDOW
)

logger = logging.getLogger(__name__)


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加市场微观结构特征"""
    try:
        new_features = {}
        
        # 1. 买卖压力指标（基础）
        price_range = df['high'] - df['low'] + MICROSTRUCTURE_PRICE_RANGE_EPS
        new_features['buying_pressure'] = (df['close'] - df['low']) / price_range
        new_features['selling_pressure'] = (df['high'] - df['close']) / price_range
        
        # 2. 价格位置百分比（捕捉支撑阻力）
        for period in MICROSTRUCTURE_POSITION_WINDOWS:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            price_position = (df['close'] - rolling_low) / (rolling_high - rolling_low + MICROSTRUCTURE_PRICE_RANGE_EPS)
            new_features[f'price_position_{period}'] = price_position
            # 超买超卖标识
            new_features[f'overbought_{period}'] = (price_position > MICROSTRUCTURE_OVERBOUGHT_THRESHOLD).astype(int)
            new_features[f'oversold_{period}'] = (price_position < MICROSTRUCTURE_OVERSOLD_THRESHOLD).astype(int)
        
        # 3. K线形态特征（实体和影线）
        body = abs(df['close'] - df['open'])
        new_features['body_range'] = body / (price_range + MICROSTRUCTURE_PRICE_RANGE_EPS)
        new_features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (price_range + MICROSTRUCTURE_PRICE_RANGE_EPS)
        new_features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (price_range + MICROSTRUCTURE_PRICE_RANGE_EPS)
        
        # 4. K线强度（看涨/看跌力量）
        new_features['bullish_candle'] = (df['close'] > df['open']).astype(int)
        new_features['strong_bullish'] = (
            (df['close'] - df['open']) / (price_range + MICROSTRUCTURE_PRICE_RANGE_EPS) >
            MICROSTRUCTURE_BODY_STRONG_RATIO
        ).astype(int)
        new_features['strong_bearish'] = (
            (df['open'] - df['close']) / (price_range + MICROSTRUCTURE_PRICE_RANGE_EPS) >
            MICROSTRUCTURE_BODY_STRONG_RATIO
        ).astype(int)
        new_features['doji'] = (body / (price_range + MICROSTRUCTURE_PRICE_RANGE_EPS) < MICROSTRUCTURE_DOJI_RATIO).astype(int)
        
        # 5. 连续K线形态（趋势延续）
        bullish = (df['close'] > df['open']).astype(int)
        new_features['consecutive_bull'] = bullish.groupby((bullish != bullish.shift()).cumsum()).cumsum()
        new_features['consecutive_bear'] = (1 - bullish).groupby((bullish == bullish.shift()).cumsum()).cumsum()
        
        # 6. 价格效率
        for period in MICROSTRUCTURE_EFFICIENCY_PERIODS:
            price_change = df['close'].diff(period)
            sum_abs_changes = df['close'].diff().abs().rolling(period).sum()
            new_features[f'price_efficiency_{period}'] = price_change.abs() / (sum_abs_changes + MICROSTRUCTURE_PRICE_RANGE_EPS)
        
        # 7. 价格加速度（捕捉拐点）
        close_for_returns = df['clean_close']
        returns = close_for_returns.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        new_features['price_jerk'] = returns.diff().diff()
        
        # 8. 分形维度
        for period in MICROSTRUCTURE_FRACTAL_PERIODS:
            new_features[f'fractal_dimension_{period}'] = calculate_fractal_dimension(df['close'], period)
        
        # 9. Hurst指数
        for period in MICROSTRUCTURE_HURST_PERIODS:
            new_features[f'hurst_exponent_{period}'] = calculate_hurst_exponent(df['close'], period)
        
        # 10. 真实波动范围占比（捕捉异常波动）
        atr_14 = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=RISK_ATR_WINDOW
        ).average_true_range()
        new_features['range_to_atr'] = price_range / (atr_14 + MICROSTRUCTURE_ATR_EPS)
        new_features['body_to_atr'] = body / (atr_14 + MICROSTRUCTURE_ATR_EPS)
        
        # 一次性添加所有特征
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"添加市场微观结构特征失败: {e}")
        logger.error(traceback.format_exc())
        return df

