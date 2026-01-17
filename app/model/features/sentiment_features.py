"""
市场情绪特征模块
"""
# StdLib
import logging
import traceback

# Third-Party
import numpy as np
import pandas as pd

from app.core.constants import (
    SENTIMENT_EPS,
    SENTIMENT_EXTREME_DECAY_WINDOW,
    SENTIMENT_EXTREME_STD_MULTIPLIER,
    SENTIMENT_EXTREME_WINDOW,
    SENTIMENT_LONG_VOL_WINDOW,
    SENTIMENT_MACD_NORM_PCT,
    SENTIMENT_PRESSURE_MA_WINDOW,
    SENTIMENT_PRICE_TREND_WINDOW,
    SENTIMENT_RSI_MOMENTUM_SHIFT,
    SENTIMENT_RSI_OVERBOUGHT,
    SENTIMENT_RSI_OVERSOLD,
    SENTIMENT_RSI_VOL_WINDOW,
    SENTIMENT_SCORE_WINDOW,
    SENTIMENT_SHORT_VOL_WINDOW,
    SENTIMENT_SMA_TREND_SHIFT,
    SENTIMENT_VOL_REGIME_LONG,
    SENTIMENT_VOL_REGIME_SHORT,
    SENTIMENT_VOLUME_DRY_MULTIPLIER,
    SENTIMENT_VOLUME_MA_WINDOW,
    SENTIMENT_VOLUME_STD_WINDOW,
    SENTIMENT_VOLUME_SURGE_MULTIPLIER,
    SENTIMENT_VOLUME_TREND_WINDOW,
    SENTIMENT_VOLUME_WEIGHTED_WINDOW
)

logger = logging.getLogger(__name__)


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加市场情绪特征"""
    try:
        new_features = {}
        
        # 1. 恐慌指数（基于价格波动）
        close_for_returns = df['clean_close']
        returns = close_for_returns.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        short_vol = returns.rolling(SENTIMENT_SHORT_VOL_WINDOW).std()
        long_vol = returns.rolling(SENTIMENT_LONG_VOL_WINDOW).std()
        new_features['fear_index'] = short_vol / (long_vol + SENTIMENT_EPS)
        
        # 2. 连续涨跌天数（捕捉趋势疲劳）
        up_days = (returns > 0).astype(int)
        down_days = (returns < 0).astype(int)
        new_features['consecutive_up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
        new_features['consecutive_down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()
        
        # 3. RSI衍生情绪指标
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14']
            new_features['extreme_overbought'] = (rsi > SENTIMENT_RSI_OVERBOUGHT).astype(int)
            new_features['extreme_oversold'] = (rsi < SENTIMENT_RSI_OVERSOLD).astype(int)
            new_features['rsi_momentum'] = rsi - rsi.shift(SENTIMENT_RSI_MOMENTUM_SHIFT)
            new_features['rsi_volatility'] = rsi.rolling(SENTIMENT_RSI_VOL_WINDOW).std()
        
        # 4. 价格加速度幅度（情绪转变强度）
        close_for_price_change = df['clean_close']
        price_change = close_for_price_change.pct_change(fill_method=None)
        price_change = price_change.replace([np.inf, -np.inf], np.nan)
        acceleration = price_change.diff()
        new_features['acceleration_magnitude'] = acceleration.abs()
        
        # 5. 成交量情绪（基于放量/缩量）
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(SENTIMENT_VOLUME_MA_WINDOW).mean()
            new_features['volume_surge'] = (df['volume'] > volume_ma * SENTIMENT_VOLUME_SURGE_MULTIPLIER).astype(int)
            new_features['volume_dry'] = (df['volume'] < volume_ma * SENTIMENT_VOLUME_DRY_MULTIPLIER).astype(int)
            
            # 价量背离
            price_trend = (price_change.rolling(SENTIMENT_PRICE_TREND_WINDOW).mean() > 0).astype(int)
            volume_for_chg = df['volume'].replace(0, np.nan) if (df['volume'] == 0).sum() > 0 else df['volume']
            volume_chg = volume_for_chg.pct_change(fill_method=None)
            volume_chg = volume_chg.replace([np.inf, -np.inf], np.nan)
            volume_trend = (volume_chg.rolling(SENTIMENT_VOLUME_TREND_WINDOW).mean() > 0).astype(int)
            new_features['price_volume_divergence'] = (price_trend != volume_trend).astype(int)
        
        # 6. 市场波动情绪
        returns_abs = returns.abs()
        new_features['volatility_regime'] = (
            returns_abs.rolling(SENTIMENT_VOL_REGIME_SHORT).mean() /
            returns_abs.rolling(SENTIMENT_VOL_REGIME_LONG).mean()
        )
        
        # 7. 趋势一致性（多个均线方向一致度）
        if 'sma_5' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma5_up = (df['sma_5'] > df['sma_5'].shift(SENTIMENT_SMA_TREND_SHIFT)).astype(int)
            sma20_up = (df['sma_20'] > df['sma_20'].shift(SENTIMENT_SMA_TREND_SHIFT)).astype(int)
            sma50_up = (df['sma_50'] > df['sma_50'].shift(SENTIMENT_SMA_TREND_SHIFT)).astype(int)
            new_features['trend_alignment'] = (sma5_up + sma20_up + sma50_up) / 3
        
        # 8. 市场情绪综合指数
        sentiment_score = 0
        if 'rsi_14' in df.columns:
            sentiment_score += ((df['rsi_14'] - 50) / 50)
        
        if 'macd_histogram' in df.columns:
            macd_norm = df['macd_histogram'] / (df['close'] * SENTIMENT_MACD_NORM_PCT + SENTIMENT_EPS)
            sentiment_score += np.clip(macd_norm, -1, 1)
        
        sentiment_score += price_change.rolling(SENTIMENT_SCORE_WINDOW).mean() * 100
        new_features['sentiment_composite'] = sentiment_score / 3
        
        # 9. 买卖压力指标（基于K线形态）
        price_range = df['high'] - df['low']
        price_range = price_range.replace(0, np.nan)
        new_features['buy_pressure'] = (df['close'] - df['low']) / price_range
        new_features['sell_pressure'] = (df['high'] - df['close']) / price_range
        new_features['pressure_diff'] = new_features['buy_pressure'] - new_features['sell_pressure']
        
        # 买卖压力趋势（多周期平均）
        new_features['buy_pressure_ma5'] = new_features['buy_pressure'].rolling(SENTIMENT_PRESSURE_MA_WINDOW).mean()
        new_features['sell_pressure_ma5'] = new_features['sell_pressure'].rolling(SENTIMENT_PRESSURE_MA_WINDOW).mean()
        
        # 10. 成交量加权情绪
        if 'volume' in df.columns:
            volume_weighted_return = price_change * df['volume']
            new_features['volume_weighted_sentiment'] = (
                volume_weighted_return.rolling(SENTIMENT_VOLUME_WEIGHTED_WINDOW).sum() /
                (df['volume'].rolling(SENTIMENT_VOLUME_WEIGHTED_WINDOW).sum() + SENTIMENT_EPS)
            )
            
            volume_std = df['volume'].rolling(SENTIMENT_VOLUME_STD_WINDOW).std()
            new_features['volume_sentiment_strength'] = (
                (df['volume'] - df['volume'].rolling(SENTIMENT_VOLUME_STD_WINDOW).mean()) /
                (volume_std + SENTIMENT_EPS)
            )
        
        # 11. 市场宽度指标（价格分布）
        if 'sma_5' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
            new_features['price_deviation_5'] = (df['close'] - df['sma_5']) / df['sma_5']
            new_features['price_deviation_20'] = (df['close'] - df['sma_20']) / df['sma_20']
            new_features['price_deviation_50'] = (df['close'] - df['sma_50']) / df['sma_50']
            
            new_features['market_breadth'] = (
                (df['sma_5'] - df['sma_20']).abs() + 
                (df['sma_20'] - df['sma_50']).abs()
            ) / df['close']
        
        # 12. 极端情绪检测
        extreme_up = (
            price_change > price_change.rolling(SENTIMENT_EXTREME_WINDOW).mean() +
            SENTIMENT_EXTREME_STD_MULTIPLIER * price_change.rolling(SENTIMENT_EXTREME_WINDOW).std()
        )
        extreme_down = (
            price_change < price_change.rolling(SENTIMENT_EXTREME_WINDOW).mean() -
            SENTIMENT_EXTREME_STD_MULTIPLIER * price_change.rolling(SENTIMENT_EXTREME_WINDOW).std()
        )
        new_features['extreme_move'] = extreme_up.astype(int) - extreme_down.astype(int)
        new_features['extreme_move_decay'] = new_features['extreme_move'].rolling(SENTIMENT_EXTREME_DECAY_WINDOW).sum()
        
        # 一次性添加所有特征
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"添加市场情绪特征失败: {e}")
        logger.error(traceback.format_exc())
        return df

