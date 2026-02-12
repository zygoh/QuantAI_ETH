# -*- coding: utf-8 -*-
"""
技术指标计算函数（纯函数）

提供图表生成所需的技术指标计算：
- EMA (指数移动平均线)
- SMA (简单移动平均线)
- RSI (相对强弱指数)
- MACD (移动平均收敛散度)
- 布林带
"""

import numpy as np


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    计算指数移动平均线 (EMA)

    使用标准 EMA 公式：
    EMA_t = price_t * k + EMA_{t-1} * (1 - k)
    其中 k = 2 / (period + 1)

    Args:
        prices: 价格序列
        period: EMA 周期

    Returns:
        EMA 序列，长度与输入相同
    """
    if len(prices) == 0:
        return np.array([])

    if len(prices) < period:
        period = max(1, len(prices))

    ema = np.zeros(len(prices))
    k = 2.0 / (period + 1)

    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)

    return ema


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    计算简单移动平均线 (SMA)

    Args:
        prices: 价格序列
        period: SMA 周期

    Returns:
        SMA 序列，前 period-1 个值为 NaN
    """
    if len(prices) == 0:
        return np.array([])

    sma = np.full(len(prices), np.nan)

    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])

    return sma


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算相对强弱指数 (RSI)

    使用 Wilder 平滑方法：
    RSI = 100 - 100 / (1 + RS)
    RS = 平均涨幅 / 平均跌幅

    Args:
        prices: 价格序列
        period: RSI 周期，默认 14

    Returns:
        RSI 序列，值在 0-100 范围内
    """
    if len(prices) < 2:
        return np.array([50.0] * len(prices))

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(len(prices), 50.0)

    if len(deltas) < period:
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

        if avg_loss == 0:
            rsi_value = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100.0 - 100.0 / (1.0 + rs)

        rsi[1:] = rsi_value
        return rsi

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    rsi = np.clip(rsi, 0.0, 100.0)
    return rsi


def calculate_macd(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 MACD (移动平均收敛散度)

    Args:
        prices: 价格序列
        fast_period: 快线周期，默认 12
        slow_period: 慢线周期，默认 26
        signal_period: 信号线周期，默认 9

    Returns:
        (macd_line, signal_line, histogram) 三元组
    """
    if len(prices) == 0:
        return np.array([]), np.array([]), np.array([])

    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算布林带

    Args:
        prices: 价格序列
        period: 周期，默认 20
        std_dev: 标准差倍数，默认 2.0

    Returns:
        (upper_band, middle_band, lower_band) 三元组
    """
    if len(prices) == 0:
        return np.array([]), np.array([]), np.array([])

    middle_band = calculate_sma(prices, period)

    std = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1:i + 1], ddof=0)

    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std

    return upper_band, middle_band, lower_band
