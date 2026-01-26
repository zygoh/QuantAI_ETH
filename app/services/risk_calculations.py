"""
风险计算纯函数（严格模式）

目标：
- 回测 / 实盘 / 风控 使用完全一致的止损止盈计算公式（仅数据来源可不同）
- 禁止在回测/实盘里分叉“魔法数字”或重复实现
"""

# StdLib
from __future__ import annotations

from typing import Any, Dict

# Local App
from app.core.constants import (
    RISK_ATR_TRAILING_STOP_MULTIPLIER,
    RISK_FIXED_TRAILING_STOP_PCT,
    STOP_LOSS_ATR_MULTIPLIER,
    STOP_LOSS_PCT,
    TAKE_PROFIT_ATR_MULTIPLIER,
    TAKE_PROFIT_PCT,
)


def calculate_atr_based_stop_levels(
    entry_price: float,
    atr: float,
    signal_type: str,
    *,
    stop_loss_atr_multiplier: float = STOP_LOSS_ATR_MULTIPLIER,
    take_profit_atr_multiplier: float = TAKE_PROFIT_ATR_MULTIPLIER,
    trailing_stop_atr_multiplier: float = RISK_ATR_TRAILING_STOP_MULTIPLIER,
) -> Dict[str, Any]:
    """
    基于 ATR 的动态止损止盈（严格模式：公式统一）。

    Args:
        entry_price: 入场价格（>0）
        atr: ATR 值（>0）
        signal_type: 'LONG' 或 'SHORT'
        stop_loss_atr_multiplier: 止损 ATR 倍数
        take_profit_atr_multiplier: 止盈 ATR 倍数
        trailing_stop_atr_multiplier: 跟踪止损 ATR 倍数

    Returns:
        与 RiskService.calculate_dynamic_stop_levels 一致的 stop_levels 字典

    Raises:
        ValueError: 参数非法
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price 必须 > 0: {entry_price}")
    if atr <= 0:
        raise ValueError(f"atr 必须 > 0: {atr}")

    side = (signal_type or "").upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(f"未知 signal_type: {signal_type}")

    if side == "LONG":
        stop_loss = entry_price - (atr * stop_loss_atr_multiplier)
        take_profit = entry_price + (atr * take_profit_atr_multiplier)
    else:
        stop_loss = entry_price + (atr * stop_loss_atr_multiplier)
        take_profit = entry_price - (atr * take_profit_atr_multiplier)

    trailing_stop_distance = atr * trailing_stop_atr_multiplier

    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    risk_reward_ratio = (reward / risk) if risk > 0 else 0.0

    return {
        "entry_price": float(entry_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "trailing_stop_enabled": True,
        "trailing_stop_distance": float(trailing_stop_distance),
        "atr": float(atr),
        "atr_percent": float((atr / entry_price) * 100),
        "risk_reward_ratio": float(risk_reward_ratio),
        "max_loss_percent": float((risk / entry_price) * 100),
        "max_profit_percent": float((reward / entry_price) * 100),
    }


def calculate_fixed_pct_stop_levels(
    entry_price: float,
    signal_type: str,
    *,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    fixed_trailing_stop_pct: float = RISK_FIXED_TRAILING_STOP_PCT,
) -> Dict[str, Any]:
    """
    固定百分比止盈止损（严格模式：回测/实盘/风控 fallback 统一）。

    Args:
        entry_price: 入场价格（>0）
        signal_type: 'LONG' 或 'SHORT'
        stop_loss_pct: 止损百分比（0~1）
        take_profit_pct: 止盈百分比（0~1）
        fixed_trailing_stop_pct: 固定跟踪止损比例（0~1）

    Returns:
        与 RiskService.calculate_dynamic_stop_levels 一致的 stop_levels 字典

    Raises:
        ValueError: 参数非法
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price 必须 > 0: {entry_price}")
    if stop_loss_pct <= 0 or take_profit_pct <= 0:
        raise ValueError(f"止损/止盈百分比必须 > 0: sl={stop_loss_pct}, tp={take_profit_pct}")
    if fixed_trailing_stop_pct <= 0:
        raise ValueError(f"fixed_trailing_stop_pct 必须 > 0: {fixed_trailing_stop_pct}")

    side = (signal_type or "").upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(f"未知 signal_type: {signal_type}")

    if side == "LONG":
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
    else:
        stop_loss = entry_price * (1 + stop_loss_pct)
        take_profit = entry_price * (1 - take_profit_pct)

    risk_reward_ratio = (take_profit_pct / stop_loss_pct) if stop_loss_pct > 0 else 0.0

    return {
        "entry_price": float(entry_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "trailing_stop_enabled": False,
        "trailing_stop_distance": float(entry_price * fixed_trailing_stop_pct),
        "atr": float(0.0),
        "atr_percent": float(0.0),
        "risk_reward_ratio": float(risk_reward_ratio),
        "max_loss_percent": float(stop_loss_pct * 100),
        "max_profit_percent": float(take_profit_pct * 100),
    }

