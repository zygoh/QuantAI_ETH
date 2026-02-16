# -*- coding: utf-8 -*-
"""
交易数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


class SignalAction(Enum):
    """信号动作"""
    WAIT = "wait"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_POSITION = "close_position"
    HOLD = "hold"
    ADJUST_STOPS = "adjust_stops"


@dataclass
class TradeSignal:
    """AI 交易信号"""
    symbol: str
    action: SignalAction
    reasoning: str
    leverage: int = 1
    position_size_usd: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: int = 0
    risk_usd: float = 0.0
    atr: float = 0.0


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    leverage: int
    stop_loss: float
    take_profit: float
    position_size_usd: float
    entry_time: datetime = field(default_factory=datetime.now)
    entry_fee: float = 0.0  # 开仓手续费
    margin_pct: float = 0.0  # 保证金占余额比例
    
    def unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """计算未实现盈亏百分比（基于保证金/本金）"""
        if self.leverage == 0 or self.position_size_usd == 0:
            return 0.0
        margin = self.position_size_usd / self.leverage
        pnl = self.unrealized_pnl(current_price)
        return (pnl / margin) * 100


@dataclass
class TradeRecord:
    """交易记录"""
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    entry_time: datetime
    exit_time: datetime
    entry_fee: float
    exit_fee: float
    pnl: float  # 净盈亏（扣除手续费）
    pnl_pct: float
    exit_reason: str  # "take_profit", "stop_loss", "signal"


@dataclass
class AccountState:
    """账户状态"""
    initial_balance: float = 3.0  # 初始资金
    balance: float = 3.0  # 当前余额
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def return_pct(self) -> float:
        """总收益率"""
        if self.initial_balance == 0:
            return 0.0
        return ((self.balance - self.initial_balance) / self.initial_balance) * 100
