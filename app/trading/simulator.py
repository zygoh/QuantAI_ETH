# -*- coding: utf-8 -*-
"""
模拟交易引擎

实现模拟交易功能，包括：
- 开仓/平仓
- 止盈止损检测
- 手续费计算
- 账户状态管理
"""

import logging
from datetime import datetime
from typing import List, Optional

from app.trading.models import (
    AccountState,
    Position,
    PositionSide,
    SignalAction,
    TradeRecord,
    TradeSignal,
)


logger = logging.getLogger(__name__)


# 手续费配置
MAKER_FEE_RATE = 0.0002  # 开仓手续费 0.02%
TAKER_FEE_RATE = 0.0005  # 平仓手续费 0.05%


class TradingSimulator:
    """
    模拟交易引擎
    
    Attributes:
        account: 账户状态
        position: 当前持仓
        trade_history: 交易历史
    """
    
    def __init__(self, initial_balance: float = 3.0) -> None:
        """初始化模拟交易引擎"""
        self.account = AccountState(
            initial_balance=initial_balance,
            balance=initial_balance
        )
        self.position: Optional[Position] = None
        self.trade_history: List[TradeRecord] = []
        self.current_symbol: Optional[str] = None
        self.is_running: bool = False
        
        logger.info(
            f"💰 模拟交易引擎初始化完成 - "
            f"初始资金: ${initial_balance:,.2f}"
        )
    
    def has_position(self) -> bool:
        """是否有持仓"""
        return self.position is not None
    
    def get_position_symbol(self) -> Optional[str]:
        """获取当前持仓币种"""
        return self.position.symbol if self.position else None
    
    def check_stop_loss_take_profit(self, current_price: float) -> Optional[str]:
        """
        检查止盈止损
        
        Args:
            current_price: 当前价格
            
        Returns:
            触发原因: "stop_loss", "take_profit", None
        """
        if not self.position:
            return None
        
        pos = self.position
        
        if pos.side == PositionSide.LONG:
            # 多仓：价格 <= 止损 或 价格 >= 止盈
            if current_price <= pos.stop_loss:
                return "stop_loss"
            if current_price >= pos.take_profit:
                return "take_profit"
        else:
            # 空仓：价格 >= 止损 或 价格 <= 止盈
            if current_price >= pos.stop_loss:
                return "stop_loss"
            if current_price <= pos.take_profit:
                return "take_profit"
        
        return None
    
    def execute_signal(
        self,
        signal: TradeSignal,
        current_price: float
    ) -> bool:
        """
        执行交易信号
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            
        Returns:
            是否执行成功
        """
        if signal.action == SignalAction.WAIT:
            logger.info(f"⏸️ {signal.symbol} 观望: {signal.reasoning}")
            return False
        
        if signal.action == SignalAction.HOLD:
            logger.info(f"✊ {signal.symbol} 继续持有: {signal.reasoning}")
            return False
        
        if signal.action == SignalAction.OPEN_LONG:
            return self._open_position(signal, current_price, PositionSide.LONG)
        
        if signal.action == SignalAction.OPEN_SHORT:
            return self._open_position(signal, current_price, PositionSide.SHORT)
        
        if signal.action == SignalAction.ADJUST_STOPS:
            return self._adjust_stops(signal)
        
        return False
    
    def close_position(
        self,
        current_price: float,
        reason: str = "signal"
    ) -> Optional[TradeRecord]:
        """
        平仓
        
        Args:
            current_price: 当前价格
            reason: 平仓原因
            
        Returns:
            交易记录
        """
        if not self.position:
            return None
        
        pos = self.position
        
        # 计算平仓手续费（基于实际仓位）
        exit_fee = pos.position_size_usd * TAKER_FEE_RATE
        
        # 计算盈亏（position_size_usd 已包含杠杆，无需再乘）
        if pos.side == PositionSide.LONG:
            price_change_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            price_change_pct = (pos.entry_price - current_price) / pos.entry_price
        
        gross_pnl = pos.position_size_usd * price_change_pct
        
        # 保证金 = 仓位 / 杠杆
        margin = pos.position_size_usd / pos.leverage
        
        # 亏损上限保护：亏损不能超过保证金（模拟爆仓）
        if gross_pnl < -margin:
            gross_pnl = -margin
            logger.warning(f"⚠️ 触发爆仓保护: 亏损限制为保证金 ${margin:.4f}")
        
        net_pnl = gross_pnl - pos.entry_fee - exit_fee
        
        # 净亏损也不能超过保证金
        if net_pnl < -margin:
            net_pnl = -margin
        
        pnl_pct = (net_pnl / margin) * 100
        
        # 更新账户
        self.account.balance += net_pnl
        self.account.total_pnl += net_pnl
        self.account.total_fees += exit_fee
        self.account.total_trades += 1
        
        if net_pnl > 0:
            self.account.winning_trades += 1
        else:
            self.account.losing_trades += 1
        
        # 创建交易记录
        record = TradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=current_price,
            quantity=pos.quantity,
            leverage=pos.leverage,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            entry_fee=pos.entry_fee,
            exit_fee=exit_fee,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )
        
        self.trade_history.append(record)
        
        # 清空持仓
        self.position = None
        
        # 日志
        pnl_emoji = "🟢" if net_pnl > 0 else "🔴"
        logger.info(
            f"{pnl_emoji} 平仓 {pos.symbol} ({pos.side.value}) - "
            f"入场: ${pos.entry_price:.4f}, "
            f"出场: ${current_price:.4f}, "
            f"盈亏: ${net_pnl:+.2f} ({pnl_pct:+.2f}%), "
            f"原因: {reason}"
        )
        
        return record

    def _open_position(
        self,
        signal: TradeSignal,
        current_price: float,
        side: PositionSide
    ) -> bool:
        """
        开仓
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            side: 持仓方向
            
        Returns:
            是否成功
        """
        # 如果已有持仓，先平仓（调用方需保证 current_price 为当前持仓币种价格）
        if self.position:
            logger.info(f"📤 已有持仓，先平仓...")
            self.close_position(current_price, "new_signal")
        
        # 全仓开仓（使用 95% 余额作为保证金，预留手续费空间）
        margin = self.account.balance * 0.95
        
        if margin <= 0:
            logger.warning(f"⚠️ 余额不足，无法开仓 (${self.account.balance:.4f})")
            return False
        
        # 实际仓位 = 保证金 × 杠杆
        position_size = margin * signal.leverage
        
        # 计算开仓手续费（基于实际仓位）
        entry_fee = position_size * MAKER_FEE_RATE
        
        # 计算数量（基于实际仓位）
        quantity = position_size / current_price

        # 校验止损止盈方向，防止 AI 设反导致一开仓就触发止损
        stop_loss, take_profit = self._clamp_stop_take_profit(
            current_price, signal.stop_loss, signal.take_profit, side
        )
        if stop_loss != signal.stop_loss or take_profit != signal.take_profit:
            logger.warning(
                f"⚠️ 止损/止盈方向已校正: 多仓止损须<入场价、止盈须>入场价；空仓相反"
            )

        # 创建持仓
        self.position = Position(
            symbol=signal.symbol,
            side=side,
            entry_price=current_price,
            quantity=quantity,
            leverage=signal.leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_usd=position_size,
            entry_fee=entry_fee,
        )
        
        # 扣除手续费
        self.account.total_fees += entry_fee
        
        # 日志
        side_emoji = "📈" if side == PositionSide.LONG else "📉"
        logger.info(
            f"{side_emoji} 开仓 {signal.symbol} ({side.value}) - "
            f"价格: ${current_price:.4f}, "
            f"保证金: ${margin:.2f}, "
            f"仓位: ${position_size:.2f}, "
            f"杠杆: {signal.leverage}x, "
            f"止损: ${stop_loss:.4f}, "
            f"止盈: ${take_profit:.4f}, "
            f"手续费: ${entry_fee:.4f}"
        )
        
        return True
    
    def _clamp_stop_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        side: PositionSide,
    ) -> tuple:
        """校正止损止盈方向：多仓止损<入场、止盈>入场；空仓相反。防止反向触发。"""
        if side == PositionSide.LONG:
            # 多仓：止损必须 < 入场价，止盈必须 > 入场价
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995
            if take_profit <= entry_price:
                take_profit = entry_price * 1.005
        else:
            # 空仓：止损必须 > 入场价，止盈必须 < 入场价
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005
            if take_profit >= entry_price:
                take_profit = entry_price * 0.995
        return stop_loss, take_profit

    def _adjust_stops(self, signal: TradeSignal) -> bool:
        """
        调整止盈止损

        Args:
            signal: 交易信号

        Returns:
            是否成功
        """
        if not self.position:
            logger.warning(f"⚠️ 无持仓，无法调整止盈止损")
            return False

        pos = self.position
        stop_loss, take_profit = self._clamp_stop_take_profit(
            pos.entry_price, signal.stop_loss, signal.take_profit, pos.side
        )
        if stop_loss != signal.stop_loss or take_profit != signal.take_profit:
            logger.warning(f"⚠️ 调整止盈止损时方向已校正")

        old_sl = pos.stop_loss
        old_tp = pos.take_profit
        pos.stop_loss = stop_loss
        pos.take_profit = take_profit

        logger.info(
            f"🔧 调整止盈止损 {pos.symbol} - "
            f"止损: ${old_sl:.4f} -> ${stop_loss:.4f}, "
            f"止盈: ${old_tp:.4f} -> ${take_profit:.4f}"
        )

        return True
    
    def get_account_summary(self) -> str:
        """获取账户摘要"""
        acc = self.account
        
        summary = (
            f"💰 账户状态:\n"
            f"   余额: ${acc.balance:,.2f} (初始: ${acc.initial_balance:,.2f})\n"
            f"   收益: ${acc.total_pnl:+,.2f} ({acc.return_pct:+.2f}%)\n"
            f"   交易: {acc.total_trades} 笔 (胜率: {acc.win_rate:.1f}%)\n"
            f"   手续费: ${acc.total_fees:.2f}"
        )
        
        if self.position:
            pos = self.position
            summary += (
                f"\n📊 当前持仓:\n"
                f"   {pos.symbol} ({pos.side.value})\n"
                f"   入场: ${pos.entry_price:.4f}\n"
                f"   止损: ${pos.stop_loss:.4f}\n"
                f"   止盈: ${pos.take_profit:.4f}"
            )
        else:
            summary += "\n📊 当前无持仓"
        
        return summary


# 全局模拟交易引擎实例
trading_simulator = TradingSimulator()
