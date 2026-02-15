# -*- coding: utf-8 -*-
"""
交易系统 API 路由

提供 REST API 端点：
- GET /api/status    - 系统状态 + 当前选中币种
- GET /api/account   - 账户状态（余额、收益、胜率）
- GET /api/position  - 当前持仓详情
- GET /api/trades    - 交易历史
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

from app.api.models import (
    AccountResponse,
    ChatHistoryResponse,
    PositionResponse,
    SystemStatusResponse,
    TradeListResponse,
)
from app.trading.ai_analyzer import ai_analyzer
from app.trading.price_monitor import price_monitor
from app.trading.price_util import get_current_price
from app.trading.simulator import trading_simulator


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["交易系统"])


@router.get("/status", response_model=SystemStatusResponse, summary="系统状态")
async def get_system_status() -> SystemStatusResponse:
    """获取系统运行状态和当前选中币种"""
    acc = trading_simulator.account
    current_price = price_monitor.current_price
    # 无仓时 WebSocket 未订阅，用 REST 拉取当前分析币种价格供前端展示
    if not trading_simulator.has_position() and trading_simulator.current_symbol:
        if not current_price or price_monitor.current_symbol != trading_simulator.current_symbol:
            current_price = await get_current_price(trading_simulator.current_symbol)

    return SystemStatusResponse(
        success=True,
        message="获取成功",
        data={
            "version": "3.0",
            "is_running": trading_simulator.is_running,
            "current_symbol": trading_simulator.current_symbol,
            "current_price": current_price,
            "has_position": trading_simulator.has_position(),
            "balance": round(acc.balance, 4),
            "total_pnl": round(acc.total_pnl, 4),
            "total_trades": acc.total_trades,
        }
    )


@router.get("/account", response_model=AccountResponse, summary="账户状态")
async def get_account() -> AccountResponse:
    """获取账户详细状态"""
    acc = trading_simulator.account

    return AccountResponse(
        success=True,
        message="获取成功",
        data={
            "initial_balance": acc.initial_balance,
            "balance": round(acc.balance, 4),
            "total_pnl": round(acc.total_pnl, 4),
            "return_pct": round(acc.return_pct, 2),
            "total_trades": acc.total_trades,
            "winning_trades": acc.winning_trades,
            "losing_trades": acc.losing_trades,
            "win_rate": round(acc.win_rate, 1),
            "total_fees": round(acc.total_fees, 6),
        }
    )


@router.get("/position", response_model=PositionResponse, summary="当前持仓")
async def get_position() -> PositionResponse:
    """获取当前持仓详情"""
    pos = trading_simulator.position

    if not pos:
        return PositionResponse(
            success=True,
            message="当前无持仓",
            data=None
        )

    current_price = price_monitor.current_price
    unrealized_pnl = pos.unrealized_pnl(current_price) if current_price > 0 else 0.0
    unrealized_pnl_pct = pos.unrealized_pnl_pct(current_price) if current_price > 0 else 0.0

    # 计算爆仓价
    if pos.side.value == "long":
        liquidation_price = pos.entry_price * (1 - 1 / pos.leverage)
    else:
        liquidation_price = pos.entry_price * (1 + 1 / pos.leverage)

    return PositionResponse(
        success=True,
        message="获取成功",
        data={
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "leverage": pos.leverage,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "position_size_usd": round(pos.position_size_usd, 4),
            "entry_fee": round(pos.entry_fee, 6),
            "entry_time": pos.entry_time.isoformat(),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            "liquidation_price": round(liquidation_price, 10),
        }
    )


@router.get("/trades", response_model=TradeListResponse, summary="交易历史")
async def get_trades(
    limit: int = Query(default=50, ge=1, le=200, description="返回数量")
) -> TradeListResponse:
    """获取交易历史记录（最新在前）"""
    history = trading_simulator.trade_history
    records = list(reversed(history))[:limit]

    data = []
    for r in records:
        data.append({
            "symbol": r.symbol,
            "side": r.side.value,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "quantity": r.quantity,
            "leverage": r.leverage,
            "entry_time": r.entry_time.isoformat(),
            "exit_time": r.exit_time.isoformat(),
            "entry_fee": round(r.entry_fee, 6),
            "exit_fee": round(r.exit_fee, 6),
            "pnl": round(r.pnl, 4),
            "pnl_pct": round(r.pnl_pct, 2),
            "exit_reason": r.exit_reason,
        })

    return TradeListResponse(
        success=True,
        message=f"获取成功，共 {len(data)} 条记录",
        data=data,
        total=len(history)
    )


@router.get("/chat", response_model=ChatHistoryResponse, summary="AI 对话历史")
async def get_chat_history(
    limit: int = Query(default=20, ge=1, le=50, description="返回数量")
) -> ChatHistoryResponse:
    """获取 AI 分析对话记录（最新在前）"""
    history = ai_analyzer.get_chat_history(limit)

    return ChatHistoryResponse(
        success=True,
        message=f"获取成功，共 {len(history)} 条记录",
        data=history,
        total=len(ai_analyzer.chat_history)
    )
