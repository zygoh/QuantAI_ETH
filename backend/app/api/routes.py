"""
API路由
"""
from fastapi import APIRouter
from app.api.endpoints import (
    account, positions, signals, trading, 
    training, performance, system, websocket
)

# 创建主路由
api_router = APIRouter()

# 注册各个端点路由
api_router.include_router(account.router, prefix="/account", tags=["账户"])
api_router.include_router(positions.router, prefix="/positions", tags=["持仓"])
api_router.include_router(signals.router, prefix="/signals", tags=["信号"])
api_router.include_router(trading.router, prefix="/trading", tags=["交易"])
api_router.include_router(training.router, prefix="/training", tags=["训练"])
api_router.include_router(performance.router, prefix="/performance", tags=["绩效"])
api_router.include_router(system.router, prefix="/system", tags=["系统"])
api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])