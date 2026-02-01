"""
API路由
"""
from fastapi import APIRouter
from app.api.endpoints import scalping, system

# 创建主路由
api_router = APIRouter()

# 注册路由
api_router.include_router(scalping.router, prefix="/scalping", tags=["剥头皮交易"])
api_router.include_router(system.router, prefix="/system", tags=["系统"])
