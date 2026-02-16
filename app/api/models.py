# -*- coding: utf-8 -*-
"""
交易系统 API 响应模型
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    detail: Optional[str] = Field(None, description="错误详情")


class AccountResponse(BaseResponse):
    """账户状态响应"""
    data: Optional[Dict[str, Any]] = Field(None, description="账户数据")


class PositionResponse(BaseResponse):
    """持仓信息响应"""
    data: Optional[Dict[str, Any]] = Field(None, description="持仓数据")


class TradeListResponse(BaseResponse):
    """交易历史响应"""
    data: List[Dict[str, Any]] = Field(default_factory=list, description="交易记录列表")
    total: int = Field(0, description="总数量")


class SystemStatusResponse(BaseResponse):
    """系统状态响应"""
    data: Optional[Dict[str, Any]] = Field(None, description="系统状态")


class ChatHistoryResponse(BaseResponse):
    """AI 对话历史响应"""
    data: List[Dict[str, Any]] = Field(default_factory=list, description="对话记录列表")
    total: int = Field(0, description="总数量")


class IndicatorsResponse(BaseResponse):
    """指标快照响应"""
    data: Optional[Dict[str, Any]] = Field(None, description="指标快照数据")
