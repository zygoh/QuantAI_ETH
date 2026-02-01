"""
API数据模型
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# 基础响应模型
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
