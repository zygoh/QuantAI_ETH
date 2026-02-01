"""
API依赖项 - 认证系统
"""
import os
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

# API Key配置（从环境变量读取）
API_KEY = os.getenv("API_KEY", "your-secret-api-key-change-this")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    获取当前用户（API Key验证）

    使用方法：
    1. 设置环境变量 API_KEY=your-secret-key
    2. 请求时添加 Header: Authorization: Bearer your-secret-key

    Args:
        credentials: HTTP Bearer Token

    Returns:
        用户标识

    Raises:
        HTTPException: 认证失败
    """
    # 开发模式：如果未设置API_KEY或使用默认值，允许无认证访问
    if API_KEY == "your-secret-api-key-change-this":
        return "dev_user"

    # 生产模式：必须提供有效的API Key
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "authenticated_user"
