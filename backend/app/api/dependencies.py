"""
API依赖项
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 安全认证
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """获取当前用户（简化版本）"""
    # 这里可以实现JWT token验证
    # 目前简化处理，返回默认用户
    return "default_user"

async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> bool:
    """验证API密钥"""
    # 简化处理，实际应该验证API密钥
    return True