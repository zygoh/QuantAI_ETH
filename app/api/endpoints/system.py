"""
系统相关API端点
"""
import logging
import platform
import sys

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.scalping.scalping_engine import scalping_engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status")
async def get_system_status(current_user: str = Depends(get_current_user)):
    """获取系统状态"""
    try:
        return {
            'success': True,
            'message': '系统状态获取成功',
            'data': {
                'scalping_engine_running': scalping_engine.is_running if scalping_engine else False,
                'scalping_status': scalping_engine.get_status() if scalping_engine and scalping_engine.is_running else None
            }
        }
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.get("/info")
async def get_system_info(current_user: str = Depends(get_current_user)):
    """获取系统信息"""
    try:
        system_info = {
            'application': {
                'name': '30天百倍剥头皮交易系统',
                'version': '1.0.0',
                'description': '高频剥头皮 + 复利滚仓'
            },
            'environment': {
                'python_version': sys.version,
                'platform': platform.platform(),
                'architecture': platform.architecture()[0]
            },
            'configuration': {
                'initial_balance': settings.SCALPING_INITIAL_BALANCE if hasattr(settings, 'SCALPING_INITIAL_BALANCE') else 5.0,
                'target_balance': 500.0,
                'target_days': 30
            }
        }

        return {
            'success': True,
            'message': '系统信息获取成功',
            'data': system_info
        }
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")


@router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    current_user: str = Depends(get_current_user)
):
    """获取系统日志"""
    try:
        import os

        log_file = os.path.join("logs", settings.LOG_FILE)
        logs = []

        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                logs = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            'success': True,
            'message': '系统日志获取成功',
            'data': {
                'logs': [line.strip() for line in logs],
                'total_lines': len(logs),
                'requested_lines': lines
            }
        }
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统日志失败: {str(e)}")
