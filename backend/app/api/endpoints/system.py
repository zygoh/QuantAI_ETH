"""
系统相关API端点
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import SystemResponse, SystemStatus, SystemControlRequest
from app.api.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例
trading_controller = None
scheduler = None

def set_services(controller, sched):
    """设置服务实例"""
    global trading_controller, scheduler
    trading_controller = controller
    scheduler = sched

@router.get("/status", response_model=SystemResponse)
async def get_system_status(current_user: str = Depends(get_current_user)):
    """获取系统状态"""
    try:
        logger.info("获取系统状态")
        
        if not trading_controller:
            raise HTTPException(status_code=503, detail="交易控制器不可用")
        
        # 获取系统状态
        status = trading_controller.get_system_status()
        
        # 转换为响应模型
        system_status = SystemStatus(
            system_state=status.get('system_state', 'UNKNOWN'),
            auto_trading_enabled=status.get('auto_trading_enabled', False),
            trading_mode=status.get('trading_mode', 'UNKNOWN'),
            services=status.get('services', {})
        )
        
        return SystemResponse(
            success=True,
            message="系统状态获取成功",
            data=system_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@router.post("/control", response_model=SystemResponse)
async def control_system(
    request: SystemControlRequest,
    current_user: str = Depends(get_current_user)
):
    """控制系统"""
    try:
        logger.info(f"系统控制: {request.action}")
        
        if not trading_controller:
            raise HTTPException(status_code=503, detail="交易控制器不可用")
        
        # 执行系统控制
        if request.action == "START":
            result = await trading_controller.start_system()
        elif request.action == "STOP":
            result = await trading_controller.stop_system()
        elif request.action == "PAUSE":
            result = await trading_controller.pause_system()
        elif request.action == "RESUME":
            result = await trading_controller.resume_system()
        else:
            raise HTTPException(status_code=400, detail=f"无效的控制动作: {request.action}")
        
        return SystemResponse(
            success=result.get('success', False),
            message=result.get('message', ''),
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"系统控制失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统控制失败: {str(e)}")

@router.get("/health")
async def get_system_health(current_user: str = Depends(get_current_user)):
    """获取系统健康状态（详细版）"""
    try:
        logger.info("获取系统健康状态")
        
        # 从健康监控服务获取状态
        from app.services.health_monitor import health_monitor
        health_status = health_monitor.get_health_status()
        
        # 如果健康状态为空或过旧，立即执行一次检查
        if not health_status or health_status.get('overall') == 'UNKNOWN':
            health_status = await health_monitor.check_system_health()
        
        return {
            'success': True,
            'message': '系统健康状态获取成功',
            'data': health_status
        }
        
    except Exception as e:
        logger.error(f"获取系统健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统健康状态失败: {str(e)}")

@router.get("/info")
async def get_system_info(current_user: str = Depends(get_current_user)):
    """获取系统信息"""
    try:
        logger.info("获取系统信息")
        
        from app.core.config import settings
        import platform
        import sys
        
        system_info = {
            'application': {
                'name': 'ETH合约中频智能交易系统',
                'version': '1.0.0',
                'description': '基于LightGBM的ETH合约中频智能交易系统'
            },
            'environment': {
                'python_version': sys.version,
                'platform': platform.platform(),
                'architecture': platform.architecture()[0]
            },
            'configuration': {
                'symbol': settings.SYMBOL,
                'leverage': settings.LEVERAGE,
                'confidence_threshold': settings.CONFIDENCE_THRESHOLD,
                'timeframes': settings.TIMEFRAMES,
                'use_gpu': settings.USE_GPU
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
        logger.info(f"获取系统日志: {lines}行")
        
        # 这里可以实现读取日志文件的逻辑
        # 简化处理，返回空列表
        logs = []
        
        return {
            'success': True,
            'message': '系统日志获取成功',
            'data': {
                'logs': logs,
                'total_lines': len(logs),
                'requested_lines': lines
            }
        }
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统日志失败: {str(e)}")

@router.get("/tasks")
async def get_system_tasks(current_user: str = Depends(get_current_user)):
    """获取系统任务状态"""
    try:
        logger.info("获取系统任务状态")
        
        if not scheduler:
            raise HTTPException(status_code=503, detail="调度器不可用")
        
        # 获取任务状态
        task_status = scheduler.get_task_status()
        
        return {
            'success': True,
            'message': '系统任务状态获取成功',
            'data': task_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取系统任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统任务状态失败: {str(e)}")

@router.post("/tasks/{task_name}/run")
async def run_system_task(
    task_name: str,
    current_user: str = Depends(get_current_user)
):
    """运行系统任务"""
    try:
        logger.info(f"运行系统任务: {task_name}")
        
        if not scheduler:
            raise HTTPException(status_code=503, detail="调度器不可用")
        
        # 运行指定任务
        success = await scheduler.run_task_now(task_name)
        
        if success:
            return {
                'success': True,
                'message': f'任务 {task_name} 已启动',
                'data': {'task_name': task_name, 'started': True}
            }
        else:
            return {
                'success': False,
                'message': f'任务 {task_name} 启动失败',
                'data': {'task_name': task_name, 'started': False}
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"运行系统任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"运行系统任务失败: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats(current_user: str = Depends(get_current_user)):
    """获取缓存统计"""
    try:
        logger.info("获取缓存统计")
        
        from app.core.cache import cache_manager
        stats = await cache_manager.get_cache_stats()
        
        return {
            'success': True,
            'message': '缓存统计获取成功',
            'data': stats
        }
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")