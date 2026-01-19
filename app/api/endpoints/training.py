"""
训练相关API端点
"""
# StdLib
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Third-Party
from fastapi import APIRouter, Depends, HTTPException

# Local App
from app.api.dependencies import get_current_user
from app.core.config import settings
from app.api.models import (
    TrainingRequest,
    TrainingResponse,
    BacktestRequest,
    BacktestResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例
ml_service = None
scheduler = None
backtest_service = None

# ✅ 回测任务状态管理（内存存储，生产环境可考虑使用Redis）
_backtest_tasks: Dict[str, Dict[str, Any]] = {}

def _cleanup_old_tasks():
    """清理超过24小时的已完成任务（避免内存泄漏）"""
    try:
        current_time = datetime.now()
        expired_tasks = []
        
        for task_id, task in _backtest_tasks.items():
            if task['status'] in ['completed', 'failed']:
                completed_at = datetime.fromisoformat(task.get('completed_at', task['created_at']))
                if (current_time - completed_at).total_seconds() > 86400:  # 24小时
                    expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            del _backtest_tasks[task_id]
            logger.debug(f"清理过期回测任务: {task_id}")
        
        if expired_tasks:
            logger.info(f"清理了 {len(expired_tasks)} 个过期回测任务")
    except Exception as e:
        logger.warning(f"清理回测任务失败: {e}")

def set_services(ml, sched, backtest=None):
    """设置服务实例"""
    global ml_service, scheduler, backtest_service
    ml_service = ml
    scheduler = sched
    backtest_service = backtest

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    current_user: str = Depends(get_current_user)
):
    """开始模型训练"""
    try:
        logger.info(f"开始模型训练: force_retrain={request.force_retrain}")
        
        if not ml_service:
            raise HTTPException(status_code=503, detail="机器学习服务不可用")
        
        # 开始训练
        metrics = await ml_service.train_model(force_retrain=request.force_retrain)
        
        if metrics:
            return TrainingResponse(
                success=True,
                message="模型训练完成",
                data=metrics
            )
        else:
            return TrainingResponse(
                success=False,
                message="模型训练失败",
                data=None
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")

@router.post("/backtest", response_model=Dict[str, Any])
async def run_backtest(
    request: BacktestRequest,
    current_user: str = Depends(get_current_user)
):
    """启动回测任务（后台异步执行，立即返回任务ID）"""
    try:
        if not backtest_service:
            raise HTTPException(status_code=503, detail="回测服务不可用")

        symbol = request.symbol or settings.SYMBOL
        
        # ✅ 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # ✅ 初始化任务状态
        _backtest_tasks[task_id] = {
            'task_id': task_id,
            'status': 'pending',  # pending, running, completed, failed
            'symbol': symbol,
            'days': request.days,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error': None
        }
        
        logger.info(f"🚀 创建回测任务: {task_id} | {symbol} {request.days}天 | "
                   f"初始资金={request.initial_balance} | 杠杆={request.leverage}x")
        
        # ✅ 后台异步执行回测（不阻塞接口返回）
        asyncio.create_task(_execute_backtest_task(
            task_id=task_id,
            symbol=symbol,
            days=request.days,
            initial_balance=request.initial_balance,
            leverage=request.leverage,
            primary_timeframe=request.primary_timeframe,
            timeframes=request.timeframes,
            include_trades=request.include_trades
        ))
        
        # ✅ 立即返回任务ID
        return {
            'success': True,
            'message': '回测任务已创建，正在后台执行',
            'data': {
                'task_id': task_id,
                'status': 'pending',
                'created_at': _backtest_tasks[task_id]['created_at']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 创建回测任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建回测任务失败: {str(e)}")


async def _execute_backtest_task(
    task_id: str,
    symbol: str,
    days: int,
    initial_balance: float,
    leverage: float,
    primary_timeframe: str,
    timeframes: Optional[list],
    include_trades: bool
):
    """后台执行回测任务"""
    try:
        # 更新任务状态为运行中
        if task_id in _backtest_tasks:
            _backtest_tasks[task_id]['status'] = 'running'
            _backtest_tasks[task_id]['started_at'] = datetime.now().isoformat()
        
        logger.info(f"🔄 开始执行回测任务: {task_id}")
        
        # 执行回测
        result = await backtest_service.run_backtest(
            symbol=symbol,
            days=days,
            initial_balance=initial_balance,
            leverage=leverage,
            primary_timeframe=primary_timeframe,
            timeframes=timeframes,
            include_trades=include_trades
        )
        
        # 更新任务状态为完成
        if task_id in _backtest_tasks:
            _backtest_tasks[task_id]['status'] = 'completed'
            _backtest_tasks[task_id]['completed_at'] = datetime.now().isoformat()
            _backtest_tasks[task_id]['result'] = result
        
        logger.info(f"✅ 回测任务完成: {task_id} | 胜率={result.get('win_rate', 0):.2%} | "
                   f"总收益={result.get('total_return', 0):.2%} | "
                   f"交易次数={result.get('total_trades', 0)}")
        
    except Exception as e:
        # 更新任务状态为失败
        if task_id in _backtest_tasks:
            _backtest_tasks[task_id]['status'] = 'failed'
            _backtest_tasks[task_id]['completed_at'] = datetime.now().isoformat()
            _backtest_tasks[task_id]['error'] = str(e)
        
        logger.error(f"❌ 回测任务失败: {task_id} | {e}", exc_info=True)


@router.get("/backtest/{task_id}", response_model=Dict[str, Any])
async def get_backtest_status(
    task_id: str,
    current_user: str = Depends(get_current_user)
):
    """查询回测任务状态"""
    try:
        # ✅ 清理过期任务
        _cleanup_old_tasks()
        
        if task_id not in _backtest_tasks:
            raise HTTPException(status_code=404, detail="回测任务不存在")
        
        task = _backtest_tasks[task_id]
        
        response_data = {
            'task_id': task['task_id'],
            'status': task['status'],
            'symbol': task['symbol'],
            'days': task['days'],
            'created_at': task['created_at'],
            'started_at': task['started_at'],
            'completed_at': task['completed_at']
        }
        
        # 如果任务完成，返回结果
        if task['status'] == 'completed':
            response_data['result'] = task['result']
            return {
                'success': True,
                'message': '回测任务已完成',
                'data': response_data
            }
        # 如果任务失败，返回错误信息
        elif task['status'] == 'failed':
            response_data['error'] = task['error']
            return {
                'success': False,
                'message': '回测任务失败',
                'data': response_data
            }
        # 如果任务还在运行中
        else:
            return {
                'success': True,
                'message': f'回测任务{task["status"]}中',
                'data': response_data
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询回测任务状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询回测任务状态失败: {str(e)}")


@router.get("/backtest", response_model=Dict[str, Any])
async def list_backtest_tasks(
    current_user: str = Depends(get_current_user)
):
    """列出所有回测任务"""
    try:
        # ✅ 清理过期任务
        _cleanup_old_tasks()
        
        tasks_list = []
        for task_id, task in _backtest_tasks.items():
            tasks_list.append({
                'task_id': task['task_id'],
                'status': task['status'],
                'symbol': task['symbol'],
                'days': task['days'],
                'created_at': task['created_at'],
                'started_at': task['started_at'],
                'completed_at': task['completed_at']
            })
        
        # 按创建时间倒序排列
        tasks_list.sort(key=lambda x: x['created_at'], reverse=True)
        
        return {
            'success': True,
            'message': '回测任务列表获取成功',
            'data': {
                'tasks': tasks_list,
                'total': len(tasks_list)
            }
        }
        
    except Exception as e:
        logger.error(f"获取回测任务列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取回测任务列表失败: {str(e)}")

@router.get("/status")
async def get_training_status(current_user: str = Depends(get_current_user)):
    """获取训练状态"""
    try:
        logger.info("获取训练状态")
        
        if not ml_service:
            raise HTTPException(status_code=503, detail="机器学习服务不可用")
        
        # 获取模型信息
        model_info = await ml_service.get_model_info()
        
        return {
            'success': True,
            'message': '训练状态获取成功',
            'data': model_info
        }
        
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")

@router.get("/metrics")
async def get_training_metrics(current_user: str = Depends(get_current_user)):
    """获取训练指标"""
    try:
        logger.info("获取训练指标")
        
        if not ml_service:
            raise HTTPException(status_code=503, detail="机器学习服务不可用")
        
        # 获取模型指标
        metrics = ml_service.model_metrics
        
        return {
            'success': True,
            'message': '训练指标获取成功',
            'data': metrics
        }
        
    except Exception as e:
        logger.error(f"获取训练指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练指标失败: {str(e)}")

@router.get("/schedule")
async def get_training_schedule(current_user: str = Depends(get_current_user)):
    """获取训练调度"""
    try:
        logger.info("获取训练调度")
        
        if not scheduler:
            raise HTTPException(status_code=503, detail="调度器不可用")
        
        # 获取任务状态
        task_status = scheduler.get_task_status()
        
        # 提取训练相关任务
        training_tasks = {
            k: v for k, v in task_status.items() 
            if 'training' in k.lower()
        }
        
        return {
            'success': True,
            'message': '训练调度获取成功',
            'data': training_tasks
        }
        
    except Exception as e:
        logger.error(f"获取训练调度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练调度失败: {str(e)}")

@router.post("/schedule/run")
async def run_training_task(current_user: str = Depends(get_current_user)):
    """手动运行训练任务"""
    try:
        logger.info("手动运行训练任务")
        
        if not scheduler:
            raise HTTPException(status_code=503, detail="调度器不可用")
        
        # 手动运行模型训练任务
        success = await scheduler.run_task_now('model_training')
        
        if success:
            return {
                'success': True,
                'message': '训练任务已启动',
                'data': {'task_started': True}
            }
        else:
            return {
                'success': False,
                'message': '训练任务启动失败',
                'data': {'task_started': False}
            }
        
    except Exception as e:
        logger.error(f"运行训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"运行训练任务失败: {str(e)}")

@router.get("/history")
async def get_training_history(current_user: str = Depends(get_current_user)):
    """获取训练历史"""
    try:
        logger.info("获取训练历史")
        
        # 这里可以从数据库获取训练历史
        # 简化处理，返回当前模型信息
        if ml_service:
            model_info = await ml_service.get_model_info()
            history = [model_info] if model_info else []
        else:
            history = []
        
        return {
            'success': True,
            'message': '训练历史获取成功',
            'data': {
                'training_sessions': history,
                'total_sessions': len(history)
            }
        }
        
    except Exception as e:
        logger.error(f"获取训练历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练历史失败: {str(e)}")

@router.get("/features")
async def get_feature_importance(current_user: str = Depends(get_current_user)):
    """获取特征重要性"""
    try:
        logger.info("获取特征重要性")
        
        if not ml_service or not ml_service.model_metrics:
            raise HTTPException(status_code=404, detail="模型未训练或指标不可用")
        
        # 获取特征重要性
        feature_importance = ml_service.model_metrics.get('feature_importance', {})
        
        # 转换为列表格式
        features = [
            {'feature': name, 'importance': importance}
            for name, importance in feature_importance.items()
        ]
        
        return {
            'success': True,
            'message': '特征重要性获取成功',
            'data': {
                'features': features,
                'total_features': len(features)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取特征重要性失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取特征重要性失败: {str(e)}")