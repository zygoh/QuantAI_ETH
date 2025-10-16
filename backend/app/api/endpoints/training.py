"""
训练相关API端点
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import TrainingResponse, TrainingRequest
from app.api.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例
ml_service = None
scheduler = None

def set_services(ml, sched):
    """设置服务实例"""
    global ml_service, scheduler
    ml_service = ml
    scheduler = sched

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