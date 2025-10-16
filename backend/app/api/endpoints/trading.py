"""
交易相关API端点
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import TradingResponse, TradeRequest, TradingModeRequest
from app.api.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例
trading_controller = None

def set_trading_controller(controller):
    """设置交易控制器实例"""
    global trading_controller
    trading_controller = controller

@router.post("/execute", response_model=TradingResponse)
async def execute_trade(
    request: TradeRequest,
    current_user: str = Depends(get_current_user)
):
    """执行交易"""
    try:
        logger.info(f"执行交易: {request.action} {request.symbol}")
        
        if not trading_controller:
            raise HTTPException(status_code=503, detail="交易控制器不可用")
        
        # 执行手动交易
        result = await trading_controller.manual_trade(
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity
        )
        
        return TradingResponse(
            success=result.get('success', False),
            message=result.get('message', ''),
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行交易失败: {e}")
        raise HTTPException(status_code=500, detail=f"执行交易失败: {str(e)}")

@router.post("/mode", response_model=TradingResponse)
async def set_trading_mode(
    request: TradingModeRequest,
    current_user: str = Depends(get_current_user)
):
    """设置交易模式"""
    try:
        logger.info(f"设置交易模式: {request.mode}")
        
        if not trading_controller:
            raise HTTPException(status_code=503, detail="交易控制器不可用")
        
        # 设置交易模式
        result = await trading_controller.set_trading_mode(request.mode)
        
        return TradingResponse(
            success=result.get('success', False),
            message=result.get('message', ''),
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置交易模式失败: {e}")
        raise HTTPException(status_code=500, detail=f"设置交易模式失败: {str(e)}")

@router.get("/status")
async def get_trading_status(current_user: str = Depends(get_current_user)):
    """获取交易状态"""
    try:
        logger.info("获取交易状态")
        
        if not trading_controller:
            raise HTTPException(status_code=503, detail="交易控制器不可用")
        
        # 获取系统状态
        status = trading_controller.get_system_status()
        
        return {
            'success': True,
            'message': '交易状态获取成功',
            'data': status
        }
        
    except Exception as e:
        logger.error(f"获取交易状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易状态失败: {str(e)}")

@router.get("/performance")
async def get_trading_performance(current_user: str = Depends(get_current_user)):
    """获取交易表现"""
    try:
        logger.info("获取交易表现")
        
        if not trading_controller:
            raise HTTPException(status_code=503, detail="交易控制器不可用")
        
        # 获取交易表现
        performance = await trading_controller.get_trading_performance()
        
        return {
            'success': True,
            'message': '交易表现获取成功',
            'data': performance
        }
        
    except Exception as e:
        logger.error(f"获取交易表现失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易表现失败: {str(e)}")

@router.get("/orders")
async def get_trading_orders(current_user: str = Depends(get_current_user)):
    """获取交易订单"""
    try:
        logger.info("获取交易订单")
        
        # 这里可以从交易引擎获取订单信息
        # 简化处理，返回空列表
        orders = []
        
        return {
            'success': True,
            'message': '交易订单获取成功',
            'data': orders
        }
        
    except Exception as e:
        logger.error(f"获取交易订单失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易订单失败: {str(e)}")

@router.get("/history")
async def get_trading_history(
    days: int = 7,
    current_user: str = Depends(get_current_user)
):
    """获取交易历史"""
    try:
        logger.info(f"获取交易历史: {days}天")
        
        # 这里可以从数据库获取交易历史
        # 简化处理，返回空列表
        history = []
        
        return {
            'success': True,
            'message': '交易历史获取成功',
            'data': {
                'period_days': days,
                'trades': history,
                'total_trades': len(history)
            }
        }
        
    except Exception as e:
        logger.error(f"获取交易历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易历史失败: {str(e)}")

@router.post("/close/{symbol}")
async def close_position(
    symbol: str,
    current_user: str = Depends(get_current_user)
):
    """平仓"""
    try:
        logger.info(f"平仓: {symbol}")
        
        if not trading_controller:
            raise HTTPException(status_code=503, detail="交易控制器不可用")
        
        # 执行平仓
        result = await trading_controller.manual_trade(
            symbol=symbol,
            action='CLOSE'
        )
        
        return {
            'success': result.get('success', False),
            'message': result.get('message', ''),
            'data': result
        }
        
    except Exception as e:
        logger.error(f"平仓失败: {e}")
        raise HTTPException(status_code=500, detail=f"平仓失败: {str(e)}")

@router.get("/limits")
async def get_trading_limits(current_user: str = Depends(get_current_user)):
    """获取交易限制"""
    try:
        logger.info("获取交易限制")
        
        from app.core.config import settings
        
        # 从实际配置获取真实限制
        limits = {
            'symbol': settings.SYMBOL,
            'leverage': settings.LEVERAGE,
            'confidence_threshold': settings.CONFIDENCE_THRESHOLD,
            'timeframes': settings.TIMEFRAMES,
            'max_drawdown_limit': settings.MAX_DRAWDOWN_LIMIT,
            'kelly_multiplier': settings.KELLY_MULTIPLIER,
            'var_confidence': settings.VAR_CONFIDENCE,
        }
        
        # 如果交易控制器可用，获取交易模式
        if trading_controller:
            limits['trading_mode'] = trading_controller.trading_mode
        
        return {
            'success': True,
            'message': '交易限制获取成功',
            'data': limits
        }
        
    except Exception as e:
        logger.error(f"获取交易限制失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易限制失败: {str(e)}")