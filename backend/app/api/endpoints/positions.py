"""
持仓相关API端点
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.models import PositionsResponse, PositionInfo
from app.api.dependencies import get_current_user
from app.services.data_service import DataService
from app.trading.position_manager import position_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例
data_service: DataService = None

def set_data_service(service: DataService):
    """设置数据服务实例"""
    global data_service
    data_service = service

@router.get("/", response_model=PositionsResponse)
async def get_positions(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    current_user: str = Depends(get_current_user)
):
    """获取持仓信息（模拟交易模式，使用position_manager）"""
    try:
        logger.info(f"获取持仓信息: {symbol}")
        
        # 🔥 模拟交易模式：使用position_manager获取虚拟持仓
        if symbol:
            position = await position_manager.get_position(symbol)
            positions = [position] if position else []
        else:
            # 获取所有持仓
            summary = await position_manager.get_position_summary()
            positions = []
            for pos_symbol in summary.get('positions', {}).keys():
                pos = await position_manager.get_position(pos_symbol)
                if pos:
                    positions.append(pos)
        
        # 转换为响应模型
        position_list = []
        for pos in positions:
            if pos:  # 确保position不为None
                position_info = PositionInfo(
                    symbol=pos.symbol if hasattr(pos, 'symbol') else pos.get('symbol', ''),
                    side=pos.side if hasattr(pos, 'side') else ('LONG' if float(pos.get('position_amt', 0)) > 0 else 'SHORT'),
                    size=pos.size if hasattr(pos, 'size') else abs(float(pos.get('position_amt', 0))),
                    entry_price=pos.entry_price if hasattr(pos, 'entry_price') else float(pos.get('entry_price', 0)),
                    mark_price=pos.mark_price if hasattr(pos, 'mark_price') else float(pos.get('mark_price', 0)),
                    unrealized_pnl=pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else float(pos.get('pnl', 0)),
                    percentage=pos.percentage if hasattr(pos, 'percentage') else float(pos.get('percentage', 0)),
                    leverage=pos.leverage if hasattr(pos, 'leverage') else int(pos.get('leverage', 1)),
                    margin_type=pos.margin_type if hasattr(pos, 'margin_type') else pos.get('margin_type', 'cross')
                )
                position_list.append(position_info)
        
        return PositionsResponse(
            success=True,
            message="持仓信息获取成功（模拟交易模式）",
            data=position_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取持仓信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取持仓信息失败: {str(e)}")

@router.get("/summary")
async def get_positions_summary(current_user: str = Depends(get_current_user)):
    """获取持仓摘要"""
    try:
        logger.info("获取持仓摘要")
        
        # 获取持仓摘要
        summary = await position_manager.get_position_summary()
        
        return {
            'success': True,
            'message': '持仓摘要获取成功',
            'data': summary
        }
        
    except Exception as e:
        logger.error(f"获取持仓摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取持仓摘要失败: {str(e)}")

@router.get("/risk")
async def get_position_risk(current_user: str = Depends(get_current_user)):
    """获取持仓风险"""
    try:
        logger.info("获取持仓风险")
        
        # 获取风险指标
        risk_metrics = await position_manager.calculate_risk_metrics()
        margin_risk = await position_manager.check_margin_call_risk()
        
        risk_info = {
            'margin_level': risk_metrics.margin_level,
            'free_margin': risk_metrics.free_margin,
            'total_margin': risk_metrics.total_margin,
            'unrealized_pnl': risk_metrics.total_unrealized_pnl,
            'risk_level': margin_risk['risk_level'],
            'risk_message': margin_risk['message']
        }
        
        return {
            'success': True,
            'message': '持仓风险获取成功',
            'data': risk_info
        }
        
    except Exception as e:
        logger.error(f"获取持仓风险失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取持仓风险失败: {str(e)}")

@router.get("/{symbol}")
async def get_position_by_symbol(
    symbol: str,
    current_user: str = Depends(get_current_user)
):
    """获取指定交易对的持仓"""
    try:
        logger.info(f"获取持仓: {symbol}")
        
        # 获取指定持仓
        position = await position_manager.get_position(symbol)
        
        if not position:
            return {
                'success': True,
                'message': '无持仓',
                'data': None
            }
        
        position_data = {
            'symbol': position.symbol,
            'side': position.side,
            'size': position.size,
            'entry_price': position.entry_price,
            'mark_price': position.mark_price,
            'unrealized_pnl': position.unrealized_pnl,
            'percentage': position.percentage,
            'leverage': position.leverage,
            'margin_type': position.margin_type,
            'liquidation_price': position.liquidation_price
        }
        
        return {
            'success': True,
            'message': '持仓信息获取成功',
            'data': position_data
        }
        
    except Exception as e:
        logger.error(f"获取持仓失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取持仓失败: {str(e)}")

@router.get("/{symbol}/value")
async def get_position_value(
    symbol: str,
    current_user: str = Depends(get_current_user)
):
    """获取持仓价值"""
    try:
        logger.info(f"获取持仓价值: {symbol}")
        
        # 计算持仓价值
        position_value = await position_manager.calculate_position_value(symbol)
        
        return {
            'success': True,
            'message': '持仓价值获取成功',
            'data': {
                'symbol': symbol,
                'position_value': position_value,
                'currency': 'USDT'
            }
        }
        
    except Exception as e:
        logger.error(f"获取持仓价值失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取持仓价值失败: {str(e)}")