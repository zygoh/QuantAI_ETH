"""
账户相关API端点
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import AccountResponse, AccountInfo
from app.api.dependencies import get_current_user
from app.services.data_service import DataService

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例（在main.py中初始化）
data_service: DataService = None

def set_data_service(service: DataService):
    """设置数据服务实例"""
    global data_service
    data_service = service

@router.get("/status", response_model=AccountResponse)
async def get_account_status(current_user: str = Depends(get_current_user)):
    """获取账户状态"""
    try:
        logger.info("获取账户状态")
        
        if not data_service:
            raise HTTPException(status_code=503, detail="数据服务不可用")
        
        # 获取账户信息
        account_info = await data_service.get_account_info()
        
        if not account_info:
            raise HTTPException(status_code=404, detail="无法获取账户信息")
        
        # 转换为响应模型
        account_data = AccountInfo(
            total_wallet_balance=account_info.get('total_wallet_balance', 0),
            total_unrealized_pnl=account_info.get('total_unrealized_pnl', 0),
            total_margin_balance=account_info.get('total_margin_balance', 0),
            available_balance=account_info.get('available_balance', 0),
            max_withdraw_amount=account_info.get('max_withdraw_amount', 0),
            can_trade=account_info.get('can_trade', False),
            can_deposit=account_info.get('can_deposit', False),
            can_withdraw=account_info.get('can_withdraw', False)
        )
        
        return AccountResponse(
            success=True,
            message="账户状态获取成功",
            data=account_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取账户状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取账户状态失败: {str(e)}")

@router.get("/balance")
async def get_account_balance(current_user: str = Depends(get_current_user)):
    """获取账户余额"""
    try:
        logger.info("获取账户余额")
        
        if not data_service:
            raise HTTPException(status_code=503, detail="数据服务不可用")
        
        account_info = await data_service.get_account_info()
        
        if not account_info:
            raise HTTPException(status_code=404, detail="无法获取账户信息")
        
        balance_info = {
            'total_balance': account_info.get('total_wallet_balance', 0),
            'available_balance': account_info.get('available_balance', 0),
            'margin_balance': account_info.get('total_margin_balance', 0),
            'unrealized_pnl': account_info.get('total_unrealized_pnl', 0),
            'currency': 'USDT'
        }
        
        return {
            'success': True,
            'message': '账户余额获取成功',
            'data': balance_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取账户余额失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取账户余额失败: {str(e)}")

@router.get("/summary")
async def get_account_summary(current_user: str = Depends(get_current_user)):
    """获取账户摘要"""
    try:
        logger.info("获取账户摘要")
        
        if not data_service:
            raise HTTPException(status_code=503, detail="数据服务不可用")
        
        # 获取账户信息
        account_info = await data_service.get_account_info()
        
        # 获取持仓信息
        positions = await data_service.get_position_info()
        
        # 计算摘要信息
        total_positions = len(positions)
        total_position_value = sum(
            abs(pos.get('position_amt', 0)) * pos.get('mark_price', 0) 
            for pos in positions
        )
        
        summary = {
            'account_balance': account_info.get('total_wallet_balance', 0) if account_info else 0,
            'unrealized_pnl': account_info.get('total_unrealized_pnl', 0) if account_info else 0,
            'total_positions': total_positions,
            'total_position_value': total_position_value,
            'margin_ratio': 0,  # 需要计算
            'last_updated': account_info.get('update_time', 0) if account_info else 0
        }
        
        return {
            'success': True,
            'message': '账户摘要获取成功',
            'data': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取账户摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取账户摘要失败: {str(e)}")