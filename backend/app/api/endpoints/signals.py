"""
信号相关API端点
"""
import logging
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.models import SignalsResponse, TradingSignal, SignalRequest
from app.api.dependencies import get_current_user
from app.core.config import settings
from app.core.database import postgresql_manager
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例
signal_generator = None
ml_service = None
data_service = None

def set_services(sg, ml, ds):
    """设置服务实例"""
    global signal_generator, ml_service, data_service
    signal_generator = sg
    ml_service = ml
    data_service = ds

@router.get("/", response_model=SignalsResponse)
async def get_signals(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    hours: int = Query(24, description="查询小时数"),
    limit: int = Query(100, description="返回数量限制"),
    current_user: str = Depends(get_current_user)
):
    """获取交易信号历史"""
    try:
        logger.info(f"获取交易信号: {symbol}, {hours}小时")
        
        if not signal_generator:
            raise HTTPException(status_code=503, detail="信号生成器不可用")
        
        # 使用默认交易对
        query_symbol = symbol or settings.SYMBOL
        
        # 获取信号历史
        signals = await signal_generator.get_recent_signals(query_symbol, hours, limit)
        
        # 转换为响应模型
        signal_list = []
        for signal in signals:
            trading_signal = TradingSignal(
                timestamp=signal.get('timestamp', datetime.now()),
                symbol=signal.get('symbol', ''),
                signal_type=signal.get('signal_type', ''),
                confidence=signal.get('confidence', 0),
                entry_price=signal.get('entry_price', 0),
                stop_loss=signal.get('stop_loss', 0),
                take_profit=signal.get('take_profit', 0),
                position_size=signal.get('position_size', 0)
            )
            signal_list.append(trading_signal)
        
        return SignalsResponse(
            success=True,
            message="交易信号获取成功",
            data=signal_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取交易信号失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易信号失败: {str(e)}")

@router.post("/generate")
async def generate_signal(
    request: SignalRequest,
    current_user: str = Depends(get_current_user)
):
    """生成交易信号"""
    try:
        logger.info(f"生成交易信号: {request.symbol}")
        
        if not signal_generator:
            raise HTTPException(status_code=503, detail="信号生成器不可用")
        
        # 生成信号
        if request.force:
            signal = await signal_generator.force_generate_signal(request.symbol)
        else:
            signal = await signal_generator.generate_signal(request.symbol)
        
        if signal:
            signal_data = {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size
            }
            
            return {
                'success': True,
                'message': '交易信号生成成功',
                'data': signal_data
            }
        else:
            return {
                'success': False,
                'message': '未生成有效信号',
                'data': None
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成交易信号失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成交易信号失败: {str(e)}")

@router.get("/latest")
async def get_latest_signal(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    current_user: str = Depends(get_current_user)
):
    """获取最新信号"""
    try:
        logger.info(f"获取最新信号: {symbol}")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 获取最近1小时的信号
        signals = await signal_generator.get_recent_signals(query_symbol, hours=1, limit=1)
        
        if signals:
            latest_signal = signals[0]
            return {
                'success': True,
                'message': '最新信号获取成功',
                'data': latest_signal
            }
        else:
            return {
                'success': True,
                'message': '暂无最新信号',
                'data': None
            }
        
    except Exception as e:
        logger.error(f"获取最新信号失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取最新信号失败: {str(e)}")

@router.get("/performance")
async def get_signal_performance(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    days: int = Query(7, description="统计天数"),
    current_user: str = Depends(get_current_user)
):
    """获取信号表现统计"""
    try:
        logger.info(f"获取信号表现: {symbol}, {days}天")
        
        if not signal_generator:
            raise HTTPException(status_code=503, detail="信号生成器不可用")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 获取信号表现
        performance = await signal_generator.get_signal_performance(query_symbol, days)
        
        return {
            'success': True,
            'message': '信号表现获取成功',
            'data': performance
        }
        
    except Exception as e:
        logger.error(f"获取信号表现失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取信号表现失败: {str(e)}")

@router.get("/statistics")
async def get_signal_statistics(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    days: int = Query(30, description="统计天数"),
    current_user: str = Depends(get_current_user)
):
    """获取信号统计"""
    try:
        logger.info(f"获取信号统计: {symbol}, {days}天")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 获取信号历史
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        signals = await postgresql_manager.query_signals(
            query_symbol, start_time, end_time, limit=1000
        )
        
        # 统计信号
        total_signals = len(signals)
        long_signals = len([s for s in signals if s.get('signal_type') == 'LONG'])
        short_signals = len([s for s in signals if s.get('signal_type') == 'SHORT'])
        
        # 计算平均置信度
        avg_confidence = 0
        if signals:
            confidences = [s.get('confidence', 0) for s in signals]
            avg_confidence = sum(confidences) / len(confidences)
        
        statistics = {
            'period_days': days,
            'total_signals': total_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'long_ratio': long_signals / total_signals if total_signals > 0 else 0,
            'short_ratio': short_signals / total_signals if total_signals > 0 else 0,
            'avg_confidence': avg_confidence,
            'signal_frequency': total_signals / days if days > 0 else 0
        }
        
        return {
            'success': True,
            'message': '信号统计获取成功',
            'data': statistics
        }
        
    except Exception as e:
        logger.error(f"获取信号统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取信号统计失败: {str(e)}")

@router.get("/model/prediction")
async def get_model_prediction(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    current_user: str = Depends(get_current_user)
):
    """获取模型预测"""
    try:
        logger.info(f"获取模型预测: {symbol}")
        
        if not ml_service or not data_service:
            raise HTTPException(status_code=503, detail="服务不可用")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 获取最新数据
        latest_data = await data_service.get_latest_klines(query_symbol, '15m', limit=200)
        
        if not latest_data:
            raise HTTPException(status_code=404, detail="无法获取市场数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(latest_data)
        
        # 模型预测
        prediction = await ml_service.predict(df)
        
        return {
            'success': True,
            'message': '模型预测获取成功',
            'data': prediction
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型预测失败: {str(e)}")