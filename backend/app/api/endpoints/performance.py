"""
绩效相关API端点
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.models import PerformanceResponse, PerformanceMetrics
from app.api.dependencies import get_current_user
from app.core.config import settings
from app.services.drawdown_monitor import drawdown_monitor
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局服务实例
risk_service = None
trading_controller = None

def set_services(risk, trading):
    """设置服务实例"""
    global risk_service, trading_controller
    risk_service = risk
    trading_controller = trading

@router.get("/", response_model=PerformanceResponse)
async def get_performance(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    days: int = Query(30, description="统计天数"),
    current_user: str = Depends(get_current_user)
):
    """获取交易绩效"""
    try:
        logger.info(f"获取交易绩效: {symbol}, {days}天")
        
        if not risk_service:
            raise HTTPException(status_code=503, detail="风险服务不可用")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 获取风险报告
        risk_report = await risk_service.get_comprehensive_risk_report(query_symbol)
        
        # 提取绩效指标
        performance_data = PerformanceMetrics(
            total_return=0.0,  # 需要计算
            sharpe_ratio=risk_report.get('performance_metrics', {}).get('sharpe_ratio', 0),
            max_drawdown=risk_report.get('drawdown_metrics', {}).get('max_drawdown', 0),
            win_rate=risk_report.get('trading_metrics', {}).get('win_rate', 0),
            profit_factor=risk_report.get('trading_metrics', {}).get('profit_factor', 0),
            total_trades=risk_report.get('trading_metrics', {}).get('total_trades', 0)
        )
        
        return PerformanceResponse(
            success=True,
            message="交易绩效获取成功",
            data=performance_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取交易绩效失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易绩效失败: {str(e)}")

@router.get("/risk")
async def get_risk_metrics(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    current_user: str = Depends(get_current_user)
):
    """获取风险指标"""
    try:
        logger.info(f"获取风险指标: {symbol}")
        
        if not risk_service:
            raise HTTPException(status_code=503, detail="风险服务不可用")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 获取综合风险报告
        risk_report = await risk_service.get_comprehensive_risk_report(query_symbol)
        
        return {
            'success': True,
            'message': '风险指标获取成功',
            'data': risk_report
        }
        
    except Exception as e:
        logger.error(f"获取风险指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取风险指标失败: {str(e)}")

@router.get("/drawdown")
async def get_drawdown_metrics(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    current_user: str = Depends(get_current_user)
):
    """获取回撤指标"""
    try:
        logger.info(f"获取回撤指标: {symbol}")
        
        if not risk_service:
            raise HTTPException(status_code=503, detail="风险服务不可用")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 计算回撤指标
        max_dd, current_dd = await risk_service.calculate_max_drawdown(query_symbol)
        
        # 获取回撤监控状态
        drawdown_status = await drawdown_monitor.get_current_drawdown_status()
        drawdown_stats = await drawdown_monitor.get_drawdown_statistics()
        
        drawdown_data = {
            'max_drawdown': max_dd,
            'current_drawdown': current_dd,
            'drawdown_limit': settings.MAX_DRAWDOWN_LIMIT * 100,
            'risk_level': drawdown_status.get('risk_level', 'UNKNOWN'),
            'statistics': drawdown_stats,
            'monitoring_status': drawdown_status.get('monitoring_status', 'INACTIVE')
        }
        
        return {
            'success': True,
            'message': '回撤指标获取成功',
            'data': drawdown_data
        }
        
    except Exception as e:
        logger.error(f"获取回撤指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取回撤指标失败: {str(e)}")

@router.get("/returns")
async def get_returns_analysis(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    days: int = Query(30, description="分析天数"),
    current_user: str = Depends(get_current_user)
):
    """获取收益分析"""
    try:
        logger.info(f"获取收益分析: {symbol}, {days}天")
        
        if not risk_service:
            raise HTTPException(status_code=503, detail="风险服务不可用")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 获取收益率数据
        returns = await risk_service._get_returns_data(query_symbol, days)
        
        if returns.empty:
            return {
                'success': False,
                'message': '无收益数据',
                'data': None
            }
        
        # 计算收益统计
        returns_stats = {
            'total_return': (returns + 1).prod() - 1,
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'max_daily_gain': returns.max(),
            'max_daily_loss': returns.min(),
            'avg_daily_return': returns.mean(),
            'median_daily_return': returns.median()
        }
        
        return {
            'success': True,
            'message': '收益分析获取成功',
            'data': returns_stats
        }
        
    except Exception as e:
        logger.error(f"获取收益分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取收益分析失败: {str(e)}")

@router.get("/ratios")
async def get_performance_ratios(
    symbol: Optional[str] = Query(None, description="交易对符号"),
    current_user: str = Depends(get_current_user)
):
    """获取绩效比率"""
    try:
        logger.info(f"获取绩效比率: {symbol}")
        
        if not risk_service:
            raise HTTPException(status_code=503, detail="风险服务不可用")
        
        query_symbol = symbol or settings.SYMBOL
        
        # 计算各种比率
        sharpe_ratio = await risk_service.calculate_sharpe_ratio(query_symbol)
        sortino_ratio = await risk_service.calculate_sortino_ratio(query_symbol)
        
        # 获取交易指标
        trading_metrics = await risk_service.calculate_trading_metrics(query_symbol)
        
        # Kelly准则
        kelly_pct = risk_service.kelly_criterion(
            trading_metrics.get('win_rate', 0),
            trading_metrics.get('avg_win', 0),
            trading_metrics.get('avg_loss', 0)
        )
        
        ratios = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'kelly_percentage': kelly_pct,
            'win_rate': trading_metrics.get('win_rate', 0),
            'profit_factor': trading_metrics.get('profit_factor', 0),
            'avg_win': trading_metrics.get('avg_win', 0),
            'avg_loss': trading_metrics.get('avg_loss', 0),
            'total_trades': trading_metrics.get('total_trades', 0)
        }
        
        return {
            'success': True,
            'message': '绩效比率获取成功',
            'data': ratios
        }
        
    except Exception as e:
        logger.error(f"获取绩效比率失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取绩效比率失败: {str(e)}")

@router.get("/summary")
async def get_performance_summary(
    days: int = Query(30, description="统计天数"),
    current_user: str = Depends(get_current_user)
):
    """获取绩效摘要"""
    try:
        logger.info(f"获取绩效摘要: {days}天")
        
        # 获取交易表现
        if trading_controller:
            performance = await trading_controller.get_trading_performance()
        else:
            performance = {}
        
        # 构建摘要
        summary = {
            'period_days': days,
            'position_summary': performance.get('position_summary', {}),
            'signal_performance': performance.get('signal_performance', {}),
            'trading_status': performance.get('trading_status', {}),
            'system_status': performance.get('system_status', {})
        }
        
        return {
            'success': True,
            'message': '绩效摘要获取成功',
            'data': summary
        }
        
    except Exception as e:
        logger.error(f"获取绩效摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取绩效摘要失败: {str(e)}")