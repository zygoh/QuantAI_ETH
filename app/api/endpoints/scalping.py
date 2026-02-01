"""
剥头皮交易系统API端点
"""
import asyncio
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.api.dependencies import get_current_user
from app.scalping.scalping_engine import scalping_engine, get_scalping_status
from app.scalping.backtest import run_scalping_backtest, BacktestResult

logger = logging.getLogger(__name__)
router = APIRouter()


class StartScalpingRequest(BaseModel):
    """启动剥头皮交易请求"""
    initial_balance: float = 5.0


class BacktestRequest(BaseModel):
    """回测请求"""
    symbol: str = "1000PEPE/USDT"
    days: int = 7
    initial_balance: float = 5.0
    leverage: int = 20


@router.post("/start")
async def start_scalping(
    request: StartScalpingRequest,
    current_user: str = Depends(get_current_user)
):
    """启动剥头皮交易"""
    try:
        if scalping_engine.is_running:
            return {
                'success': False,
                'message': '交易引擎已在运行',
                'data': get_scalping_status()
            }

        await scalping_engine.start(request.initial_balance)

        return {
            'success': True,
            'message': '剥头皮交易引擎已启动',
            'data': get_scalping_status()
        }

    except Exception as e:
        logger.error(f"启动剥头皮交易失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_scalping(current_user: str = Depends(get_current_user)):
    """停止剥头皮交易"""
    try:
        if not scalping_engine.is_running:
            return {
                'success': False,
                'message': '交易引擎未在运行'
            }

        await scalping_engine.stop()

        return {
            'success': True,
            'message': '剥头皮交易引擎已停止',
            'data': get_scalping_status()
        }

    except Exception as e:
        logger.error(f"停止剥头皮交易失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status(current_user: str = Depends(get_current_user)):
    """获取交易状态"""
    try:
        return {
            'success': True,
            'message': '状态获取成功',
            'data': get_scalping_status()
        }

    except Exception as e:
        logger.error(f"获取状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-position")
async def close_position(current_user: str = Depends(get_current_user)):
    """手动平仓"""
    try:
        result = scalping_engine.manual_close_position()

        if result is None:
            return {
                'success': False,
                'message': '没有持仓或无法获取价格'
            }

        return {
            'success': True,
            'message': '平仓成功',
            'data': result
        }

    except Exception as e:
        logger.error(f"手动平仓失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# 回测任务存储
_backtest_tasks: Dict[str, Dict[str, Any]] = {}


@router.post("/backtest")
async def run_backtest(
    request: BacktestRequest,
    current_user: str = Depends(get_current_user)
):
    """运行回测"""
    try:
        import uuid
        from datetime import datetime

        task_id = str(uuid.uuid4())

        _backtest_tasks[task_id] = {
            'task_id': task_id,
            'status': 'running',
            'symbol': request.symbol,
            'days': request.days,
            'created_at': datetime.now().isoformat(),
            'result': None,
            'error': None
        }

        # 后台执行回测
        asyncio.create_task(_execute_backtest(
            task_id=task_id,
            symbol=request.symbol,
            days=request.days,
            initial_balance=request.initial_balance,
            leverage=request.leverage
        ))

        return {
            'success': True,
            'message': '回测任务已创建',
            'data': {
                'task_id': task_id,
                'status': 'running'
            }
        }

    except Exception as e:
        logger.error(f"创建回测任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_backtest(
    task_id: str,
    symbol: str,
    days: int,
    initial_balance: float,
    leverage: int
):
    """执行回测任务"""
    try:
        result = await run_scalping_backtest(
            symbol=symbol,
            days=days,
            initial_balance=initial_balance,
            leverage=leverage
        )

        # 转换为可序列化的格式
        result_dict = {
            'symbol': result.symbol,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'initial_balance': result.initial_balance,
            'final_balance': result.final_balance,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_return': result.total_return,
            'total_return_pct': result.total_return_pct,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_pct': result.max_drawdown_pct,
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl_pct': t.pnl_pct,
                    'pnl_amount': t.pnl_amount,
                    'exit_reason': t.exit_reason
                }
                for t in result.trades[:50]  # 只返回前50笔
            ]
        }

        _backtest_tasks[task_id]['status'] = 'completed'
        _backtest_tasks[task_id]['result'] = result_dict

        logger.info(f"✅ 回测任务完成: {task_id}")

    except Exception as e:
        _backtest_tasks[task_id]['status'] = 'failed'
        _backtest_tasks[task_id]['error'] = str(e)
        logger.error(f"❌ 回测任务失败: {task_id} - {e}")


@router.get("/backtest/{task_id}")
async def get_backtest_status(
    task_id: str,
    current_user: str = Depends(get_current_user)
):
    """获取回测任务状态"""
    if task_id not in _backtest_tasks:
        raise HTTPException(status_code=404, detail="回测任务不存在")

    task = _backtest_tasks[task_id]

    return {
        'success': task['status'] != 'failed',
        'message': f"回测任务{task['status']}",
        'data': task
    }


@router.get("/scan-symbols")
async def scan_symbols(current_user: str = Depends(get_current_user)):
    """扫描高波动币种"""
    try:
        from app.scalping.symbol_scanner import symbol_scanner

        logger.info("🔍 手动触发币种扫描...")

        await symbol_scanner.scan_all_symbols()
        top_symbols = symbol_scanner.get_top_symbols(20)

        # 获取详细指标
        symbols_data = []
        for s in top_symbols:
            metrics = symbol_scanner.cache.get(
                s.symbol.replace("/", ""),
                None
            )
            if metrics:
                symbols_data.append({
                    'symbol': s.symbol,
                    'phase': s.phase.value,
                    'price': metrics.price,
                    'volatility': metrics.volatility,
                    'volume_24h': metrics.volume_24h,
                    'price_change_24h': metrics.price_change_24h,
                    'total_score': metrics.total_score
                })
            else:
                symbols_data.append({
                    'symbol': s.symbol,
                    'phase': s.phase.value
                })

        return {
            'success': True,
            'message': f'扫描完成，找到 {len(top_symbols)} 个高波动币种',
            'data': {
                'symbols': symbols_data,
                'scan_time': symbol_scanner.last_scan_time.isoformat() if symbol_scanner.last_scan_time else None,
                'total_scanned': len(symbol_scanner.cache)
            }
        }

    except Exception as e:
        logger.error(f"扫描币种失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug")
async def debug_info(current_user: str = Depends(get_current_user)):
    """调试信息：查看市场数据和信号分析状态"""
    try:
        monitor = scalping_engine.signal_generator.monitor
        analyzer = scalping_engine.signal_generator.analyzer

        debug_data = {
            'monitor_running': monitor.is_running,
            'symbols_data': {},
            'signal_stats': {}
        }

        for symbol, data in monitor.symbol_data.items():
            debug_data['symbols_data'][symbol] = {
                'last_price': data.last_price,
                'orderbook_bids': len(data.orderbook.bids) if data.orderbook else 0,
                'orderbook_asks': len(data.orderbook.asks) if data.orderbook else 0,
                'trades_count': len(data.trades),
                'price_history_count': len(data.price_history),
                'volume_imbalance': data.orderbook.get_volume_imbalance(20) if data.orderbook and data.orderbook.bids and data.orderbook.asks else None
            }

            # 信号统计
            debug_data['signal_stats'][symbol] = analyzer.get_signal_stats(symbol)

        return {
            'success': True,
            'message': '调试信息',
            'data': debug_data
        }

    except Exception as e:
        logger.error(f"获取调试信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-daily")
async def reset_daily_stats(current_user: str = Depends(get_current_user)):
    """重置每日统计（解除每日亏损限制）"""
    try:
        scalping_engine.position_manager.reset_daily_stats()

        return {
            'success': True,
            'message': '每日统计已重置',
            'data': scalping_engine.position_manager.get_status()
        }

    except Exception as e:
        logger.error(f"重置每日统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
