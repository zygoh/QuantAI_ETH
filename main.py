# -*- coding: utf-8 -*-
"""
AI 模拟交易系统（本地端）

功能：
- 从云端获取选币结果
- 生成 5m 和 15m 图表
- 调用 Claude AI 分析图表
- 根据 AI 信号进行模拟交易
- WebSocket 实时价格监控，止盈止损
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.core.config import settings
from app.chart.generator import chart_generator
from app.trading.ai_analyzer import ai_analyzer
from app.trading.simulator import trading_simulator
from app.trading.models import PositionSide, SignalAction
from app.trading.price_monitor import price_monitor
from app.trading.price_util import get_current_price
from app.trading.market_context import build_market_context
from app.exchange.clients.binance.binance_client import binance_client


# 云端选币接口
CLOUD_COIN_SELECT_URL = "https://n8n.do2ge.com/tail/tro"


# 配置日志
def setup_logging() -> logging.Logger:
    """配置日志"""
    os.makedirs("logs", exist_ok=True)

    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 文件处理器
    log_filename = f"logs/selector_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 降低第三方库日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = setup_logging()


# 上次价格日志时间
_last_price_log_time: Optional[datetime] = None


async def on_price_update(symbol: str, price: float) -> None:
    """
    WebSocket 价格更新回调

    每次收到价格更新时检查止盈止损
    """
    global _last_price_log_time

    if not trading_simulator.has_position():
        return

    pos = trading_simulator.position
    if pos.symbol != symbol:
        return

    # 检查止盈止损
    trigger = trading_simulator.check_stop_loss_take_profit(price)

    if trigger:
        logger.info(f"🎯 触发 {trigger}: {symbol} @ ${price:.4f}")
        trading_simulator.close_position(price, trigger)
        logger.info(trading_simulator.get_account_summary())

        # 取消订阅
        await price_monitor.unsubscribe()
        return

    # 检查爆仓：浮亏 >= 保证金时强制平仓
    margin = pos.position_size_usd / pos.leverage
    unrealized = pos.unrealized_pnl(price)
    if unrealized <= -margin:
        logger.warning(f"💥 触发爆仓: {symbol} @ ${price:.4f}, 浮亏: ${unrealized:.4f}, 保证金: ${margin:.4f}")
        trading_simulator.close_position(price, "liquidation")
        logger.info(trading_simulator.get_account_summary())
        await price_monitor.unsubscribe()
        return

    # 每 10 秒打印一次价格日志（避免刷屏）
    now = datetime.now()
    if _last_price_log_time is None or (now - _last_price_log_time).total_seconds() >= 10:
        _last_price_log_time = now
        pnl = pos.unrealized_pnl(price)
        pnl_pct = pos.unrealized_pnl_pct(price)
        # 计算爆仓价
        if pos.side == PositionSide.LONG:
            liq_price = pos.entry_price * (1 - 1 / pos.leverage)
        else:
            liq_price = pos.entry_price * (1 + 1 / pos.leverage)
        logger.info(
            f"📊 实时监控: {symbol} ({pos.side.value}) - "
            f"价格: ${price:.4f}, "
            f"浮盈: ${pnl:+.2f} ({pnl_pct:+.2f}%), "
            f"爆仓价: ${liq_price:.6f}"
        )


def is_5min_chart_time() -> bool:
    """
    检查当前是否是 5 分钟周期后 1 秒（生成图表时间）

    时间: 0:00:01, 0:05:01, 0:10:01, 0:15:01, ...
    """
    now = datetime.now()
    return now.minute % 5 == 0 and 1 <= now.second <= 5


def get_seconds_until_next_5min_chart() -> int:
    """计算距离下一个 5 分钟图表生成时间的秒数"""
    now = datetime.now()
    current_min = now.minute
    current_sec = now.second

    # 当前 5 分钟周期的开始分钟
    current_5min_start = (current_min // 5) * 5

    # 如果当前秒数 < 1，等待到当前周期的第 1 秒
    if current_min % 5 == 0 and current_sec < 1:
        return 1 - current_sec

    # 否则等待到下一个 5 分钟周期的第 1 秒
    next_5min = current_5min_start + 5
    if next_5min >= 60:
        target = now.replace(minute=0, second=1, microsecond=0) + timedelta(hours=1)
    else:
        target = now.replace(minute=next_5min, second=1, microsecond=0)

    delta = (target - now).total_seconds()
    return max(1, int(delta))


# 当前选中的币种
_current_selected_symbol: Optional[str] = None
_startup_done: bool = False
_last_chart_time: Optional[datetime] = None


async def trading_loop() -> None:
    """
    主交易循环

    选币与分析：
    - 启动时：无仓位则从云端选币并分析；有仓位则只分析持仓币种（一般启动时无仓）
    - 之后每 5 分钟：
      有仓位 → 只对持仓币种生成图表并 AI 分析（平/持/调仓/反向）
      无仓位 → 先云端选币，再对选中币种生成图表并 AI 分析（找开仓机会）

    WebSocket：
    - 有仓位时一定订阅持仓币种，用于止盈止损与爆仓检查；无仓位时取消订阅
    """
    global _current_selected_symbol, _startup_done, _last_chart_time

    # 首次启动等待
    await asyncio.sleep(3)

    # 启动时：有仓分析持仓币种，无仓则云端选币并分析
    if not _startup_done:
        if trading_simulator.has_position():
            pos = trading_simulator.position
            trading_simulator.current_symbol = pos.symbol
            logger.info("🚀 系统启动，当前有持仓，分析持仓币种...")
            await do_chart_and_analyze(pos.symbol)
        else:
            logger.info("🚀 系统启动，执行首次选币...")
            await do_select_coin()
        _startup_done = True
        trading_simulator.is_running = True

    while True:
        try:
            now = datetime.now()

            # 有持仓时：WebSocket 一定订阅持仓币种（用于止盈止损）；图表与 AI 只分析持仓币种
            if trading_simulator.has_position():
                pos = trading_simulator.position
                if price_monitor.current_symbol != pos.symbol:
                    await price_monitor.subscribe(pos.symbol, on_price_update)
            else:
                if price_monitor.current_symbol:
                    await price_monitor.unsubscribe()

            # 每 5 分钟：有仓分析持仓币种，无仓先选币再分析云端币种
            if is_5min_chart_time():
                if _last_chart_time is None or (now - _last_chart_time).total_seconds() > 60:
                    _last_chart_time = now
                    if trading_simulator.has_position():
                        pos = trading_simulator.position
                        trading_simulator.current_symbol = pos.symbol
                        await do_chart_and_analyze(pos.symbol)
                    else:
                        await do_select_coin()
                    await asyncio.sleep(30)
                    continue

            # 计算下一个事件的等待时间
            wait_5m = get_seconds_until_next_5min_chart()
            wait_seconds = min(wait_5m, 30)
            await asyncio.sleep(wait_seconds)

        except asyncio.CancelledError:
            logger.info("交易循环被取消")
            break
        except Exception as e:
            logger.error(f"❌ 交易循环异常: {e}", exc_info=True)
            await asyncio.sleep(30)


async def do_select_coin() -> None:
    """从云端获取选币结果"""
    global _current_selected_symbol, _last_chart_time

    # 固定币种模式：跳过云端选币，直接使用指定币种
    FIXED_SYMBOL = "ETHUSDT"
    if FIXED_SYMBOL:
        _current_selected_symbol = FIXED_SYMBOL
        trading_simulator.current_symbol = FIXED_SYMBOL
        logger.info(f"📌 固定币种模式: {FIXED_SYMBOL}")
        _last_chart_time = datetime.now()
        await do_chart_and_analyze(FIXED_SYMBOL)
        return

    logger.info("🔍 从云端获取选币结果...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(CLOUD_COIN_SELECT_URL)
            response.raise_for_status()
            data = response.json()

        # 失败格式: {"detail": "选币服务暂不可用: ..."}
        if "detail" in data:
            logger.warning(f"⚠️ 云端选币不可用: {data['detail']}")
            if _current_selected_symbol:
                logger.info(f"📊 使用上一轮币种继续分析: {_current_selected_symbol}")
                _last_chart_time = datetime.now()
                await do_chart_and_analyze(_current_selected_symbol)
            return

        symbol = data.get("symbol")
        if not symbol:
            logger.warning("⚠️ 云端未返回有效币种")
            if _current_selected_symbol:
                logger.info(f"📊 使用上一轮币种继续分析: {_current_selected_symbol}")
                _last_chart_time = datetime.now()
                await do_chart_and_analyze(_current_selected_symbol)
            return

        _current_selected_symbol = symbol
        trading_simulator.current_symbol = symbol

        score = data.get("score", 0)
        price = data.get("price", 0)
        change_24h = data.get("change_24h", 0)

        logger.info(
            f"🏆 云端选中币种: {_current_selected_symbol} - "
            f"综合评分: {score:.1f}, "
            f"价格: ${price:.4f}, "
            f"24h涨幅: {change_24h:+.2f}%"
        )

        # 获取到币种后立即生成图表并分析
        _last_chart_time = datetime.now()
        await do_chart_and_analyze(_current_selected_symbol)

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ 云端接口返回错误: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"❌ 云端接口请求失败: {e}")
    except Exception as e:
        logger.error(f"❌ 获取云端选币结果失败: {e}", exc_info=True)


async def do_chart_and_analyze(symbol: str) -> None:
    """生成图表并调用 AI 分析"""
    logger.info(f"📊 生成 {symbol} 图表并分析...")

    # 先获取当前价格，确保图表和 AI 分析使用同一价格
    current_price = await get_current_price(symbol)
    if current_price <= 0:
        logger.warning(f"⚠️ 无法获取 {symbol} 当前价格")
        return

    # 生成图表（传入实时价格，确保两张图显示一致）
    loop = asyncio.get_event_loop()
    chart_result = await loop.run_in_executor(
        None,
        chart_generator.generate_charts,
        symbol,
        current_price
    )

    if not chart_result.chart_5m or not chart_result.chart_15m:
        logger.warning(
            f"⚠️ {symbol} 图表生成失败 - "
            f"5m: {'✅' if chart_result.chart_5m else '❌'}, "
            f"15m: {'✅' if chart_result.chart_15m else '❌'} "
            f"(可能是新币种，K线数据不足)"
        )
        return

    logger.info(f"📊 图表生成完成: {chart_result.chart_5m}, {chart_result.chart_15m}")

    # 构建持仓信息（传给 AI 做决策参考）；浮盈必须用持仓币种价格
    position_info = None
    if trading_simulator.has_position():
        pos = trading_simulator.position
        pos_price = current_price if symbol == pos.symbol else await get_current_price(pos.symbol)
        if pos_price <= 0 and price_monitor.current_symbol == pos.symbol:
            pos_price = price_monitor.current_price
        if pos_price <= 0:
            pos_price = current_price
        position_info = {
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "position_size_usd": pos.position_size_usd,
            "leverage": pos.leverage,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "unrealized_pnl": pos.unrealized_pnl(pos_price),
            "unrealized_pnl_pct": pos.unrealized_pnl_pct(pos_price)
        }

    # 并行拉取订单簿与近期成交，生成市场微观上下文供 AI 参考
    orderbook: Optional[dict] = None
    agg_trades: Optional[list] = None
    try:
        orderbook, agg_trades = await asyncio.gather(
            binance_client.get_orderbook_async(symbol, limit=20),
            binance_client.get_agg_trades_async(symbol, limit=100),
        )
    except Exception as e:
        logger.debug(f"拉取订单簿/成交失败: {e}")
    market_context = build_market_context(symbol, current_price, orderbook, agg_trades)

    # 调用 AI 分析（使用同一个 current_price，并注入市场微观数据和指标数值）
    indicator_data = {
        "5m": chart_result.indicators_5m,
        "15m": chart_result.indicators_15m,
    }
    signal = await ai_analyzer.analyze_charts(
        symbol=symbol,
        chart_5m_path=chart_result.chart_5m,
        chart_15m_path=chart_result.chart_15m,
        current_price=current_price,
        position_info=position_info,
        market_context=market_context,
        indicator_data=indicator_data,
    )

    if not signal:
        logger.warning("⚠️ AI 分析未返回信号")
        return

    # 执行交易信号
    if signal.action == SignalAction.WAIT:
        logger.info(f"⏸️ AI 建议观望")
    elif signal.action == SignalAction.HOLD:
        logger.info(f"✊ AI 建议继续持有: {signal.reasoning}")
    elif signal.action == SignalAction.CLOSE_POSITION:
        if trading_simulator.has_position():
            pos = trading_simulator.position
            # 平仓必须使用持仓币种的实时价，避免 4H 选币后分析的是新币种导致用错价格
            close_price = await get_current_price(pos.symbol)
            if close_price <= 0 and price_monitor.current_symbol == pos.symbol:
                close_price = price_monitor.current_price
            if close_price <= 0:
                close_price = current_price if symbol == pos.symbol else 0.0
                if close_price <= 0:
                    logger.warning(f"⚠️ 无法获取 {pos.symbol} 平仓价格，跳过本次平仓")
                    return
            logger.info(f"📤 AI 建议主动平仓: {signal.reasoning}")
            await price_monitor.unsubscribe()
            trading_simulator.close_position(close_price, "ai_close")
        else:
            logger.info("⚠️ 无持仓，忽略 close_position 信号")
    elif signal.action == SignalAction.ADJUST_STOPS:
        if trading_simulator.has_position():
            trading_simulator.execute_signal(signal, current_price)
        else:
            logger.info("⚠️ 无持仓，忽略 adjust_stops 信号")
    elif signal.action in [SignalAction.OPEN_LONG, SignalAction.OPEN_SHORT]:
        # 置信度分级杠杆：≥90 用最大，≥85 用 70%，≥80 用 50%，<80 用 40%（均去小数）
        max_leverage = await binance_client.get_max_leverage_async(symbol)
        confidence = signal.confidence
        if confidence >= 90:
            signal.leverage = max_leverage  # 直接用最大杠杆
        elif confidence >= 85:
            signal.leverage = max(1, int(max_leverage * 0.70))
        elif confidence >= 80:
            signal.leverage = max(1, int(max_leverage * 0.50))
        else:
            signal.leverage = max(1, int(max_leverage * 0.40))
        logger.info(f"📊 {symbol} 最大杠杆: {max_leverage}x, 置信度: {confidence}%, 使用: {signal.leverage}x")

        # 从5m指标快照提取ATR赋值给signal
        atr_5m = chart_result.indicators_5m.get("atr", 0.0)
        if atr_5m:
            signal.atr = atr_5m
            logger.info(f"📊 ATR(5m): {atr_5m:.6f}")

        if trading_simulator.has_position():
            # 有持仓时，只处理「同币种反向」：先平仓再开新仓；同币种同向或新币种开仓信号均忽略
            pos = trading_simulator.position
            is_same_symbol = symbol == pos.symbol
            is_reverse = is_same_symbol and (
                (pos.side.value == "long" and signal.action == SignalAction.OPEN_SHORT) or
                (pos.side.value == "short" and signal.action == SignalAction.OPEN_LONG)
            )
            if is_reverse:
                logger.info("🔄 AI 建议反向操作，执行平仓并开新仓")
                # 平仓必须使用持仓币种的实时价，避免用错价格
                close_price = await get_current_price(pos.symbol)
                if close_price <= 0 and price_monitor.current_symbol == pos.symbol:
                    close_price = price_monitor.current_price
                if close_price <= 0:
                    close_price = current_price if is_same_symbol else 0.0
                if close_price <= 0:
                    logger.warning(f"⚠️ 无法获取 {pos.symbol} 平仓价格，使用当前分析价作为 fallback")
                    close_price = current_price
                await price_monitor.unsubscribe()
                trading_simulator.close_position(close_price, "ai_reverse")
                trading_simulator.execute_signal(signal, current_price)
                if trading_simulator.has_position():
                    await price_monitor.subscribe(
                        trading_simulator.position.symbol,
                        on_price_update
                    )
            elif not is_same_symbol:
                logger.info(f"⏭️ 当前持仓 {pos.symbol}，忽略新币种 {symbol} 的开仓信号，保持现有仓位")
        else:
            # 无持仓，直接开仓
            trading_simulator.execute_signal(signal, current_price)
            if trading_simulator.has_position():
                await price_monitor.subscribe(
                    trading_simulator.position.symbol,
                    on_price_update
                )

        logger.info(trading_simulator.get_account_summary())


# 应用生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("   🎯 AI 模拟交易系统 v3.0")
    logger.info("   选币: 云端 (n8n)")
    logger.info("   数据源: Binance Futures")
    logger.info("   AI: Claude API")
    logger.info("=" * 60)

    # 启动后台任务
    trading_task = asyncio.create_task(trading_loop())

    logger.info("🚀 系统启动完成")
    logger.info(f"💰 初始资金: ${trading_simulator.account.initial_balance:,.2f}")
    logger.info(f"📊 手续费: 开仓 0.02% (Maker), 平仓 0.05% (Taker)")

    yield

    # 停止后台任务
    logger.info("🛑 正在停止系统...")
    trading_task.cancel()

    try:
        await trading_task
    except asyncio.CancelledError:
        pass

    # 输出最终账户状态
    logger.info("=" * 60)
    logger.info("📊 最终账户状态:")
    logger.info(trading_simulator.get_account_summary())
    logger.info("=" * 60)
    logger.info("✅ 系统已停止")


# 创建 FastAPI 应用
app = FastAPI(
    title="AI 模拟交易系统",
    description="基于 Claude AI 的模拟交易系统",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(api_router)

# 静态文件服务
os.makedirs("static", exist_ok=True)
os.makedirs("image", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/image", StaticFiles(directory="image"), name="image")


# 首页路由
@app.get("/", include_in_schema=False)
async def index():
    """首页"""
    return FileResponse("static/index.html")





# 主函数
def main() -> None:
    """主函数"""
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
