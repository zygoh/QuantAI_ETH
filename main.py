"""
量化交易系统 - 主应用入口
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.api.routes import api_router
from app.api.middleware import LoggingMiddleware, ErrorHandlingMiddleware
from app.services.data_service import DataService
from app.model.ensemble_ml_service import ensemble_ml_service  # 🆕 使用Stacking集成
from app.trading.trading_engine import TradingEngine
from app.services.risk_service import RiskService
from app.trading.signal_generator import SignalGenerator
from app.trading.trading_controller import TradingController
from app.services.scheduler import TaskScheduler
from app.services.backtest_service import BacktestService
from app.services.drawdown_monitor import drawdown_monitor
from app.services.health_monitor import health_monitor
from app.core.database import init_database, cleanup_database, close_database
from app.core.executor import global_executor  # 🆕 全局线程池
from app.core.gpu_config import log_gpu_config  # 🆕 GPU配置

# 配置日志

# 创建logs目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 日志文件路径（使用配置文件）
log_file = os.path.join(log_dir, settings.LOG_FILE)

# 配置日志格式
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 创建文件处理器（按日期分割，每天午夜轮转，保留30天）
file_handler = TimedRotatingFileHandler(
    log_file,
    when='midnight',  # 每天午夜
    interval=1,       # 间隔1天
    backupCount=30,  # 保留30天
    encoding='utf-8',
    utc=False        # 使用本地时间
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))

# 配置根日志器
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

# 清除已有的 handlers（避免重复添加）
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# 添加 handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_file}")

# 全局服务实例
data_service = None
ml_service = None
trading_engine = None
risk_service = None
signal_generator = None
trading_controller = None
scheduler = None
backtest_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global data_service, ml_service, trading_engine, risk_service
    global signal_generator, trading_controller, scheduler, backtest_service

    logger.info("启动量化交易系统...")

    try:
        # 🎮 记录GPU配置
        if settings.USE_GPU:
            log_gpu_config()
        
        # 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")

        # 🔥 系统启动时清理数据库和缓存（确保干净启动）
        from app.core.database import cleanup_database, postgresql_manager, redis_manager
        from app.core.cache import cache_manager
        from app.trading.position_manager import position_manager
        
        logger.info("=" * 70)
        logger.info("🧹 系统启动清理：清理数据库和缓存...")
        logger.info("=" * 70)
        
        # 1. 清理数据库（所有交易相关表）
        try:
            await cleanup_database()
        except Exception as e:
            logger.error(f"❌ 数据库清理失败: {e}", exc_info=True)
            # 不阻止启动，但记录错误
        
        # 2. 清理Redis缓存
        try:
            cache_patterns = [
                "market_data:*",          # 市场数据
                "prediction:*",           # 模型预测
                "signal:*",               # 交易信号
                "model_metrics:*",        # 模型指标
                "account_info",           # 账户信息
                "position_info",          # 持仓信息
                "risk_metrics",           # 风险指标
                "system_status",          # 系统状态
                "virtual_account:*",      # 虚拟账户
                "recent_trades:*",        # 最近交易
                "system:trading_mode",    # 交易模式
                "warmup:*",               # 🔥 预热阶段数据（信号计数器等）
                "alert:*",                # 🔥 健康监控告警
                "lock:*"                  # 🔥 资源锁
            ]
            
            cleared_count = 0
            for pattern in cache_patterns:
                try:
                    keys = []
                    async for key in cache_manager.redis.client.scan_iter(match=pattern):
                        keys.append(key)
                    if keys:
                        await cache_manager.redis.client.delete(*keys)
                        cleared_count += len(keys)
                        logger.debug(f"   清理缓存模式 {pattern}: {len(keys)}个键")
                except Exception as e:
                    logger.warning(f"   清理缓存模式 {pattern} 失败: {e}")
            
            logger.info(f"✅ Redis缓存清理完成（共清理{cleared_count}个键）")
        except Exception as e:
            logger.error(f"❌ Redis缓存清理失败: {e}", exc_info=True)
            # 不阻止启动，但记录错误
        
        # 3. 重置虚拟账户余额
        try:
            await position_manager.reset_virtual_balance()
            logger.info("✅ 虚拟账户余额已重置")
        except Exception as e:
            logger.error(f"❌ 虚拟账户余额重置失败: {e}", exc_info=True)
        
        # 4. 🔥 重置回测累积余额（内存清理）
        try:
            # 注意：backtest_service 在后面才创建，这里只是记录需要重置
            # 实际重置会在 backtest_service 创建后执行
            logger.info("✅ 回测累积余额将在服务创建后重置")
        except Exception as e:
            logger.error(f"❌ 回测余额重置失败: {e}", exc_info=True)
        
        logger.info("✅ 系统启动清理完成")
        logger.info("=" * 70)

        # 🔥 检查模型是否存在，如果不存在则在初始化WebSocket之前清理数据库
        from app.services.scheduler import TaskScheduler
        
        # 检查模型是否存在（不创建服务实例）
        model_dir = "models"
        has_model = False
        safe_symbol = settings.SYMBOL.replace('/', '_')
        
        if os.path.exists(model_dir):
            for timeframe in settings.TIMEFRAMES:
                required_models = ['lgb', 'xgb', 'cat', 'meta']
                timeframe_has_all_models = True
                
                for model_name in required_models:
                    model_file = os.path.join(model_dir, f"{safe_symbol}_{timeframe}_{model_name}_model.pkl")
                    if not os.path.exists(model_file):
                        timeframe_has_all_models = False
                        break
                
                if timeframe_has_all_models:
                    has_model = True
                    break
        
        # 如果未找到模型，在初始化WebSocket之前清理数据库
        if not has_model:
            logger.warning("⚠️ 未找到已保存的Stacking集成模型文件，需要训练")
            logger.info("🧹 训练前清理：开始清理数据库、Redis缓存（在初始化WebSocket之前）...")
            
            # 清理数据库
            await postgresql_manager.cleanup_old_data(days=0)
            logger.info("✅ 数据库清理完成")
            
            # 清理Redis缓存
            cache_patterns = [
                "market_data:*",
                "prediction:*",
                "signal:*",
                "model_metrics:*"
            ]
            cleared_count = 0
            for pattern in cache_patterns:
                try:
                    keys = []
                    async for key in cache_manager.redis.client.scan_iter(match=pattern):
                        keys.append(key)
                    if keys:
                        await cache_manager.redis.client.delete(*keys)
                        cleared_count += len(keys)
                except Exception as e:
                    logger.warning(f"   清理缓存模式 {pattern} 失败: {e}")
            logger.info(f"✅ Redis缓存清理完成（共清理{cleared_count}个键）")
            logger.info("✅ 训练前清理完成")
        
        # 初始化服务
        data_service = DataService()
        ml_service = ensemble_ml_service  # 🆕 使用Stacking集成ML服务
        trading_engine = TradingEngine(data_service=data_service)  # 🔑 传入data_service
        risk_service = RiskService(data_service)
        signal_generator = SignalGenerator(ml_service, data_service)
        trading_controller = TradingController(
            trading_engine, signal_generator, ml_service, data_service
        )
        scheduler = TaskScheduler(ml_service, data_service, signal_generator)  # 🔥 传入signal_generator
        backtest_service = BacktestService(ml_service)
        
        # 🔥 重置回测累积余额（确保每次启动都是干净状态）
        backtest_service.reset_backtest_balance()
        logger.info("✅ 回测累积余额已重置（内存清理）")

        # 设置API端点的服务依赖（已移除account端点，仅支持模拟交易）
        from app.api.endpoints import positions, signals, trading, training, performance, system, websocket

        positions.set_data_service(data_service)
        signals.set_services(signal_generator, ml_service, data_service)
        trading.set_trading_controller(trading_controller)
        training.set_services(ml_service, scheduler, backtest_service)
        performance.set_services(risk_service, trading_controller)
        system.set_services(trading_controller, scheduler)
        websocket.set_services(data_service, signal_generator, trading_controller)

        # 启动数据服务（此时WebSocket数据缓冲区将被初始化）
        await data_service.start()
        logger.info("数据服务启动完成")

        # 启动机器学习服务
        await ml_service.start()
        logger.info("机器学习服务启动完成")

        # 启动交易引擎
        await trading_engine.start()
        logger.info("交易引擎启动完成")

        # 启动信号生成器
        await signal_generator.start()
        logger.info("信号生成器启动完成")

        # 启动回撤监控
        await drawdown_monitor.start()
        logger.info("回撤监控启动完成")

        # 启动任务调度器
        await scheduler.start()
        logger.info("任务调度器启动完成")

        # 启动健康监控服务（由scheduler在每天00:00执行）
        health_monitor.set_signal_generator(signal_generator)
        await health_monitor.start()
        logger.info("健康监控服务启动完成（检查时间: 每天00:00）")

        # 启动WebSocket推送任务
        from app.api.endpoints.websocket import start_websocket_tasks, on_risk_alert
        await start_websocket_tasks()

        # 🔥 修复：移除重复的回调注册
        # trading_controller已经注册了_on_signal_generated回调，这里不需要再注册
        # on_signal_generated只用于WebSocket推送，不执行交易（避免重复开仓）
        # signal_generator.add_signal_callback(on_signal_generated)  # 已移除，避免重复执行
        
        # 注册回撤监控回调（用于WebSocket推送）
        drawdown_monitor.add_alert_callback(on_risk_alert)

        logger.info("系统启动完成")

        yield

    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise
    finally:
        # 清理资源
        logger.info("正在关闭系统...")

        if health_monitor:
            await health_monitor.stop()
        if scheduler:
            await scheduler.stop()
        if drawdown_monitor:
            await drawdown_monitor.stop()
        if signal_generator:
            await signal_generator.stop()
        if trading_engine:
            await trading_engine.stop()
        if ml_service:
            await ml_service.stop()
        if data_service:
            await data_service.stop()

        # 关闭全局线程池
        global_executor.shutdown(wait=True)

        # 关闭数据库连接
        await close_database()

        logger.info("系统关闭完成")


# 创建FastAPI应用
app = FastAPI(
    title="量化交易系统",
    description="基于机器学习的合约中频智能交易系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(api_router, prefix="/api")

# 静态文件服务（前端构建文件）
try:
    app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static")
except Exception:
    logger.warning("前端静态文件目录不存在，跳过挂载")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "services": {
            "data_service": data_service.is_running if data_service else False,
            "ml_service": ml_service.is_running if ml_service else False,
            "trading_engine": trading_engine.is_running if trading_engine else False,
            "signal_generator": signal_generator.is_running if signal_generator else False,
            "scheduler": scheduler.is_running if scheduler else False,
        }
    }


if __name__ == "__main__":
    # 禁用自动重载（避免日志文件触发频繁重载）
    # 生产环境应该禁用 reload
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # 改为 False，避免日志文件触发重载
        log_level="info"
    )
