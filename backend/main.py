"""
ETH合约中频智能交易系统 - 主应用入口
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.api.routes import api_router
from app.api.middleware import LoggingMiddleware, ErrorHandlingMiddleware
from app.services.data_service import DataService
from app.services.ensemble_ml_service import ensemble_ml_service  # 🆕 使用Stacking集成
from app.services.trading_engine import TradingEngine
from app.services.risk_service import RiskService
from app.services.signal_generator import SignalGenerator
from app.services.trading_controller import TradingController
from app.services.scheduler import TaskScheduler
from app.services.drawdown_monitor import drawdown_monitor
from app.services.health_monitor import health_monitor
from app.core.database import init_database, cleanup_database, close_database

# 配置日志

# 创建logs目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 日志文件路径
log_file = os.path.join(log_dir, "trading_system.log")

# 配置日志格式
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 创建文件处理器（支持日志轮转，单文件最大10MB，保留5个备份）
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global data_service, ml_service, trading_engine, risk_service
    global signal_generator, trading_controller, scheduler
    
    logger.info("启动ETH合约中频智能交易系统...")
    
    try:
        # 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")
        
        # 清理旧数据（必须在启动时完成，避免新旧数据混合）
        await cleanup_database()
        logger.info("数据库清理完成")
        
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
        
        # 设置API端点的服务依赖
        from app.api.endpoints import account, positions, signals, trading, training, performance, system, websocket
        
        account.set_data_service(data_service)
        positions.set_data_service(data_service)
        signals.set_services(signal_generator, ml_service, data_service)
        trading.set_trading_controller(trading_controller)
        training.set_services(ml_service, scheduler)
        performance.set_services(risk_service, trading_controller)
        system.set_services(trading_controller, scheduler)
        websocket.set_services(data_service, signal_generator, trading_controller)
        
        # 启动数据服务
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
        from app.api.endpoints.websocket import start_websocket_tasks, on_signal_generated, on_risk_alert
        await start_websocket_tasks()
        
        # 注册回调函数
        signal_generator.add_signal_callback(on_signal_generated)
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
        
        # 关闭数据库连接
        await close_database()
            
        logger.info("系统关闭完成")

# 创建FastAPI应用
app = FastAPI(
    title="ETH合约中频智能交易系统",
    description="基于LightGBM的ETH合约中频智能交易系统",
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