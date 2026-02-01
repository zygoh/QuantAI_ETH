"""
30天百倍剥头皮交易系统 - 主应用入口
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.api.routes import api_router
from app.api.middleware import LoggingMiddleware, ErrorHandlingMiddleware
from app.scalping.scalping_engine import scalping_engine
from app.scalping.config import scalping_config

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, settings.LOG_FILE)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 文件处理器（按日期分割）
file_handler = TimedRotatingFileHandler(
    log_file,
    when='midnight',
    interval=1,
    backupCount=30,
    encoding='utf-8',
    utc=False
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))

# 配置根日志器
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

if root_logger.hasHandlers():
    root_logger.handlers.clear()

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_file}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 70)
    logger.info("启动30天百倍剥头皮交易系统...")
    logger.info("=" * 70)

    try:
        # 启动剥头皮交易引擎
        await scalping_engine.start(scalping_config.initial_balance)
        logger.info(f"初始资金: {scalping_config.initial_balance}U")
        logger.info(f"目标资金: {scalping_config.target_balance}U")
        logger.info(f"自动扫描币种: {'开启' if scalping_config.auto_scan_symbols else '关闭'}")
        logger.info("系统启动完成")

        yield

    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise
    finally:
        logger.info("正在关闭系统...")

        if scalping_engine and scalping_engine.is_running:
            await scalping_engine.stop()
            logger.info("剥头皮交易引擎已停止")

        logger.info("系统关闭完成")


# 创建FastAPI应用
app = FastAPI(
    title="30天百倍剥头皮交易系统",
    description="高频剥头皮 + 复利滚仓",
    version="1.0.0",
    lifespan=lifespan
)

# 添加中间件（注意顺序：先错误处理，再日志记录）
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "scalping_engine": scalping_engine.is_running if scalping_engine else False,
        "scalping_status": scalping_engine.get_status() if scalping_engine and scalping_engine.is_running else None
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )
