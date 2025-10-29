"""
系统配置管理
"""
import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """系统配置"""
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Binance API配置
    BINANCE_API_KEY: str = "pEnCcceHaD72df4o2PkLu8EWeddfVfuUvzJwdSxLVVbZRYv90mEXcd7oj5PCz4rb"
    BINANCE_SECRET_KEY: str = "vnjKNZ6nK48xaSRAwuATq30pWZJUK6JmzqjTqJewL3UK3t2dHva6PB8ckfn1dSGj"
    BINANCE_TESTNET: bool = True  # 使用生产环境
    
    # 交易配置
    SYMBOL: str = "ETHUSDT"
    LEVERAGE: int = 50  # 20x杠杆（1.5%止损 × 20x = 30%单次风险，风险回报1:2.67）
    CONFIDENCE_THRESHOLD: float = 0.35  # 降低到0.35以增加信号数量（81%准确率下合理阈值）
    
    # 交易模式配置
    TRADING_MODE: str = "SIGNAL_ONLY"  # 默认交易模式：SIGNAL_ONLY（信号模式/虚拟交易）或 AUTO（自动交易/实盘）
    
    # 时间框架配置（以5m为主，3m和15m为辅助）
    TIMEFRAMES: list = ["3m", "5m", "15m"]
    
    # PostgreSQL + TimescaleDB 配置
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "Kuan12345"
    PG_DATABASE: str = "trading-data"
    PG_POOL_SIZE: int = 10
    PG_MAX_OVERFLOW: int = 20
    
    # Redis配置（缓存）
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    
    # 机器学习配置
    # 注意：模型训练由scheduler统一管理（每天00:01执行）
    TRAINING_SPLIT: float = 0.8  # 训练集/验证集分割比例
    
    # GPU配置
    USE_GPU: bool = True
    GPU_DEVICE: str = "cuda:0"
    
    # 风险管理配置
    VAR_CONFIDENCE: float = 0.95
    MAX_DRAWDOWN_LIMIT: float = 0.15  # 15%
    KELLY_MULTIPLIER: float = 0.25  # Kelly系数乘数
    
    # 日志配置
    LOG_LEVEL: str = "DEBUG"  # ✅ 临时改为DEBUG查看详细日志
    LOG_FILE: str = "trading_system.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 创建全局配置实例
settings = Settings()