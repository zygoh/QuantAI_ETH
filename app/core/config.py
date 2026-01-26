"""
系统配置管理
"""
# StdLib
import logging
import os

# Third-Party
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

# Local App
from app.core.constants import (
    DEFAULT_SYMBOL,
    DEFAULT_LEVERAGE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TIMEFRAMES
)

class Settings(BaseSettings):
    """系统配置"""
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 信号系统：仅使用Binance公共接口获取市场数据（无需API Key）
    # 注意：系统仅用于信号生成和虚拟交易，不进行实际交易
    
    # 代理配置（可选）
    USE_PROXY: bool = True  # 是否使用代理（REST API）
    USE_PROXY_WS: bool = True  # 是否为WebSocket使用代理（SOCKS5更稳定）
    PROXY_HOST: str = "127.0.0.1"  # 代理主机
    PROXY_PORT: int = 10808  # 代理端口（SOCKS5通常10808，HTTP通常10809）
    PROXY_TYPE: str = "socks5"  # 代理类型：http, https, socks5（V2Ray SOCKS5用socks5，HTTP用http）
    
    # 🎯 交易配置（从 constants.py 读取默认值，确保回测和模拟交易一致）
    SYMBOL: str = DEFAULT_SYMBOL  # 默认交易对
    LEVERAGE: int = DEFAULT_LEVERAGE  # 杠杆倍数（回测和模拟交易都使用）
    CONFIDENCE_THRESHOLD: float = DEFAULT_CONFIDENCE_THRESHOLD  # 置信度阈值（回测和模拟交易都使用）
    TIMEFRAMES: list = DEFAULT_TIMEFRAMES  # 多时间框架配置（回测和模拟交易都使用）
    
    # 交易模式配置
    TRADING_MODE: str = "SIGNAL_ONLY"  # 默认交易模式：SIGNAL_ONLY（信号模式/虚拟交易）或 AUTO（自动交易/实盘）
    
    # PostgreSQL + TimescaleDB 配置
    PG_HOST: str = "172.22.22.93"
    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "Kuan12345"
    PG_DATABASE: str = "trading-data"
    PG_POOL_SIZE: int = 20
    PG_MAX_OVERFLOW: int = 40
    
    # Redis配置（缓存）
    REDIS_URL: str = "redis://172.22.22.93:6379"
    REDIS_DB: int = 0
    
    
    # GPU配置
    USE_GPU: bool = True
    GPU_DEVICE: str = "cuda:0"
    
    
    # 日志配置
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "trading_system.log"
    
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def validate_exchange_config(self) -> bool:
        """
        验证交易所配置的完整性（信号系统仅使用Binance）
        
        Returns:
            配置是否有效
        """
        logger = logging.getLogger(__name__)
        
        # 信号系统固定使用Binance公共接口获取市场数据
        logger.info(f"✅ 信号系统配置: 使用Binance公共接口（仅数据获取，无实际交易）")
        return True
    
    def validate_config(self):
        """
        验证配置参数的合理性
        
        Raises:
            ValueError: 配置参数不合法时抛出
        """
        errors = []
        
        # 验证交易所配置
        self.validate_exchange_config()
        
        # 🎯 验证配置一致性（确保与 constants.py 一致）
        logger = logging.getLogger(__name__)
        if self.LEVERAGE != DEFAULT_LEVERAGE:
            logger.warning(f"⚠️ 杠杆倍数配置不一致: config.py={self.LEVERAGE}, constants.py={DEFAULT_LEVERAGE}")
            logger.warning(f"   建议: 使用 constants.py 的默认值 {DEFAULT_LEVERAGE}，确保回测和模拟交易一致")
        
        if self.CONFIDENCE_THRESHOLD != DEFAULT_CONFIDENCE_THRESHOLD:
            logger.warning(f"⚠️ 置信度阈值配置不一致: config.py={self.CONFIDENCE_THRESHOLD}, constants.py={DEFAULT_CONFIDENCE_THRESHOLD}")
            logger.warning(f"   建议: 使用 constants.py 的默认值 {DEFAULT_CONFIDENCE_THRESHOLD}，确保回测和模拟交易一致")
        
        if self.TIMEFRAMES != DEFAULT_TIMEFRAMES:
            logger.warning(f"⚠️ 多时间框架配置不一致: config.py={self.TIMEFRAMES}, constants.py={DEFAULT_TIMEFRAMES}")
            logger.warning(f"   建议: 使用 constants.py 的默认值 {DEFAULT_TIMEFRAMES}，确保回测和模拟交易一致")
        
        # 如果有错误，抛出异常
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

# 创建全局配置实例
settings = Settings()

# 验证配置
try:
    settings.validate_config()
except ValueError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ 配置验证失败: {e}")
    raise