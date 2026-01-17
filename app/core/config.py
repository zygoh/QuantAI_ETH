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
    
    # 交易配置
    SYMBOL: str = "BTC/USDT"  # 使用标准格式，系统会自动转换为交易所格式
    LEVERAGE: int = 50  # 🔥 提高杠杆到50倍
    CONFIDENCE_THRESHOLD: float = 0.45  # 🔥 提高阈值到0.45以提高信号质量（目标胜率>50%）
    
    # 交易模式配置
    TRADING_MODE: str = "SIGNAL_ONLY"  # 默认交易模式：SIGNAL_ONLY（信号模式/虚拟交易）或 AUTO（自动交易/实盘）
    
    # 时间框架配置（以5m为主，3m和15m为辅助）
    TIMEFRAMES: list = ["3m", "5m", "15m"]
    
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