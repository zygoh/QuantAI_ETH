"""
系统配置管理
"""
import logging
import os

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """系统配置"""

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # 代理配置
    USE_PROXY: bool = True
    USE_PROXY_WS: bool = True
    PROXY_HOST: str = "127.0.0.1"
    PROXY_PORT: int = 10808
    PROXY_TYPE: str = "socks5"

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "trading_system.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()
