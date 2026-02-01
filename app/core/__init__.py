"""
核心基础设施模块

包含：
- config: 应用配置
- constants: 全局常量
- exceptions: 统一异常体系
- logging: 日志配置
"""
from app.core.config import settings
from app.core.exceptions import (
    ScalpingBaseException,
    ExchangeError,
    ConnectionError,
    WebSocketError,
    APIError,
    RateLimitError,
    SignalError,
    RiskError,
    TradingError,
    ConfigError,
    DataError,
)

__all__ = [
    'settings',
    'ScalpingBaseException',
    'ExchangeError',
    'ConnectionError',
    'WebSocketError',
    'APIError',
    'RateLimitError',
    'SignalError',
    'RiskError',
    'TradingError',
    'ConfigError',
    'DataError',
]