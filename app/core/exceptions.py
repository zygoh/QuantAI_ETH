"""
统一异常体系

定义系统中所有自定义异常类，便于错误处理和日志记录。
"""


class ScalpingBaseException(Exception):
    """剥头皮交易系统基础异常"""

    def __init__(self, message: str = "", code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


# ==================== 交易所相关异常 ====================

class ExchangeError(ScalpingBaseException):
    """交易所相关错误"""
    pass


class ConnectionError(ExchangeError):
    """连接错误"""
    pass


class WebSocketError(ExchangeError):
    """WebSocket错误"""
    pass


class APIError(ExchangeError):
    """API调用错误"""

    def __init__(self, message: str = "", code: str = None, status_code: int = None):
        super().__init__(message, code)
        self.status_code = status_code


class RateLimitError(APIError):
    """API限流错误"""
    pass


class AuthenticationError(APIError):
    """认证错误"""
    pass


# ==================== 信号相关异常 ====================

class SignalError(ScalpingBaseException):
    """信号相关错误"""
    pass


class InsufficientDataError(SignalError):
    """数据不足错误"""
    pass


class SignalValidationError(SignalError):
    """信号验证错误"""
    pass


# ==================== 风控相关异常 ====================

class RiskError(ScalpingBaseException):
    """风控相关错误"""
    pass


class PositionLimitError(RiskError):
    """持仓限制错误"""
    pass


class DailyLossLimitError(RiskError):
    """每日亏损限制错误"""
    pass


class CooldownError(RiskError):
    """冷却期错误"""
    pass


# ==================== 交易相关异常 ====================

class TradingError(ScalpingBaseException):
    """交易相关错误"""
    pass


class OrderError(TradingError):
    """订单错误"""
    pass


class InsufficientBalanceError(TradingError):
    """余额不足错误"""
    pass


class PositionNotFoundError(TradingError):
    """持仓不存在错误"""
    pass


# ==================== 配置相关异常 ====================

class ConfigError(ScalpingBaseException):
    """配置相关错误"""
    pass


class InvalidConfigError(ConfigError):
    """无效配置错误"""
    pass


# ==================== 数据相关异常 ====================

class DataError(ScalpingBaseException):
    """数据相关错误"""
    pass


class DataValidationError(DataError):
    """数据验证错误"""
    pass


class DataParseError(DataError):
    """数据解析错误"""
    pass
