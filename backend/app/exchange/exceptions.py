"""
交易所异常类定义

定义统一的异常层次结构，用于处理不同交易所的错误
"""


class ExchangeError(Exception):
    """
    交易所错误基类
    
    所有交易所相关的异常都应该继承此类
    """
    pass


class ExchangeConnectionError(ExchangeError):
    """
    连接错误
    
    当无法连接到交易所API时抛出
    """
    pass


class ExchangeAPIError(ExchangeError):
    """
    API调用错误
    
    当API返回错误响应时抛出
    
    Attributes:
        code: 错误代码
        message: 错误消息
    """
    
    def __init__(self, code: str, message: str):
        """
        初始化API错误
        
        Args:
            code: 错误代码
            message: 错误消息
        """
        self.code = code
        self.message = message
        super().__init__(f"API Error {code}: {message}")


class ExchangeRateLimitError(ExchangeError):
    """
    限流错误
    
    当触发API限流时抛出
    """
    pass


class ExchangeAuthError(ExchangeError):
    """
    认证错误
    
    当API密钥无效或权限不足时抛出
    """
    pass


class ExchangeInvalidParameterError(ExchangeError):
    """
    参数错误
    
    当传入的参数无效时抛出
    """
    pass


class ExchangeOrderError(ExchangeError):
    """
    订单错误
    
    当订单操作失败时抛出
    """
    pass


class ExchangeInsufficientBalanceError(ExchangeError):
    """
    余额不足错误
    
    当账户余额不足以执行操作时抛出
    """
    pass


class ExchangeTimeoutError(ExchangeError):
    """
    超时错误
    
    当API请求超时时抛出
    """
    pass


class ExchangeWebSocketError(ExchangeError):
    """
    WebSocket错误
    
    当WebSocket连接或通信出现问题时抛出
    """
    pass
