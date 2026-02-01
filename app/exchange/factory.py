"""
交易所工厂

提供统一的交易所客户端创建接口，支持多交易所扩展。
"""
import logging
from typing import Optional, Dict, Any, Type

from app.core.constants import Exchange
from app.exchange.base.client import BaseExchangeClient
from app.exchange.base.websocket import BaseWebSocketClient

logger = logging.getLogger(__name__)


class ExchangeFactory:
    """
    交易所工厂类

    用于创建和管理交易所客户端实例。
    """

    # 注册的交易所客户端类
    _rest_clients: Dict[Exchange, Type[BaseExchangeClient]] = {}
    _ws_clients: Dict[Exchange, Type[BaseWebSocketClient]] = {}

    # 单例实例缓存
    _rest_instances: Dict[Exchange, BaseExchangeClient] = {}
    _ws_instances: Dict[Exchange, BaseWebSocketClient] = {}

    @classmethod
    def register_rest_client(cls, exchange: Exchange, client_class: Type[BaseExchangeClient]):
        """
        注册REST客户端类

        Args:
            exchange: 交易所标识
            client_class: 客户端类
        """
        cls._rest_clients[exchange] = client_class
        logger.info(f"注册REST客户端: {exchange.value} -> {client_class.__name__}")

    @classmethod
    def register_ws_client(cls, exchange: Exchange, client_class: Type[BaseWebSocketClient]):
        """
        注册WebSocket客户端类

        Args:
            exchange: 交易所标识
            client_class: 客户端类
        """
        cls._ws_clients[exchange] = client_class
        logger.info(f"注册WebSocket客户端: {exchange.value} -> {client_class.__name__}")

    @classmethod
    def get_rest_client(
        cls,
        exchange: Exchange = Exchange.BINANCE,
        config: Optional[Dict[str, Any]] = None,
        use_singleton: bool = True
    ) -> BaseExchangeClient:
        """
        获取REST客户端实例

        Args:
            exchange: 交易所标识
            config: 配置参数
            use_singleton: 是否使用单例模式

        Returns:
            交易所REST客户端实例

        Raises:
            ValueError: 交易所未注册
        """
        if exchange not in cls._rest_clients:
            # 尝试自动注册Binance
            if exchange == Exchange.BINANCE:
                cls._auto_register_binance()
            else:
                raise ValueError(f"交易所 {exchange.value} 未注册REST客户端")

        if use_singleton:
            if exchange not in cls._rest_instances:
                cls._rest_instances[exchange] = cls._rest_clients[exchange](config)
            return cls._rest_instances[exchange]

        return cls._rest_clients[exchange](config)

    @classmethod
    def get_ws_client(
        cls,
        exchange: Exchange = Exchange.BINANCE,
        use_singleton: bool = True
    ) -> BaseWebSocketClient:
        """
        获取WebSocket客户端实例

        Args:
            exchange: 交易所标识
            use_singleton: 是否使用单例模式

        Returns:
            交易所WebSocket客户端实例

        Raises:
            ValueError: 交易所未注册
        """
        if exchange not in cls._ws_clients:
            # 尝试自动注册Binance
            if exchange == Exchange.BINANCE:
                cls._auto_register_binance()
            else:
                raise ValueError(f"交易所 {exchange.value} 未注册WebSocket客户端")

        if use_singleton:
            if exchange not in cls._ws_instances:
                cls._ws_instances[exchange] = cls._ws_clients[exchange]()
            return cls._ws_instances[exchange]

        return cls._ws_clients[exchange]()

    @classmethod
    def _auto_register_binance(cls):
        """自动注册Binance客户端"""
        try:
            from app.exchange.clients.binance.binance_client import (
                BinanceClient,
                BinanceWebSocketClient
            )
            cls.register_rest_client(Exchange.BINANCE, BinanceClient)
            cls.register_ws_client(Exchange.BINANCE, BinanceWebSocketClient)
        except ImportError as e:
            logger.error(f"自动注册Binance客户端失败: {e}")
            raise

    @classmethod
    def clear_instances(cls):
        """清除所有单例实例"""
        cls._rest_instances.clear()
        cls._ws_instances.clear()
        logger.info("已清除所有交易所客户端实例")

    @classmethod
    def get_supported_exchanges(cls) -> list:
        """
        获取支持的交易所列表

        Returns:
            交易所列表
        """
        return list(Exchange)


# 便捷函数
def get_exchange_client(
    exchange: Exchange = Exchange.BINANCE,
    config: Optional[Dict[str, Any]] = None
) -> BaseExchangeClient:
    """
    获取交易所REST客户端

    Args:
        exchange: 交易所标识
        config: 配置参数

    Returns:
        交易所客户端实例
    """
    return ExchangeFactory.get_rest_client(exchange, config)


def get_ws_client(exchange: Exchange = Exchange.BINANCE) -> BaseWebSocketClient:
    """
    获取交易所WebSocket客户端

    Args:
        exchange: 交易所标识

    Returns:
        WebSocket客户端实例
    """
    return ExchangeFactory.get_ws_client(exchange)
