"""
交易所客户端工厂

使用工厂模式集中管理不同交易所客户端的创建和生命周期
"""
import logging
from typing import Dict, Optional, Any
from enum import Enum

from app.exchange.base_exchange_client import BaseExchangeClient

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """支持的交易所类型"""
    BINANCE = "BINANCE"
    OKX = "OKX"
    MOCK = "MOCK"


class ExchangeFactory:
    """
    交易所客户端工厂
    
    使用单例模式管理交易所客户端实例，确保每个交易所类型只有一个实例
    """
    
    _instances: Dict[ExchangeType, BaseExchangeClient] = {}
    
    @classmethod
    def create_client(
        cls,
        exchange_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseExchangeClient:
        """
        创建交易所客户端实例（单例模式）
        
        Args:
            exchange_type: 交易所类型（BINANCE, OKX, MOCK）
            config: 可选的配置参数
        
        Returns:
            交易所客户端实例
        
        Raises:
            ValueError: 不支持的交易所类型
        """
        try:
            exchange_enum = ExchangeType(exchange_type.upper())
        except ValueError:
            logger.error(f"❌ 不支持的交易所类型: {exchange_type}")
            logger.error(f"   支持的类型: {[e.value for e in ExchangeType]}")
            raise ValueError(f"Unsupported exchange type: {exchange_type}")
        
        # 单例模式：如果实例已存在，直接返回
        if exchange_enum in cls._instances:
            logger.info(f"✅ 返回已存在的{exchange_type}客户端实例")
            return cls._instances[exchange_enum]
        
        # 创建新实例
        logger.info(f"🔧 创建新的{exchange_type}客户端实例...")
        
        try:
            if exchange_enum == ExchangeType.BINANCE:
                from app.exchange.binance_client import BinanceClient
                client = BinanceClient()
                logger.info(f"✅ Binance客户端创建成功")
            elif exchange_enum == ExchangeType.OKX:
                from app.exchange.okx_client import OKXClient
                client = OKXClient(config)
                logger.info(f"✅ OKX客户端创建成功")
            elif exchange_enum == ExchangeType.MOCK:
                from app.exchange.mock_client import MockExchangeClient
                client = MockExchangeClient(config)
                logger.info(f"✅ Mock客户端创建成功")
            else:
                raise ValueError(f"Unsupported exchange type: {exchange_type}")
            
            cls._instances[exchange_enum] = client
            return client
            
        except ImportError as e:
            logger.error(f"❌ 导入{exchange_type}客户端失败: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 创建{exchange_type}客户端失败: {e}")
            raise
    
    @classmethod
    def get_current_client(cls) -> BaseExchangeClient:
        """
        获取当前配置的交易所客户端
        
        Returns:
            当前交易所客户端实例
        
        Raises:
            ValueError: 配置的交易所类型不支持
        """
        from app.core.config import settings
        
        exchange_type = settings.EXCHANGE_TYPE
        logger.info(f"📋 从配置读取交易所类型: {exchange_type}")
        
        return cls.create_client(exchange_type)
    
    @classmethod
    def reset(cls):
        """
        重置所有客户端实例
        
        主要用于测试，清除所有缓存的客户端实例
        """
        cls._instances.clear()
        logger.info("🔄 所有交易所客户端实例已重置")
    
    @classmethod
    def get_instance_count(cls) -> int:
        """
        获取当前缓存的客户端实例数量
        
        Returns:
            实例数量
        """
        return len(cls._instances)
    
    @classmethod
    def has_instance(cls, exchange_type: str) -> bool:
        """
        检查指定交易所类型的实例是否已存在
        
        Args:
            exchange_type: 交易所类型
        
        Returns:
            实例是否存在
        """
        try:
            exchange_enum = ExchangeType(exchange_type.upper())
            return exchange_enum in cls._instances
        except ValueError:
            return False
