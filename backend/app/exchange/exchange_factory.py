"""
交易所客户端工厂（信号系统：仅使用Binance获取市场数据）

信号系统仅用于数据获取和虚拟交易，不进行实际交易
"""
# StdLib
import logging
from typing import Optional

# Local App
from app.exchange.base_exchange_client import BaseExchangeClient
from app.exchange.clients.binance import BinanceClient

logger = logging.getLogger(__name__)


class ExchangeFactory:
    """
    交易所客户端工厂（信号系统：固定使用Binance）
    
    使用单例模式管理Binance客户端实例
    """
    
    _instance: Optional[BaseExchangeClient] = None
    
    @classmethod
    def get_current_client(cls) -> BaseExchangeClient:
        """
        获取Binance客户端实例（信号系统固定使用Binance）
        
        Returns:
            Binance客户端实例
        """
        if cls._instance is None:
            cls._instance = BinanceClient()
            logger.info("✅ Binance客户端创建成功（信号系统：仅数据获取）")
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """
        重置客户端实例
        
        主要用于测试，清除缓存的客户端实例
        """
        cls._instance = None
        logger.info("🔄 交易所客户端实例已重置")
