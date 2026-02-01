"""
交易所客户端抽象基类

定义交易所REST API客户端的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from app.exchange.base.types import (
    UnifiedKlineData,
    UnifiedTickerData,
    UnifiedOrderBook,
    UnifiedPosition,
    UnifiedOrder,
)


class BaseExchangeClient(ABC):
    """
    交易所客户端抽象基类

    所有交易所客户端实现都应继承此类，并实现所有抽象方法。
    """

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        测试API连接

        Returns:
            连接是否成功
        """
        pass

    @abstractmethod
    def get_server_time(self) -> int:
        """
        获取服务器时间

        Returns:
            服务器时间（毫秒时间戳）
        """
        pass

    @abstractmethod
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        获取交易所信息

        Returns:
            交易所信息字典
        """
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取交易对信息

        Args:
            symbol: 交易对（标准格式，如 "BTC/USDT"）

        Returns:
            交易对信息字典，不存在返回None
        """
        pass

    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[UnifiedKlineData]:
        """
        获取K线数据

        Args:
            symbol: 交易对（标准格式）
            interval: K线间隔（如 "1m", "5m", "1h"）
            limit: 获取数量
            start_time: 开始时间（毫秒时间戳）
            end_time: 结束时间（毫秒时间戳）

        Returns:
            K线数据列表
        """
        pass

    @abstractmethod
    def get_ticker_price(self, symbol: str) -> Optional[UnifiedTickerData]:
        """
        获取实时价格

        Args:
            symbol: 交易对（标准格式）

        Returns:
            行情数据，不存在返回None
        """
        pass

    # ==================== 以下为需要认证的接口 ====================
    # 信号系统模式下这些方法返回空值

    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息

        Returns:
            账户信息字典
        """
        return {}

    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取持仓信息

        Args:
            symbol: 交易对（可选，不指定则返回所有持仓）

        Returns:
            持仓信息列表
        """
        return []

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        下单

        Args:
            symbol: 交易对
            side: 买卖方向 (BUY/SELL)
            order_type: 订单类型 (MARKET/LIMIT)
            quantity: 下单数量
            price: 下单价格（限价单必填）
            **kwargs: 其他参数

        Returns:
            订单信息字典
        """
        return {}

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        撤销订单

        Args:
            symbol: 交易对
            order_id: 订单ID

        Returns:
            撤销结果
        """
        return {}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取未成交订单

        Args:
            symbol: 交易对（可选）

        Returns:
            订单列表
        """
        return []

    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        修改杠杆倍数

        Args:
            symbol: 交易对
            leverage: 杠杆倍数

        Returns:
            修改结果
        """
        return {}

    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        修改保证金模式

        Args:
            symbol: 交易对
            margin_type: 保证金模式 (ISOLATED/CROSSED)

        Returns:
            修改结果
        """
        return {}

    # ==================== 工具方法 ====================

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        安全地将值转换为float

        Args:
            value: 要转换的值
            default: 默认值

        Returns:
            转换后的float值
        """
        if value is None or value == '' or value == 'None':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """
        安全地将值转换为int

        Args:
            value: 要转换的值
            default: 默认值

        Returns:
            转换后的int值
        """
        if value is None or value == '' or value == 'None':
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
