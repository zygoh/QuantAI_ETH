"""
统一交易所客户端接口

定义所有交易所必须实现的抽象基类和统一数据格式
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UnifiedKlineData:
    """
    统一的K线数据格式
    
    Attributes:
        timestamp: 开盘时间（毫秒时间戳）
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        volume: 成交量（基础货币）
        close_time: 收盘时间（毫秒时间戳）
        quote_volume: 成交额（计价货币）
        trades: 成交笔数
        taker_buy_base_volume: 主动买入成交量（基础货币）
        taker_buy_quote_volume: 主动买入成交额（计价货币）
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float


@dataclass
class UnifiedTickerData:
    """
    统一的价格数据格式
    
    Attributes:
        symbol: 交易对符号
        price: 最新价格
        timestamp: 时间戳（毫秒）
    """
    symbol: str
    price: float
    timestamp: int


@dataclass
class UnifiedOrderData:
    """
    统一的订单数据格式
    
    Attributes:
        order_id: 交易所订单ID
        client_order_id: 客户端订单ID
        symbol: 交易对符号
        side: 订单方向（BUY, SELL）
        type: 订单类型（MARKET, LIMIT）
        status: 订单状态（NEW, FILLED, CANCELED等）
        quantity: 订单数量
        price: 订单价格（限价单）
        filled_quantity: 已成交数量
        avg_price: 平均成交价
        commission: 手续费
        created_at: 创建时间（毫秒时间戳）
        updated_at: 更新时间（毫秒时间戳）
    """
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    type: str
    status: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    avg_price: float
    commission: float
    created_at: int
    updated_at: int


class BaseExchangeClient(ABC):
    """
    交易所客户端抽象基类
    
    定义所有交易所客户端必须实现的接口方法
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
            symbol: 交易对符号
        
        Returns:
            交易对信息字典，如果不存在返回None
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
            symbol: 交易对符号
            interval: K线周期（如1m, 5m, 15m）
            limit: 返回数量限制
            start_time: 开始时间（毫秒时间戳，可选）
            end_time: 结束时间（毫秒时间戳，可选）
        
        Returns:
            K线数据列表
        """
        pass
    
    @abstractmethod
    def get_klines_paginated(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        rate_limit_delay: float = 0.1
    ) -> List[UnifiedKlineData]:
        """
        分页获取K线数据（自动处理超过单次请求限制的情况）
        
        Args:
            symbol: 交易对符号
            interval: K线周期
            limit: 需要获取的总数量
            start_time: 开始时间（毫秒时间戳，可选）
            end_time: 结束时间（毫秒时间戳，可选）
            rate_limit_delay: API限流延迟（秒）
        
        Returns:
            K线数据列表
        """
        pass
    
    @abstractmethod
    def get_ticker_price(self, symbol: str) -> Optional[UnifiedTickerData]:
        """
        获取实时价格
        
        Args:
            symbol: 交易对符号
        
        Returns:
            价格数据，如果获取失败返回None
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息
        
        Returns:
            账户信息字典
        """
        pass
    
    @abstractmethod
    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取持仓信息
        
        Args:
            symbol: 交易对符号（可选，不指定则返回所有持仓）
        
        Returns:
            持仓信息列表
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_position: bool = False,
        stop_price: Optional[float] = None,
        callback_rate: Optional[float] = None,
        working_type: str = "MARK_PRICE"
    ) -> Dict[str, Any]:
        """
        下单
        
        Args:
            symbol: 交易对符号
            side: 订单方向（BUY, SELL）
            order_type: 订单类型（MARKET, LIMIT）
            quantity: 订单数量
            price: 订单价格（限价单必需）
            time_in_force: 有效方式（GTC, IOC, FOK）
            reduce_only: 是否只减仓
            close_position: 是否平仓
            stop_price: 触发价格（止损/止盈单）
            callback_rate: 回调比率（追踪止损）
            working_type: 触发价格类型（MARK_PRICE, CONTRACT_PRICE）
        
        Returns:
            订单结果字典
        """
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        取消订单
        
        Args:
            symbol: 交易对符号
            order_id: 订单ID
        
        Returns:
            取消结果字典
        """
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取未成交订单
        
        Args:
            symbol: 交易对符号（可选，不指定则返回所有未成交订单）
        
        Returns:
            未成交订单列表
        """
        pass
    
    @abstractmethod
    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        修改杠杆倍数
        
        Args:
            symbol: 交易对符号
            leverage: 杠杆倍数
        
        Returns:
            修改结果字典
        """
        pass
    
    @abstractmethod
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        修改保证金模式
        
        Args:
            symbol: 交易对符号
            margin_type: 保证金模式（ISOLATED, CROSSED）
        
        Returns:
            修改结果字典
        """
        pass
