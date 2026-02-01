"""
WebSocket客户端抽象基类

定义WebSocket客户端的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class WebSocketState(Enum):
    """WebSocket连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


@dataclass
class ReconnectRecord:
    """重连历史记录"""
    timestamp: datetime
    attempt_number: int
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    delay_seconds: float
    connection_duration_before_failure: Optional[float]


class BaseWebSocketClient(ABC):
    """
    WebSocket客户端抽象基类

    所有交易所WebSocket客户端实现都应继承此类。
    """

    def __init__(self):
        self.is_connected: bool = False
        self.is_running: bool = False
        self.callbacks: Dict[str, Callable] = {}
        self.subscriptions: List[Dict] = []
        self.last_message_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None

    @abstractmethod
    def start_websocket(self):
        """启动WebSocket连接"""
        pass

    @abstractmethod
    def stop_websocket(self):
        """停止WebSocket连接"""
        pass

    @abstractmethod
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """
        订阅K线数据

        Args:
            symbol: 交易对（标准格式）
            interval: K线间隔
            callback: 数据回调函数
        """
        pass

    @abstractmethod
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """
        订阅价格变动数据

        Args:
            symbol: 交易对（标准格式）
            callback: 数据回调函数
        """
        pass

    @abstractmethod
    def subscribe_depth(self, symbol: str, level: int, callback: Callable):
        """
        订阅订单簿深度数据

        Args:
            symbol: 交易对（标准格式）
            level: 深度档位
            callback: 数据回调函数
        """
        pass

    @abstractmethod
    def subscribe_agg_trade(self, symbol: str, callback: Callable):
        """
        订阅聚合成交流数据

        Args:
            symbol: 交易对（标准格式）
            callback: 数据回调函数
        """
        pass

    @abstractmethod
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        获取连接统计信息

        Returns:
            统计信息字典
        """
        pass

    # ==================== 通用方法 ====================

    def add_callback(self, stream_name: str, callback: Callable):
        """
        添加回调函数

        Args:
            stream_name: 流名称
            callback: 回调函数
        """
        self.callbacks[stream_name] = callback

    def remove_callback(self, stream_name: str):
        """
        移除回调函数

        Args:
            stream_name: 流名称
        """
        if stream_name in self.callbacks:
            del self.callbacks[stream_name]

    def get_state(self) -> WebSocketState:
        """
        获取当前连接状态

        Returns:
            连接状态
        """
        if not self.is_running:
            return WebSocketState.DISCONNECTED
        if self.is_connected:
            return WebSocketState.CONNECTED
        return WebSocketState.RECONNECTING

    @property
    def uptime_seconds(self) -> float:
        """获取连接运行时间（秒）"""
        if self.connection_start_time:
            return (datetime.now() - self.connection_start_time).total_seconds()
        return 0.0

    @property
    def idle_seconds(self) -> float:
        """获取空闲时间（秒）"""
        if self.last_message_time:
            return (datetime.now() - self.last_message_time).total_seconds()
        return 0.0
