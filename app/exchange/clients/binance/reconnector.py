"""
WebSocket重连策略

实现智能重连机制：
- 前3次：使用指数退避策略重连
- 3次之后：每隔2分钟重连一次，不再限制次数
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from app.core.constants import (
    WS_RECONNECT_INITIAL_DELAY,
    WS_RECONNECT_MAX_DELAY,
    WS_RECONNECT_BACKOFF_FACTOR,
    WS_MAX_INITIAL_RETRIES,
    WS_PERIODIC_RETRY_INTERVAL_SECONDS,
    WebSocketErrorType,
)

logger = logging.getLogger(__name__)


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


class ExponentialBackoffReconnector:
    """
    智能重连策略

    实现重连策略：
    - 前3次：使用指数退避策略重连
    - 3次之后：每隔2分钟重连一次，不再限制次数
    """

    def __init__(self):
        """初始化重连器"""
        self.initial_delay = WS_RECONNECT_INITIAL_DELAY
        self.max_delay = WS_RECONNECT_MAX_DELAY
        self.backoff_factor = WS_RECONNECT_BACKOFF_FACTOR
        self.max_initial_retries = WS_MAX_INITIAL_RETRIES
        self.periodic_retry_interval = WS_PERIODIC_RETRY_INTERVAL_SECONDS

        self.current_delay = self.initial_delay
        self.retry_count = 0
        self.reconnect_history: List[ReconnectRecord] = []
        self.connection_start_time: Optional[datetime] = None

        logger.info(f"重连器初始化: 初始延迟={self.initial_delay}s, "
                   f"最大延迟={self.max_delay}s, 退避因子={self.backoff_factor}")
        logger.info(f"   前{self.max_initial_retries}次使用指数退避，"
                   f"之后每隔{self.periodic_retry_interval}秒重连一次")

    def calculate_next_delay(self) -> float:
        """
        计算下次重连延迟

        Returns:
            下次重连延迟（秒）
        """
        if self.retry_count < self.max_initial_retries:
            # 前3次：使用指数退避
            delay = min(
                self.initial_delay * (self.backoff_factor ** self.retry_count),
                self.max_delay
            )
        else:
            # 3次之后：固定为2分钟
            delay = self.periodic_retry_interval
        return delay

    def should_retry(self) -> bool:
        """
        检查是否应该继续重试（始终返回True，不再限制次数）

        Returns:
            是否应该重试（始终为True）
        """
        return True  # 不再限制重连次数

    def on_reconnect_attempt(self) -> float:
        """
        记录重连尝试，返回应该等待的延迟

        Returns:
            等待延迟（秒）
        """
        self.retry_count += 1
        self.current_delay = self.calculate_next_delay()

        if self.retry_count <= self.max_initial_retries:
            logger.info(f"重连尝试 {self.retry_count}/{self.max_initial_retries} "
                       f"(指数退避阶段), 延迟: {self.current_delay:.1f}秒")
        else:
            logger.info(f"重连尝试 {self.retry_count} (周期性重连阶段), "
                       f"延迟: {self.current_delay:.1f}秒 (每2分钟)")

        return self.current_delay

    def on_reconnect_success(self):
        """记录重连成功"""
        connection_duration = None
        if self.connection_start_time:
            connection_duration = (datetime.now() - self.connection_start_time).total_seconds()

        record = ReconnectRecord(
            timestamp=datetime.now(),
            attempt_number=self.retry_count,
            success=True,
            error_type=None,
            error_message=None,
            delay_seconds=self.current_delay,
            connection_duration_before_failure=connection_duration
        )

        self._add_history(record)

        # 重置状态
        self.retry_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now()

        logger.info("重连成功！连接已恢复，重置重连计数器")

    def on_reconnect_failure(self, error: Exception):
        """
        记录重连失败

        Args:
            error: 错误对象
        """
        connection_duration = None
        if self.connection_start_time:
            connection_duration = (datetime.now() - self.connection_start_time).total_seconds()

        error_type = self._classify_error(error)

        record = ReconnectRecord(
            timestamp=datetime.now(),
            attempt_number=self.retry_count,
            success=False,
            error_type=error_type.value,
            error_message=str(error),
            delay_seconds=self.current_delay,
            connection_duration_before_failure=connection_duration
        )

        self._add_history(record)

        if self.retry_count <= self.max_initial_retries:
            logger.error(f"重连失败 (尝试 {self.retry_count}/{self.max_initial_retries}): "
                        f"{error_type.value}")
        else:
            logger.error(f"重连失败 (尝试 {self.retry_count}, 周期性重连阶段): "
                        f"{error_type.value}")
        logger.error(f"   错误信息: {str(error)[:200]}")

    def reset(self):
        """重置重连状态"""
        self.retry_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now()
        logger.info("重连器状态已重置")

    def _add_history(self, record: ReconnectRecord):
        """
        添加历史记录（保留最近10次）

        Args:
            record: 重连记录
        """
        self.reconnect_history.append(record)

        # 只保留最近10次记录
        if len(self.reconnect_history) > 10:
            self.reconnect_history = self.reconnect_history[-10:]

    def _classify_error(self, error: Exception) -> WebSocketErrorType:
        """
        分类错误类型

        Args:
            error: 错误对象

        Returns:
            错误类型
        """
        error_str = str(error).lower()

        if "ssl" in error_str or "decryption" in error_str or "bad record mac" in error_str:
            return WebSocketErrorType.SSL_ERROR
        elif "timeout" in error_str:
            return WebSocketErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "network" in error_str:
            return WebSocketErrorType.NETWORK_ERROR
        elif "protocol" in error_str:
            return WebSocketErrorType.PROTOCOL_ERROR
        else:
            return WebSocketErrorType.UNKNOWN_ERROR

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取重连统计信息

        Returns:
            统计信息字典
        """
        if not self.reconnect_history:
            return {
                'total_attempts': 0,
                'success_count': 0,
                'failure_count': 0,
                'success_rate': 0.0,
                'avg_delay': 0.0,
                'current_retry_count': self.retry_count,
                'current_delay': self.current_delay
            }

        success_count = sum(1 for r in self.reconnect_history if r.success)
        failure_count = len(self.reconnect_history) - success_count

        # 按错误类型统计
        error_types = {}
        for record in self.reconnect_history:
            if not record.success and record.error_type:
                error_types[record.error_type] = error_types.get(record.error_type, 0) + 1

        return {
            'total_attempts': len(self.reconnect_history),
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': success_count / len(self.reconnect_history) if self.reconnect_history else 0.0,
            'avg_delay': sum(r.delay_seconds for r in self.reconnect_history) / len(self.reconnect_history),
            'current_retry_count': self.retry_count,
            'current_delay': self.current_delay,
            'error_types': error_types,
            'recent_history': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'attempt': r.attempt_number,
                    'success': r.success,
                    'error_type': r.error_type,
                    'delay': r.delay_seconds
                }
                for r in self.reconnect_history[-5:]  # 最近5次
            ]
        }
