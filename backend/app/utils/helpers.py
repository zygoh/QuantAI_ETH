"""
辅助工具函数
"""
import logging
from typing import Any, Dict, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def format_currency(amount: float, currency: str = "USDT") -> str:
    """格式化货币显示"""
    return f"{amount:.2f} {currency}"

def format_percentage(value: float) -> str:
    """格式化百分比显示"""
    return f"{value:.2f}%"

def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为整数"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def timestamp_to_datetime(timestamp: int) -> datetime:
    """时间戳转换为datetime对象"""
    return datetime.fromtimestamp(timestamp / 1000)

def datetime_to_timestamp(dt: datetime) -> int:
    """datetime对象转换为时间戳"""
    return int(dt.timestamp() * 1000)

def serialize_datetime(obj: Any) -> Any:
    """序列化datetime对象为JSON"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """记录错误日志"""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {}
    }
    logger.error(f"Error occurred: {json.dumps(error_info, default=serialize_datetime)}")

def calculate_pnl_percentage(entry_price: float, current_price: float, side: str) -> float:
    """计算盈亏百分比"""
    if entry_price == 0:
        return 0.0
    
    if side.upper() == 'LONG':
        return ((current_price - entry_price) / entry_price) * 100
    else:  # SHORT
        return ((entry_price - current_price) / entry_price) * 100

def format_signal_type(signal_type: str) -> str:
    """格式化信号类型显示（图标+中文）"""
    signal_map = {
        'LONG': '📈 做多',
        'SHORT': '📉 做空',
        'HOLD': '⏸️ 持有'
    }
    return signal_map.get(signal_type.upper(), signal_type)