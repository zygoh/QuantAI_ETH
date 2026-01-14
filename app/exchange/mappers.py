"""
交易所数据格式映射器

提供交易对符号和K线周期在不同交易所之间的格式转换
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SymbolMapper:
    """交易对格式转换器（动态转换，无需硬编码）"""
    
    @classmethod
    def to_exchange_format(cls, symbol: str, exchange_type: str = "BINANCE") -> str:
        """
        将标准格式转换为交易所格式（信号系统：固定使用Binance）
        
        注意：exchange_type参数保留以保持API兼容性，但信号系统固定使用Binance
        """
        # 信号系统：固定使用Binance格式
        return symbol.replace("/", "")
    
    @classmethod
    def to_standard_format(cls, symbol: str, exchange_type: str = "BINANCE") -> str:
        """
        将交易所格式转换为标准格式（信号系统：固定使用Binance）
        
        注意：exchange_type参数保留以保持API兼容性，但信号系统固定使用Binance
        
        Examples:
            - "BTCUSDT" -> "BTC/USDT"
            - "BTC/USDT" -> "BTC/USDT" (已经是标准格式，保持不变)
        """
        # 🔥 修复：如果已经是标准格式（包含 /），直接返回
        if "/" in symbol:
            return symbol
        
        # 信号系统：固定使用Binance格式转换
        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}/USDT"
        return symbol


class IntervalMapper:
    """
    K线周期格式转换器
    
    不同交易所使用不同的K线周期格式：
    - Binance: 1m, 3m, 5m, 15m, 1h, 4h, 1d
    - OKX: 1m, 3m, 5m, 15m, 1H, 4H, 1D
    """
    
    # 标准格式 -> Binance格式
    BINANCE_INTERVALS: Dict[str, str] = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M"
    }
    
    # 标准格式 -> OKX格式
    OKX_INTERVALS: Dict[str, str] = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "8h": "8H",
        "12h": "12H",
        "1d": "1D",
        "3d": "3D",
        "1w": "1W",
        "1M": "1M"
    }
    
    @classmethod
    def to_exchange_format(cls, interval: str, exchange_type: str) -> str:
        """
        将标准格式转换为交易所格式
        
        Args:
            interval: 标准格式周期（如5m, 1h）
            exchange_type: 交易所类型（BINANCE, OKX）
        
        Returns:
            交易所格式的周期
        """
        exchange_type = exchange_type.upper()
        
        if exchange_type == "BINANCE":
            return cls.BINANCE_INTERVALS.get(interval, interval)
        elif exchange_type == "OKX":
            return cls.OKX_INTERVALS.get(interval, interval)
        else:
            logger.warning(f"⚠️ 未知的交易所类型: {exchange_type}，返回原始周期")
            return interval
    
    @classmethod
    def to_standard_format(cls, interval: str, exchange_type: str = "BINANCE") -> str:
        """
        将交易所格式转换为标准格式
        
        Args:
            interval: 交易所格式周期
            exchange_type: 交易所类型（BINANCE, OKX）
        
        Returns:
            标准格式的周期
        """
        exchange_type = exchange_type.upper()
        
        if exchange_type == "BINANCE":
            # Binance格式与标准格式相同
            return interval
        elif exchange_type == "OKX":
            # 反向查找映射表
            for std, exch in cls.OKX_INTERVALS.items():
                if exch == interval:
                    return std
        
        logger.warning(f"⚠️ 无法转换周期格式: {interval} ({exchange_type})")
        return interval
