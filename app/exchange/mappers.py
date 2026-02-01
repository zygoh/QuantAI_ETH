"""
交易所数据格式映射器

提供交易对符号格式转换（Binance专用）
"""
import logging

logger = logging.getLogger(__name__)


class SymbolMapper:
    """交易对格式转换器（Binance专用）"""

    @classmethod
    def to_exchange_format(cls, symbol: str, exchange_type: str = "BINANCE") -> str:
        """
        将标准格式转换为Binance格式

        Examples:
            - "BTC/USDT" -> "BTCUSDT"
        """
        return symbol.replace("/", "")

    @classmethod
    def to_standard_format(cls, symbol: str, exchange_type: str = "BINANCE") -> str:
        """
        将Binance格式转换为标准格式

        Examples:
            - "BTCUSDT" -> "BTC/USDT"
            - "BTC/USDT" -> "BTC/USDT" (已经是标准格式，保持不变)
        """
        if "/" in symbol:
            return symbol

        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}/USDT"
        return symbol
