"""
统一数据类型定义

定义交易所数据的统一格式，便于多交易所支持。
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class UnifiedKlineData:
    """统一K线数据格式"""
    timestamp: int              # 开盘时间（毫秒时间戳）
    open: float                 # 开盘价
    high: float                 # 最高价
    low: float                  # 最低价
    close: float                # 收盘价
    volume: float               # 成交量
    close_time: int = 0         # 收盘时间（毫秒时间戳）
    quote_volume: float = 0.0   # 成交额
    trades: int = 0             # 成交笔数
    taker_buy_base_volume: float = 0.0   # 主动买入成交量
    taker_buy_quote_volume: float = 0.0  # 主动买入成交额

    @property
    def datetime(self) -> datetime:
        """转换为datetime对象"""
        return datetime.fromtimestamp(self.timestamp / 1000)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'close_time': self.close_time,
            'quote_volume': self.quote_volume,
            'trades': self.trades,
            'taker_buy_base_volume': self.taker_buy_base_volume,
            'taker_buy_quote_volume': self.taker_buy_quote_volume,
        }


@dataclass
class UnifiedTickerData:
    """统一行情数据格式"""
    symbol: str                 # 交易对
    price: float                # 最新价格
    timestamp: int              # 时间戳（毫秒）

    @property
    def datetime(self) -> datetime:
        """转换为datetime对象"""
        return datetime.fromtimestamp(self.timestamp / 1000)


@dataclass
class OrderBookLevel:
    """订单簿价格档位"""
    price: float                # 价格
    quantity: float             # 数量

    @property
    def value(self) -> float:
        """档位价值"""
        return self.price * self.quantity


@dataclass
class UnifiedOrderBook:
    """统一订单簿数据"""
    symbol: str                 # 交易对
    timestamp: int              # 时间戳（毫秒）
    bids: List[OrderBookLevel] = field(default_factory=list)  # 买单（价格降序）
    asks: List[OrderBookLevel] = field(default_factory=list)  # 卖单（价格升序）

    @property
    def best_bid(self) -> Optional[float]:
        """最优买价"""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """最优卖价"""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """中间价"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """买卖价差"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """买卖价差百分比"""
        if self.spread and self.mid_price:
            return self.spread / self.mid_price
        return None

    def get_bid_volume(self, depth: int = 10) -> float:
        """获取买单总量"""
        return sum(level.quantity for level in self.bids[:depth])

    def get_ask_volume(self, depth: int = 10) -> float:
        """获取卖单总量"""
        return sum(level.quantity for level in self.asks[:depth])

    def get_volume_imbalance(self, depth: int = 10) -> float:
        """
        获取买卖量不平衡度

        返回值范围 [-1, 1]
        正值表示买压大，负值表示卖压大
        """
        bid_vol = self.get_bid_volume(depth)
        ask_vol = self.get_ask_volume(depth)
        total = bid_vol + ask_vol
        if total == 0:
            return 0
        return (bid_vol - ask_vol) / total


@dataclass
class UnifiedTrade:
    """统一成交数据"""
    symbol: str                 # 交易对
    timestamp: int              # 时间戳（毫秒）
    price: float                # 成交价格
    quantity: float             # 成交数量
    is_buyer_maker: bool        # True=卖单成交（主动卖），False=买单成交（主动买）

    @property
    def side(self) -> str:
        """成交方向"""
        return "SELL" if self.is_buyer_maker else "BUY"

    @property
    def value(self) -> float:
        """成交价值"""
        return self.price * self.quantity

    @property
    def datetime(self) -> datetime:
        """转换为datetime对象"""
        return datetime.fromtimestamp(self.timestamp / 1000)


@dataclass
class UnifiedPosition:
    """统一持仓数据"""
    symbol: str                 # 交易对
    side: str                   # 持仓方向 (LONG/SHORT)
    quantity: float             # 持仓数量
    entry_price: float          # 入场价格
    unrealized_pnl: float       # 未实现盈亏
    leverage: int               # 杠杆倍数
    margin: float               # 保证金
    liquidation_price: float = 0.0  # 强平价格


@dataclass
class UnifiedOrder:
    """统一订单数据"""
    order_id: str               # 订单ID
    symbol: str                 # 交易对
    side: str                   # 买卖方向 (BUY/SELL)
    order_type: str             # 订单类型 (MARKET/LIMIT)
    quantity: float             # 下单数量
    price: float                # 下单价格（市价单为0）
    status: str                 # 订单状态
    filled_quantity: float = 0.0  # 已成交数量
    avg_price: float = 0.0      # 平均成交价
    create_time: int = 0        # 创建时间
    update_time: int = 0        # 更新时间
