"""
多币种实时数据监控

功能：
- 同时监控多个币种的订单簿深度
- 实时K线数据
- 成交流数据
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from collections import deque

from app.scalping.config import scalping_config, SymbolConfig
from app.exchange.mappers import SymbolMapper

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """订单簿价格档位"""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """订单簿数据"""
    symbol: str
    timestamp: int
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
class TradeData:
    """成交数据"""
    symbol: str
    timestamp: int
    price: float
    quantity: float
    is_buyer_maker: bool  # True=卖单成交（主动卖），False=买单成交（主动买）

    @property
    def side(self) -> str:
        """成交方向"""
        return "SELL" if self.is_buyer_maker else "BUY"


@dataclass
class KlineData:
    """K线数据"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SymbolData:
    """单个币种的实时数据"""
    symbol: str
    orderbook: Optional[OrderBook] = None
    last_price: float = 0.0
    price_change_1m: float = 0.0  # 1分钟价格变化
    volume_1m: float = 0.0        # 1分钟成交量
    trades: deque = field(default_factory=lambda: deque(maxlen=1000))  # 最近成交
    price_history: deque = field(default_factory=lambda: deque(maxlen=60))  # 价格历史（秒级）
    kline_history: deque = field(default_factory=lambda: deque(maxlen=100))  # K线历史
    last_kline: Optional[KlineData] = None  # 最新K线
    last_update: Optional[datetime] = None


class MultiSymbolMonitor:
    """多币种实时监控"""

    def __init__(self):
        self.symbol_data: Dict[str, SymbolData] = {}
        self.ws_client = None
        self.is_running = False
        self.callbacks: List[Callable[[str, SymbolData], None]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self, symbols: Optional[List[SymbolConfig]] = None):
        """启动监控"""
        if self.is_running:
            logger.warning("监控已在运行")
            return

        self._loop = asyncio.get_running_loop()

        # 使用配置中的币种或传入的币种
        if symbols is None:
            symbols = scalping_config.symbols

        # 初始化数据结构
        for sym_config in symbols:
            self.symbol_data[sym_config.symbol] = SymbolData(symbol=sym_config.symbol)

        logger.info(f"🚀 启动多币种监控: {[s.symbol for s in symbols]}")

        # 创建独立的WebSocket连接（使用binance库直接创建）
        from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
        from app.core.config import settings

        # 配置代理
        proxies = None
        if settings.USE_PROXY_WS:
            proxy_url = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            proxies = {'https': proxy_url}
            logger.info(f"📡 使用代理: {proxy_url}")

        # 创建WebSocket客户端
        self._ws_callbacks = {}  # 存储回调

        def on_message(_, msg):
            self._handle_ws_message(msg)

        def on_close(_):
            logger.warning("⚠️ Scalping WebSocket连接关闭")
            # 尝试重连
            if self.is_running:
                asyncio.run_coroutine_threadsafe(self._reconnect(), self._loop)

        def on_error(_, error):
            logger.error(f"❌ Scalping WebSocket错误: {error}")

        self._raw_ws_client = UMFuturesWebsocketClient(
            on_message=on_message,
            on_close=on_close,
            on_error=on_error,
            proxies=proxies
        )

        logger.info("✅ Scalping WebSocket连接成功")

        # 先设置运行状态，确保断开时能触发重连
        self.is_running = True

        # 订阅各币种数据
        for sym_config in symbols:
            await self._subscribe_symbol(sym_config.symbol)
            await asyncio.sleep(0.2)  # 增加间隔，避免订阅太快触发限制

        # 启动价格历史记录任务
        asyncio.create_task(self._record_price_history())

        logger.info(f"✅ 多币种监控启动完成，监控 {len(symbols)} 个币种")

    def _handle_ws_message(self, msg: Dict[str, Any]):
        """处理WebSocket消息"""
        try:
            if isinstance(msg, str):
                import json
                msg = json.loads(msg)

            # 获取流名称
            stream = msg.get('stream', '')
            data = msg.get('data', msg)

            # 根据事件类型处理
            event_type = data.get('e', '')

            if event_type == 'depthUpdate':
                # 深度更新
                symbol = data.get('s', '')
                std_symbol = SymbolMapper.to_standard_format(symbol, "BINANCE")
                if std_symbol in self.symbol_data:
                    self._on_depth_update(std_symbol, data)

            elif event_type == 'aggTrade':
                # 聚合成交
                symbol = data.get('s', '')
                std_symbol = SymbolMapper.to_standard_format(symbol, "BINANCE")
                if std_symbol in self.symbol_data:
                    self._on_trade(std_symbol, data)

            elif event_type == 'kline':
                # K线
                symbol = data.get('s', '')
                std_symbol = SymbolMapper.to_standard_format(symbol, "BINANCE")
                if std_symbol in self.symbol_data:
                    self._on_kline(std_symbol, data)

        except Exception as e:
            logger.debug(f"处理WebSocket消息失败: {e}")

    async def _reconnect(self):
        """重连WebSocket"""
        if not self.is_running:
            return

        logger.info("🔄 尝试重连Scalping WebSocket...")
        await asyncio.sleep(3)

        try:
            from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
            from app.core.config import settings

            # 配置代理
            proxies = None
            if settings.USE_PROXY_WS:
                proxy_url = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                proxies = {'https': proxy_url}

            # 重新创建WebSocket客户端
            def on_message(_, msg):
                self._handle_ws_message(msg)

            def on_close(_):
                logger.warning("⚠️ Scalping WebSocket连接关闭")
                if self.is_running:
                    asyncio.run_coroutine_threadsafe(self._reconnect(), self._loop)

            def on_error(_, error):
                logger.error(f"❌ Scalping WebSocket错误: {error}")

            self._raw_ws_client = UMFuturesWebsocketClient(
                on_message=on_message,
                on_close=on_close,
                on_error=on_error,
                proxies=proxies
            )

            logger.info("✅ Scalping WebSocket重连成功")

            # 重新订阅
            for symbol in self.symbol_data.keys():
                await self._subscribe_symbol(symbol)
                await asyncio.sleep(0.2)

            logger.info(f"✅ 重新订阅 {len(self.symbol_data)} 个币种完成")

        except Exception as e:
            logger.error(f"❌ 重连失败: {e}")
            # 5秒后重试
            await asyncio.sleep(5)
            if self.is_running:
                asyncio.create_task(self._reconnect())

    async def _subscribe_symbol(self, symbol: str):
        """订阅单个币种的数据"""
        exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")

        try:
            # 订阅订单簿深度
            self._raw_ws_client.partial_book_depth(
                symbol=exchange_symbol,
                level=scalping_config.orderbook_depth,
                speed=100,
                id=hash(f"{symbol}_depth") % 10000
            )
            await asyncio.sleep(0.05)  # 每个流之间增加小延迟

            # 订阅成交流
            self._raw_ws_client.agg_trade(
                symbol=exchange_symbol,
                id=hash(f"{symbol}_trade") % 10000
            )
            await asyncio.sleep(0.05)

            # 订阅1分钟K线（用于ATR和成交量计算）
            self._raw_ws_client.kline(
                symbol=exchange_symbol,
                interval="1m",
                id=hash(f"{symbol}_kline") % 10000
            )

            logger.info(f"  ✓ 订阅 {symbol} 数据流 (深度+成交+K线)")

        except Exception as e:
            logger.error(f"订阅 {symbol} 失败: {e}")

    def _on_depth_update(self, symbol: str, data: Dict[str, Any]):
        """处理订单簿更新"""
        try:
            # 解析数据
            if 'data' in data:
                depth_data = data['data']
            else:
                depth_data = data

            bids = [
                OrderBookLevel(price=float(b[0]), quantity=float(b[1]))
                for b in depth_data.get('bids', depth_data.get('b', []))
            ]
            asks = [
                OrderBookLevel(price=float(a[0]), quantity=float(a[1]))
                for a in depth_data.get('asks', depth_data.get('a', []))
            ]

            orderbook = OrderBook(
                symbol=symbol,
                timestamp=int(time.time() * 1000),
                bids=bids,
                asks=asks
            )

            if symbol in self.symbol_data:
                self.symbol_data[symbol].orderbook = orderbook
                self.symbol_data[symbol].last_update = datetime.now()

                # 更新最新价格
                if orderbook.mid_price:
                    self.symbol_data[symbol].last_price = orderbook.mid_price

                # 触发回调
                self._notify_callbacks(symbol)

        except Exception as e:
            logger.error(f"处理订单簿更新失败 {symbol}: {e}")

    def _on_trade(self, symbol: str, data: Dict[str, Any]):
        """处理成交数据"""
        try:
            if 'data' in data:
                trade_data = data['data']
            else:
                trade_data = data

            trade = TradeData(
                symbol=symbol,
                timestamp=trade_data.get('T', trade_data.get('E', int(time.time() * 1000))),
                price=float(trade_data.get('p', 0)),
                quantity=float(trade_data.get('q', 0)),
                is_buyer_maker=trade_data.get('m', False)
            )

            if symbol in self.symbol_data:
                self.symbol_data[symbol].trades.append(trade)
                self.symbol_data[symbol].last_price = trade.price
                self.symbol_data[symbol].last_update = datetime.now()

        except Exception as e:
            logger.error(f"处理成交数据失败 {symbol}: {e}")

    def _on_kline(self, symbol: str, data: Dict[str, Any]):
        """处理K线数据"""
        try:
            if 'data' in data:
                kline_data = data['data'].get('k', {})
            else:
                kline_data = data.get('k', {})

            if symbol in self.symbol_data:
                # 解析K线数据
                kline = KlineData(
                    timestamp=int(kline_data.get('t', 0)),
                    open=float(kline_data.get('o', 0)),
                    high=float(kline_data.get('h', 0)),
                    low=float(kline_data.get('l', 0)),
                    close=float(kline_data.get('c', 0)),
                    volume=float(kline_data.get('v', 0))
                )

                # 检查是否是已完成的K线（x=true表示K线已完成）
                is_closed = kline_data.get('x', False)

                # 更新数据
                self.symbol_data[symbol].last_kline = kline
                self.symbol_data[symbol].last_price = kline.close
                self.symbol_data[symbol].volume_1m = kline.volume

                if kline.open > 0:
                    self.symbol_data[symbol].price_change_1m = (kline.close - kline.open) / kline.open

                # 如果K线已完成，添加到历史
                if is_closed:
                    self.symbol_data[symbol].kline_history.append(kline)
                    logger.debug(f"📊 K线完成 {symbol}: O={kline.open:.6f} H={kline.high:.6f} L={kline.low:.6f} C={kline.close:.6f} V={kline.volume:.2f}")

                # 触发回调（让momentum_analyzer更新ATR）
                self._notify_callbacks(symbol)

        except Exception as e:
            logger.error(f"处理K线数据失败 {symbol}: {e}")

    async def _record_price_history(self):
        """记录价格历史（每秒）"""
        while self.is_running:
            try:
                for symbol, data in self.symbol_data.items():
                    if data.last_price > 0:
                        data.price_history.append({
                            'timestamp': int(time.time()),
                            'price': data.last_price
                        })
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"记录价格历史失败: {e}")
                await asyncio.sleep(1)

    def _notify_callbacks(self, symbol: str):
        """通知回调"""
        if symbol in self.symbol_data:
            data = self.symbol_data[symbol]
            for callback in self.callbacks:
                try:
                    callback(symbol, data)
                except Exception as e:
                    logger.error(f"回调执行失败: {e}")

    def add_callback(self, callback: Callable[[str, SymbolData], None]):
        """添加数据更新回调"""
        self.callbacks.append(callback)

    def get_symbol_data(self, symbol: str) -> Optional[SymbolData]:
        """获取币种数据"""
        return self.symbol_data.get(symbol)

    def get_all_data(self) -> Dict[str, SymbolData]:
        """获取所有币种数据"""
        return self.symbol_data

    def get_price_momentum(self, symbol: str, lookback_seconds: int = 10) -> float:
        """
        获取价格动量
        返回过去N秒的价格变化百分比
        """
        if symbol not in self.symbol_data:
            return 0.0

        history = list(self.symbol_data[symbol].price_history)
        if len(history) < 2:
            return 0.0

        current_time = int(time.time())
        recent_prices = [
            h['price'] for h in history
            if current_time - h['timestamp'] <= lookback_seconds
        ]

        if len(recent_prices) < 2:
            return 0.0

        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

    async def stop(self):
        """停止监控"""
        self.is_running = False
        if hasattr(self, '_raw_ws_client') and self._raw_ws_client:
            try:
                self._raw_ws_client.stop()
            except Exception as e:
                logger.warning(f"停止WebSocket时出错: {e}")
        logger.info("🛑 多币种监控已停止")
