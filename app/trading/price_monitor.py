# -*- coding: utf-8 -*-
"""
实时价格监控器

通过 WebSocket 订阅 Binance Futures K 线数据，实现实时止盈止损监控。
"""

import asyncio
import json
import logging
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from app.core.config import settings


logger = logging.getLogger(__name__)


class PriceMonitor:
    """
    实时价格监控器
    
    通过 WebSocket 订阅指定币种的 K 线数据，
    每次价格更新时触发回调函数检查止盈止损。
    """
    
    # Binance Futures WebSocket 地址
    WS_BASE_URL = "wss://fstream.binance.com/ws"
    
    def __init__(self) -> None:
        """初始化价格监控器"""
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_symbol: Optional[str] = None
        self._price_callback: Optional[Callable[[str, float], None]] = None
        self._running: bool = False
        self._reconnect_delay: float = 5.0
        self._current_price: float = 0.0
        
        logger.info("📡 价格监控器初始化完成")
    
    async def subscribe(
        self,
        symbol: str,
        callback: Callable[[str, float], None]
    ) -> None:
        """
        订阅币种价格
        
        Args:
            symbol: 交易对 (如 "BTCUSDT")
            callback: 价格更新回调函数 (symbol, price) -> None
        """
        # 如果已订阅其他币种，先取消
        if self._running and self._current_symbol != symbol:
            await self.unsubscribe()
        
        self._current_symbol = symbol.upper()
        self._price_callback = callback
        self._running = True
        
        logger.info(f"📡 开始订阅 {self._current_symbol} 实时价格")
        
        # 启动 WebSocket 连接
        asyncio.create_task(self._connect_and_listen())
    
    async def unsubscribe(self) -> None:
        """取消订阅"""
        self._running = False
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        
        if self._current_symbol:
            logger.info(f"📡 已取消订阅 {self._current_symbol}")
            self._current_symbol = None
    
    async def _connect_and_listen(self) -> None:
        """连接 WebSocket 并监听价格"""
        while self._running:
            try:
                # 构建 WebSocket URL
                # 订阅 1m K 线，获取实时价格
                stream = f"{self._current_symbol.lower()}@kline_1m"
                ws_url = f"{self.WS_BASE_URL}/{stream}"
                
                # 如果使用代理
                if settings.USE_PROXY:
                    proxy = f"socks5://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                    # websockets 库不直接支持 socks5，需要用 aiohttp 或其他方式
                    # 这里先不使用代理，直接连接
                    pass
                
                logger.debug(f"📡 连接 WebSocket: {ws_url}")
                
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10
                ) as ws:
                    self._ws = ws
                    logger.info(f"✅ WebSocket 连接成功: {self._current_symbol}")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        await self._handle_message(message)
                
            except ConnectionClosed as e:
                logger.warning(f"⚠️ WebSocket 连接关闭: {e}")
            except Exception as e:
                logger.error(f"❌ WebSocket 错误: {e}")
            
            # 重连
            if self._running:
                logger.info(f"🔄 {self._reconnect_delay}秒后重连...")
                await asyncio.sleep(self._reconnect_delay)
    
    async def _handle_message(self, message: str) -> None:
        """处理 WebSocket 消息"""
        try:
            data = json.loads(message)
            
            # K 线数据格式
            # {"e":"kline","E":1234567890,"s":"BTCUSDT","k":{...}}
            if data.get("e") == "kline":
                kline = data.get("k", {})
                symbol = data.get("s", "")
                close_price = float(kline.get("c", 0))  # 当前收盘价（实时价格）
                
                if close_price > 0 and self._price_callback:
                    self._current_price = close_price
                    # 触发回调
                    await self._trigger_callback(symbol, close_price)
                    
        except json.JSONDecodeError:
            logger.warning(f"⚠️ 无法解析消息: {message[:100]}")
        except Exception as e:
            logger.error(f"❌ 处理消息失败: {e}")
    
    async def _trigger_callback(self, symbol: str, price: float) -> None:
        """触发价格回调"""
        if self._price_callback:
            try:
                # 回调可能是同步或异步函数
                result = self._price_callback(symbol, price)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"❌ 价格回调执行失败: {e}")
    
    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._ws is not None and self._ws.open
    
    @property
    def current_symbol(self) -> Optional[str]:
        """当前订阅的币种"""
        return self._current_symbol

    @property
    def current_price(self) -> float:
        """当前实时价格"""
        return self._current_price


# 全局价格监控器实例
price_monitor = PriceMonitor()
