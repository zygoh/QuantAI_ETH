# -*- coding: utf-8 -*-
"""Binance API客户端（httpx 异步版本）"""
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode

import httpx
from binance.um_futures import UMFutures

from app.core.config import settings
from app.exchange.mappers import SymbolMapper

logger = logging.getLogger(__name__)
BINANCE_API_LIMIT_LARGE = 1000


@dataclass
class UnifiedKlineData:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int = 0
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_base_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0


@dataclass
class UnifiedTickerData:
    symbol: str
    price: float
    timestamp: int


class BinanceClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.base_url = "https://fapi.binance.com"
        client_kwargs = {"base_url": self.base_url, "timeout": 60}
        if settings.USE_PROXY:
            proxy_type = settings.PROXY_TYPE.lower()
            proxy_url = (
                f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                if proxy_type == "socks5"
                else f"{proxy_type}://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            )
            client_kwargs["proxies"] = {"http": proxy_url, "https": proxy_url}
        self.client = UMFutures(**client_kwargs)
        self._async_client: Optional[httpx.AsyncClient] = None
        self._server_time_offset_ms: int = 0
        logger.info(f"Binance客户端初始化完成 - {self.base_url}")

    # =========================================================================
    # 服务器时间
    # =========================================================================

    def _get_server_time_ms(self) -> int:
        """
        获取 Binance 服务器时间（毫秒）

        通过同步接口获取服务器时间，计算与本地时钟的偏移量并缓存。
        如果请求失败，回退到本地时间 + 已缓存偏移量。
        """
        try:
            result = self.client.time()
            server_time = result.get("serverTime", 0)
            if server_time > 0:
                local_time = int(time.time() * 1000)
                self._server_time_offset_ms = server_time - local_time
                return server_time
        except Exception as e:
            logger.debug(f"获取服务器时间失败，使用本地时间: {e}")
        return int(time.time() * 1000) + self._server_time_offset_ms

    async def _get_server_time_ms_async(self) -> int:
        """异步获取 Binance 服务器时间（毫秒）"""
        try:
            client = await self._get_async_client()
            response = await client.get("/fapi/v1/time")
            response.raise_for_status()
            data = response.json()
            server_time = data.get("serverTime", 0)
            if server_time > 0:
                local_time = int(time.time() * 1000)
                self._server_time_offset_ms = server_time - local_time
                return server_time
        except Exception as e:
            logger.debug(f"异步获取服务器时间失败，使用本地时间: {e}")
        return int(time.time() * 1000) + self._server_time_offset_ms

    # =========================================================================
    # 异步 HTTP 客户端
    # =========================================================================

    async def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            proxy_url = None
            if settings.USE_PROXY:
                proxy_type = settings.PROXY_TYPE.lower()
                proxy_url = (
                    f"socks5://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                    if proxy_type == "socks5"
                    else f"http://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                )
            limits = httpx.Limits(
                max_keepalive_connections=100,
                max_connections=200,
                keepalive_expiry=30.0,
            )
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
                limits=limits,
                proxy=proxy_url,
                http2=False,
            )
        return self._async_client

    async def close(self) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    # =========================================================================
    # K 线数据（核心方法）
    # =========================================================================

    def _parse_klines(
        self,
        klines: List[Any],
        server_time_ms: int,
    ) -> List[UnifiedKlineData]:
        """
        解析原始 K 线数据，过滤未关闭和无效的 K 线

        Args:
            klines: Binance API 返回的原始 K 线数组
            server_time_ms: 服务器时间（毫秒），用于过滤未关闭 K 线

        Returns:
            已关闭的有效 K 线列表
        """
        result = []
        for k in klines:
            try:
                # close_time >= server_time 表示该 K 线尚未关闭
                if k[6] >= server_time_ms:
                    continue
                # 排除无效数据
                if float(k[4]) <= 0 or float(k[5]) <= 0:
                    continue
                result.append(UnifiedKlineData(
                    timestamp=k[0],
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    close_time=k[6],
                    quote_volume=float(k[7]),
                    trades=int(k[8]),
                    taker_buy_base_volume=float(k[9]) if len(k) > 9 else 0.0,
                    taker_buy_quote_volume=float(k[10]) if len(k) > 10 else 0.0,
                ))
            except (IndexError, ValueError, TypeError):
                continue
        return result

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedKlineData]:
        """
        获取已关闭的 K 线数据（同步）

        先获取服务器时间，再用 endTime 参数让 Binance 服务端
        只返回 open_time <= endTime 的 K 线，从源头减少未关闭数据。
        客户端再用 close_time < server_time 做二次过滤，彻底排除
        竞态条件导致的中间状态数据。

        Args:
            symbol: 交易对
            interval: 时间周期
            limit: 返回数量上限
            start_time: 起始时间（毫秒）
            end_time: 结束时间（毫秒），默认使用服务器时间

        Returns:
            已关闭的 K 线数据列表
        """
        try:
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            limit = min(max(limit, 1), BINANCE_API_LIMIT_LARGE)

            # 先获取服务器时间，用作 endTime 过滤
            server_time_ms = self._get_server_time_ms()

            params: Dict[str, Any] = {
                "symbol": exchange_symbol,
                "interval": interval,
                "limit": limit,
                "endTime": end_time if end_time else server_time_ms,
            }
            if start_time:
                params["startTime"] = start_time

            klines = self.client.klines(**params)
            return self._parse_klines(klines, server_time_ms)
        except Exception as e:
            logger.debug(f"获取K线失败: {e}")
            return []

    async def get_klines_async(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[UnifiedKlineData]:
        """
        获取已关闭的 K 线数据（异步）

        先获取服务器时间，再用 endTime 参数让 Binance 服务端
        只返回 open_time <= endTime 的 K 线，从源头减少未关闭数据。
        客户端再用 close_time < server_time 做二次过滤。

        Args:
            symbol: 交易对
            interval: 时间周期
            limit: 返回数量上限

        Returns:
            已关闭的 K 线数据列表
        """
        try:
            client = await self._get_async_client()
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")

            # 先获取服务器时间，用作 endTime 过滤
            server_time_ms = await self._get_server_time_ms_async()

            params: Dict[str, Any] = {
                "symbol": exchange_symbol,
                "interval": interval,
                "limit": min(limit, BINANCE_API_LIMIT_LARGE),
                "endTime": server_time_ms,
            }
            response = await client.get("/fapi/v1/klines", params=params)
            response.raise_for_status()
            klines = response.json()
            return self._parse_klines(klines, server_time_ms)
        except Exception as e:
            logger.debug(f"异步获取K线失败: {symbol} - {e}")
            return []

    # =========================================================================
    # 其他异步接口
    # =========================================================================

    async def get_open_interest_async(self, symbol: str) -> Dict[str, Any]:
        try:
            client = await self._get_async_client()
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            response = await client.get(
                "/fapi/v1/openInterest",
                params={"symbol": exchange_symbol},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"异步获取持仓量失败: {symbol} - {e}")
            return {}

    async def get_max_leverage_async(self, symbol: str) -> int:
        """
        获取币种最大杠杆倍数

        通过 Binance leverageBracket 接口查询，需要 API Key 签名。
        上限 30x，失败时回退到 20x。

        Args:
            symbol: 交易对

        Returns:
            最大杠杆倍数（上限 30）
        """
        api_key = settings.BINANCE_API_KEY
        api_secret = settings.BINANCE_API_SECRET
        if not api_key or not api_secret:
            logger.debug("未配置 Binance API Key，使用默认杠杆 20x")
            return 20

        try:
            client = await self._get_async_client()
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            timestamp = int(time.time() * 1000) + self._server_time_offset_ms

            params = {
                "symbol": exchange_symbol,
                "timestamp": timestamp,
            }
            query_string = urlencode(params)
            signature = hmac.new(
                api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            params["signature"] = signature

            response = await client.get(
                "/fapi/v1/leverageBracket",
                params=params,
                headers={"X-MBX-APIKEY": api_key},
            )
            response.raise_for_status()
            data = response.json()

            # 解析最大杠杆
            if isinstance(data, list) and len(data) > 0:
                brackets = data[0].get("brackets", [])
                if brackets:
                    max_lev = brackets[0].get("initialLeverage", 20)
                    return min(int(max_lev), 30)

            return 20
        except Exception as e:
            logger.warning(f"⚠️ 获取 {symbol} 最大杠杆失败: {e}，使用默认 20x")
            return 20

    async def get_orderbook_async(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        try:
            client = await self._get_async_client()
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            response = await client.get(
                "/fapi/v1/depth",
                params={"symbol": exchange_symbol, "limit": limit},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"异步获取订单簿失败: {symbol} - {e}")
            return {"bids": [], "asks": []}

    async def get_agg_trades_async(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """近期聚合成交（逐笔聚合），用于成交分布与买卖压力"""
        try:
            client = await self._get_async_client()
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            response = await client.get(
                "/fapi/v1/aggTrades",
                params={"symbol": exchange_symbol, "limit": min(limit, 1000)},
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.debug(f"异步获取近期成交失败: {symbol} - {e}")
            return []

    async def get_24hr_ticker_async(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            client = await self._get_async_client()
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            response = await client.get("/fapi/v1/ticker/24hr", params=params)
            response.raise_for_status()
            data = response.json()
            return [data] if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"异步获取24小时行情失败: {e}")
            return []

    # =========================================================================
    # 其他同步接口
    # =========================================================================

    def get_24hr_ticker(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            if symbol:
                exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
                result = self.client.ticker_24hr_price_change(
                    symbol=exchange_symbol
                )
                return [result] if result else []
            return self.client.ticker_24hr_price_change()
        except Exception as e:
            logger.error(f"获取24小时行情失败: {e}")
            return []

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        try:
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            return self.client.depth(symbol=exchange_symbol, limit=limit)
        except Exception as e:
            logger.debug(f"获取订单簿失败: {symbol} - {e}")
            return {"bids": [], "asks": []}

    def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        try:
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")
            return self.client.open_interest(symbol=exchange_symbol)
        except Exception as e:
            logger.debug(f"获取持仓量失败: {symbol} - {e}")
            return {}


binance_client = BinanceClient()
