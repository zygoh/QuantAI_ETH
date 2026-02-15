# -*- coding: utf-8 -*-
"""
价格工具：REST 获取当前价，供主循环与 API 共用
"""

import logging

from app.exchange.clients.binance.binance_client import binance_client


logger = logging.getLogger(__name__)


async def get_current_price(symbol: str) -> float:
    """获取指定币种当前价格（REST ticker），供图表、AI、API 展示使用"""
    try:
        tickers = await binance_client.get_24hr_ticker_async(symbol)
        if tickers and len(tickers) > 0:
            price = float(tickers[0].get("lastPrice", 0))
            if price > 0:
                return price
    except Exception as e:
        logger.warning(f"⚠️ 获取 {symbol} 价格失败: {e}")
    return 0.0
