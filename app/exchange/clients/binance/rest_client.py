"""
Binance REST API客户端

提供Binance期货REST API的封装。
"""
import logging
import time
from typing import Optional, List, Dict, Any

from binance.um_futures import UMFutures

from app.core.config import settings
from app.core.constants import (
    BINANCE_API_LIMIT_LARGE,
    BINANCE_API_LIMIT_MEDIUM,
    BINANCE_RECV_WINDOW_MS,
    BINANCE_RATE_LIMIT_DELAY_SECONDS,
)
from app.exchange.base.types import UnifiedKlineData, UnifiedTickerData
from app.exchange.mappers import SymbolMapper

logger = logging.getLogger(__name__)


class BinanceRestClient:
    """Binance REST API客户端"""

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """安全地将值转换为float"""
        if value is None or value == '' or value == 'None':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"无法转换为float: value={repr(value)}, 使用默认值={default}")
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """安全地将值转换为int"""
        if value is None or value == '' or value == 'None':
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"无法转换为int: value={repr(value)}, 使用默认值={default}")
            return default

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化REST客户端

        Args:
            config: 配置参数（可选）
        """
        # 仅使用公共接口，无需API Key
        self.account_endpoints_enabled = False

        self.base_url = "https://fapi.binance.com"
        client_kwargs = {
            "base_url": self.base_url,
            "timeout": 60
        }

        # 添加代理配置
        if settings.USE_PROXY:
            proxy_type = settings.PROXY_TYPE.lower()

            if proxy_type == "socks5":
                # SOCKS5代理（使用 socks5h 协议以支持远程 DNS 解析）
                proxy_url = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                client_kwargs["proxies"] = {
                    "http": proxy_url,
                    "https": proxy_url
                }
                logger.info(f"REST API使用SOCKS5代理 (Remote DNS): "
                           f"{settings.PROXY_HOST}:{settings.PROXY_PORT}")
            else:
                # HTTP/HTTPS代理
                proxy_url = f"{proxy_type}://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                client_kwargs["proxies"] = {
                    "http": proxy_url,
                    "https": proxy_url
                }
                logger.info(f"REST API使用{proxy_type.upper()}代理: "
                           f"{settings.PROXY_HOST}:{settings.PROXY_PORT}")
        else:
            logger.info("REST API直连（未使用代理）")

        # REST API客户端
        self.client = UMFutures(**client_kwargs)

        # 设置默认的recvWindow
        self.recv_window = BINANCE_RECV_WINDOW_MS

        logger.info(f"Binance REST客户端初始化完成（公共接口模式）")
        logger.info(f"  - REST URL: {self.base_url}")

    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            server_time = self.client.time()
            logger.info(f"服务器时间获取成功: {server_time.get('serverTime')}")
            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"服务器时间获取失败: {e}")

            # 检查是否是代理连接问题
            if settings.USE_PROXY:
                if "10061" in error_msg or "积极拒绝" in error_msg or "Connection refused" in error_msg:
                    logger.error("代理连接失败：代理服务器可能未运行")
                    logger.error(f"   请检查代理服务是否在 {settings.PROXY_HOST}:{settings.PROXY_PORT} 运行")
                elif "SOCKS" in error_msg or "socks" in error_msg:
                    logger.error("SOCKS代理连接失败")

            return False

    def get_server_time(self) -> int:
        """获取服务器时间"""
        try:
            result = self.client.time()
            return result['serverTime']
        except Exception as e:
            logger.error(f"获取服务器时间失败: {e}")
            return int(time.time() * 1000)

    def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        try:
            return self.client.exchange_info()
        except Exception as e:
            logger.error(f"获取交易所信息失败: {e}")
            return {}

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取交易对信息"""
        try:
            exchange_info = self.get_exchange_info()
            symbols = exchange_info.get('symbols', [])

            # 将标准格式转换为 Binance 格式
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")

            for symbol_info in symbols:
                if symbol_info['symbol'] == exchange_symbol:
                    return symbol_info

            return None

        except Exception as e:
            logger.error(f"获取交易对信息失败: {e}")
            return None

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[UnifiedKlineData]:
        """获取K线数据"""
        try:
            # 将标准格式转换为 Binance 格式
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")

            if limit > BINANCE_API_LIMIT_LARGE:
                logger.warning(f"limit={limit} 超过Binance最大限制{BINANCE_API_LIMIT_LARGE}，"
                              f"自动调整为{BINANCE_API_LIMIT_LARGE}")
                limit = BINANCE_API_LIMIT_LARGE
            elif limit <= 0:
                logger.warning(f"limit={limit} 无效，使用默认值{BINANCE_API_LIMIT_MEDIUM}")
                limit = BINANCE_API_LIMIT_MEDIUM

            params = {
                'symbol': exchange_symbol,
                'interval': interval,
                'limit': limit
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            klines = self.client.klines(**params)

            # 转换为统一格式
            current_time_ms = int(time.time() * 1000)
            formatted_klines = []
            skipped_incomplete = 0
            skipped_invalid = 0

            for idx, kline in enumerate(klines):
                try:
                    if len(kline) < 11:
                        taker_buy_base = 0.0
                        taker_buy_quote = 0.0
                    else:
                        taker_buy_base = float(kline[9]) if kline[9] else 0.0
                        taker_buy_quote = float(kline[10]) if kline[10] else 0.0

                    close_time = kline[6]
                    if close_time >= current_time_ms:
                        skipped_incomplete += 1
                        continue

                    close_price = float(kline[4])
                    volume = float(kline[5])

                    if close_price <= 0:
                        skipped_invalid += 1
                        continue

                    if volume is None or volume <= 0 or (isinstance(volume, float) and (volume != volume)):
                        skipped_invalid += 1
                        continue

                    formatted_kline = UnifiedKlineData(
                        timestamp=kline[0],
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=close_price,
                        volume=volume,
                        close_time=kline[6],
                        quote_volume=float(kline[7]),
                        trades=int(kline[8]),
                        taker_buy_base_volume=taker_buy_base,
                        taker_buy_quote_volume=taker_buy_quote
                    )
                    formatted_klines.append(formatted_kline)
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"解析K线数据失败 (索引{idx}): {e}")
                    skipped_invalid += 1
                    continue

            if skipped_incomplete > 0:
                logger.debug(f"过滤了 {skipped_incomplete} 根未完成K线")
            if skipped_invalid > 0:
                logger.info(f"已过滤 {skipped_invalid} 条无效K线")
            logger.debug(f"获取K线数据: {symbol} {interval} {len(formatted_klines)}条")
            return formatted_klines

        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return []

    def get_klines_paginated(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        rate_limit_delay: float = BINANCE_RATE_LIMIT_DELAY_SECONDS
    ) -> List[UnifiedKlineData]:
        """
        分页获取K线数据（自动处理超过1000的情况）

        Args:
            symbol: 交易对符号
            interval: K线间隔
            limit: 需要获取的总数量
            start_time: 开始时间（毫秒时间戳，可选）
            end_time: 结束时间（毫秒时间戳，可选，默认当前时间）
            rate_limit_delay: API限流延迟（秒）

        Returns:
            K线数据列表（按时间升序排列）
        """
        try:
            if limit <= BINANCE_API_LIMIT_LARGE:
                return self.get_klines(symbol, interval, limit, start_time, end_time)

            # 超过1000，需要分页获取
            all_klines = []
            max_per_request = BINANCE_API_LIMIT_LARGE
            batches_needed = (limit + max_per_request - 1) // max_per_request

            logger.debug(f"分页获取K线: {symbol} {interval} 需要{limit}条，分{batches_needed}批获取")

            current_end_time = end_time

            for batch in range(batches_needed):
                remaining = limit - len(all_klines)
                batch_limit = min(max_per_request, remaining)

                if batch_limit <= 0:
                    break

                # 获取一批数据
                klines = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=batch_limit,
                    start_time=start_time,
                    end_time=current_end_time
                )

                if not klines:
                    break

                all_klines.extend(klines)

                if len(all_klines) >= limit:
                    break

                # 设置下一批次的 end_time 为当前批次最早的时间 - 1ms
                current_end_time = klines[0].timestamp - 1

                # API限流（最后一批不需要延迟）
                if batch < batches_needed - 1:
                    time.sleep(rate_limit_delay)

            # 按时间戳排序（确保顺序正确）
            all_klines.sort(key=lambda x: x.timestamp)

            # 去重（防止批次边界重复）
            seen_timestamps = set()
            unique_klines = []
            for kline in all_klines:
                ts = kline.timestamp
                if ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    unique_klines.append(kline)

            logger.debug(f"分页获取完成: {symbol} {interval} 共{len(unique_klines)}条")
            return unique_klines[:limit]

        except Exception as e:
            logger.error(f"分页获取K线数据失败: {symbol} {interval} - {e}")
            return []

    def get_ticker_price(self, symbol: str) -> Optional[UnifiedTickerData]:
        """获取实时价格"""
        try:
            # 将标准格式转换为 Binance 格式
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, "BINANCE")

            ticker = self.client.ticker_price(symbol=exchange_symbol)

            if ticker:
                return UnifiedTickerData(
                    symbol=ticker.get('symbol', symbol),
                    price=self._safe_float(ticker.get('price'), 0.0),
                    timestamp=int(time.time() * 1000)
                )
            return None

        except Exception as e:
            logger.error(f"获取实时价格失败: {symbol} - {e}")
            return None

    # ==================== 信号系统模式：以下方法返回空值 ====================

    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息（信号系统：不支持实际交易，返回空）"""
        logger.debug("信号系统：get_account_info被调用（不支持实际交易，返回空）")
        return {}

    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息（信号系统：不支持实际交易，返回空）"""
        logger.debug("信号系统：get_position_info被调用（不支持实际交易，返回空）")
        return []

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_position: bool = False,
        stop_price: Optional[float] = None,
        callback_rate: Optional[float] = None,
        working_type: str = "MARK_PRICE"
    ) -> Dict[str, Any]:
        """下单（信号系统：不支持实际交易，返回空）"""
        logger.warning(f"信号系统：place_order被调用（不支持实际交易）"
                      f"symbol={symbol}, side={side}, type={order_type}")
        logger.warning("   提示：请使用trading_engine的虚拟交易功能")
        return {}

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """撤销订单（信号系统：不支持实际交易，返回空）"""
        logger.warning(f"信号系统：cancel_order被调用（不支持实际交易）"
                      f"symbol={symbol}, order_id={order_id}")
        return {}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取未成交订单（信号系统：不支持实际交易，返回空）"""
        logger.debug("信号系统：get_open_orders被调用（不支持实际交易，返回空）")
        return []

    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """修改杠杆倍数（信号系统：不支持实际交易，返回空）"""
        logger.warning(f"信号系统：change_leverage被调用（不支持实际交易）"
                      f"symbol={symbol}, leverage={leverage}")
        return {}

    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """修改保证金模式（信号系统：不支持实际交易，返回空）"""
        logger.warning(f"信号系统：change_margin_type被调用（不支持实际交易）"
                      f"symbol={symbol}, margin_type={margin_type}")
        return {}
