"""
Binance API客户端
"""
import asyncio
import logging
import traceback
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import time
import hmac
import hashlib
import requests
import os
import ssl
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
import websocket

from app.core.config import settings
from app.exchange.base_exchange_client import (
    BaseExchangeClient,
    UnifiedKlineData,
    UnifiedTickerData,
    UnifiedOrderData
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


class WebSocketErrorType(Enum):
    """WebSocket错误类型"""
    SSL_ERROR = "ssl_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    PROTOCOL_ERROR = "protocol_error"
    UNKNOWN_ERROR = "unknown_error"


class ExponentialBackoffReconnector:
    """
    指数退避重连策略
    
    实现智能重连策略，避免频繁重连导致服务端封禁
    """
    
    def __init__(self):
        """初始化重连器"""
        self.initial_delay = settings.WS_RECONNECT_INITIAL_DELAY
        self.max_delay = settings.WS_RECONNECT_MAX_DELAY
        self.backoff_factor = settings.WS_RECONNECT_BACKOFF_FACTOR
        self.max_retries = settings.WS_RECONNECT_MAX_RETRIES
        
        self.current_delay = self.initial_delay
        self.retry_count = 0
        self.reconnect_history: List[ReconnectRecord] = []
        self.connection_start_time: Optional[datetime] = None
        
        logger.info(f"🔧 重连器初始化: 初始延迟={self.initial_delay}s, 最大延迟={self.max_delay}s, 退避因子={self.backoff_factor}")
    
    def calculate_next_delay(self) -> float:
        """
        计算下次重连延迟（指数退避）
        
        Returns:
            下次重连延迟（秒）
        """
        delay = min(
            self.initial_delay * (self.backoff_factor ** self.retry_count),
            self.max_delay
        )
        return delay
    
    def should_retry(self) -> bool:
        """
        检查是否应该继续重试
        
        Returns:
            是否应该重试
        """
        return self.retry_count < self.max_retries
    
    def on_reconnect_attempt(self) -> float:
        """
        记录重连尝试，返回应该等待的延迟
        
        Returns:
            等待延迟（秒）
        """
        self.retry_count += 1
        self.current_delay = self.calculate_next_delay()
        
        logger.info(f"🔄 重连尝试 {self.retry_count}/{self.max_retries}, 延迟: {self.current_delay:.1f}秒")
        
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
        
        logger.info(f"✅ 重连成功！连接已恢复，重置重连计数器")
    
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
        
        logger.error(f"❌ 重连失败 (尝试 {self.retry_count}/{self.max_retries}): {error_type.value}")
        logger.error(f"   错误信息: {str(error)[:200]}")
    
    def reset(self):
        """重置重连状态"""
        self.retry_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now()
        logger.info("🔄 重连器状态已重置")
    
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


class WebSocketHeartbeat:
    """
    WebSocket心跳保活机制
    
    定期发送ping消息保持连接活跃，检测pong超时
    """
    
    def __init__(self, ws_client):
        """
        初始化心跳机制
        
        Args:
            ws_client: WebSocket客户端实例
        """
        self.ws_client = ws_client
        self.ping_interval = settings.WS_PING_INTERVAL
        self.pong_timeout = settings.WS_PONG_TIMEOUT
        self.last_ping_time: Optional[datetime] = None
        self.last_pong_time: Optional[datetime] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(f"💓 心跳机制初始化: ping间隔={self.ping_interval}s, pong超时={self.pong_timeout}s")
    
    async def start(self):
        """启动心跳任务"""
        if self.is_running:
            logger.warning("⚠️ 心跳任务已在运行")
            return
        
        self.is_running = True
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("💓 心跳任务已启动")
    
    async def stop(self):
        """停止心跳任务"""
        self.is_running = False
        
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        logger.info("💓 心跳任务已停止")
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.ping_interval)
                
                # 发送ping
                await self.send_ping()
                
                # 检查pong超时
                if self.last_ping_time and self.last_pong_time:
                    time_since_pong = (datetime.now() - self.last_pong_time).total_seconds()
                    if time_since_pong > self.pong_timeout:
                        logger.warning(f"⚠️ Pong超时: {time_since_pong:.1f}秒未收到pong响应")
                        # 注意：不在这里触发重连，由健康检查任务处理
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 心跳循环异常: {e}")
                await asyncio.sleep(5)  # 出错后等待5秒再继续
    
    async def send_ping(self):
        """发送ping消息"""
        try:
            if hasattr(self.ws_client, 'ws') and self.ws_client.ws:
                # 发送ping帧
                self.ws_client.ws.ping()
                self.last_ping_time = datetime.now()
                logger.debug("📤 发送Ping消息")
            else:
                logger.debug("⚠️ WebSocket未连接，跳过ping")
        except Exception as e:
            logger.error(f"❌ 发送ping失败: {e}")
    
    def on_pong_received(self):
        """处理pong响应"""
        self.last_pong_time = datetime.now()
        
        if self.last_ping_time:
            rtt = (self.last_pong_time - self.last_ping_time).total_seconds()
            logger.debug(f"📥 收到Pong响应 (RTT: {rtt*1000:.1f}ms)")
        else:
            logger.debug("📥 收到Pong响应")
    
    def is_alive(self) -> bool:
        """
        检查连接是否存活
        
        Returns:
            连接是否存活
        """
        if not self.last_pong_time:
            return True  # 还没有收到过pong，认为是活的
        
        time_since_pong = (datetime.now() - self.last_pong_time).total_seconds()
        return time_since_pong <= self.pong_timeout


class BinanceClient(BaseExchangeClient):
    """Binance API客户端"""
    
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        安全地将值转换为float
        
        Args:
            value: 要转换的值
            default: 默认值
        
        Returns:
            转换后的float值，如果转换失败则返回默认值
        """
        if value is None or value == '' or value == 'None':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ 无法转换为float: value={repr(value)}, 使用默认值={default}")
            return default
    
    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """
        安全地将值转换为int
        
        Args:
            value: 要转换的值
            default: 默认值
        
        Returns:
            转换后的int值，如果转换失败则返回默认值
        """
        if value is None or value == '' or value == 'None':
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"⚠️ 无法转换为int: value={repr(value)}, 使用默认值={default}")
            return default
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = settings.BINANCE_API_KEY
        self.secret_key = settings.BINANCE_SECRET_KEY
        self.testnet = settings.BINANCE_TESTNET
        # ⛔️ 当前账户不可用，临时禁用需要签名的账户相关接口
        # TODO: 恢复账户权限后将该标记改为 True 并恢复相关调用
        self.account_endpoints_enabled = False
        
        # 配置代理地址
        # REST API: https://n8n.do2ge.com/tail/http/relay/fapi/v1/... -> https://fapi.binance.com/fapi/v1/...
        self.base_url = "https://n8n.do2ge.com/tail/http/relay"
        
        # 🔧 配置REST API代理（如果启用）
        client_kwargs = {
            "key": self.api_key,
            "secret": self.secret_key,
            "base_url": self.base_url,
            "timeout": 30  # 增加超时时间
        }
        
        # 添加代理配置
        if settings.USE_PROXY:
            proxy_type = settings.PROXY_TYPE.lower()
            
            if proxy_type == "socks5":
                # SOCKS5代理（需要PySocks库支持）
                proxy_url = f"socks5://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                client_kwargs["proxies"] = {
                    "http": proxy_url,
                    "https": proxy_url
                }
                logger.info(f"🔧 REST API使用SOCKS5代理: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
            else:
                # HTTP/HTTPS代理
                proxy_url = f"{proxy_type}://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                client_kwargs["proxies"] = {
                    "http": proxy_url,
                    "https": proxy_url
                }
                logger.info(f"🔧 REST API使用{proxy_type.upper()}代理: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
        
        # REST API客户端
        self.client = UMFutures(**client_kwargs)
        
        # 设置默认的recvWindow（在API调用时使用）
        self.recv_window = 60000  # 60秒的时间窗口（默认5000ms）
        
        # WebSocket客户端
        self.ws_client: Optional[UMFuturesWebsocketClient] = None
        self.ws_callbacks: Dict[str, Callable] = {}
        
        logger.info(f"Binance客户端初始化完成")
        logger.info(f"  - 模式: {'测试网' if self.testnet else '生产环境'}")
        logger.info(f"  - REST URL: {self.base_url}")
        logger.info(f"  - API Key 长度: {len(self.api_key)} 字符")
        logger.info(f"  - API Key (前8位): {self.api_key[:8]}...")
        logger.info(f"  - Secret Key 长度: {len(self.secret_key)} 字符")
        logger.info(f"  - Secret Key (前8位): {self.secret_key[:8]}...")
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            # 测试REST API
            server_time = self.client.time()
            logger.info(f"✓ 服务器时间获取成功: {server_time.get('serverTime')}")
            
            # 测试账户信息（需要签名）
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 已跳过账户信息检测：账户相关接口暂时禁用")
                return True
            logger.info("正在测试账户信息获取（需要 API Key 签名）...")
            try:
                account = self.client.account(recvWindow=self.recv_window)
                logger.info(f"✓ 账户余额: {account.get('totalWalletBalance', 0)} USDT")
                return True
            except Exception as account_error:
                logger.error(f"✗ 账户信息获取失败: {account_error}")
                logger.error("可能的原因：")
                logger.error("  1. API Key 未启用期货交易权限")
                logger.error("  2. API Key 或 Secret Key 不正确")
                return False
            
        except Exception as e:
            logger.error(f"服务器时间获取失败: {e}")
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
            
            for symbol_info in symbols:
                if symbol_info['symbol'] == symbol:
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
            # ✅ 关键修复：Binance API limit 最大值为 1500
            if limit > 1500:
                logger.warning(f"⚠️ limit={limit} 超过Binance最大限制1500，自动调整为1500")
                limit = 1500
            elif limit <= 0:
                logger.warning(f"⚠️ limit={limit} 无效，使用默认值500")
                limit = 500
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            klines = self.client.klines(**params)
            
            # ✅ 诊断：检查REST API返回的数组长度
            if klines and len(klines) > 0:
                first_kline = klines[0]
                logger.debug(f"📊 REST API K线数组长度: {len(first_kline)} (预期11个字段)")
                if len(first_kline) < 11:
                    logger.warning(f"⚠️ REST API返回的K线数组长度不足: {len(first_kline)} < 11")
                    logger.warning(f"   数组内容: {first_kline}")
                    logger.warning(f"   可能原因: Binance API版本或代理过滤了某些字段")
            
            # 转换为统一格式
            current_time_ms = int(time.time() * 1000)  # 当前时间（毫秒）
            formatted_klines = []
            skipped_incomplete = 0
            for idx, kline in enumerate(klines):
                try:
                    # ✅ 安全访问：检查数组长度
                    if len(kline) < 11:
                        logger.warning(f"⚠️ K线数组{idx}长度不足: {len(kline)} < 11，使用默认值0")
                        taker_buy_base = 0.0
                        taker_buy_quote = 0.0
                    else:
                        taker_buy_base = float(kline[9]) if kline[9] else 0.0
                        taker_buy_quote = float(kline[10]) if kline[10] else 0.0
                    
                    close_time = kline[6]  # K线结束时
                    # 🔥 关键修复：过滤未完成的K线
                    if close_time >= current_time_ms:
                        skipped_incomplete += 1
                        logger.debug(f"⏸️ 跳过未完成K线: 索引={idx}")
                        continue

                    formatted_kline = UnifiedKlineData(
                        timestamp=kline[0],
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5]),
                        close_time=kline[6],
                        quote_volume=float(kline[7]),
                        trades=int(kline[8]),
                        taker_buy_base_volume=taker_buy_base,
                        taker_buy_quote_volume=taker_buy_quote
                    )
                    formatted_klines.append(formatted_kline)
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"❌ 解析K线数据失败 (索引{idx}): {e}")
                    logger.error(f"   数组长度: {len(kline)}, 数组内容: {kline}")
                    continue
            if skipped_incomplete > 0:
                logger.debug(f"⏸️ 过滤了 {skipped_incomplete} 根未完成K线")
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
        rate_limit_delay: float = 0.1
    ) -> List[UnifiedKlineData]:
        """
        分页获取K线数据（自动处理超过1500的情况）
        
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
            # 如果 limit <= 1500，直接调用单次获取
            if limit <= 1500:
                return self.get_klines(symbol, interval, limit, start_time, end_time)
            
            # 超过1500，需要分页获取
            all_klines = []
            max_per_request = 1500
            batches_needed = (limit + max_per_request - 1) // max_per_request
            
            logger.debug(f"📊 分页获取K线: {symbol} {interval} 需要{limit}条，分{batches_needed}批获取")
            
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
                    logger.warning(f"⚠️ 批次 {batch + 1}/{batches_needed} 未获取到数据")
                    break
                
                # 添加到总列表
                all_klines.extend(klines)
                
                # 如果已经获取到足够的数据，退出
                if len(all_klines) >= limit:
                    break
                
                # 如果返回的数据少于请求的数量，说明没有更多数据了
                if len(klines) < batch_limit:
                    logger.debug(f"📊 批次 {batch + 1}/{batches_needed} 返回{len(klines)}条 < 请求{batch_limit}条，数据已获取完毕")
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
            
            logger.debug(f"✅ 分页获取完成: {symbol} {interval} 共{len(unique_klines)}条（去重后）")
            return unique_klines[:limit]  # 确保不超过请求的数量
            
        except Exception as e:
            logger.error(f"分页获取K线数据失败: {symbol} {interval} - {e}")
            return []
    
    def get_ticker_price(self, symbol: str) -> Optional[UnifiedTickerData]:
        """获取实时价格（24hr ticker）"""
        try:
            # ✅ 使用代理的 REST API
            ticker = self.client.ticker_price(symbol=symbol)
            
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
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 账户接口已临时禁用，返回空账户信息")
                return {}

            account = self.client.account(recvWindow=self.recv_window)
            
            # 格式化账户信息 - 使用安全转换
            formatted_account = {
                'total_wallet_balance': self._safe_float(account.get('totalWalletBalance'), 0.0),
                'total_unrealized_pnl': self._safe_float(account.get('totalUnrealizedPnL'), 0.0),
                'total_margin_balance': self._safe_float(account.get('totalMarginBalance'), 0.0),
                'total_position_initial_margin': self._safe_float(account.get('totalPositionInitialMargin'), 0.0),
                'total_open_order_initial_margin': self._safe_float(account.get('totalOpenOrderInitialMargin'), 0.0),
                'available_balance': self._safe_float(account.get('availableBalance'), 0.0),
                'max_withdraw_amount': self._safe_float(account.get('maxWithdrawAmount'), 0.0),
                'can_trade': account.get('canTrade', False),
                'can_deposit': account.get('canDeposit', False),
                'can_withdraw': account.get('canWithdraw', False),
                'update_time': account.get('updateTime', 0)
            }
            
            return formatted_account
            
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return {}
    
    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        try:
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 持仓接口已临时禁用，返回空列表")
                return []

            params = {'recvWindow': self.recv_window}
            if symbol:
                params['symbol'] = symbol
            
            positions = self.client.get_position_risk(**params)
            
            # 过滤有持仓的合约
            active_positions = []
            for position in positions:
                position_amt = self._safe_float(position.get('positionAmt'), 0.0)
                if position_amt != 0:
                    formatted_position = {
                        'symbol': position.get('symbol', ''),
                        'position_amt': position_amt,
                        'entry_price': self._safe_float(position.get('entryPrice'), 0.0),
                        'mark_price': self._safe_float(position.get('markPrice'), 0.0),
                        'pnl': self._safe_float(position.get('unRealizedProfit'), 0.0),
                        'percentage': self._safe_float(position.get('percentage'), 0.0),
                        'position_side': position.get('positionSide', 'BOTH'),
                        'isolated': position.get('isolated', False),
                        'margin_type': position.get('marginType', 'cross'),
                        'leverage': self._safe_int(position.get('leverage'), 1),
                        'max_notional_value': self._safe_float(position.get('maxNotionalValue'), 0.0),
                        'update_time': position.get('updateTime', 0)
                    }
                    active_positions.append(formatted_position)
            
            return active_positions
            
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
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
        """下单"""
        try:
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 下单接口已临时禁用，返回空结果")
                return {}

            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'timeInForce': time_in_force,
                'reduceOnly': reduce_only,
                'closePosition': close_position,
                'workingType': working_type
            }
            
            if price is not None:
                params['price'] = price
            
            if stop_price is not None:
                params['stopPrice'] = stop_price
            
            if callback_rate is not None:
                params['callbackRate'] = callback_rate
            
            # 添加recvWindow参数
            params['recvWindow'] = self.recv_window
            
            result = self.client.new_order(**params)
            
            logger.info(f"下单成功: {symbol} {side} {quantity} @ {price}")
            return result
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {}
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        撤销订单
        
        Args:
            symbol: 交易对符号
            order_id: 订单ID（字符串格式，内部转换为整数）
        
        Returns:
            取消结果字典
        """
        try:
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 撤单接口已临时禁用，返回空结果")
                return {}

            # Binance API需要整数类型的orderId，但接口统一使用str
            order_id_int = int(order_id)
            result = self.client.cancel_order(symbol=symbol, orderId=order_id_int, recvWindow=self.recv_window)
            logger.info(f"撤销订单成功: {symbol} {order_id}")
            return result
        except ValueError as e:
            logger.error(f"撤销订单失败: 订单ID格式错误 '{order_id}': {e}")
            return {}
        except Exception as e:
            logger.error(f"撤销订单失败: {e}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        try:
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 未成交订单接口已临时禁用，返回空列表")
                return []

            params = {'recvWindow': self.recv_window}
            if symbol:
                params['symbol'] = symbol
            
            orders = self.client.get_orders(**params)
            return orders
            
        except Exception as e:
            logger.error(f"获取未成交订单失败: {e}")
            return []
    
    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """修改杠杆倍数"""
        try:
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 杠杆调整接口已临时禁用，返回空结果")
                return {}

            result = self.client.change_leverage(symbol=symbol, leverage=leverage, recvWindow=self.recv_window)
            logger.info(f"修改杠杆成功: {symbol} {leverage}x")
            return result
        except Exception as e:
            logger.error(f"修改杠杆失败: {e}")
            return {}
    
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """修改保证金模式（可能已设置，失败不影响）"""
        try:
            if not self.account_endpoints_enabled:
                logger.warning("⏸️ 保证金模式调整接口已临时禁用，返回空结果")
                return {}

            result = self.client.change_margin_type(symbol=symbol, marginType=margin_type, recvWindow=self.recv_window)
            logger.info(f"修改保证金模式成功: {symbol} {margin_type}")
            return result
        except Exception as e:
            # 如果提示"No need to change"，说明已经是目标模式，使用warning而非error
            error_msg = str(e)
            if 'No need to change' in error_msg or '-4046' in error_msg:
                logger.warning(f"保证金模式无需修改（已是 {margin_type} 模式）: {symbol}")
            else:
                logger.error(f"修改保证金模式失败: {e}")
            return {}

class BinanceWebSocketClient:
    """Binance WebSocket客户端（支持自动重连和心跳保活）"""
    
    def __init__(self):
        self.testnet = settings.BINANCE_TESTNET
        self.ws_client: Optional[UMFuturesWebsocketClient] = None
        self.callbacks: Dict[str, Callable] = {}
        self.is_connected = False
        self.is_running = False
        self.is_reconnecting = False  # 🔒 重连锁，防止重复重连
        self.subscriptions = []  # 保存订阅信息以便重连后恢复
        self.reconnect_task = None
        self.monitor_task = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None  # 🔥 保存事件循环
        self.last_message_time = None  # 最后收到消息的时间
        self.health_check_task = None  # 健康检查任务
        
        # 🔥 使用指数退避重连策略
        self.reconnector = ExponentialBackoffReconnector()
        
        # 💓 心跳保活机制
        self.heartbeat: Optional[WebSocketHeartbeat] = None
        
    def start_websocket(self):
        """启动WebSocket连接"""
        try:
            # 🔥 如果事件循环还未设置，尝试获取当前循环
            if self.loop is None:
                try:
                    self.loop = asyncio.get_running_loop()
                    logger.info("✅ 事件循环已保存")
                except RuntimeError:
                    logger.warning("⚠️ 当前没有运行的事件循环，重连功能可能受限")
            else:
                logger.debug("✅ 使用已设置的事件循环")
            
            # WebSocket: wss://n8n.do2ge.com/tail/ws/relay -> wss://fstream.binance.com
            stream_url = "wss://n8n.do2ge.com/tail/ws/relay"
            
            # 🔒 配置SSL上下文（增强安全性和稳定性）
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            # 禁用旧的不安全协议
            ssl_context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
            # 配置安全密码套件
            ssl_context.set_ciphers('HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4')
            
            # 🔧 配置WebSocket参数
            ws_kwargs = {
                "stream_url": stream_url,
                "on_message": self._on_message,
                "on_error": self._on_error,
                "on_close": self._on_close,
                "on_open": self._on_open,
                "on_ping": self._on_ping,
                "on_pong": self._on_pong,
                "sslopt": {
                    "context": ssl_context,
                    "check_hostname": True,
                    "cert_reqs": ssl.CERT_REQUIRED,
                    "ssl_version": ssl.PROTOCOL_TLS,  # 使用最新TLS版本
                    "timeout": settings.WS_SSL_TIMEOUT  # SSL握手超时
                },
                "timeout": settings.WS_SSL_TIMEOUT,  # 整体超时
                "ping_interval": settings.WS_PING_INTERVAL,  # 启用内置ping
                "ping_timeout": settings.WS_PONG_TIMEOUT
            }
            
            # 添加代理配置（仅在USE_PROXY_WS启用时）
            if settings.USE_PROXY and settings.USE_PROXY_WS:
                # 🔧 WebSocket代理通过环境变量设置（websocket-client库要求）
                proxy_type = settings.PROXY_TYPE.lower()
                proxy_url = f"socks5://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                os.environ['http_proxy'] = proxy_url
                os.environ['https_proxy'] = proxy_url
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url
            elif settings.USE_PROXY and not settings.USE_PROXY_WS:
                logger.info("✅ WebSocket直连（不使用代理），仅REST API使用代理")
            
            self.ws_client = UMFuturesWebsocketClient(**ws_kwargs)
            
            self.is_running = True
            self.connection_start_time = datetime.now()
            self.last_message_time = datetime.now()
            
            # 💓 初始化并启动心跳机制
            if self.heartbeat is None:
                self.heartbeat = WebSocketHeartbeat(self.ws_client)
            asyncio.create_task(self.heartbeat.start())
            
            # 启动连接监控任务（24小时重建连接）
            if self.monitor_task is None or self.monitor_task.done():
                self.monitor_task = asyncio.create_task(self._monitor_connection())
            
            # 启动健康检查任务（检测消息超时）
            if self.health_check_task is None or self.health_check_task.done():
                self.health_check_task = asyncio.create_task(self._health_check())
            
            logger.info(f"WebSocket客户端启动 (URL: {stream_url})")
            
        except Exception as e:
            logger.error(f"启动WebSocket失败: {e}")
            raise
    
    def _on_open(self, ws):
        """WebSocket连接打开"""
        self.is_connected = True
        # 🔥 重置重连器状态
        self.reconnector.reset()
        logger.info("✅ WebSocket连接已建立")
    
    def _on_close(self, ws, close_status_code=None, close_msg=None):
        """WebSocket连接关闭（同步回调，在WebSocket线程）"""
        self.is_connected = False
        logger.warning(f"WebSocket连接关闭: {close_status_code} {close_msg}")
        
        # 如果系统还在运行，且没有正在重连，尝试重连
        if self.is_running and not self.is_reconnecting:
            self.is_reconnecting = True  # 🔒 设置重连锁
            logger.info(f"将在 {self.current_reconnect_delay} 秒后尝试重连...")
            
            # 🔥 使用run_coroutine_threadsafe将重连任务提交到主事件循环
            if self.loop:
                future = asyncio.run_coroutine_threadsafe(self._reconnect(), self.loop)
                # 保存future，避免被GC
                self.reconnect_task = future
                logger.info("✅ 重连任务已提交到主事件循环")
            else:
                logger.error("❌ 事件循环未初始化，无法自动重连！请检查系统状态")
                self.is_reconnecting = False  # 释放锁
        elif self.is_reconnecting:
            logger.debug("重连任务已在进行中，跳过重复重连")
    
    def _on_error(self, ws, error):
        """WebSocket错误（可能不会触发 on_close，需要主动重连）"""
        error_msg = str(error)
        
        # 降低常见错误的日志级别
        if "Lost websocket connection" in error_msg or "Connection to remote host was lost" in error_msg:
            logger.warning(f"⚠️ WebSocket连接丢失: {error_msg}")
        else:
            logger.error(f"❌ WebSocket错误: {error}")
        
        # 标记连接断开
        self.is_connected = False
        
        # 主动触发重连（防止只触发 error 不触发 close 的情况）
        if self.is_running and not self.is_reconnecting:
            self.is_reconnecting = True  # 🔒 设置重连锁
            logger.warning("检测到错误，主动触发重连机制...")
            
            if self.loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(self._reconnect(), self.loop)
                    self.reconnect_task = future
                    logger.info("✅ 重连任务已提交到主事件循环")
                except Exception as e:
                    logger.error(f"❌ 提交重连任务失败: {e}")
                    self.is_reconnecting = False  # 释放锁
            else:
                logger.error("❌ 事件循环未初始化，无法自动重连！")
                self.is_reconnecting = False  # 释放锁
        elif self.is_reconnecting:
            logger.debug("重连任务已在进行中，跳过重复重连")
    
    def _on_ping(self, ws, message):
        """处理WebSocket Ping消息（服务端每3分钟发送）"""
        logger.debug("📥 收到服务端Ping帧（保持连接活跃）")
        # Binance库会自动回复PONG，无需手动处理
    
    def _on_pong(self, ws):
        """处理WebSocket Pong消息"""
        logger.debug("📥 收到服务端Pong帧")
        # 更新最后消息时间（用于健康检查）
        self.last_message_time = datetime.now()
        # 💓 通知心跳机制收到pong
        if self.heartbeat:
            self.heartbeat.on_pong_received()
    
    def _on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            # 更新最后消息时间（用于健康检查）
            self.last_message_time = datetime.now()
            
            data = json.loads(message)
            
            # 🔥 修复：兼容两种消息格式
            # 格式1（多流订阅）: {"stream":"ethusdt@kline_15m", "data":{...}}
            # 格式2（单流订阅）: {"e":"kline", "s":"ETHUSDT", "k":{"i":"15m",...}}
            stream = data.get('stream', '')
            
            if not stream:
                # 没有stream字段，根据消息内容构造
                event_type = data.get('e', '')
                if event_type == 'kline':
                    # K线数据
                    symbol = data.get('s', '').lower()
                    kline_data = data.get('k', {})
                    interval = kline_data.get('i', '')
                    if symbol and interval:
                        stream = f"{symbol}@kline_{interval}"
                        logger.debug(f"📨 收到K线消息，构造stream: {stream}")
                elif event_type == '24hrTicker':
                    # 价格数据
                    symbol = data.get('s', '').lower()
                    if symbol:
                        stream = f"{symbol}@ticker"
                        logger.debug(f"📨 收到价格消息，构造stream: {stream}")
            else:
                logger.debug(f"📨 收到WebSocket消息: stream={stream}")
            
            # 根据流类型调用相应的回调函数
            matched = False
            for pattern, callback in self.callbacks.items():
                if pattern in stream:
                    matched = True
                    logger.debug(f"✓ 匹配回调成功: pattern={pattern}")  # 改为DEBUG，减少日志
                    callback(data)
                    break  # 匹配后退出
            
            if not matched and stream:
                logger.warning(f"⚠️ 未匹配任何回调: stream={stream}")
                logger.warning(f"   已注册的回调: {list(self.callbacks.keys())}")
                logger.warning(f"   消息内容: {json.dumps(data, indent=2)[:200]}")
                    
        except Exception as e:
            logger.error(f"❌ 处理WebSocket消息失败: {e}", exc_info=True)
            logger.error(f"   原始消息: {message[:500]}")
    
    async def _reconnect(self):
        """自动重连（使用指数退避策略）"""
        logger.warning(f"🔄 重连任务开始执行...")
        
        try:
            # 🔥 检查是否应该继续重试
            if not self.reconnector.should_retry():
                logger.error(f"❌ 已达到最大重连次数 ({self.reconnector.max_retries})，停止重连")
                self.is_reconnecting = False
                self.is_running = False
                return
            
            # 🔥 计算并等待重连延迟（指数退避）
            delay = self.reconnector.on_reconnect_attempt()
            logger.info(f"⏱️ 等待 {delay:.1f} 秒后开始重连...")
            await asyncio.sleep(delay)
            
            # 停止旧连接
            if self.ws_client:
                try:
                    logger.info("🛑 停止旧WebSocket连接...")
                    self.ws_client.stop()
                    await asyncio.sleep(0.5)  # 等待连接完全关闭
                    logger.info("✅ 旧连接已停止")
                except Exception as stop_error:
                    logger.warning(f"⚠️ 停止旧连接时出错: {stop_error}")
            
            # 重新启动
            logger.info("🚀 启动新WebSocket连接...")
            self.start_websocket()
            
            # 等待连接建立
            max_wait_time = 10  # 最多等待10秒
            wait_time = 0
            while not self.is_connected and wait_time < max_wait_time:
                await asyncio.sleep(0.5)
                wait_time += 0.5
                if wait_time % 2 == 0:  # 每2秒输出一次等待状态
                    logger.debug(f"⏳ 等待连接建立中... ({wait_time:.1f}s/{max_wait_time}s)")
            
            if self.is_connected:
                logger.info("✅ WebSocket连接已建立，开始恢复订阅...")
                # 再等待一点时间确保连接稳定
                await asyncio.sleep(1)
                self._restore_subscriptions()
                
                # 🔥 记录重连成功
                self.reconnector.on_reconnect_success()
                self.is_reconnecting = False  # 🔓 释放重连锁
                logger.warning("✅ ✅ ✅ WebSocket重连成功！连接已恢复正常 ✅ ✅ ✅")
            else:
                logger.error(f"❌ WebSocket连接建立超时（等待了{max_wait_time}秒），重连失败")
                logger.error(f"   当前状态: is_connected={self.is_connected}, is_running={self.is_running}")
                raise Exception("连接建立超时")
            
        except Exception as e:
            # 🔥 记录重连失败
            self.reconnector.on_reconnect_failure(e)
            
            # 🔄 重连失败后，再次尝试重连
            self.is_reconnecting = False  # 释放锁，允许下次重连
            
            # 再次调度重连任务（如果还在运行且未超过最大次数）
            if self.is_running and self.loop and self.reconnect_count < self.max_reconnect_attempts:
                logger.info(f"📅 调度下次重连... (还剩 {self.max_reconnect_attempts - self.reconnect_count} 次机会)")
                future = asyncio.run_coroutine_threadsafe(self._reconnect(), self.loop)
                self.reconnect_task = future
                logger.info("✅ 下次重连任务已提交")
            elif self.reconnect_count >= self.max_reconnect_attempts:
                logger.error("❌ ❌ ❌ 已达到最大重连次数，停止重连尝试 ❌ ❌ ❌")
                logger.error("   系统将继续运行，但WebSocket数据流已中断")
                self.is_running = False
            else:
                logger.error(f"❌ 无法调度重连: is_running={self.is_running}, loop={self.loop is not None}")
    
    def _restore_subscriptions(self):
        """恢复所有订阅"""
        try:
            logger.info(f"📋 开始恢复 {len(self.subscriptions)} 个订阅...")
            logger.debug(f"   当前状态: ws_client={self.ws_client is not None}, is_connected={self.is_connected}")
            success_count = 0
            failed_subs = []
            
            for sub_info in self.subscriptions:
                try:
                    if sub_info['type'] == 'kline':
                        self._do_subscribe_kline(
                            sub_info['symbol'],
                            sub_info['interval']
                        )
                        success_count += 1
                    elif sub_info['type'] == 'ticker':
                        self._do_subscribe_ticker(sub_info['symbol'])
                        success_count += 1
                    else:
                        logger.warning(f"  ⚠️ 未知订阅类型: {sub_info.get('type')}")
                except Exception as sub_error:
                    logger.error(f"  └─ ❌ 恢复订阅失败: {sub_info}")
                    logger.error(f"     错误类型: {type(sub_error).__name__}")
                    logger.error(f"     错误详情: {sub_error}")
                    logger.error(traceback.format_exc())
                    failed_subs.append(sub_info)
            
            if success_count == len(self.subscriptions):
                logger.info(f"✅ 订阅恢复完成: {success_count}/{len(self.subscriptions)} 全部成功")
            else:
                logger.warning(f"⚠️ 订阅恢复完成: {success_count}/{len(self.subscriptions)} 成功")
                if failed_subs:
                    logger.error(f"  失败列表: {failed_subs}")
                    
        except Exception as e:
            logger.error(f"恢复订阅失败: {e}")
            logger.error(traceback.format_exc())
    
    async def _health_check(self):
        """健康检查（检测消息超时）"""
        # 15m K线周期需要更长的超时时间（至少20分钟）
        message_timeout = 1200  # 20分钟（考虑最长15m周期 + 缓冲）
        warning_timeout = 600  # 10分钟警告（但不重连）
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                if self.is_connected and self.last_message_time:
                    elapsed = (datetime.now() - self.last_message_time).total_seconds()
                    
                    if elapsed > message_timeout:
                        logger.error(f"❌ WebSocket已 {elapsed:.0f} 秒未收到消息，连接异常！")
                        logger.info("🔄 主动触发重连...")
                        
                        # 标记连接断开并触发重连
                        self.is_connected = False
                        if not self.is_reconnecting and self.loop:
                            self.is_reconnecting = True
                            future = asyncio.run_coroutine_threadsafe(self._reconnect(), self.loop)
                            self.reconnect_task = future
                    elif elapsed > warning_timeout:
                        # 只警告，不重连（可能是正常的15m周期等待）
                        logger.debug(f"ℹ️ WebSocket已 {elapsed:.0f} 秒未收到消息（正常，15m周期最长15分钟）")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查异常: {e}")
    
    async def _monitor_connection(self):
        """监控连接状态（每24小时重建连接，Binance要求）"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                
                if self.connection_start_time:
                    elapsed = (datetime.now() - self.connection_start_time).total_seconds()
                    # 23小时后重建连接（预留1小时缓冲）
                    if elapsed > 23 * 3600:
                        logger.info("⏰ WebSocket连接已运行23小时，重建连接...")
                        await self._rebuild_connection()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"连接监控异常: {e}")
    
    async def _rebuild_connection(self):
        """重建连接（定期维护）"""
        try:
            logger.info("🔧 开始重建WebSocket连接（定期维护）...")
            
            # 标记为正在重连，防止其他重连任务干扰
            if self.is_reconnecting:
                logger.warning("已有重连任务在进行，跳过定期重建")
                return
            
            self.is_reconnecting = True
            
            # 停止旧连接
            if self.ws_client:
                self.ws_client.stop()
            
            await asyncio.sleep(2)
            
            # 重新启动
            self.start_websocket()
            
            # 等待连接建立
            max_wait_time = 10
            wait_time = 0
            while not self.is_connected and wait_time < max_wait_time:
                await asyncio.sleep(0.5)
                wait_time += 0.5
            
            if self.is_connected:
                await asyncio.sleep(1)
                self._restore_subscriptions()
                logger.info("✅ WebSocket连接重建完成")
            else:
                logger.error("❌ WebSocket连接重建失败")
            
            self.is_reconnecting = False
            
        except Exception as e:
            logger.error(f"❌ 重建连接失败: {e}")
            self.is_reconnecting = False
    
    def _do_subscribe_kline(self, symbol: str, interval: str):
        """执行K线订阅（内部方法）"""
        if not self.ws_client:
            raise Exception("WebSocket客户端未初始化")
        if not self.is_connected:
            raise Exception("WebSocket未连接")
        
        try:
            self.ws_client.kline(symbol=symbol, interval=interval, id=1)
            logger.info(f"✓ 订阅K线: {symbol} {interval}")
        except Exception as e:
            logger.error(f"✗ 订阅K线失败: {symbol} {interval} - {e}")
            raise  # 🔑 向上抛出，让调用方知道失败
    
    def _do_subscribe_ticker(self, symbol: str):
        """执行价格订阅（内部方法）"""
        if not self.ws_client:
            raise Exception("WebSocket客户端未初始化")
        if not self.is_connected:
            raise Exception("WebSocket未连接")
        
        try:
            self.ws_client.ticker(symbol=symbol, id=2)
            logger.info(f"✓ 订阅价格: {symbol}")
        except Exception as e:
            logger.error(f"✗ 订阅价格失败: {symbol} - {e}")
            raise  # 🔑 向上抛出，让调用方知道失败
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """订阅K线数据"""
        try:
            stream_name = f"{symbol.lower()}@kline_{interval}"
            self.callbacks[stream_name] = callback
            
            # 保存订阅信息以便重连后恢复
            sub_info = {
                'type': 'kline',
                'symbol': symbol,
                'interval': interval
            }
            if sub_info not in self.subscriptions:
                self.subscriptions.append(sub_info)
            
            self._do_subscribe_kline(symbol, interval)
                
        except Exception as e:
            logger.error(f"订阅K线数据失败: {e}")
    
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """订阅价格变动数据"""
        try:
            stream_name = f"{symbol.lower()}@ticker"
            self.callbacks[stream_name] = callback
            
            # 保存订阅信息以便重连后恢复
            sub_info = {
                'type': 'ticker',
                'symbol': symbol
            }
            if sub_info not in self.subscriptions:
                self.subscriptions.append(sub_info)
            
            self._do_subscribe_ticker(symbol)
                
        except Exception as e:
            logger.error(f"订阅价格数据失败: {e}")
    
    def stop_websocket(self):
        """停止WebSocket连接"""
        try:
            logger.info("🛑 正在停止WebSocket连接...")
            self.is_running = False
            
            # 💓 停止心跳任务
            if self.heartbeat:
                asyncio.create_task(self.heartbeat.stop())
                logger.debug("心跳任务已取消")
            
            # 取消健康检查任务
            if self.health_check_task and not self.health_check_task.done():
                self.health_check_task.cancel()
                logger.debug("健康检查任务已取消")
            
            # 取消监控任务
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                logger.debug("连接监控任务已取消")
            
            # 取消重连任务
            if self.reconnect_task and not self.reconnect_task.done():
                self.reconnect_task.cancel()
                logger.debug("重连任务已取消")
            
            # 停止WebSocket
            if self.ws_client:
                self.ws_client.stop()
                self.is_connected = False
                logger.info("✅ WebSocket连接已停止")
        except Exception as e:
            logger.error(f"❌ 停止WebSocket失败: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        # 🔥 获取重连器统计信息
        reconnect_stats = self.reconnector.get_statistics()
        
        stats = {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'is_reconnecting': self.is_reconnecting,
            'subscriptions_count': len(self.subscriptions),
            'callbacks_count': len(self.callbacks),
            'reconnect_statistics': reconnect_stats
        }
        
        if self.reconnector.connection_start_time:
            uptime = (datetime.now() - self.reconnector.connection_start_time).total_seconds()
            stats['uptime_seconds'] = uptime
            stats['uptime_hours'] = uptime / 3600
        
        if self.last_message_time:
            idle_time = (datetime.now() - self.last_message_time).total_seconds()
            stats['last_message_seconds_ago'] = idle_time
        
        return stats

# 全局客户端实例
binance_client = BinanceClient()
binance_ws_client = BinanceWebSocketClient()