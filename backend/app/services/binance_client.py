"""
Binance API客户端
"""
import asyncio
import logging
import traceback
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
import json
import time
import hmac
import hashlib
import requests
import os
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
import websocket

from app.core.config import settings

logger = logging.getLogger(__name__)

class BinanceClient:
    """Binance API客户端"""
    
    def __init__(self):
        self.api_key = settings.BINANCE_API_KEY
        self.secret_key = settings.BINANCE_SECRET_KEY
        self.testnet = settings.BINANCE_TESTNET
        
        # 配置代理地址
        # REST API: https://n8n.do2ge.com/tail/http/relay/fapi/v1/... -> https://fapi.binance.com/fapi/v1/...
        self.base_url = "https://n8n.do2ge.com/tail/http/relay"
        
        # REST API客户端
        self.client = UMFutures(
            key=self.api_key,
            secret=self.secret_key,
            base_url=self.base_url,
            timeout=30  # 增加超时时间
        )
        
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
    ) -> List[Dict[str, Any]]:
        """获取K线数据"""
        try:
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
            
            # 转换为标准格式
            formatted_klines = []
            for kline in klines:
                formatted_kline = {
                    'timestamp': kline[0],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': kline[6],
                    'quote_volume': float(kline[7]),
                    'trades': int(kline[8]),
                    'taker_buy_base_volume': float(kline[9]),
                    'taker_buy_quote_volume': float(kline[10])
                }
                formatted_klines.append(formatted_kline)
            
            logger.debug(f"获取K线数据: {symbol} {interval} {len(formatted_klines)}条")
            return formatted_klines
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            account = self.client.account(recvWindow=self.recv_window)
            
            # 格式化账户信息
            formatted_account = {
                'total_wallet_balance': float(account.get('totalWalletBalance', 0)),
                'total_unrealized_pnl': float(account.get('totalUnrealizedPnL', 0)),
                'total_margin_balance': float(account.get('totalMarginBalance', 0)),
                'total_position_initial_margin': float(account.get('totalPositionInitialMargin', 0)),
                'total_open_order_initial_margin': float(account.get('totalOpenOrderInitialMargin', 0)),
                'available_balance': float(account.get('availableBalance', 0)),
                'max_withdraw_amount': float(account.get('maxWithdrawAmount', 0)),
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
            params = {'recvWindow': self.recv_window}
            if symbol:
                params['symbol'] = symbol
            
            positions = self.client.get_position_risk(**params)
            
            # 过滤有持仓的合约
            active_positions = []
            for position in positions:
                position_amt = float(position.get('positionAmt', 0))
                if position_amt != 0:
                    formatted_position = {
                        'symbol': position['symbol'],
                        'position_amt': position_amt,
                        'entry_price': float(position.get('entryPrice', 0)),
                        'mark_price': float(position.get('markPrice', 0)),
                        'pnl': float(position.get('unRealizedProfit', 0)),
                        'percentage': float(position.get('percentage', 0)),
                        'position_side': position.get('positionSide', 'BOTH'),
                        'isolated': position.get('isolated', False),
                        'margin_type': position.get('marginType', 'cross'),
                        'leverage': int(position.get('leverage', 1)),
                        'max_notional_value': float(position.get('maxNotionalValue', 0)),
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
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """撤销订单"""
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id, recvWindow=self.recv_window)
            logger.info(f"撤销订单成功: {symbol} {order_id}")
            return result
        except Exception as e:
            logger.error(f"撤销订单失败: {e}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        try:
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
            result = self.client.change_leverage(symbol=symbol, leverage=leverage, recvWindow=self.recv_window)
            logger.info(f"修改杠杆成功: {symbol} {leverage}x")
            return result
        except Exception as e:
            logger.error(f"修改杠杆失败: {e}")
            return {}
    
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """修改保证金模式（可能已设置，失败不影响）"""
        try:
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
        self.reconnect_delay = 5  # 重连延迟（秒）
        self.max_reconnect_delay = 60  # 最大重连延迟（秒）
        self.current_reconnect_delay = 5
        self.reconnect_task = None
        self.is_reconnecting = False  # 🔒 重连锁，防止重复重连
        self.subscriptions = []  # 保存订阅信息以便重连后恢复
        self.connection_start_time = None
        self.monitor_task = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None  # 🔥 保存事件循环
        self.reconnect_count = 0  # 重连次数统计
        self.max_reconnect_attempts = 10  # 最大连续重连次数
        self.last_message_time = None  # 最后收到消息的时间
        self.health_check_task = None  # 健康检查任务
        
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
            
            self.ws_client = UMFuturesWebsocketClient(
                stream_url=stream_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
                on_ping=self._on_ping,
                on_pong=self._on_pong
            )
            
            self.is_running = True
            self.connection_start_time = datetime.now()
            self.last_message_time = datetime.now()
            
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
        self.reconnect_count = 0  # 重置重连计数
        self.current_reconnect_delay = self.reconnect_delay  # 重置重连延迟
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
        """自动重连"""
        # ✅ 立即输出日志，确认重连任务已开始执行
        logger.warning(f"🔄 重连任务开始执行 (当前重连次数: {self.reconnect_count})...")
        
        try:
            # 检查是否超过最大重连次数
            self.reconnect_count += 1
            if self.reconnect_count > self.max_reconnect_attempts:
                logger.error(f"❌ 已达到最大重连次数 ({self.max_reconnect_attempts})，停止重连")
                self.is_reconnecting = False
                self.is_running = False
                return
            
            logger.info(f"🔌 尝试重新建立WebSocket连接 (第 {self.reconnect_count}/{self.max_reconnect_attempts} 次)...")
            logger.info(f"⏱️ 等待 {self.current_reconnect_delay} 秒后开始重连...")
            await asyncio.sleep(self.current_reconnect_delay)
            
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
            
            # 🔥 等待连接建立，增加重试机制和详细日志
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
                
                # 重置重连延迟和计数
                self.current_reconnect_delay = self.reconnect_delay
                self.reconnect_count = 0
                self.is_reconnecting = False  # 🔓 释放重连锁
                logger.warning("✅ ✅ ✅ WebSocket重连成功！连接已恢复正常 ✅ ✅ ✅")
            else:
                logger.error(f"❌ WebSocket连接建立超时（等待了{max_wait_time}秒），重连失败")
                logger.error(f"   当前状态: is_connected={self.is_connected}, is_running={self.is_running}")
                raise Exception("连接建立超时")
            
        except Exception as e:
            logger.error(f"❌ WebSocket重连失败 (第 {self.reconnect_count}/{self.max_reconnect_attempts} 次): {e}")
            logger.error(f"   错误类型: {type(e).__name__}")
            
            # 指数退避，增加重连延迟
            self.current_reconnect_delay = min(
                self.current_reconnect_delay * 2,
                self.max_reconnect_delay
            )
            logger.warning(f"⏱️ 下次重连延迟: {self.current_reconnect_delay}秒 (指数退避策略)")
            
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
        stats = {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'is_reconnecting': self.is_reconnecting,
            'reconnect_count': self.reconnect_count,
            'current_reconnect_delay': self.current_reconnect_delay,
            'subscriptions_count': len(self.subscriptions),
            'callbacks_count': len(self.callbacks)
        }
        
        if self.connection_start_time:
            uptime = (datetime.now() - self.connection_start_time).total_seconds()
            stats['uptime_seconds'] = uptime
            stats['uptime_hours'] = uptime / 3600
        
        if self.last_message_time:
            idle_time = (datetime.now() - self.last_message_time).total_seconds()
            stats['last_message_seconds_ago'] = idle_time
        
        return stats

# 全局客户端实例
binance_client = BinanceClient()
binance_ws_client = BinanceWebSocketClient()