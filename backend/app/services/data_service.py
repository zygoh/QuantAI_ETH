"""
数据获取服务
"""
import asyncio
import logging
import random
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass

from app.core.config import settings
from app.core.database import postgresql_manager
from app.core.cache import cache_manager
from app.exchange.exchange_factory import ExchangeFactory
from app.exchange.mappers import SymbolMapper

logger = logging.getLogger(__name__)


@dataclass
class KlineData:
    """K线数据模型（时间戳保持为Binance原始格式）"""
    symbol: str
    interval: str
    open_time: int  # ✅ 毫秒时间戳（UTC），不转换
    close_time: int  # ✅ 毫秒时间戳（UTC），不转换
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    trades: int = 0
    taker_buy_base_volume: float = 0.0  # ✅ 主动买入成交量
    taker_buy_quote_volume: float = 0.0  # ✅ 主动买入成交额
    is_closed: bool = False  # 🔑 K线是否完成（修复预测频率问题）


class DataService:
    """数据获取服务"""
    
    def __init__(self):
        self.is_running = False
        self.subscriptions: Dict[str, bool] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # 秒
        
        # 🔥 保存主事件循环引用（用于WebSocket回调）
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 数据回调函数
        self.data_callbacks: List[Callable] = []
        
        # 🆕 价格更新回调函数（用于虚拟仓位止损止盈监控）
        self.price_callbacks: List[Callable] = []
        
        # WebSocket重连回调函数
        self.reconnect_callbacks: List[Callable] = []
        
        # WebSocket状态监控
        self._last_connection_state = False  # 上次连接状态
        self._monitor_task = None  # 监控任务
        
        # 🔑 获取交易所客户端（使用工厂模式）
        self.exchange_client = ExchangeFactory.get_current_client()

        # 🔑 WebSocket客户端（根据交易所类型动态获取）
        self.ws_client = None

    async def start(self):
        """启动数据服务"""
        try:
            logger.info("启动数据获取服务...")
            
            # ✅ 显式输出交易所客户端初始化状态（确保日志可见）
            exchange_type = settings.EXCHANGE_TYPE
            logger.info(f"✅ {exchange_type}客户端初始化完成")
            logger.info(f"   - 交易所类型: {exchange_type}")
            
            # 🔥 保存当前事件循环（用于WebSocket回调）
            self.loop = asyncio.get_running_loop()
            
            # 测试API连接
            if not await self.exchange_client.test_connection():
                raise Exception(f"{exchange_type} API连接失败")

            # 🔑 根据交易所类型初始化WebSocket客户端
            if exchange_type == "BINANCE":
                from app.exchange.binance_client import binance_ws_client
                self.ws_client = binance_ws_client
            elif exchange_type == "OKX":
                from app.exchange.okx_client import OKXWebSocketClient
                self.ws_client = OKXWebSocketClient()
            else:
                logger.warning(f"⚠️ {exchange_type}暂不支持WebSocket，仅使用REST API")
                self.ws_client = None
            
            # 设置杠杆
            await self._setup_leverage()
            
            # 启动WebSocket连接
            await self._start_websocket()
            
            # 订阅数据流
            await self._subscribe_data_streams()
            
            # 注释：历史数据由模型训练时获取并写入（90天完整数据）
            # 这里不再重复获取，避免数据不一致和冗余操作
            # await self._fetch_historical_data()
            
            self.is_running = True
            
            # 启动WebSocket连接监控（检测重连事件）
            if self.ws_client and hasattr(self.ws_client, 'is_connected'):
                self._last_connection_state = self.ws_client.is_connected
            else:
                self._last_connection_state = False
            self._monitor_task = asyncio.create_task(self._monitor_websocket_connection())
            
            logger.info("数据获取服务启动完成")
            
        except Exception as e:
            logger.error(f"启动数据服务失败: {e}")
            raise
    
    async def stop(self):
        """停止数据服务"""
        try:
            logger.info("停止数据获取服务...")
            
            self.is_running = False
            
            # 停止监控任务
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            # 停止WebSocket连接
            if self.ws_client and hasattr(self.ws_client, 'stop_websocket'):
                self.ws_client.stop_websocket()
            
            logger.info("数据获取服务已停止")
            
        except Exception as e:
            logger.error(f"停止数据服务失败: {e}")
    
    async def _setup_leverage(self):
        """设置杠杆（可选，失败不影响系统运行）"""
        try:
            symbol = settings.SYMBOL
            leverage = settings.LEVERAGE
            
            # 尝试修改保证金模式为全仓（可能已经是全仓模式，失败不影响）
            try:
                self.exchange_client.change_margin_type(symbol, "CROSSED")
                logger.info(f"✓ 保证金模式设置成功: {symbol} CROSSED")
            except Exception as e:
                logger.warning(f"⚠️ 保证金模式设置失败（可能已是全仓模式，可忽略）: {e}")
            
            # 设置杠杆倍数
            try:
                result = self.exchange_client.change_leverage(symbol, leverage)
                if result:
                    logger.info(f"✓ 杠杆设置成功: {symbol} {leverage}x")
                else:
                    logger.warning(f"⚠️ 杠杆设置返回空结果（可能已设置，可忽略）")
            except Exception as e:
                logger.warning(f"⚠️ 杠杆设置失败（可能已设置，可忽略）: {e}")
                
        except Exception as e:
            logger.warning(f"⚠️ 杠杆设置过程出现异常（不影响系统运行）: {e}")
    
    async def _start_websocket(self):
        """启动WebSocket连接"""
        try:
            if not self.ws_client:
                logger.warning("⚠️ WebSocket客户端未初始化，跳过WebSocket连接")
                return

            # 🔥 传递事件循环给WebSocket客户端（用于重连）
            if hasattr(self.ws_client, 'loop'):
                self.ws_client.loop = asyncio.get_running_loop()
            
            # 启动WebSocket连接
            if hasattr(self.ws_client, 'start_websocket'):
                self.ws_client.start_websocket()
            else:
                logger.warning("⚠️ WebSocket客户端不支持start_websocket方法")
                return
            
            # 等待连接建立
            for i in range(10):
                if hasattr(self.ws_client, 'is_connected') and self.ws_client.is_connected:
                    break
                await asyncio.sleep(1)
            
            if hasattr(self.ws_client, 'is_connected') and not self.ws_client.is_connected:
                raise Exception("WebSocket连接超时")
                
        except Exception as e:
            logger.error(f"启动WebSocket失败: {e}")
            raise
    
    async def _subscribe_data_streams(self):
        """订阅数据流"""
        try:
            if not self.ws_client:
                logger.warning("⚠️ WebSocket客户端未初始化，跳过数据流订阅")
                return

            symbol = settings.SYMBOL
            timeframes = settings.TIMEFRAMES
            
            # 订阅K线数据
            for interval in timeframes:
                if hasattr(self.ws_client, 'subscribe_kline'):
                    self.ws_client.subscribe_kline(
                        symbol, 
                        interval, 
                        self._on_kline_data
                    )
                    self.subscriptions[f"{symbol}_{interval}"] = True
                else:
                    logger.warning(f"⚠️ WebSocket客户端不支持subscribe_kline方法")
            
            # 订阅价格变动数据
            if hasattr(self.ws_client, 'subscribe_ticker'):
                self.ws_client.subscribe_ticker(symbol, self._on_ticker_data)
            
            logger.info(f"数据流订阅完成: {symbol} {timeframes}")
            
        except Exception as e:
            logger.error(f"订阅数据流失败: {e}")
            raise
    
    def _on_kline_data(self, data: Any, symbol: Optional[str] = None, interval: Optional[str] = None):
        """
        处理K线数据（支持Binance和OKX格式）
        
        Args:
            data: K线数据，格式因交易所而异：
                  - Binance: {"e":"kline", "k":{...}} 或 {"data": {...}}
                  - OKX: [[timestamp, open, high, low, close, volume, quote_volume, is_closed], ...]
            symbol: 交易对（OKX格式需要，Binance格式从data中提取）
            interval: 时间框架（OKX格式需要，Binance格式从data中提取）
        """
        try:
            logger.debug(f"📥 _on_kline_data被调用: data类型={type(data)}, symbol={symbol}, interval={interval}, data长度={len(data) if isinstance(data, list) else 'N/A'}")
            
            # 🔥 自动检测数据格式：OKX格式是数组，Binance格式是字典
            if isinstance(data, list):
                # OKX格式：数组格式
                if not symbol or not interval:
                    logger.error("❌ OKX格式K线数据缺少symbol或interval参数")
                    return
                
                if not data or len(data) == 0:
                    logger.debug("❌ OKX K线数据为空")
                    return
                
                # 取最新的一条K线（数组第一个元素）
                kline_array = data[0] if isinstance(data[0], list) else data
                
                if len(kline_array) < 9:
                    logger.error(f"❌ OKX K线数组长度不足: {len(kline_array)} < 9")
                    return
                
                # 🔥 OKX格式：[timestamp, open, high, low, close, volume, volCcyQuote, volCcy, confirm]
                # 注意：数组有9个元素，confirm是最后一个（索引8），confirm=1表示K线已完成
                timestamp = int(kline_array[0])
                open_price = float(kline_array[1])
                high_price = float(kline_array[2])
                low_price = float(kline_array[3])
                close_price = float(kline_array[4])
                volume = float(kline_array[5])
                quote_volume = float(kline_array[6])  # volCcyQuote
                # kline_array[7] 是 volCcy（另一个字段，我们不需要）
                confirm = kline_array[8]  # OKX使用confirm字段（索引8）
                is_closed = (str(confirm) == "1" or confirm == 1)  # confirm=1表示已完成
                
                # 计算close_time（OKX不提供，需要根据interval计算）
                from app.exchange.mappers import IntervalMapper
                okx_interval = IntervalMapper.to_exchange_format(interval, "OKX")
                interval_ms = self._interval_to_ms(interval)
                close_time = timestamp + interval_ms - 1
                
                logger.debug(f"✅ 处理OKX K线: {symbol} {interval} is_closed={is_closed} close={close_price}")
                
                # 验证数据
                if close_price <= 0:
                    logger.error(f"❌ 收到无效OKX K线数据: {symbol} {interval} close={close_price}")
                    return
                
                # 创建K线数据对象
                kline = KlineData(
                    symbol=symbol,
                    interval=interval,
                    open_time=timestamp,
                    close_time=close_time,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume,
                    quote_volume=quote_volume,
                    trades=0,  # OKX不提供
                    taker_buy_base_volume=0.0,  # OKX不提供
                    taker_buy_quote_volume=0.0,  # OKX不提供
                    is_closed=is_closed
                )
                
                # 只处理已完成的K线
                if not is_closed:
                    logger.debug(f"⏸️ 跳过未完成OKX K线: {symbol} {interval}")
                    return
                
            else:
                # Binance格式：字典格式
                kline_data = data.get('data', data) if isinstance(data, dict) else data
                
                k = kline_data.get('k', {}) if isinstance(kline_data, dict) else {}
                if not k:
                    logger.debug("❌ WebSocket消息无k字段（非Binance格式）")
                    return
                
                symbol = k.get('s', 'UNKNOWN')
                interval = k.get('i', 'UNKNOWN')
                is_closed = k.get('x', False)
                
                # 🔑 增强日志验证（新增）
                logger.debug(f"📥 收到Binance K线: {symbol} {interval} is_closed={is_closed} t={k.get('t')} c={k.get('c')}")
                
                # 只处理已完成的K线
                if not is_closed:
                    logger.debug(f"⏸️ 跳过未完成Binance K线: {symbol} {interval}")
                    return
                
                # 已完成的K线
                logger.debug(f"✅ 处理Binance K线: {symbol} {interval} close={k.get('c')}")
                
                # ✅ 关键修复：数据质量验证（防止close/volume为0）- 增强诊断
                close_price = float(k['c'])
                volume = float(k['v'])
                open_price = float(k['o'])
                high_price = float(k['h'])
                low_price = float(k['l'])
                
                # ✅ 详细诊断：记录原始接收到的数据
                logger.debug(f"📥 原始Binance K线数据: {symbol} {interval}")
                logger.debug(f"   open={open_price}, high={high_price}, low={low_price}, close={close_price}, volume={volume}")
                logger.debug(f"   时间戳: t={k.get('t')}, T={k.get('T')}, is_closed={is_closed}")
                
                # ✅ 关键诊断：检查V和Q字段是否存在（taker buy volume）
                has_V = 'V' in k
                has_Q = 'Q' in k
                V_value = k.get('V', None)
                Q_value = k.get('Q', None)
                logger.debug(f"   taker_buy字段检查: V存在={has_V}, Q存在={has_Q}, V值={V_value}, Q值={Q_value}")
                logger.debug(f"   k对象所有字段: {list(k.keys())}")
                if not has_V or not has_Q:
                    logger.warning(f"⚠️ Binance WebSocket K线数据缺少taker_buy字段: V={has_V}, Q={has_Q}")
                    logger.warning(f"   可用字段: {list(k.keys())}")
                    logger.warning(f"   完整k对象: {k}")
                
                # 验证价格数据
                if close_price <= 0:
                    logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    logger.error(f"❌ 收到无效Binance K线数据: {symbol} {interval} close={close_price}（价格不应为0或负数）")
                    logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    logger.error(f"   完整K线数据:")
                    logger.error(f"      open={open_price}, high={high_price}, low={low_price}, close={close_price}")
                    logger.error(f"      volume={volume}, quote_volume={k.get('q', 'N/A')}")
                    logger.error(f"      trades={k.get('n', 'N/A')}, is_closed={is_closed}")
                    logger.error(f"      时间戳: t={k.get('t')}, T={k.get('T')}")
                    logger.error(f"   原始JSON数据（前1000字符）: {str(k)[:1000]}")
                    logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    return  # 跳过无效数据
                
                if volume < 0:
                    logger.warning(f"⚠️ 收到异常Binance K线数据: {symbol} {interval} volume={volume}（成交量不应为负数）")
                    logger.warning(f"   完整数据: o={open_price}, h={high_price}, l={low_price}, c={close_price}, v={volume}")
                    volume = 0  # 设为0而不是负数
                
                # ✅ 详细诊断：检查数据合理性
                if close_price < low_price or close_price > high_price:
                    logger.warning(f"⚠️ 价格数据异常: close={close_price}不在[low={low_price}, high={high_price}]范围内")
                
                if high_price < low_price:
                    logger.error(f"❌ 价格数据严重异常: high={high_price} < low={low_price}")
                
                if volume == 0:
                    logger.debug(f"   ℹ️ volume=0（可能是极低流动性时段，但会导致pct_change产生inf）")
                
                # 创建K线数据对象（保留Binance原始时间戳，不转换）
                kline = KlineData(
                    symbol=symbol,
                    interval=interval,
                    open_time=k['t'],  # ✅ 保留毫秒时间戳（整数）
                    close_time=k['T'],  # ✅ 保留毫秒时间戳（整数）
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume,
                    quote_volume=float(k['q']),
                    trades=int(k['n']),
                    taker_buy_base_volume=float(k.get('V', 0)),  # ✅ 主动买入量
                    taker_buy_quote_volume=float(k.get('Q', 0)),  # ✅ 主动买入额
                    is_closed=True  # 🔑 K线已完成（只处理已完成的K线）
                )
            
            # 🔥 直接通知回调函数（signal_generator），不需要额外处理
            # 删除了不必要的Redis缓存和数据库写入
            if self.loop:
                if not self.data_callbacks:
                    logger.warning("⚠️ 没有注册的数据回调函数，K线数据将被丢弃")
                else:
                    # 🔥 只在K线完成时记录INFO，其他时候用DEBUG
                    if kline.is_closed:
                        logger.info(f"📤 通知 {len(self.data_callbacks)} 个数据回调: {kline.symbol} {kline.interval} ✅已完成")
                    else:
                        logger.debug(f"📤 通知 {len(self.data_callbacks)} 个数据回调: {kline.symbol} {kline.interval} ⏳进行中")
                    for idx, callback in enumerate(self.data_callbacks):
                        try:
                            logger.debug(f"   调用回调 {idx+1}/{len(self.data_callbacks)}: {callback.__name__ if hasattr(callback, '__name__') else type(callback).__name__}")
                            asyncio.run_coroutine_threadsafe(
                                callback(kline),
                                self.loop
                            )
                        except Exception as e:
                            logger.error(f"   ❌ 回调 {idx+1} 调用失败: {e}")
            else:
                logger.warning("⚠️ 事件循环未初始化，跳过K线处理")
            
        except Exception as e:
            logger.error(f"❌ 处理K线数据失败: {e}", exc_info=True)
    

    def _on_ticker_data(self, data: Any):
        """
        处理价格变动数据（支持多交易所格式）
        
        Args:
            data: WebSocket返回的数据，格式因交易所而异：
                  - Binance: {"e":"24hrTicker", "s":"ETHUSDT", "c":"2000.5", ...} 或 {"stream":"...", "data":{...}}
                  - OKX: [{"instId": "ETH-USDT-SWAP", "last": "2000.5", ...}] 或 {"data": [...]}
        """
        try:
            ticker_item = None
            symbol = None
            price = None
            
            # 🔧 步骤1: 提取ticker数据项（处理不同数据结构）
            if isinstance(data, list):
                # OKX格式：直接传递的列表
                if not data:
                    return
                ticker_item = data[0]
            elif isinstance(data, dict):
                # 检查是否是Binance多流订阅格式: {"stream":"...", "data":{...}}
                if 'data' in data and isinstance(data['data'], dict):
                    # Binance多流格式：使用data字段
                    ticker_item = data['data']
                elif 'data' in data and isinstance(data['data'], list):
                    # OKX格式：包含data数组的字典
                    if not data['data']:
                        return
                    ticker_item = data['data'][0]
                elif 'e' in data and data.get('e') == '24hrTicker':
                    # Binance单流格式：直接是ticker消息
                    ticker_item = data
                elif 's' in data and 'c' in data:
                    # Binance格式：有s和c字段
                    ticker_item = data
                elif 'instId' in data and 'last' in data:
                    # OKX格式：直接是ticker对象
                    ticker_item = data
                else:
                    logger.warning(f"⚠️ 无法识别的ticker数据格式: {list(data.keys())}")
                    return
            else:
                logger.warning(f"⚠️ 未知的ticker数据格式: {type(data)}")
                return
            
            if not ticker_item:
                return
            
            # 🔧 步骤2: 根据字段名自动识别交易所格式并提取数据
            # 优先检查Binance格式（字段：s, c）
            if 's' in ticker_item and 'c' in ticker_item:
                # Binance格式
                binance_symbol = ticker_item.get('s', '')
                if not binance_symbol:
                    logger.warning("⚠️ ticker数据中缺少symbol字段(s)")
                    return
                
                # 转换为标准格式（ETHUSDT -> ETH/USDT）
                symbol = SymbolMapper.to_standard_format(binance_symbol, "BINANCE")
                
                # 获取最新价格
                price_str = ticker_item.get('c', '0')
                try:
                    price = float(price_str)
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ 无法解析价格: {price_str}")
                    return
                    
            # 检查OKX格式（字段：instId, last）
            elif 'instId' in ticker_item and 'last' in ticker_item:
                # OKX格式
                okx_symbol = ticker_item.get('instId', '')
                if not okx_symbol:
                    logger.warning("⚠️ ticker数据中缺少instId字段")
                    return
                
                # 转换为标准格式（ETH-USDT-SWAP -> ETH/USDT）
                symbol = SymbolMapper.to_standard_format(okx_symbol, "OKX")
                
                # 获取最新价格
                price_str = ticker_item.get('last', '0')
                try:
                    price = float(price_str)
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ 无法解析价格: {price_str}")
                    return
            else:
                # 无法识别的格式
                logger.warning(f"⚠️ 无法识别的ticker数据格式，字段: {list(ticker_item.keys())}")
                return
            
            # 🔧 步骤3: 验证数据有效性
            if not symbol:
                logger.warning("⚠️ 无法提取交易对符号")
                return
                
            if price is None or price <= 0:
                logger.warning(f"⚠️ 价格无效: {price}")
                return
            
            # 🔧 缓存最新价格（使用run_coroutine_threadsafe，因为WebSocket回调不在异步上下文）
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                cache_manager.set_market_data(
                    symbol, 
                    "ticker", 
                    {
                        "price": price,
                        "timestamp": datetime.now().isoformat()
                    },
                    expire=30
                    ),
                    self.loop
                )
            else:
                logger.warning("⚠️ 事件循环未初始化，跳过价格缓存")
            
            # 🆕 通知价格更新回调（用于虚拟仓位止损止盈检查）
            if self.loop and self.price_callbacks:
                for callback in self.price_callbacks:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            callback(symbol, price),
                            self.loop
                        )
                    except Exception as e:
                        logger.error(f"执行价格更新回调失败: {e}", exc_info=True)
            
            # 🔥 添加调试日志（每100次记录一次，避免日志过多）
            if random.random() < 0.01:  # 1%的概率记录调试日志
                logger.debug(f"📊 价格更新: {symbol} @{price:.2f}, 回调数: {len(self.price_callbacks)}")
            
        except Exception as e:
            logger.error(f"处理价格数据失败: {e}", exc_info=True)
    

    # ✅ 已删除 _process_kline_data 方法
    # 理由：
    # 1. Redis缓存K线数据无实际用途（前端查数据库）
    # 2. 数据库写入由 signal_generator 负责
    # 3. callback通知已在 _on_kline_data 中完成
    
    async def _fetch_historical_data(self):
        """获取历史数据"""
        try:
            symbol = settings.SYMBOL
            timeframes = settings.TIMEFRAMES
            
            for interval in timeframes:
                await self._fetch_historical_klines(symbol, interval)
            
            logger.info("历史数据获取完成")
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
    

    async def _fetch_historical_klines(self, symbol: str, interval: str, limit: int = 1000):
        """获取历史K线数据"""
        try:
            # ✅ 统一使用分页方法（自动处理超过1500的情况）
            klines = self.exchange_client.get_klines_paginated(symbol, interval, limit)
            
            if not klines:
                logger.warning(f"未获取到历史数据: {symbol} {interval}")
                return
            
            # 批量写入数据库（添加 symbol 和 interval）
            klines_with_meta = [
                {**kline, 'symbol': symbol, 'interval': interval}
                for kline in klines
            ]
            await postgresql_manager.write_kline_data(klines_with_meta)
            
            logger.info(f"历史K线数据获取完成: {symbol} {interval} {len(klines)}条")
            
        except Exception as e:
            logger.error(f"获取历史K线数据失败: {e}")
    

    async def get_latest_klines(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取最新K线数据（仅供前端API调用，模型训练和信号生成不使用此方法）"""
        try:
            # 先尝试从缓存获取
            cached_data = await cache_manager.get_market_data(symbol, interval)
            
            # 从数据库查询（供前端展示）
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit)
            
            df = await postgresql_manager.query_kline_data(
                symbol, interval, start_time, end_time, limit
            )
            
            if df.empty:
                # 如果数据库没有数据，从API获取
                logger.debug(f"数据库无数据，从API获取: {symbol} {interval}")
                # ✅ 统一使用分页方法（自动处理超过1500的情况）
                klines = self.exchange_client.get_klines_paginated(symbol, interval, limit)
                # 🔥 转换为字典列表（UnifiedKlineData对象转换为字典）
                klines_dict = []
                for kline in klines:
                    if isinstance(kline, dict):
                        klines_dict.append(kline)
                    else:
                        # UnifiedKlineData对象转换为字典
                        klines_dict.append({
                            'timestamp': kline.timestamp,
                            'open': kline.open,
                            'high': kline.high,
                            'low': kline.low,
                            'close': kline.close,
                            'volume': kline.volume,
                            'close_time': kline.close_time,
                            'quote_volume': kline.quote_volume,
                            'trades': kline.trades,
                            'taker_buy_base_volume': kline.taker_buy_base_volume,
                            'taker_buy_quote_volume': kline.taker_buy_quote_volume
                        })
                return klines_dict
            
            # 转换为字典列表
            klines = []
            for _, row in df.iterrows():
                kline = {
                    "timestamp": int(row['timestamp'].timestamp() * 1000),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "quote_volume": float(row['quote_volume'])
                }
                klines.append(kline)
            
            return klines
            
        except Exception as e:
            logger.error(f"获取最新K线数据失败: {e}")
            return []
    

    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            # 先尝试从缓存获取
            cached_info = await cache_manager.get_account_info()
            if cached_info:
                return cached_info
            
            # 从API获取
            account_info = self.exchange_client.get_account_info()
            
            # 缓存结果
            if account_info:
                await cache_manager.set_account_info(account_info, expire=30)
            
            return account_info
            
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return {}
    

    async def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        try:
            # 先尝试从缓存获取
            cached_positions = await cache_manager.get_position_info()
            if cached_positions:
                if symbol:
                    return [pos for pos in cached_positions if pos['symbol'] == symbol]
                return cached_positions
            
            # 从API获取
            positions = self.exchange_client.get_position_info(symbol)
            
            # 缓存结果
            if positions:
                await cache_manager.set_position_info(positions, expire=30)
            
            return positions
            
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
            return []
    

    def add_data_callback(self, callback: Callable):
        """添加数据回调函数"""
        self.data_callbacks.append(callback)
        logger.info(f"✅ 注册K线数据回调: {callback.__name__ if hasattr(callback, '__name__') else type(callback).__name__}, 当前回调数: {len(self.data_callbacks)}")
    

    def add_price_callback(self, callback: Callable):
        """添加价格更新回调函数（用于虚拟仓位止损止盈监控）"""
        self.price_callbacks.append(callback)
        logger.debug(f"注册价格更新回调: {callback.__name__}")
    

    def add_reconnect_callback(self, callback: Callable):
        """添加WebSocket重连回调函数"""
        self.reconnect_callbacks.append(callback)
        logger.debug(f"注册WebSocket重连回调: {callback.__name__}")
    

    async def _notify_reconnect(self):
        """通知所有注册的重连回调"""
        try:
            logger.info(f"🔄 通知 {len(self.reconnect_callbacks)} 个重连回调...")
            for callback in self.reconnect_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"执行重连回调失败: {e}")
            logger.info("✅ 重连回调通知完成")
        except Exception as e:
            logger.error(f"通知重连回调失败: {e}")
    

    async def _monitor_websocket_connection(self):
        """监控WebSocket连接状态，检测重连事件"""
        try:
            logger.info("启动WebSocket连接状态监控...")
            
            while self.is_running:
                try:
                    # 获取当前连接状态
                    if self.ws_client and hasattr(self.ws_client, 'is_connected'):
                        current_state = self.ws_client.is_connected
                    else:
                        current_state = False
                    
                    # 检测状态变化：从断开到连接（重连成功）
                    if not self._last_connection_state and current_state:
                        logger.info("🔔 检测到WebSocket重连成功！")
                        # 通知所有注册的重连回调
                        await self._notify_reconnect()
                    
                    # 更新状态
                    self._last_connection_state = current_state
                    
                    # 每5秒检查一次
                    await asyncio.sleep(5)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"监控WebSocket连接状态失败: {e}")
                    await asyncio.sleep(5)
            
            logger.info("WebSocket连接状态监控已停止")
            
        except Exception as e:
            logger.error(f"WebSocket连接监控异常: {e}")
    

    def remove_data_callback(self, callback: Callable):
        """移除数据回调函数"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    

    def _interval_to_ms(self, interval: str) -> int:
        """将K线周期转换为毫秒数"""
        unit = interval[-1].lower()
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        elif unit == 'M':
            return value * 30 * 24 * 60 * 60 * 1000
        else:
            return 60 * 1000  # 默认1分钟
    
    async def reconnect(self):
        """重连WebSocket"""
        try:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error("达到最大重连次数，停止重连")
                return False
            
            self.reconnect_attempts += 1
            logger.info(f"尝试重连WebSocket ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
            
            # 停止当前连接
            if self.ws_client and hasattr(self.ws_client, 'stop_websocket'):
                self.ws_client.stop_websocket()
            
            # 等待一段时间后重连
            await asyncio.sleep(self.reconnect_delay)
            
            # 重新启动WebSocket
            await self._start_websocket()
            await self._subscribe_data_streams()
            
            self.reconnect_attempts = 0
            logger.info("WebSocket重连成功")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket重连失败: {e}")
            return False
