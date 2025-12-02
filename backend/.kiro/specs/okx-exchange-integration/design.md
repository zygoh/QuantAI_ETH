# Design Document

## Overview

本设计文档描述了为QuantAI-ETH交易系统添加OKX交易所支持的技术方案。系统将采用抽象工厂模式和统一接口设计，实现多交易所架构，允许用户在Binance和OKX之间灵活切换，同时保持现有业务逻辑的完整性。

### Design Goals

1. **可扩展性**: 采用统一接口设计，便于未来添加更多交易所
2. **零侵入性**: 现有业务模块无需修改即可支持多交易所
3. **高可靠性**: 实现完善的错误处理和自动重连机制
4. **高性能**: 优化API调用和数据处理流程
5. **可测试性**: 支持模拟交易所客户端，便于单元测试
6. **官方SDK集成**: 使用python-okx 0.4.0 SDK处理OKX底层API调用，提高稳定性和可维护性

### Key Design Decisions

1. **统一接口**: 定义`BaseExchangeClient`抽象基类，所有交易所客户端必须实现该接口
2. **工厂模式**: 使用`ExchangeFactory`集中管理客户端创建和生命周期
3. **配置驱动**: 通过配置文件控制交易所选择，无需修改代码
4. **数据标准化**: 所有交易所返回统一格式的数据结构
5. **独立配置**: 每个交易所拥有独立的API密钥和参数配置
6. **SDK封装**: OKXClient使用python-okx SDK处理认证、签名和API调用，提供统一接口适配层

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Trading Engine│  │Signal Generator│ │Data Service  │      │
│  └──────┬───────┘  └──────┬─────────┘ └──────┬───────┘      │
│         │                  │                   │              │
│         └──────────────────┼───────────────────┘              │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────────────┐
│                   Exchange Abstraction Layer                  │
│                            │                                  │
│                   ┌────────▼────────┐                         │
│                   │ ExchangeFactory │                         │
│                   └────────┬────────┘                         │
│                            │                                  │
│              ┌─────────────┴─────────────┐                   │
│              │                           │                   │
│     ┌────────▼────────┐         ┌───────▼────────┐          │
│     │BaseExchangeClient│         │BaseExchangeClient│         │
│     │   (Interface)    │         │   (Interface)    │         │
│     └────────┬────────┘         └───────┬────────┘          │
│              │                           │                   │
└──────────────┼───────────────────────────┼───────────────────┘
               │                           │
┌──────────────┼───────────────────────────┼───────────────────┐
│         Exchange Implementation Layer                         │
│              │                           │                   │
│     ┌────────▼────────┐         ┌───────▼────────┐          │
│     │ BinanceClient   │         │   OKXClient    │          │
│     │  + REST API     │         │  + REST API    │          │
│     │  + WebSocket    │         │  + WebSocket   │          │
│     └────────┬────────┘         └───────┬────────┘          │
│              │                           │                   │
└──────────────┼───────────────────────────┼───────────────────┘
               │                           │
┌──────────────┼───────────────────────────┼───────────────────┐
│                    External Services                          │
│              │                           │                   │
│     ┌────────▼────────┐         ┌───────▼────────┐          │
│     │ Binance API     │         │   OKX API      │          │
│     │ (fapi.binance)  │         │ (www.okx.com)  │          │
│     └─────────────────┘         └────────────────┘          │
└───────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Config → ExchangeFactory → Specific Client → External API
     ↓              ↓                  ↓                ↓
  EXCHANGE=OKX  create_client()   okx_client.get_klines()  OKX API
     ↓              ↓                  ↓                ↓
  Settings      OKXClient         Unified Data      Raw Data
```


## Components and Interfaces

### 1. BaseExchangeClient (Abstract Base Class)

统一的交易所客户端接口，定义所有交易所必须实现的方法。

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class UnifiedKlineData:
    """统一的K线数据格式"""
    timestamp: int  # 毫秒时间戳
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

@dataclass
class UnifiedTickerData:
    """统一的价格数据格式"""
    symbol: str
    price: float
    timestamp: int

@dataclass
class UnifiedOrderData:
    """统一的订单数据格式"""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # BUY, SELL
    type: str  # MARKET, LIMIT
    status: str  # NEW, FILLED, CANCELED
    quantity: float
    price: Optional[float]
    filled_quantity: float
    avg_price: float
    commission: float
    created_at: int
    updated_at: int

class BaseExchangeClient(ABC):
    """交易所客户端抽象基类"""
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """测试API连接"""
        pass
    
    @abstractmethod
    def get_server_time(self) -> int:
        """获取服务器时间（毫秒时间戳）"""
        pass
    
    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[UnifiedKlineData]:
        """获取K线数据"""
        pass
    
    @abstractmethod
    def get_ticker_price(self, symbol: str) -> Optional[UnifiedTickerData]:
        """获取实时价格"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        pass
    
    @abstractmethod
    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> UnifiedOrderData:
        """下单"""
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """取消订单"""
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[UnifiedOrderData]:
        """获取未成交订单"""
        pass
    
    @abstractmethod
    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """修改杠杆倍数"""
        pass
```

### 2. ExchangeFactory

工厂类，负责创建和管理交易所客户端实例。

```python
from typing import Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """支持的交易所类型"""
    BINANCE = "BINANCE"
    OKX = "OKX"
    MOCK = "MOCK"  # 用于测试

class ExchangeFactory:
    """交易所客户端工厂"""
    
    _instances: Dict[ExchangeType, BaseExchangeClient] = {}
    
    @classmethod
    def create_client(
        cls,
        exchange_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseExchangeClient:
        """
        创建交易所客户端实例（单例模式）
        
        Args:
            exchange_type: 交易所类型（BINANCE, OKX, MOCK）
            config: 可选的配置参数
        
        Returns:
            交易所客户端实例
        
        Raises:
            ValueError: 不支持的交易所类型
        """
        try:
            exchange_enum = ExchangeType(exchange_type.upper())
        except ValueError:
            logger.error(f"不支持的交易所类型: {exchange_type}")
            raise ValueError(f"Unsupported exchange type: {exchange_type}")
        
        # 单例模式：如果实例已存在，直接返回
        if exchange_enum in cls._instances:
            logger.info(f"返回已存在的{exchange_type}客户端实例")
            return cls._instances[exchange_enum]
        
        # 创建新实例
        if exchange_enum == ExchangeType.BINANCE:
            from app.exchange.binance_client import BinanceClient
            client = BinanceClient(config)
        elif exchange_enum == ExchangeType.OKX:
            from app.exchange.okx_client import OKXClient
            client = OKXClient(config)
        elif exchange_enum == ExchangeType.MOCK:
            from app.exchange.mock_client import MockExchangeClient
            client = MockExchangeClient(config)
        else:
            raise ValueError(f"Unsupported exchange type: {exchange_type}")
        
        cls._instances[exchange_enum] = client
        logger.info(f"创建新的{exchange_type}客户端实例")
        
        return client
    
    @classmethod
    def get_current_client(cls) -> BaseExchangeClient:
        """
        获取当前配置的交易所客户端
        
        Returns:
            当前交易所客户端实例
        """
        from app.core.config import settings
        return cls.create_client(settings.EXCHANGE_TYPE)
    
    @classmethod
    def reset(cls):
        """重置所有客户端实例（主要用于测试）"""
        cls._instances.clear()
        logger.info("所有交易所客户端实例已重置")
```


### 3. OKXClient Implementation

OKX交易所客户端的具体实现，使用python-okx 0.4.0 SDK作为底层。

```python
import logging
from typing import List, Dict, Any, Optional
from okx import Account, MarketData, Trade, PublicData
from okx.exceptions import OkxAPIException, OkxRequestException, OkxParamsException

logger = logging.getLogger(__name__)

class OKXClient(BaseExchangeClient):
    """OKX交易所客户端（基于python-okx SDK）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化OKX客户端
        
        Args:
            config: 可选配置参数，如果为None则从settings读取
        """
        from app.core.config import settings
        
        # 读取配置
        self.api_key = config.get('api_key') if config else settings.OKX_API_KEY
        self.secret_key = config.get('secret_key') if config else settings.OKX_SECRET_KEY
        self.passphrase = config.get('passphrase') if config else settings.OKX_PASSPHRASE
        
        # 配置代理
        proxy = None
        if settings.USE_PROXY:
            proxy_type = settings.PROXY_TYPE.lower()
            if proxy_type == "socks5":
                proxy = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            else:
                proxy = f"{proxy_type}://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            logger.info(f"🔧 OKX SDK使用代理: {proxy}")
        
        # 初始化python-okx SDK客户端
        # SDK会自动处理认证、签名、请求头等
        try:
            self.account_api = Account(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag='0',  # 0: 实盘, 1: 模拟盘
                proxy=proxy
            )
            
            self.market_api = MarketData(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag='0',
                proxy=proxy
            )
            
            self.trade_api = Trade(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag='0',
                proxy=proxy
            )
            
            self.public_api = PublicData(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag='0',
                proxy=proxy
            )
            
            logger.info("✅ OKX SDK客户端初始化完成")
            
        except Exception as e:
            logger.error(f"❌ OKX SDK初始化失败: {e}")
            raise ExchangeConnectionError(f"Failed to initialize OKX SDK: {e}")
    
    def _handle_sdk_exception(self, e: Exception) -> None:
        """
        处理SDK异常，转换为统一异常类型
        
        Args:
            e: SDK抛出的异常
        
        Raises:
            ExchangeError: 统一的交易所异常
        """
        if isinstance(e, OkxAPIException):
            # API错误
            code = e.code
            message = e.message
            
            # 处理限流错误
            if code in ['50011', '50014']:
                raise ExchangeRateLimitError(f"Rate limit exceeded: {message}")
            
            # 处理认证错误
            if code in ['50100', '50101', '50102', '50103']:
                raise ExchangeAuthError(f"Authentication failed: {message}")
            
            raise ExchangeAPIError(code, message)
            
        elif isinstance(e, OkxRequestException):
            # 请求错误（网络问题等）
            raise ExchangeConnectionError(f"Request failed: {str(e)}")
            
        elif isinstance(e, OkxParamsException):
            # 参数错误
            raise ExchangeInvalidParameterError(f"Invalid parameters: {str(e)}")
            
        else:
            # 其他未知错误
            raise ExchangeError(f"Unknown error: {str(e)}")
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            # 测试公共接口
            server_time = self.get_server_time()
            logger.info(f"✓ OKX服务器时间获取成功: {server_time}")
            
            # 测试私有接口
            account_info = self.get_account_info()
            if account_info:
                logger.info("✓ OKX账户信息获取成功")
                return True
            else:
                logger.warning("⚠️ OKX账户信息为空")
                return False
                
        except Exception as e:
            logger.error(f"❌ OKX连接测试失败: {e}")
            return False
    
    def get_server_time(self) -> int:
        """获取服务器时间"""
        try:
            # 使用SDK的公共API获取服务器时间
            response = self.public_api.get_system_time()
            
            if response['code'] == '0':
                return int(response['data'][0]['ts'])
            else:
                logger.error(f"获取服务器时间失败: {response['msg']}")
                import time
                return int(time.time() * 1000)
                
        except Exception as e:
            self._handle_sdk_exception(e)
            import time
            return int(time.time() * 1000)
    
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
            # 转换格式
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            okx_interval = IntervalMapper.to_exchange_format(interval, "OKX")
            
            # OKX API limit 最大值为 300
            if limit > 300:
                logger.warning(f"⚠️ limit={limit} 超过OKX最大限制300，自动调整为300")
                limit = 300
            
            # 使用SDK的市场数据API获取K线
            response = self.market_api.get_candlesticks(
                instId=okx_symbol,
                bar=okx_interval,
                limit=str(limit),
                after=str(end_time) if end_time else None,
                before=str(start_time) if start_time else None
            )
            
            if response['code'] != '0':
                logger.error(f"获取K线失败: {response['msg']}")
                return []
            
            klines = response.get('data', [])
            
            # 转换为统一格式
            formatted_klines = []
            for kline in klines:
                try:
                    formatted_kline = UnifiedKlineData(
                        timestamp=int(kline[0]),
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5]),
                        close_time=int(kline[0]) + self._interval_to_ms(interval) - 1,
                        quote_volume=float(kline[6]),
                        trades=0,
                        taker_buy_base_volume=0.0,
                        taker_buy_quote_volume=0.0
                    )
                    formatted_klines.append(formatted_kline)
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"❌ 解析K线数据失败: {e}")
                    continue
            
            # OKX返回的数据是倒序的，需要反转
            formatted_klines.reverse()
            
            logger.debug(f"✅ 获取OKX K线数据: {symbol} {interval} {len(formatted_klines)}条")
            return formatted_klines
            
        except Exception as e:
            self._handle_sdk_exception(e)
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            # 使用SDK的账户API获取余额
            response = self.account_api.get_account_balance()
            
            if response['code'] != '0':
                logger.error(f"获取账户信息失败: {response['msg']}")
                return {}
            
            balance_data = response.get('data', [])
            if not balance_data:
                return {}
            
            account = balance_data[0]
            
            # 格式化账户信息
            formatted_account = {
                'total_wallet_balance': float(account.get('totalEq', 0)),
                'total_unrealized_pnl': 0.0,
                'total_margin_balance': float(account.get('totalEq', 0)),
                'available_balance': float(account.get('availEq', 0)),
                'can_trade': True,
                'update_time': int(account.get('uTime', 0))
            }
            
            return formatted_account
            
        except Exception as e:
            self._handle_sdk_exception(e)
            return {}
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> UnifiedOrderData:
        """下单"""
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 构建订单参数
            order_params = {
                'instId': okx_symbol,
                'tdMode': 'cross',
                'side': 'buy' if side == 'BUY' else 'sell',
                'ordType': 'market' if order_type == 'MARKET' else 'limit',
                'sz': str(quantity)
            }
            
            if price is not None:
                order_params['px'] = str(price)
            
            # 使用SDK的交易API下单
            response = self.trade_api.place_order(**order_params)
            
            if response['code'] != '0':
                logger.error(f"下单失败: {response['msg']}")
                raise ExchangeAPIError(response['code'], response['msg'])
            
            order_data = response.get('data', [])
            if order_data:
                result = order_data[0]
                logger.info(f"✅ OKX下单成功: {symbol} {side} {quantity} @ {price}")
                
                return UnifiedOrderData(
                    order_id=result.get('ordId', ''),
                    client_order_id=result.get('clOrdId', ''),
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    status='NEW',
                    quantity=quantity,
                    price=price,
                    filled_quantity=0.0,
                    avg_price=0.0,
                    commission=0.0,
                    created_at=int(result.get('cTime', 0)),
                    updated_at=int(result.get('uTime', 0))
                )
            else:
                raise ExchangeAPIError('EMPTY_RESPONSE', 'Order response is empty')
            
        except Exception as e:
            self._handle_sdk_exception(e)
            raise
    
    # ... 其他方法实现
```

**SDK集成优势**：
1. **认证和签名**: SDK自动处理API密钥认证和请求签名，无需手动实现HMAC-SHA256算法
2. **请求封装**: SDK提供了类型安全的API方法，减少参数错误
3. **错误处理**: SDK定义了标准异常类型，便于统一处理
4. **维护性**: 官方SDK会持续更新以适配API变化
5. **代理支持**: SDK原生支持HTTP/SOCKS5代理配置

### 4. OKXWebSocketClient

OKX WebSocket客户端，支持实时数据订阅。

**注意**: python-okx SDK 0.4.0 主要提供 REST API 封装，WebSocket 功能需要使用 websocket-client 库手动实现，但可以复用 SDK 的认证机制。

```python
import asyncio
import json
import logging
from typing import Dict, Callable, List, Optional, Any
import websocket
import ssl
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class OKXWebSocketClient:
    """OKX WebSocket客户端（支持自动重连和心跳保活）"""
    
    def __init__(self):
        """初始化WebSocket客户端"""
        from app.core.config import settings
        
        # WebSocket URL
        if settings.OKX_TESTNET:
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/public"  # 模拟盘
        else:
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"  # 实盘
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.callbacks: Dict[str, Callable] = {}
        self.subscriptions: List[Dict[str, Any]] = []
        self.is_connected = False
        self.is_running = False
        self.is_reconnecting = False
        
        # 重连和心跳机制
        self.reconnect_task = None
        self.monitor_task = None
        self.health_check_task = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.last_message_time = None
        self.connection_start_time = None
        
        # 配置代理
        self.proxy_config = None
        if settings.USE_PROXY and settings.USE_PROXY_WS:
            proxy_type = settings.PROXY_TYPE.lower()
            if proxy_type == "socks5":
                # SOCKS5代理通过环境变量配置
                import os
                proxy_url = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                os.environ['http_proxy'] = proxy_url
                os.environ['https_proxy'] = proxy_url
                logger.info(f"🔧 OKX WebSocket使用SOCKS5代理: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
            else:
                # HTTP/HTTPS代理
                self.proxy_config = {
                    'http_proxy_host': settings.PROXY_HOST,
                    'http_proxy_port': settings.PROXY_PORT
                }
                logger.info(f"🔧 OKX WebSocket使用{proxy_type.upper()}代理: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
        
        logger.info("✅ OKX WebSocket客户端初始化完成")
        logger.info(f"   - WebSocket URL: {self.ws_url}")
    
    def start_websocket(self):
        """启动WebSocket连接"""
        try:
            # 保存事件循环
            if self.loop is None:
                try:
                    self.loop = asyncio.get_running_loop()
                    logger.info("✅ 事件循环已保存")
                except RuntimeError:
                    logger.warning("⚠️ 当前没有运行的事件循环，重连功能可能受限")
            
            # 配置WebSocket参数
            ws_kwargs = {
                "on_open": self._on_open,
                "on_message": self._on_message,
                "on_error": self._on_error,
                "on_close": self._on_close
            }
            
            # 添加代理配置
            if self.proxy_config:
                ws_kwargs.update(self.proxy_config)
            
            # 创建WebSocket连接
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                **ws_kwargs
            )
            
            self.is_running = True
            self.connection_start_time = datetime.now()
            self.last_message_time = datetime.now()
            
            # 启动WebSocket连接（在后台线程运行）
            ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            ws_thread.start()
            
            logger.info(f"✅ OKX WebSocket客户端启动 (URL: {self.ws_url})")
            
        except Exception as e:
            logger.error(f"❌ 启动OKX WebSocket失败: {e}")
            raise
    
    def _run_websocket(self):
        """在后台线程运行WebSocket"""
        try:
            # 配置SSL选项
            sslopt = {
                "cert_reqs": ssl.CERT_REQUIRED,
                "check_hostname": True
            }
            
            # 运行WebSocket
            self.ws.run_forever(sslopt=sslopt)
            
        except Exception as e:
            logger.error(f"❌ WebSocket运行失败: {e}")
            if not self.is_reconnecting:
                self._schedule_reconnect()
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """
        订阅K线数据
        
        Args:
            symbol: 交易对（如ETHUSDT）
            interval: K线周期（如1m, 5m, 15m）
            callback: 回调函数
        """
        from app.exchange.mappers import SymbolMapper, IntervalMapper
        
        # 转换格式
        okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
        okx_interval = IntervalMapper.to_exchange_format(interval, "OKX")
        
        channel = f"candle{okx_interval}"
        
        sub_msg = {
            "op": "subscribe",
            "args": [{
                "channel": channel,
                "instId": okx_symbol
            }]
        }
        
        # 保存订阅信息
        self.subscriptions.append({
            'type': 'kline',
            'symbol': symbol,
            'interval': interval,
            'channel': channel,
            'inst_id': okx_symbol
        })
        
        # 保存回调
        callback_key = f"{channel}:{okx_symbol}"
        self.callbacks[callback_key] = callback
        
        # 发送订阅消息
        if self.ws and self.is_connected:
            self.ws.send(json.dumps(sub_msg))
            logger.info(f"✅ 订阅OKX K线: {okx_symbol} {okx_interval}")
    
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """
        订阅价格数据
        
        Args:
            symbol: 交易对（如ETHUSDT）
            callback: 回调函数
        """
        from app.exchange.mappers import SymbolMapper
        
        okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
        
        sub_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "tickers",
                "instId": okx_symbol
            }]
        }
        
        # 保存订阅信息
        self.subscriptions.append({
            'type': 'ticker',
            'symbol': symbol,
            'channel': 'tickers',
            'inst_id': okx_symbol
        })
        
        # 保存回调
        callback_key = f"tickers:{okx_symbol}"
        self.callbacks[callback_key] = callback
        
        # 发送订阅消息
        if self.ws and self.is_connected:
            self.ws.send(json.dumps(sub_msg))
            logger.info(f"✅ 订阅OKX价格: {okx_symbol}")
    
    def _on_open(self, ws):
        """WebSocket连接建立回调"""
        self.is_connected = True
        self.is_reconnecting = False
        logger.info("✅ OKX WebSocket连接已建立")
        
        # 恢复订阅
        self._restore_subscriptions()
    
    def _on_message(self, ws, message):
        """WebSocket消息接收回调"""
        try:
            self.last_message_time = datetime.now()
            data = json.loads(message)
            
            # 处理订阅确认
            if data.get('event') == 'subscribe':
                logger.info(f"✅ 订阅确认: {data.get('arg', {})}")
                return
            
            # 处理数据推送
            if 'data' in data:
                arg = data.get('arg', {})
                channel = arg.get('channel', '')
                inst_id = arg.get('instId', '')
                
                callback_key = f"{channel}:{inst_id}"
                if callback_key in self.callbacks:
                    self.callbacks[callback_key](data['data'])
            
        except Exception as e:
            logger.error(f"❌ 处理WebSocket消息失败: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket错误回调"""
        logger.error(f"❌ OKX WebSocket错误: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭回调"""
        self.is_connected = False
        logger.warning(f"⚠️ OKX WebSocket连接已关闭: {close_status_code} - {close_msg}")
        
        if self.is_running and not self.is_reconnecting:
            self._schedule_reconnect()
    
    def _restore_subscriptions(self):
        """恢复所有订阅"""
        for sub in self.subscriptions:
            try:
                sub_msg = {
                    "op": "subscribe",
                    "args": [{
                        "channel": sub['channel'],
                        "instId": sub['inst_id']
                    }]
                }
                self.ws.send(json.dumps(sub_msg))
                logger.info(f"✅ 恢复订阅: {sub['channel']} {sub['inst_id']}")
            except Exception as e:
                logger.error(f"❌ 恢复订阅失败: {e}")
    
    def _schedule_reconnect(self):
        """安排重连"""
        if self.is_reconnecting:
            return
        
        self.is_reconnecting = True
        logger.info("🔄 准备重连OKX WebSocket...")
        
        # 使用指数退避策略重连
        import time
        time.sleep(5)  # 简单延迟，实际应使用ExponentialBackoffReconnector
        
        if self.is_running:
            self.start_websocket()
    
    def stop_websocket(self):
        """停止WebSocket连接"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("✅ OKX WebSocket已停止")
```

**WebSocket 实现说明**：
1. **手动实现**: python-okx SDK 不提供 WebSocket 封装，需要使用 websocket-client 库
2. **代理支持**: 支持 HTTP/HTTPS/SOCKS5 代理配置
3. **自动重连**: 实现连接断开后的自动重连机制
4. **订阅恢复**: 重连后自动恢复所有订阅
5. **心跳保活**: 通过监控消息时间实现健康检查


## Data Models

### Configuration Model

```python
class ExchangeConfig(BaseSettings):
    """交易所配置模型"""
    
    # 当前使用的交易所
    EXCHANGE_TYPE: str = "BINANCE"  # BINANCE, OKX, MOCK
    
    # Binance配置
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET_KEY: str = ""
    BINANCE_TESTNET: bool = True
    
    # OKX配置
    OKX_API_KEY: str = ""
    OKX_SECRET_KEY: str = ""
    OKX_PASSPHRASE: str = ""
    OKX_TESTNET: bool = False
    
    # 代理配置
    USE_PROXY: bool = True
    PROXY_HOST: str = "127.0.0.1"
    PROXY_PORT: int = 10808
    PROXY_TYPE: str = "socks5"
    
    # WebSocket配置
    WS_RECONNECT_INITIAL_DELAY: float = 1.0
    WS_RECONNECT_MAX_DELAY: float = 60.0
    WS_RECONNECT_BACKOFF_FACTOR: float = 2.0
    WS_RECONNECT_MAX_RETRIES: int = 10
    WS_PING_INTERVAL: int = 30
    WS_PONG_TIMEOUT: int = 10
    
    def validate_exchange_config(self) -> bool:
        """
        验证交易所配置的完整性
        
        Returns:
            配置是否有效
        """
        if self.EXCHANGE_TYPE == "BINANCE":
            if not self.BINANCE_API_KEY or not self.BINANCE_SECRET_KEY:
                logger.warning("Binance API密钥未配置")
                return False
        elif self.EXCHANGE_TYPE == "OKX":
            if not self.OKX_API_KEY or not self.OKX_SECRET_KEY or not self.OKX_PASSPHRASE:
                logger.warning("OKX API密钥未配置")
                return False
        
        return True
```

### Symbol Mapping Model

不同交易所的交易对格式不同，需要进行映射转换。

```python
class SymbolMapper:
    """交易对格式转换器"""
    
    # 标准格式 -> Binance格式
    BINANCE_MAPPING = {
        "ETH/USDT": "ETHUSDT",
        "BTC/USDT": "BTCUSDT"
    }
    
    # 标准格式 -> OKX格式
    OKX_MAPPING = {
        "ETH/USDT": "ETH-USDT-SWAP",
        "BTC/USDT": "BTC-USDT-SWAP"
    }
    
    @classmethod
    def to_exchange_format(cls, symbol: str, exchange_type: str) -> str:
        """
        将标准格式转换为交易所格式
        
        Args:
            symbol: 标准格式交易对（如ETH/USDT）
            exchange_type: 交易所类型
        
        Returns:
            交易所格式的交易对
        """
        if exchange_type == "BINANCE":
            return cls.BINANCE_MAPPING.get(symbol, symbol.replace("/", ""))
        elif exchange_type == "OKX":
            return cls.OKX_MAPPING.get(symbol, symbol.replace("/", "-") + "-SWAP")
        return symbol
    
    @classmethod
    def to_standard_format(cls, symbol: str, exchange_type: str) -> str:
        """
        将交易所格式转换为标准格式
        
        Args:
            symbol: 交易所格式交易对
            exchange_type: 交易所类型
        
        Returns:
            标准格式的交易对
        """
        if exchange_type == "BINANCE":
            # ETHUSDT -> ETH/USDT
            for std, exch in cls.BINANCE_MAPPING.items():
                if exch == symbol:
                    return std
        elif exchange_type == "OKX":
            # ETH-USDT-SWAP -> ETH/USDT
            for std, exch in cls.OKX_MAPPING.items():
                if exch == symbol:
                    return std
        return symbol
```

### Interval Mapping Model

不同交易所的K线周期格式也不同。

```python
class IntervalMapper:
    """K线周期格式转换器"""
    
    # 标准格式 -> Binance格式
    BINANCE_INTERVALS = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d"
    }
    
    # 标准格式 -> OKX格式
    OKX_INTERVALS = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D"
    }
    
    @classmethod
    def to_exchange_format(cls, interval: str, exchange_type: str) -> str:
        """
        将标准格式转换为交易所格式
        
        Args:
            interval: 标准格式周期（如5m）
            exchange_type: 交易所类型
        
        Returns:
            交易所格式的周期
        """
        if exchange_type == "BINANCE":
            return cls.BINANCE_INTERVALS.get(interval, interval)
        elif exchange_type == "OKX":
            return cls.OKX_INTERVALS.get(interval, interval)
        return interval
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Configuration Reading Consistency
*For any* valid configuration file, when the system starts, it should correctly identify and load the specified exchange type.
**Validates: Requirements 1.1**

### Property 2: Factory Returns Correct Client Type
*For any* valid exchange type string, the ExchangeFactory should return a client instance of the corresponding type.
**Validates: Requirements 3.1**

### Property 3: Unified Data Format Consistency
*For any* exchange client and any data retrieval method, the returned data structure should conform to the unified format specification.
**Validates: Requirements 2.4**

### Property 4: Exception Type Consistency
*For any* exchange client, when a method fails, it should throw exceptions of the unified exception type.
**Validates: Requirements 2.5**

### Property 5: Singleton Pattern Enforcement
*For any* exchange type, multiple requests to the factory should return the same client instance.
**Validates: Requirements 3.5**

### Property 6: Configuration Validation Completeness
*For any* exchange configuration, the system should detect all missing required parameters.
**Validates: Requirements 4.4**

### Property 7: Configuration Fallback Behavior
*For any* invalid configuration parameter, the system should use default values and log warnings.
**Validates: Requirements 4.5**

### Property 8: K-line Data Transformation Correctness
*For any* OKX format K-line data, transformation to unified format should preserve all essential fields.
**Validates: Requirements 5.3**

### Property 9: K-line Data Integrity Validation
*For any* K-line data, the system should detect missing required fields.
**Validates: Requirements 5.4**

### Property 10: K-line Error Handling
*For any* K-line data retrieval failure, the client should return an empty list without throwing exceptions.
**Validates: Requirements 5.5**

### Property 11: Price Data Transformation Correctness
*For any* OKX format price data, transformation to unified format should preserve symbol and price fields.
**Validates: Requirements 6.3, 6.4**

### Property 12: Price Error Handling
*For any* price data retrieval failure, the client should return None without throwing exceptions.
**Validates: Requirements 6.5**

### Property 13: Order Parameter Validation
*For any* order request with invalid parameters, the client should validate and throw exceptions before sending the request.
**Validates: Requirements 7.5**

### Property 14: Order Success Response Completeness
*For any* successful order placement, the response should contain order ID and order details.
**Validates: Requirements 7.3**

### Property 15: Order Failure Handling
*For any* failed order placement, the client should log errors and return failure status.
**Validates: Requirements 7.4**

### Property 16: Order Query Correctness
*For any* order query request, the client should call the correct API and return order details.
**Validates: Requirements 8.1**

### Property 17: Open Orders Filtering
*For any* order list, the query should return only unfilled orders.
**Validates: Requirements 8.3**

### Property 18: Order Query Error Handling
*For any* order query failure, the client should return empty results without throwing exceptions.
**Validates: Requirements 8.4**

### Property 19: Account Balance Retrieval
*For any* account with multiple currencies, the client should return balance information for all currencies.
**Validates: Requirements 9.3**

### Property 20: Position PnL Calculation
*For any* position, the client should calculate and return unrealized PnL data.
**Validates: Requirements 9.4**

### Property 21: Account Query Error Handling
*For any* account query failure, the client should return an empty dictionary without throwing exceptions.
**Validates: Requirements 9.5**

### Property 22: Leverage Setting Validation
*For any* valid leverage value, the client should successfully set leverage and return confirmation.
**Validates: Requirements 10.2**

### Property 23: Leverage Query Extraction
*For any* position information, the client should extract leverage data.
**Validates: Requirements 10.5**

### Property 24: WebSocket Auto-Reconnect Trigger
*For any* WebSocket disconnection, the client should automatically attempt to reconnect.
**Validates: Requirements 11.1**

### Property 25: Exponential Backoff Strategy
*For any* consecutive reconnection failures, the delay time should increase exponentially.
**Validates: Requirements 11.2**

### Property 26: Subscription Recovery After Reconnect
*For any* successful reconnection, all previous subscriptions should be restored.
**Validates: Requirements 11.3**

### Property 27: Heartbeat Ping Regularity
*For any* established WebSocket connection, ping messages should be sent at regular intervals.
**Validates: Requirements 12.1**

### Property 28: Pong Response Time Update
*For any* received pong response, the last response time should be updated.
**Validates: Requirements 12.2**

### Property 29: Pong Timeout Reconnect Trigger
*For any* pong timeout, the client should log warnings and trigger reconnection.
**Validates: Requirements 12.3**

### Property 30: Health Check Trigger
*For any* extended period without messages, the client should trigger health checks.
**Validates: Requirements 12.4**

### Property 31: Health Check Failure Response
*For any* failed health check, the client should actively disconnect and reconnect.
**Validates: Requirements 12.5**

### Property 32: Trading Engine Factory Usage
*For any* Trading Engine initialization, it should obtain the exchange client through the factory.
**Validates: Requirements 13.1**

### Property 33: Trading Engine Interface Usage
*For any* trading operation, Trading Engine should call unified interface methods.
**Validates: Requirements 13.2**

### Property 34: Trading Engine Configuration Switch
*For any* exchange configuration change, Trading Engine should use the new client after restart.
**Validates: Requirements 13.3**

### Property 35: Trading Engine Error Handling
*For any* exchange client method failure, Trading Engine should log errors and execute fallback strategies.
**Validates: Requirements 13.4**

### Property 36: Virtual Trading Interface Consistency
*For any* virtual trading mode, Trading Engine should use the same interface for simulated trades.
**Validates: Requirements 13.5**

### Property 37: Signal Generator Factory Usage
*For any* Signal Generator initialization, it should obtain the exchange client through the factory.
**Validates: Requirements 14.1**

### Property 38: Signal Generator Interface Usage
*For any* market data retrieval, Signal Generator should call unified interface methods.
**Validates: Requirements 14.2**

### Property 39: Signal Generator Data Format Consistency
*For any* exchange, Signal Generator should receive data in unified format.
**Validates: Requirements 14.3**

### Property 40: Signal Generator Error Handling
*For any* data retrieval failure, Signal Generator should log errors and skip the current signal generation cycle.
**Validates: Requirements 14.4**

### Property 41: Data Service Factory Usage
*For any* Data Service initialization, it should obtain the exchange client through the factory.
**Validates: Requirements 15.1**

### Property 42: Data Service Interface Usage
*For any* real-time data subscription, Data Service should use unified interface methods.
**Validates: Requirements 15.2**

### Property 43: Data Service Storage Consistency
*For any* received data, Data Service should store it in the database.
**Validates: Requirements 15.3**

### Property 44: Data Service Query Format
*For any* historical data query, Data Service should return data in unified format.
**Validates: Requirements 15.4**

### Property 45: API Call Logging
*For any* exchange API call, the client should log request parameters and response results.
**Validates: Requirements 16.1**

### Property 46: API Failure Logging
*For any* API call failure, the client should log detailed error information and stack traces.
**Validates: Requirements 16.2**

### Property 47: WebSocket Event Logging
*For any* WebSocket connection state change, the client should log connection events.
**Validates: Requirements 16.3**

### Property 48: Trading Operation Logging
*For any* trading operation, the client should log order details and execution results.
**Validates: Requirements 16.4**

### Property 49: Debug Level Logging
*For any* DEBUG log level, the client should log all API interaction details.
**Validates: Requirements 16.5**

### Property 50: Rate Limit Auto-Delay
*For any* detected API rate limit error, the client should automatically delay subsequent requests.
**Validates: Requirements 17.1**

### Property 51: Rate Limit Adaptive Delay
*For any* consecutive rate limit triggers, the delay time should increase.
**Validates: Requirements 17.2**

### Property 52: Rate Limit Recovery
*For any* rate limit recovery, the client should gradually restore normal request frequency.
**Validates: Requirements 17.3**

### Property 53: Pagination Request Delay
*For any* paginated large data retrieval, the client should add delays between requests.
**Validates: Requirements 17.4**

### Property 54: Mock Client Test Data
*For any* mock client method call, it should return predefined test data.
**Validates: Requirements 18.2**

### Property 55: Mock Trading No Real Requests
*For any* mock trading execution, the client should log operations but not send real requests.
**Validates: Requirements 18.3**

### Property 56: Mock Error Simulation
*For any* error testing, the mock client should be able to simulate various error scenarios.
**Validates: Requirements 18.5**

### Property 57: Startup Connection Test
*For any* system startup, the exchange client should execute connection tests.
**Validates: Requirements 19.1**

### Property 58: Connection Success Continuation
*For any* successful connection test, the client should log success and continue startup.
**Validates: Requirements 19.4**

### Property 59: Connection Failure Handling
*For any* failed connection test, the client should log detailed errors and decide whether to continue based on configuration.
**Validates: Requirements 19.5**

### Property 60: SDK Authentication Initialization
*For any* OKX client initialization, the SDK should be configured with correct API credentials.
**Validates: Requirements 21.1**

### Property 61: SDK API Method Usage
*For any* OKX REST API call, the client should use SDK-provided methods instead of manual HTTP requests.
**Validates: Requirements 21.2**

### Property 62: SDK Signature Delegation
*For any* API request requiring signature, the SDK should handle signature generation automatically.
**Validates: Requirements 21.3**

### Property 63: SDK Response Transformation
*For any* SDK API response, the client should transform it to unified data format.
**Validates: Requirements 21.4**

### Property 64: SDK Exception Conversion
*For any* SDK exception, the client should catch and convert it to unified exception types.
**Validates: Requirements 21.5**


## Error Handling

### Error Hierarchy

```python
class ExchangeError(Exception):
    """交易所错误基类"""
    pass

class ExchangeConnectionError(ExchangeError):
    """连接错误"""
    pass

class ExchangeAPIError(ExchangeError):
    """API调用错误"""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"API Error {code}: {message}")

class ExchangeRateLimitError(ExchangeError):
    """限流错误"""
    pass

class ExchangeAuthError(ExchangeError):
    """认证错误"""
    pass

class ExchangeInvalidParameterError(ExchangeError):
    """参数错误"""
    pass
```

### Error Handling Strategy

1. **网络错误**: 自动重试，使用指数退避策略
2. **认证错误**: 记录错误，停止操作，通知管理员
3. **限流错误**: 自动延迟，调整请求频率
4. **参数错误**: 记录错误，返回失败状态
5. **未知错误**: 记录详细信息，执行降级策略

### Error Logging Format

```python
logger.error(f"❌ {operation_name}失败")
logger.error(f"   错误类型: {type(error).__name__}")
logger.error(f"   错误信息: {str(error)}")
logger.error(f"   请求参数: {params}")
logger.error(traceback.format_exc())
```

## Testing Strategy

### Unit Testing

**测试框架**: pytest

**测试覆盖范围**:
1. ExchangeFactory客户端创建逻辑
2. 数据格式转换函数
3. 签名生成算法
4. 配置验证逻辑
5. 错误处理分支

**示例测试**:
```python
def test_factory_creates_binance_client():
    """测试工厂创建Binance客户端"""
    client = ExchangeFactory.create_client("BINANCE")
    assert isinstance(client, BinanceClient)

def test_factory_creates_okx_client():
    """测试工厂创建OKX客户端"""
    client = ExchangeFactory.create_client("OKX")
    assert isinstance(client, OKXClient)

def test_factory_raises_error_for_invalid_type():
    """测试工厂对无效类型抛出异常"""
    with pytest.raises(ValueError):
        ExchangeFactory.create_client("INVALID")

def test_symbol_mapper_binance_format():
    """测试交易对转换为Binance格式"""
    result = SymbolMapper.to_exchange_format("ETH/USDT", "BINANCE")
    assert result == "ETHUSDT"

def test_symbol_mapper_okx_format():
    """测试交易对转换为OKX格式"""
    result = SymbolMapper.to_exchange_format("ETH/USDT", "OKX")
    assert result == "ETH-USDT-SWAP"
```

### Property-Based Testing

**测试框架**: Hypothesis (Python的property-based testing库)

**配置**: 每个属性测试运行至少100次迭代

**测试策略**:
- 使用Hypothesis生成随机输入数据
- 验证系统在各种输入下的行为符合属性定义
- 每个正确性属性对应一个property-based test

**示例测试**:
```python
from hypothesis import given, strategies as st
from unittest.mock import Mock, patch

@given(st.text(min_size=1))
def test_property_1_config_reading_consistency(exchange_type):
    """
    Property 1: Configuration Reading Consistency
    Feature: okx-exchange-integration, Property 1
    Validates: Requirements 1.1
    
    For any valid configuration file, when the system starts,
    it should correctly identify and load the specified exchange type.
    """
    # 假设exchange_type是有效的配置值
    if exchange_type.upper() in ["BINANCE", "OKX", "MOCK"]:
        # 设置配置
        config = {"EXCHANGE_TYPE": exchange_type.upper()}
        
        # 创建客户端
        client = ExchangeFactory.create_client(exchange_type.upper(), config)
        
        # 验证客户端类型正确
        if exchange_type.upper() == "BINANCE":
            assert isinstance(client, BinanceClient)
        elif exchange_type.upper() == "OKX":
            assert isinstance(client, OKXClient)
        elif exchange_type.upper() == "MOCK":
            assert isinstance(client, MockExchangeClient)

@given(st.sampled_from(["api_key_123", "test_key", "prod_key"]))
def test_property_60_sdk_authentication_initialization(api_key):
    """
    Property 60: SDK Authentication Initialization
    Feature: okx-exchange-integration, Property 60
    Validates: Requirements 21.1
    
    For any OKX client initialization, the SDK should be configured
    with correct API credentials.
    """
    with patch('okx.Account') as mock_account, \
         patch('okx.MarketData') as mock_market, \
         patch('okx.Trade') as mock_trade, \
         patch('okx.PublicData') as mock_public:
        
        # 配置
        config = {
            'api_key': api_key,
            'secret_key': 'secret_123',
            'passphrase': 'pass_123'
        }
        
        # 创建客户端
        client = OKXClient(config)
        
        # 验证SDK被正确初始化
        mock_account.assert_called_once()
        call_kwargs = mock_account.call_args[1]
        assert call_kwargs['api_key'] == api_key
        assert call_kwargs['api_secret_key'] == 'secret_123'
        assert call_kwargs['passphrase'] == 'pass_123'

def test_property_61_sdk_api_method_usage():
    """
    Property 61: SDK API Method Usage
    Feature: okx-exchange-integration, Property 61
    Validates: Requirements 21.2
    
    For any OKX REST API call, the client should use SDK-provided
    methods instead of manual HTTP requests.
    """
    with patch('okx.MarketData') as mock_market:
        # 模拟SDK响应
        mock_instance = Mock()
        mock_instance.get_candlesticks.return_value = {
            'code': '0',
            'data': []
        }
        mock_market.return_value = mock_instance
        
        # 创建客户端
        client = OKXClient()
        
        # 调用获取K线方法
        client.get_klines('ETHUSDT', '5m', limit=100)
        
        # 验证使用了SDK方法而非手动HTTP请求
        mock_instance.get_candlesticks.assert_called_once()
        
        # 验证没有使用requests库
        with patch('requests.get') as mock_requests:
            client.get_klines('ETHUSDT', '5m', limit=100)
            mock_requests.assert_not_called()

def test_property_64_sdk_exception_conversion():
    """
    Property 64: SDK Exception Conversion
    Feature: okx-exchange-integration, Property 64
    Validates: Requirements 21.5
    
    For any SDK exception, the client should catch and convert it
    to unified exception types.
    """
    from okx.exceptions import OkxAPIException, OkxRequestException
    
    with patch('okx.MarketData') as mock_market:
        mock_instance = Mock()
        
        # 模拟SDK抛出API异常
        mock_instance.get_candlesticks.side_effect = OkxAPIException(
            code='50011',
            message='Rate limit exceeded'
        )
        mock_market.return_value = mock_instance
        
        client = OKXClient()
        
        # 验证异常被转换为统一类型
        with pytest.raises(ExchangeRateLimitError):
            client.get_klines('ETHUSDT', '5m')
        
        # 模拟SDK抛出请求异常
        mock_instance.get_candlesticks.side_effect = OkxRequestException('Network error')
        
        with pytest.raises(ExchangeConnectionError):
            client.get_klines('ETHUSDT', '5m')

@given(st.sampled_from(["BINANCE", "OKX", "MOCK"]))
def test_property_5_singleton_pattern(exchange_type):
    """
    Property 5: Singleton Pattern Enforcement
    Feature: okx-exchange-integration, Property 5
    Validates: Requirements 3.5
    
    For any exchange type, multiple requests to the factory
    should return the same client instance.
    """
    # 重置工厂状态
    ExchangeFactory.reset()
    
    # 创建两次客户端
    client1 = ExchangeFactory.create_client(exchange_type)
    client2 = ExchangeFactory.create_client(exchange_type)
    
    # 验证是同一个实例
    assert client1 is client2

@given(
    st.floats(min_value=0.0, max_value=1000000.0),
    st.floats(min_value=0.0, max_value=1000000.0),
    st.floats(min_value=0.0, max_value=1000000.0)
)
def test_property_8_kline_transformation(open_price, high_price, low_price):
    """
    Property 8: K-line Data Transformation Correctness
    Feature: okx-exchange-integration, Property 8
    Validates: Requirements 5.3
    
    For any OKX format K-line data, transformation to unified format
    should preserve all essential fields.
    """
    # 构造OKX格式的K线数据
    okx_kline = {
        "ts": "1609459200000",
        "o": str(open_price),
        "h": str(high_price),
        "l": str(low_price),
        "c": str((open_price + high_price + low_price) / 3),
        "vol": "1000",
        "volCcy": "50000"
    }
    
    # 转换为统一格式
    unified_kline = convert_okx_kline_to_unified(okx_kline)
    
    # 验证所有字段都被保留
    assert unified_kline.timestamp == 1609459200000
    assert unified_kline.open == open_price
    assert unified_kline.high == high_price
    assert unified_kline.low == low_price
    assert unified_kline.volume == 1000.0
```

### Integration Testing

**测试范围**:
1. Trading Engine与ExchangeFactory的集成
2. Signal Generator与ExchangeClient的集成
3. Data Service与WebSocket的集成
4. 完整的交易流程测试

**测试环境**: 使用MockExchangeClient模拟交易所

### Mock Testing

**MockExchangeClient功能**:
- 模拟所有API响应
- 可配置返回数据
- 可模拟各种错误场景
- 记录所有API调用

```python
class MockExchangeClient(BaseExchangeClient):
    """模拟交易所客户端（用于测试）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.call_history = []
        self.mock_responses = {}
        self.error_mode = None
    
    def set_mock_response(self, method: str, response: Any):
        """设置模拟响应"""
        self.mock_responses[method] = response
    
    def set_error_mode(self, error_type: str):
        """设置错误模式"""
        self.error_mode = error_type
    
    def get_klines(self, symbol: str, interval: str, **kwargs) -> List[UnifiedKlineData]:
        """模拟获取K线数据"""
        self.call_history.append(("get_klines", symbol, interval, kwargs))
        
        if self.error_mode == "network_error":
            raise ExchangeConnectionError("Network error")
        
        if "get_klines" in self.mock_responses:
            return self.mock_responses["get_klines"]
        
        # 返回默认测试数据
        return [
            UnifiedKlineData(
                timestamp=1609459200000,
                open=1000.0,
                high=1100.0,
                low=900.0,
                close=1050.0,
                volume=10000.0,
                close_time=1609459259999,
                quote_volume=10500000.0,
                trades=1000,
                taker_buy_base_volume=5000.0,
                taker_buy_quote_volume=5250000.0
            )
        ]
```


## Implementation Considerations

### 1. Backward Compatibility

**原则**: 确保现有代码无需修改即可继续工作

**策略**:
- 保留`binance_client`全局实例，但标记为deprecated
- 在`binance_client`中添加警告日志，建议使用ExchangeFactory
- 提供迁移指南和示例代码

**迁移示例**:
```python
# 旧代码（仍然可用，但会有警告）
from app.exchange.binance_client import binance_client
klines = binance_client.get_klines("ETHUSDT", "5m")

# 新代码（推荐）
from app.exchange.exchange_factory import ExchangeFactory
client = ExchangeFactory.get_current_client()
klines = client.get_klines("ETHUSDT", "5m")
```

### 2. Performance Optimization

**缓存策略**:
- 缓存交易所客户端实例（单例模式）
- 缓存交易对和周期映射关系
- 缓存服务器时间偏移量

**连接池**:
- 使用requests.Session复用HTTP连接
- 配置合理的连接池大小
- 设置适当的超时时间

**批量操作**:
- 支持批量获取K线数据
- 支持批量查询订单状态
- 使用WebSocket减少REST API调用

### 3. Security Considerations

**API密钥管理**:
- 从环境变量读取API密钥
- 不在日志中输出完整密钥
- 支持密钥加密存储

**签名安全**:
- 使用HMAC-SHA256算法
- 包含时间戳防止重放攻击
- 验证响应签名（如果交易所支持）

**网络安全**:
- 强制使用HTTPS/WSS
- 验证SSL证书
- 支持代理配置

### 4. Monitoring and Alerting

**关键指标**:
- API调用成功率
- API响应时间
- WebSocket连接稳定性
- 重连次数和频率
- 限流触发次数

**告警规则**:
- API调用失败率超过5%
- WebSocket连接中断超过5分钟
- 连续重连失败超过3次
- 检测到认证错误

**日志级别**:
- DEBUG: 所有API交互细节
- INFO: 正常操作和状态变化
- WARNING: 可恢复的错误和异常
- ERROR: 严重错误和失败操作

### 5. Configuration Management

**配置文件结构**:
```python
# config.py
class Settings(BaseSettings):
    # 交易所选择
    EXCHANGE_TYPE: str = "BINANCE"
    
    # Binance配置
    BINANCE_API_KEY: str = Field(default="", env="BINANCE_API_KEY")
    BINANCE_SECRET_KEY: str = Field(default="", env="BINANCE_SECRET_KEY")
    
    # OKX配置（用于python-okx SDK）
    OKX_API_KEY: str = Field(default="", env="OKX_API_KEY")
    OKX_SECRET_KEY: str = Field(default="", env="OKX_SECRET_KEY")
    OKX_PASSPHRASE: str = Field(default="", env="OKX_PASSPHRASE")
    OKX_TESTNET: bool = Field(default=False, env="OKX_TESTNET")  # SDK flag参数
    
    # 代理配置（SDK和WebSocket共用）
    USE_PROXY: bool = Field(default=True, env="USE_PROXY")
    USE_PROXY_WS: bool = Field(default=False, env="USE_PROXY_WS")  # WebSocket是否使用代理
    PROXY_HOST: str = Field(default="127.0.0.1", env="PROXY_HOST")
    PROXY_PORT: int = Field(default=10808, env="PROXY_PORT")
    PROXY_TYPE: str = Field(default="socks5", env="PROXY_TYPE")  # http, https, socks5
    
    class Config:
        env_file = ".env"
        case_sensitive = True
```

**环境变量示例**:
```bash
# .env
# 交易所选择
EXCHANGE_TYPE=OKX

# OKX API配置（用于python-okx SDK）
OKX_API_KEY=your_api_key
OKX_SECRET_KEY=your_secret_key
OKX_PASSPHRASE=your_passphrase
OKX_TESTNET=false  # false=实盘, true=模拟盘

# 代理配置
USE_PROXY=true
USE_PROXY_WS=false  # WebSocket直连，REST API使用代理
PROXY_HOST=127.0.0.1
PROXY_PORT=10808
PROXY_TYPE=socks5
```

**SDK 配置说明**:
- `flag='0'`: 实盘模式（OKX_TESTNET=false）
- `flag='1'`: 模拟盘模式（OKX_TESTNET=true）
- `proxy`: SDK 原生支持代理配置，格式为 `protocol://host:port`
- SDK 会自动处理 API 密钥的认证和签名

### 6. Documentation Requirements

**代码文档**:
- 所有公共方法必须有docstring
- 包含参数说明和返回值说明
- 包含使用示例
- 标注可能抛出的异常

**API文档**:
- 记录所有接口方法
- 提供完整的参数说明
- 包含请求和响应示例
- 说明错误码和处理方式

**用户文档**:
- 配置指南
- 快速开始教程
- 常见问题解答
- 故障排查指南

### 7. Deployment Strategy

**部署步骤**:
1. 添加新的配置参数（保持默认值为BINANCE）
2. 部署新代码（不修改配置）
3. 验证系统正常运行
4. 逐步切换到OKX（先测试环境，后生产环境）
5. 监控系统运行状态
6. 收集反馈并优化

**回滚计划**:
- 保留旧版本代码
- 支持快速切换回Binance
- 准备数据恢复方案

### 8. Testing in Production

**灰度发布**:
- 先在测试账户验证
- 使用小额资金测试
- 逐步增加交易量
- 监控关键指标

**A/B测试**:
- 同时运行Binance和OKX
- 对比交易性能
- 分析数据质量
- 评估系统稳定性

## Migration Guide

### For Developers

**步骤1: 安装依赖**
```bash
# 安装python-okx SDK
pip install python-okx==0.4.0

# 或使用uv（推荐）
uv pip install python-okx==0.4.0
```

**步骤2: 更新导入**
```python
# 旧代码
from app.exchange.binance_client import binance_client

# 新代码
from app.exchange.exchange_factory import ExchangeFactory
```

**步骤3: 获取客户端**
```python
# 旧代码
client = binance_client

# 新代码
client = ExchangeFactory.get_current_client()
```

**步骤4: 使用统一接口**
```python
# 接口方法保持不变
klines = client.get_klines("ETHUSDT", "5m", limit=100)
price = client.get_ticker_price("ETHUSDT")

# OKXClient内部使用python-okx SDK处理API调用
# 开发者无需关心SDK细节，只需使用统一接口
```

**步骤5: 理解SDK集成**
```python
# OKXClient内部实现示例
class OKXClient(BaseExchangeClient):
    def __init__(self, config):
        # SDK自动处理认证和签名
        self.market_api = MarketData(
            api_key=config['api_key'],
            api_secret_key=config['secret_key'],
            passphrase=config['passphrase'],
            proxy=proxy_url  # SDK原生支持代理
        )
    
    def get_klines(self, symbol, interval, limit):
        # 使用SDK方法而非手动HTTP请求
        response = self.market_api.get_candlesticks(
            instId=okx_symbol,
            bar=okx_interval,
            limit=str(limit)
        )
        # 转换为统一格式
        return self._convert_to_unified_format(response)
```

### For System Administrators

**步骤1: 添加OKX配置**
```bash
# 在.env文件中添加
OKX_API_KEY=your_api_key
OKX_SECRET_KEY=your_secret_key
OKX_PASSPHRASE=your_passphrase
OKX_TESTNET=false  # false=实盘, true=模拟盘

# 代理配置（可选）
USE_PROXY=true
USE_PROXY_WS=false
PROXY_HOST=127.0.0.1
PROXY_PORT=10808
PROXY_TYPE=socks5
```

**步骤2: 切换交易所**
```bash
# 修改EXCHANGE_TYPE
EXCHANGE_TYPE=OKX
```

**步骤3: 安装依赖**
```bash
# 安装python-okx SDK
pip install -r requirements.txt
```

**步骤4: 重启系统**
```powershell
# Windows PowerShell
python main.py
```

**步骤5: 验证连接**
- 检查日志中的 "✅ OKX SDK客户端初始化完成"
- 检查日志中的连接测试结果
- 验证数据正常接收
- 确认交易功能正常

**SDK 优势**:
- ✅ 官方维护，API变更会及时更新
- ✅ 自动处理认证和签名，减少错误
- ✅ 类型安全的API方法
- ✅ 原生支持代理配置
- ✅ 标准化的异常处理

## Future Enhancements

### Phase 2: Additional Exchanges
- 添加Bybit支持
- 添加Gate.io支持
- 添加Huobi支持

### Phase 3: Advanced Features
- 跨交易所套利
- 多交易所聚合行情
- 智能路由选择最优交易所

### Phase 4: Performance Optimization
- 实现连接池管理
- 优化数据序列化
- 减少内存占用

## Conclusion

本设计文档提供了添加OKX交易所支持的完整技术方案。通过采用统一接口和工厂模式，系统能够灵活支持多个交易所，同时保持代码的可维护性和可扩展性。设计充分考虑了向后兼容性、性能优化、安全性和可测试性，确保系统能够稳定可靠地运行。
