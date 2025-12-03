"""
OKX交易所API客户端（基于python-okx 0.4.0 SDK）

使用官方python-okx SDK处理认证、签名和API调用
保持BaseExchangeClient统一接口，提供数据格式转换层
"""
import logging
import time
import asyncio
import websocket
import ssl
import threading
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from app.core.config import settings
from app.exchange.base_exchange_client import (
    BaseExchangeClient,
    UnifiedKlineData,
    UnifiedTickerData,
    UnifiedOrderData
)
from app.exchange.exceptions import (
    ExchangeError,
    ExchangeConnectionError,
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeRateLimitError,
    ExchangeInvalidParameterError
)
from app.exchange.mappers import SymbolMapper, IntervalMapper

# 导入python-okx SDK
logger = logging.getLogger(__name__)

try:
    import okx.Account as AccountModule
    import okx.MarketData as MarketDataModule
    import okx.Trade as TradeModule
    import okx.PublicData as PublicDataModule
    
    # 尝试导入异常类
    try:
        from okx.exceptions import OkxAPIException, OkxRequestException, OkxParamsException
    except ImportError:
        # 如果异常类不存在，定义占位符
        class OkxAPIException(Exception):
            def __init__(self, code, message):
                self.code = code
                self.message = message
                super().__init__(f"OKX API Error {code}: {message}")
        
        class OkxRequestException(Exception):
            pass
        
        class OkxParamsException(Exception):
            pass
    
    SDK_AVAILABLE = True
    logger.info("✅ python-okx SDK 模块导入成功")
    logger.debug(f"   Account 模块: {type(AccountModule)}")
    logger.debug(f"   MarketData 模块: {type(MarketDataModule)}")
    
except ImportError as e:
    SDK_AVAILABLE = False
    logger.error(f"❌ python-okx SDK 导入失败: {e}")
    logger.error("   请运行: pip install python-okx==0.4.0")
    AccountModule = None
    MarketDataModule = None
    TradeModule = None
    PublicDataModule = None



class OKXClient(BaseExchangeClient):
    """OKX交易所客户端（基于python-okx SDK）"""
    
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
        except (ValueError, TypeError) as e:
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
            return int(float(value))  # 先转float再转int，处理 "123.0" 这种情况
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ 无法转换为int: value={repr(value)}, 使用默认值={default}")
            return default
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化OKX客户端
        
        Args:
            config: 可选配置参数，如果为None则从settings读取
        """
        if not SDK_AVAILABLE:
            raise ImportError("python-okx SDK未安装，请运行: pip install python-okx==0.4.0")
        
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
            flag = '1' if settings.OKX_TESTNET else '0'  # 0: 实盘, 1: 模拟盘
            
            logger.info("🔧 开始初始化 OKX SDK API 客户端...")
            logger.info(f"  - 模式: {'模拟盘' if settings.OKX_TESTNET else '实盘'} (flag={flag})")
            logger.info(f"  - API Key: {self.api_key[:8]}..." if self.api_key else "  - API Key: 未设置")
            logger.info(f"  - 代理: {proxy if proxy else '不使用代理'}")
            
            # 初始化 Account API
            logger.debug("  初始化 Account API...")
            logger.debug(f"    AccountModule 类型: {type(AccountModule)}")
            logger.debug(f"    AccountModule 属性: {[x for x in dir(AccountModule) if not x.startswith('_')][:10]}")
            
            # 尝试找到正确的 API 类
            if hasattr(AccountModule, 'AccountAPI'):
                AccountAPIClass = AccountModule.AccountAPI
            elif hasattr(AccountModule, 'Account'):
                AccountAPIClass = AccountModule.Account
            else:
                # 如果找不到类，尝试直接使用模块
                AccountAPIClass = AccountModule
            
            logger.debug(f"    使用 API 类: {AccountAPIClass}")
            
            self.account_api = AccountAPIClass(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=flag,
                proxy=proxy if proxy else {}
            )
            logger.debug("  ✅ Account API 初始化成功")
            
            # 初始化 Market Data API
            logger.debug("  初始化 MarketData API...")
            if hasattr(MarketDataModule, 'MarketAPI'):
                MarketAPIClass = MarketDataModule.MarketAPI
            elif hasattr(MarketDataModule, 'MarketData'):
                MarketAPIClass = MarketDataModule.MarketData
            else:
                MarketAPIClass = MarketDataModule
            
            self.market_api = MarketAPIClass(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=flag,
                proxy=proxy if proxy else {}
            )
            logger.debug("  ✅ MarketData API 初始化成功")
            
            # 初始化 Trade API
            logger.debug("  初始化 Trade API...")
            if hasattr(TradeModule, 'TradeAPI'):
                TradeAPIClass = TradeModule.TradeAPI
            elif hasattr(TradeModule, 'Trade'):
                TradeAPIClass = TradeModule.Trade
            else:
                TradeAPIClass = TradeModule
            
            self.trade_api = TradeAPIClass(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=flag,
                proxy=proxy if proxy else {}
            )
            logger.debug("  ✅ Trade API 初始化成功")
            
            # 初始化 Public Data API
            logger.debug("  初始化 PublicData API...")
            if hasattr(PublicDataModule, 'PublicAPI'):
                PublicAPIClass = PublicDataModule.PublicAPI
            elif hasattr(PublicDataModule, 'PublicData'):
                PublicAPIClass = PublicDataModule.PublicData
            else:
                PublicAPIClass = PublicDataModule
            
            self.public_api = PublicAPIClass(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=flag,
                proxy=proxy if proxy else {}
            )
            logger.debug("  ✅ PublicData API 初始化成功")
            
            logger.info("✅ OKX SDK 所有 API 客户端初始化完成")
            
        except Exception as e:
            logger.error(f"❌ OKX SDK初始化失败: {e}")
            logger.error(f"   错误类型: {type(e).__name__}")
            logger.error(f"   错误详情: {str(e)}")
            import traceback
            logger.error(f"   堆栈跟踪:\n{traceback.format_exc()}")
            raise ExchangeConnectionError(f"Failed to initialize OKX SDK: {e}")

    
    def _handle_sdk_exception(self, e: Exception) -> None:
        """
        处理SDK异常，转换为统一异常类型
        
        Args:
            e: SDK抛出的异常
        
        Raises:
            ExchangeError: 统一的交易所异常
        """
        logger.debug(f"🔍 处理 SDK 异常: {type(e).__name__}")
        
        if isinstance(e, OkxAPIException):
            # API错误
            code = e.code
            message = e.message
            logger.error(f"❌ OKX API 错误: code={code}, message={message}")
            
            # 处理限流错误
            if code in ['50011', '50014']:
                logger.warning(f"⚠️ 触发限流: {message}")
                raise ExchangeRateLimitError(f"Rate limit exceeded: {message}")
            
            # 处理认证错误
            if code in ['50100', '50101', '50102', '50103']:
                logger.error(f"🔐 认证失败: {message}")
                raise ExchangeAuthError(f"Authentication failed: {message}")
            
            logger.error(f"❌ API 错误: {code} - {message}")
            raise ExchangeAPIError(code, message)
            
        elif isinstance(e, OkxRequestException):
            # 请求错误（网络问题等）
            logger.error(f"🌐 请求失败: {str(e)}")
            raise ExchangeConnectionError(f"Request failed: {str(e)}")
            
        elif isinstance(e, OkxParamsException):
            # 参数错误
            logger.error(f"📝 参数错误: {str(e)}")
            raise ExchangeInvalidParameterError(f"Invalid parameters: {str(e)}")
            
        else:
            # 其他未知错误
            logger.error(f"❓ 未知错误: {type(e).__name__} - {str(e)}")
            import traceback
            logger.error(f"   堆栈跟踪:\n{traceback.format_exc()}")
            raise ExchangeError(f"Unknown error: {str(e)}")

    
    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            # 测试公共接口
            server_time = self.get_server_time()
            logger.info(f"✓ OKX服务器时间获取成功: {server_time}")
            
            # 测试私有接口
            try:
                account_info = self.get_account_info()
                if account_info:
                    logger.info("✓ OKX账户信息获取成功")
                    return True
                else:
                    logger.warning("⚠️ OKX账户信息为空")
                    return False
            except ExchangeAuthError as e:
                logger.error(f"✗ OKX账户信息获取失败: {e}")
                logger.error("  可能的原因：")
                logger.error("    1. API Key 未启用合约交易权限")
                logger.error("    2. API Key、Secret Key 或 Passphrase 不正确")
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
                return int(time.time() * 1000)
                
        except Exception as e:
            self._handle_sdk_exception(e)
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
            logger.debug(f"📊 请求获取K线: symbol={symbol}, interval={interval}, limit={limit}")
            
            # OKX API limit 最大值为 300
            if limit > 300:
                logger.warning(f"⚠️ limit={limit} 超过OKX最大限制300，自动调整为300")
                limit = 300
            elif limit <= 0:
                logger.warning(f"⚠️ limit={limit} 无效，使用默认值100")
                limit = 100
            
            # 转换格式
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            okx_interval = IntervalMapper.to_exchange_format(interval, "OKX")
            logger.debug(f"  转换后: okx_symbol={okx_symbol}, okx_interval={okx_interval}")
            
            # 使用SDK的市场数据API获取K线
            logger.debug(f"  调用 SDK market_api.get_candlesticks()...")
            response = self.market_api.get_candlesticks(
                instId=okx_symbol,
                bar=okx_interval,
                limit=str(limit),
                after=str(end_time) if end_time else None,
                before=str(start_time) if start_time else None
            )
            
            logger.debug(f"  SDK 响应: code={response.get('code')}, msg={response.get('msg')}")
            
            if response['code'] != '0':
                logger.error(f"❌ 获取K线失败: code={response['code']}, msg={response['msg']}")
                return []
            
            klines = response.get('data', [])
            logger.debug(f"  收到 {len(klines)} 条原始K线数据")
            
            # 转换为统一格式
            formatted_klines = []
            for idx, kline in enumerate(klines):
                try:
                    # 使用安全转换处理K线数据
                    formatted_kline = UnifiedKlineData(
                        timestamp=self._safe_int(kline[0]),
                        open=self._safe_float(kline[1]),
                        high=self._safe_float(kline[2]),
                        low=self._safe_float(kline[3]),
                        close=self._safe_float(kline[4]),
                        volume=self._safe_float(kline[5]),
                        close_time=self._safe_int(kline[0]) + self._interval_to_ms(interval) - 1,
                        quote_volume=self._safe_float(kline[6]),
                        trades=0,  # OKX不提供此字段
                        taker_buy_base_volume=0.0,  # OKX不提供此字段
                        taker_buy_quote_volume=0.0  # OKX不提供此字段
                    )
                    formatted_klines.append(formatted_kline)
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"❌ 解析第 {idx} 条K线数据失败: {e}")
                    logger.error(f"   原始数据: {kline}")
                    continue
            
            # OKX返回的数据是倒序的，需要反转
            formatted_klines.reverse()
            
            logger.info(f"✅ 获取OKX K线数据成功: {symbol} {interval} {len(formatted_klines)}条")
            return formatted_klines
            
        except Exception as e:
            logger.error(f"❌ 获取K线数据异常: {type(e).__name__} - {str(e)}")
            self._handle_sdk_exception(e)
            return []
    
    def _interval_to_ms(self, interval: str) -> int:
        """将K线周期转换为毫秒数"""
        unit = interval[-1]
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

    
    def get_klines_paginated(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        rate_limit_delay: float = 0.1
    ) -> List[UnifiedKlineData]:
        """分页获取K线数据"""
        try:
            if limit <= 300:
                return self.get_klines(symbol, interval, limit, start_time, end_time)
            
            all_klines = []
            max_per_request = 300
            batches_needed = (limit + max_per_request - 1) // max_per_request
            
            logger.debug(f"📊 分页获取OKX K线: {symbol} {interval} 需要{limit}条，分{batches_needed}批获取")
            
            current_end_time = end_time
            
            for batch in range(batches_needed):
                remaining = limit - len(all_klines)
                batch_limit = min(max_per_request, remaining)
                
                if batch_limit <= 0:
                    break
                
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
                
                all_klines.extend(klines)
                
                if len(all_klines) >= limit:
                    break
                
                if len(klines) < batch_limit:
                    logger.debug(f"📊 批次 {batch + 1}/{batches_needed} 返回{len(klines)}条 < 请求{batch_limit}条，数据已获取完毕")
                    break
                
                # 设置下一批次的 end_time
                current_end_time = klines[0].timestamp - 1
                
                # API限流
                if batch < batches_needed - 1:
                    time.sleep(rate_limit_delay)
            
            # 按时间戳排序
            all_klines.sort(key=lambda x: x.timestamp)
            
            # 去重
            seen_timestamps = set()
            unique_klines = []
            for kline in all_klines:
                if kline.timestamp not in seen_timestamps:
                    seen_timestamps.add(kline.timestamp)
                    unique_klines.append(kline)
            
            logger.debug(f"✅ 分页获取完成: {symbol} {interval} 共{len(unique_klines)}条（去重后）")
            return unique_klines[:limit]
            
        except Exception as e:
            logger.error(f"❌ 分页获取OKX K线数据失败: {e}")
            return []
    
    def get_ticker_price(self, symbol: str) -> Optional[UnifiedTickerData]:
        """获取实时价格"""
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的市场数据API获取价格
            response = self.market_api.get_ticker(instId=okx_symbol)
            
            if response['code'] != '0':
                logger.error(f"获取价格失败: {response['msg']}")
                return None
            
            tickers = response.get('data', [])
            
            if tickers:
                ticker = tickers[0]
                logger.debug(f"  价格数据: last={repr(ticker.get('last'))}, ts={repr(ticker.get('ts'))}")
                return UnifiedTickerData(
                    symbol=symbol,
                    price=self._safe_float(ticker.get('last'), 0.0),
                    timestamp=self._safe_int(ticker.get('ts'), 0)
                )
            
            return None
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX实时价格失败: {symbol} - {e}")
            return None

    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            logger.debug("💰 请求获取账户信息...")
            
            # 使用SDK的账户API获取余额
            logger.debug("  调用 SDK account_api.get_account_balance()...")
            response = self.account_api.get_account_balance()
            
            logger.debug(f"  SDK 响应: code={response.get('code')}, msg={response.get('msg')}")
            
            if response['code'] != '0':
                logger.error(f"❌ 获取账户信息失败: code={response['code']}, msg={response['msg']}")
                return {}
            
            balance_data = response.get('data', [])
            if not balance_data:
                logger.warning("⚠️ 账户余额数据为空")
                return {}
            
            account = balance_data[0]
            logger.debug(f"  账户原始数据: {account}")
            logger.debug(f"  totalEq={repr(account.get('totalEq'))}, availEq={repr(account.get('availEq'))}")
            
            # 格式化账户信息 - 使用安全转换
            formatted_account = {
                'total_wallet_balance': self._safe_float(account.get('totalEq'), 0.0),
                'total_unrealized_pnl': 0.0,  # 需要从持仓信息计算
                'total_margin_balance': self._safe_float(account.get('totalEq'), 0.0),
                'available_balance': self._safe_float(account.get('availEq'), 0.0),
                'can_trade': True,
                'update_time': self._safe_int(account.get('uTime'), 0)
            }
            
            logger.info(f"✅ 获取账户信息成功: 总资产={formatted_account['total_wallet_balance']}, 可用={formatted_account['available_balance']}")
            return formatted_account
            
        except Exception as e:
            logger.error(f"❌ 获取账户信息异常: {type(e).__name__} - {str(e)}")
            self._handle_sdk_exception(e)
            return {}
    
    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        try:
            # 使用SDK的账户API获取持仓
            if symbol:
                okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
                response = self.account_api.get_positions(instId=okx_symbol)
            else:
                response = self.account_api.get_positions()
            
            if response['code'] != '0':
                logger.error(f"获取持仓信息失败: {response['msg']}")
                return []
            
            positions = response.get('data', [])
            
            # 过滤有持仓的合约
            active_positions = []
            for position in positions:
                logger.debug(f"  持仓原始数据: {position}")
                position_amt = self._safe_float(position.get('pos'), 0.0)
                if position_amt != 0:
                    formatted_position = {
                        'symbol': SymbolMapper.to_standard_format(position.get('instId', ''), "OKX"),
                        'position_amt': position_amt,
                        'entry_price': self._safe_float(position.get('avgPx'), 0.0),
                        'mark_price': self._safe_float(position.get('markPx'), 0.0),
                        'pnl': self._safe_float(position.get('upl'), 0.0),
                        'percentage': self._safe_float(position.get('uplRatio'), 0.0) * 100,
                        'position_side': position.get('posSide', 'net'),
                        'margin_type': position.get('mgnMode', 'cross'),
                        'leverage': self._safe_int(position.get('lever'), 1),
                        'update_time': self._safe_int(position.get('uTime'), 0)
                    }
                    active_positions.append(formatted_position)
                    logger.debug(f"  ✅ 添加持仓: {formatted_position['symbol']}, 数量={formatted_position['position_amt']}")
            
            return active_positions
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX持仓信息失败: {e}")
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
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 构建订单参数
            order_params = {
                'instId': okx_symbol,
                'tdMode': 'cross',  # 全仓模式
                'side': 'buy' if side == 'BUY' else 'sell',
                'ordType': 'market' if order_type == 'MARKET' else 'limit',
                'sz': str(quantity)
            }
            
            if price is not None:
                order_params['px'] = str(price)
            
            if reduce_only:
                order_params['reduceOnly'] = 'true'
            
            # 使用SDK的交易API下单
            response = self.trade_api.place_order(**order_params)
            
            if response['code'] != '0':
                logger.error(f"下单失败: {response['msg']}")
                raise ExchangeAPIError(response['code'], response['msg'])
            
            order_data = response.get('data', [])
            if order_data:
                result = order_data[0]
                logger.info(f"✅ OKX下单成功: {symbol} {side} {quantity} @ {price}")
                return {
                    'orderId': result.get('ordId', ''),
                    'status': 'NEW',
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity,
                    'price': price
                }
            else:
                logger.error(f"❌ OKX下单失败: 返回数据为空")
                return {}
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ OKX下单失败: {e}")
            return {}
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """取消订单"""
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的交易API取消订单
            response = self.trade_api.cancel_order(instId=okx_symbol, ordId=order_id)
            
            if response['code'] != '0':
                logger.error(f"取消订单失败: {response['msg']}")
                return {}
            
            logger.info(f"✅ OKX撤销订单成功: {symbol} {order_id}")
            return response
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ OKX撤销订单失败: {e}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        try:
            # 使用SDK的交易API获取未成交订单
            if symbol:
                okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
                response = self.trade_api.get_order_list(instType='SWAP', instId=okx_symbol)
            else:
                response = self.trade_api.get_order_list(instType='SWAP')
            
            if response['code'] != '0':
                logger.error(f"获取未成交订单失败: {response['msg']}")
                return []
            
            return response.get('data', [])
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX未成交订单失败: {e}")
            return []

    
    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """修改杠杆倍数"""
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的账户API设置杠杆
            response = self.account_api.set_leverage(
                instId=okx_symbol,
                lever=str(leverage),
                mgnMode='cross'
            )
            
            if response['code'] != '0':
                logger.error(f"修改杠杆失败: {response['msg']}")
                return {}
            
            logger.info(f"✅ OKX修改杠杆成功: {symbol} {leverage}x")
            return response
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ OKX修改杠杆失败: {e}")
            return {}
    
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """修改保证金模式"""
        try:
            # OKX的保证金模式在下单时指定，这里返回成功
            logger.info(f"✅ OKX保证金模式: {margin_type}")
            return {'success': True, 'margin_type': margin_type}
            
        except Exception as e:
            logger.error(f"❌ OKX修改保证金模式失败: {e}")
            return {}
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        try:
            # 使用SDK的公共API获取交易所信息
            response = self.public_api.get_instruments(instType='SWAP')
            
            if response['code'] != '0':
                logger.error(f"获取交易所信息失败: {response['msg']}")
                return {}
            
            return response
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX交易所信息失败: {e}")
            return {}
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取交易对信息"""
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的公共API获取交易对信息
            response = self.public_api.get_instruments(instType='SWAP', instId=okx_symbol)
            
            if response['code'] != '0':
                logger.error(f"获取交易对信息失败: {response['msg']}")
                return None
            
            instruments = response.get('data', [])
            return instruments[0] if instruments else None
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX交易对信息失败: {e}")
            return None



class OKXWebSocketClient:
    """
    OKX WebSocket客户端（支持自动重连和心跳保活）
    
    注意：python-okx SDK不提供WebSocket封装，需要手动实现
    """
    
    def __init__(self):
        """初始化OKX WebSocket客户端"""
        self.ws: Optional[websocket.WebSocketApp] = None
        self.callbacks: Dict[str, Callable] = {}
        self.is_connected = False
        self.is_running = False
        self.is_reconnecting = False
        self.subscriptions = []  # 保存订阅信息以便重连后恢复
        self.reconnect_task = None
        self.monitor_task = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.last_message_time = None
        self.health_check_task = None
        
        # WebSocket URL
        # 🔥 根据OKX文档，K线频道需要使用business地址
        if settings.OKX_TESTNET:
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/business"  # 模拟盘
        else:
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/business"  # 实盘（K线频道使用business地址）
        
        logger.info(f"✅ OKX WebSocket客户端初始化完成")
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
            
            # 添加代理配置（仅在USE_PROXY_WS启用时）
            if settings.USE_PROXY and settings.USE_PROXY_WS:
                proxy_type = settings.PROXY_TYPE.lower()
                if proxy_type == "socks5":
                    # SOCKS5代理（websocket-client通过http_proxy环境变量支持）
                    proxy_url = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                    import os
                    os.environ['http_proxy'] = proxy_url
                    os.environ['https_proxy'] = proxy_url
                    logger.info(f"🔧 OKX WebSocket使用SOCKS5代理: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
                else:
                    # HTTP/HTTPS代理
                    ws_kwargs["http_proxy_host"] = settings.PROXY_HOST
                    ws_kwargs["http_proxy_port"] = settings.PROXY_PORT
                    logger.info(f"🔧 OKX WebSocket使用{proxy_type.upper()}代理: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
            elif settings.USE_PROXY and not settings.USE_PROXY_WS:
                logger.info("✅ OKX WebSocket直连（不使用代理），仅REST API使用代理")
            
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
            
            # 运行WebSocket，添加ping/pong机制防止连接超时
            # OKX要求每30秒发送一次ping，否则会断开连接
            self.ws.run_forever(
                sslopt=sslopt,
                ping_interval=25,  # 每25秒发送一次ping（小于OKX的30秒超时）
                ping_timeout=10    # ping超时时间10秒
            )
            
        except Exception as e:
            logger.error(f"❌ WebSocket运行失败: {e}")
            if not self.is_reconnecting:
                self._schedule_reconnect()

    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """订阅K线数据"""
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
        
        # 🔥 创建包装回调，传递symbol和interval信息
        def wrapped_callback(data):
            logger.debug(f"📞 wrapped_callback被调用: {symbol} {interval}, 数据长度={len(data) if isinstance(data, list) else 'N/A'}")
            callback(data, symbol, interval)
        
        # 保存回调
        callback_key = f"{channel}:{okx_symbol}"
        self.callbacks[callback_key] = wrapped_callback
        logger.info(f"✅ 注册OKX K线回调: {callback_key}, 回调函数: {callback.__name__ if hasattr(callback, '__name__') else type(callback).__name__}, 已注册回调数: {len(self.callbacks)}")
        
        # 发送订阅消息
        if self.ws and self.is_connected:
            sub_msg_str = json.dumps(sub_msg)
            self.ws.send(sub_msg_str)
            logger.info(f"✅ 发送订阅消息: {okx_symbol} {okx_interval}, channel={channel}, 消息={sub_msg_str}")
        else:
            logger.warning(f"⚠️ WebSocket未连接，订阅将在连接建立后自动恢复: {okx_symbol} {okx_interval}")
    
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """订阅价格数据"""
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
            
            # 🔥 记录所有收到的WebSocket消息（用于调试）
            logger.debug(f"📥 收到WebSocket原始消息: {message[:200]}...")  # 只记录前200字符
            
            # 处理订阅确认（OKX可能返回多种格式）
            if data.get('event') == 'subscribe':
                arg = data.get('arg', {})
                channel = arg.get('channel', '')
                inst_id = arg.get('instId', '')
                code = data.get('code', '')
                msg = data.get('msg', '')
                # 🔥 OKX订阅成功时code为空字符串或'0'，失败时code不为空
                if code and code != '0' and code != 0:
                    logger.error(f"❌ 订阅失败: channel={channel}, instId={inst_id}, code={code}, msg={msg}")
                else:
                    logger.info(f"✅ 订阅成功: channel={channel}, instId={inst_id}")
                return
            
            # 处理错误消息（可能包含订阅失败信息）
            if data.get('event') == 'error':
                code = data.get('code', '')
                msg = data.get('msg', '')
                logger.error(f"❌ WebSocket错误: code={code}, msg={msg}")
                return
            
            # 处理数据推送
            if 'data' in data:
                arg = data.get('arg', {})
                channel = arg.get('channel', '')
                inst_id = arg.get('instId', '')
                
                callback_key = f"{channel}:{inst_id}"
                
                # 🔥 记录所有收到的数据推送（用于诊断）
                logger.debug(f"📥 收到数据推送: channel={channel}, instId={inst_id}, 数据长度={len(data['data']) if isinstance(data['data'], list) else 'N/A'}")
                
                # 🔥 区分日志级别：只记录已完成的K线，tickers用DEBUG
                if channel.startswith('candle'):
                    # K线数据：只记录已完成的K线
                    data_list = data['data'] if isinstance(data['data'], list) else []
                    if isinstance(data_list, list) and len(data_list) > 0:
                        first_item = data_list[0]
                        if isinstance(first_item, list) and len(first_item) >= 8:
                            # 🔥 OKX文档：第8个字段是confirm（不是is_closed）
                            # confirm=1表示K线已完成，confirm=0表示K线未完成
                            confirm = first_item[7]
                            is_closed = (str(confirm) == "1" or confirm == 1)
                            close_price = first_item[4]
                            # 🔥 只记录已完成的K线
                            if is_closed:
                                logger.info(f"📊 收到已完成K线: {channel} {inst_id}, close={close_price}, confirm={confirm}")
                            else:
                                logger.debug(f"📥 收到进行中K线: {channel} {inst_id}, close={close_price}, confirm={confirm}")
                        else:
                            logger.debug(f"📥 收到K线数据: {channel} {inst_id}, 数据长度={len(data_list)}")
                    else:
                        logger.debug(f"📥 收到K线数据: {channel} {inst_id}, 数据为空")
                else:
                    # tickers等其他数据：只记录DEBUG级别
                    logger.debug(f"📥 收到WebSocket数据: channel={channel}, instId={inst_id}, 数据长度={len(data['data']) if isinstance(data['data'], list) else 'N/A'}")
                
                if callback_key in self.callbacks:
                    if channel.startswith('candle'):
                        logger.debug(f"✅ 调用K线回调: {callback_key}")
                    else:
                        logger.debug(f"✅ 调用回调: {callback_key}")
                    self.callbacks[callback_key](data['data'])
                else:
                    logger.warning(f"⚠️ 未找到回调函数: {callback_key}, 已注册的回调: {list(self.callbacks.keys())}")
            else:
                # 记录其他类型的消息（用于调试）
                logger.debug(f"📥 收到其他类型WebSocket消息: {list(data.keys())}")
            
        except Exception as e:
            logger.error(f"❌ 处理WebSocket消息失败: {e}")
            logger.error(f"   原始消息: {message[:500] if len(message) > 500 else message}")
    
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
        if not self.ws or not self.is_connected:
            logger.warning("⚠️ WebSocket未连接，无法恢复订阅")
            return
        
        logger.info(f"📋 开始恢复 {len(self.subscriptions)} 个订阅...")
        for sub in self.subscriptions:
            try:
                sub_msg = {
                    "op": "subscribe",
                    "args": [{
                        "channel": sub['channel'],
                        "instId": sub['inst_id']
                    }]
                }
                sub_msg_str = json.dumps(sub_msg)
                self.ws.send(sub_msg_str)
                logger.info(f"✅ 恢复订阅: {sub['channel']} {sub['inst_id']}, 消息={sub_msg_str}")
            except Exception as e:
                logger.error(f"❌ 恢复订阅失败: {sub}, 错误={e}")
    
    def _schedule_reconnect(self):
        """安排重连"""
        if self.is_reconnecting:
            return
        
        self.is_reconnecting = True
        logger.info("🔄 准备重连OKX WebSocket...")
        
        # 使用指数退避策略重连
        time.sleep(5)  # 简单延迟
        
        if self.is_running:
            self.start_websocket()
    
    def stop_websocket(self):
        """停止WebSocket连接"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("✅ OKX WebSocket已停止")
