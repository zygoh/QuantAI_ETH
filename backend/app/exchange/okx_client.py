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
    # ✅ 新增：导入交易大数据模块 (Rubik)
    import okx.TradingData as TradingDataModule
    
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
    TradingDataModule = None



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
            
            # ✅ 新增：初始化 TradingDataAPI (Rubik)
            logger.debug("  初始化 TradingData API...")
            if TradingDataModule is None:
                logger.warning("  ⚠️ TradingDataModule 未导入，跳过初始化")
                self.trading_data_api = None
            else:
                if hasattr(TradingDataModule, 'TradingDataAPI'):
                    TradingDataAPIClass = TradingDataModule.TradingDataAPI
                elif hasattr(TradingDataModule, 'TradingData'):
                    TradingDataAPIClass = TradingDataModule.TradingData
                else:
                    TradingDataAPIClass = TradingDataModule
                
                self.trading_data_api = TradingDataAPIClass(
                    api_key=self.api_key,
                    api_secret_key=self.secret_key,
                    passphrase=self.passphrase,
                    flag=flag,
                    proxy=proxy if proxy else {}
                )
                logger.debug("  ✅ TradingData API 初始化成功")
            
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
            # 🔥 OKX K线数组格式：[timestamp, open, high, low, close, volume, volCcyQuote, volCcy, confirm]
            # 索引：            [0,       1,    2,    3,    4,     5,      6,           7,       8]
            # 根据OKX文档：https://www.okx.com/docs-v5/zh/#order-book-trading-market-data-get-candlesticks-history
            formatted_klines = []
            skipped_incomplete = 0
            skipped_invalid = 0
            
            for idx, kline in enumerate(klines):
                try:
                    # ✅ 修复：检查数组长度应为9（包含confirm字段）
                    if len(kline) < 9:
                        logger.warning(f"⚠️ 第 {idx} 条K线数据长度不足: {len(kline)} < 9（期望9个元素）")
                        logger.warning(f"   原始数据: {kline}")
                        skipped_invalid += 1
                        continue
                    
                    # ✅ 修复：提取confirm字段（索引8）判断K线是否完成
                    confirm = kline[8]
                    # OKX可能返回字符串"1"或数字1，统一处理
                    confirm_str = str(confirm).strip()
                    is_closed = (confirm_str == "1" or confirm == 1)
                    
                    # ✅ 修复：只处理已完成的K线（confirm=1）
                    if not is_closed:
                        skipped_incomplete += 1
                        logger.debug(f"⏸️ 跳过未完成K线: 索引={idx}, confirm={confirm}")
                        continue
                    
                    # ✅ 验证关键字段有效性
                    close_price = self._safe_float(kline[4])
                    volume = self._safe_float(kline[5])
                    
                    # 过滤无效数据（close<=0或volume<0）
                    if close_price <= 0:
                        logger.warning(f"⚠️ 跳过无效K线（close<=0）: 索引={idx}, close={close_price}")
                        skipped_invalid += 1
                        continue
                    
                    # 使用安全转换处理K线数据
                    formatted_kline = UnifiedKlineData(
                        timestamp=self._safe_int(kline[0]),
                        open=self._safe_float(kline[1]),
                        high=self._safe_float(kline[2]),
                        low=self._safe_float(kline[3]),
                        close=close_price,
                        volume=volume,
                        close_time=self._safe_int(kline[0]) + self._interval_to_ms(interval) - 1,
                        quote_volume=self._safe_float(kline[6]),  # volCcyQuote
                        trades=0,  # OKX不提供此字段
                        taker_buy_base_volume=0.0,  # OKX不提供此字段
                        taker_buy_quote_volume=0.0  # OKX不提供此字段
                    )
                    formatted_klines.append(formatted_kline)
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"❌ 解析第 {idx} 条K线数据失败: {e}")
                    logger.error(f"   原始数据: {kline}, 长度={len(kline) if isinstance(kline, list) else 'N/A'}")
                    skipped_invalid += 1
                    continue
            
            # ✅ 记录过滤统计
            if skipped_incomplete > 0:
                logger.info(f"📊 已过滤 {skipped_incomplete} 条未完成K线（confirm!=1）")
            if skipped_invalid > 0:
                logger.warning(f"⚠️ 已跳过 {skipped_invalid} 条无效K线（格式错误或数据无效）")
            
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
    
    def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取资金费率
        
        Args:
            symbol: 交易对符号
        
        Returns:
            资金费率数据字典，包含：
            - funding_rate: 当前资金费率
            - next_funding_time: 下次资金费率时间
            - funding_rate_8h: 8小时资金费率
        """
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的公共API获取资金费率
            # OKX API: GET /api/v5/public/funding-rate
            response = self.public_api.get_funding_rate(instId=okx_symbol)
            
            if response['code'] != '0':
                logger.error(f"获取资金费率失败: {response['msg']}")
                return None
            
            data_list = response.get('data', [])
            if not data_list:
                logger.warning(f"资金费率数据为空: {okx_symbol}")
                return None
            
            # 取最新的一条
            data = data_list[0]
            
            return {
                'funding_rate': self._safe_float(data.get('fundingRate'), 0.0),
                'next_funding_time': int(data.get('nextFundingTime', 0)),
                'funding_rate_8h': self._safe_float(data.get('fundingRate'), 0.0),  # 8小时资金费率
                'timestamp': int(data.get('ts', time.time() * 1000))
            }
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX资金费率失败: {e}")
            return None
    
    def get_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取持仓量（Open Interest）
        
        Args:
            symbol: 交易对符号
        
        Returns:
            持仓量数据字典，包含：
            - open_interest: 持仓量（张数）
            - open_interest_usd: 持仓量（USDT价值）
            - timestamp: 时间戳
        """
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的公共API获取持仓量
            # OKX API: GET /api/v5/public/open-interest
            response = self.public_api.get_open_interest(instId=okx_symbol)
            
            if response['code'] != '0':
                logger.error(f"获取持仓量失败: {response['msg']}")
                return None
            
            data_list = response.get('data', [])
            if not data_list:
                logger.warning(f"持仓量数据为空: {okx_symbol}")
                return None
            
            # 取最新的一条
            data = data_list[0]
            
            return {
                'open_interest': self._safe_float(data.get('oi', 0.0)),  # 持仓量（张数）
                'open_interest_usd': self._safe_float(data.get('oiCcy', 0.0)),  # 持仓量（USDT价值）
                'timestamp': int(data.get('ts', time.time() * 1000))
            }
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX持仓量失败: {e}")
            return None
    
    def get_long_short_ratio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取多空持仓人数比
        
        Args:
            symbol: 交易对符号
        
        Returns:
            多空比数据字典，包含：
            - long_short_ratio: 多空持仓人数比
            - long_account: 多头账户数
            - short_account: 空头账户数
            - timestamp: 时间戳
        """
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的公共API获取多空持仓人数比
            # OKX API: GET /api/v5/public/retail-margin
            response = self.public_api.get_retail_margin(instId=okx_symbol)
            
            if response['code'] != '0':
                logger.error(f"获取多空比失败: {response['msg']}")
                return None
            
            data_list = response.get('data', [])
            if not data_list:
                logger.warning(f"多空比数据为空: {okx_symbol}")
                return None
            
            # 取最新的一条
            data = data_list[0]
            
            long_account = self._safe_float(data.get('longRatio', 0.0))
            short_account = self._safe_float(data.get('shortRatio', 0.0))
            
            # 计算多空比（避免除零）
            long_short_ratio = long_account / (short_account + 1e-10) if short_account > 0 else 0.0
            
            return {
                'long_short_ratio': long_short_ratio,
                'long_account': long_account,
                'short_account': short_account,
                'timestamp': int(data.get('ts', time.time() * 1000))
            }
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX多空比失败: {e}")
            return None
    
    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[Dict[str, Any]]:
        """
        获取订单簿数据
        
        Args:
            symbol: 交易对符号
            depth: 深度（默认5档）
        
        Returns:
            订单簿数据字典，包含：
            - bids: 买单列表 [[price, size], ...]
            - asks: 卖单列表 [[price, size], ...]
            - bid_volume_top5: 前5档买单总量
            - ask_volume_top5: 前5档卖单总量
            - order_book_imbalance: 订单簿不平衡度
            - timestamp: 时间戳
        """
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的市场数据API获取订单簿
            # OKX API: GET /api/v5/market/books
            response = self.market_api.get_books(instId=okx_symbol, sz=str(depth))
            
            if response['code'] != '0':
                logger.error(f"获取订单簿失败: {response['msg']}")
                return None
            
            data_list = response.get('data', [])
            if not data_list:
                logger.warning(f"订单簿数据为空: {okx_symbol}")
                return None
            
            # 取最新的一条
            data = data_list[0]
            
            bids = data.get('bids', [])  # [[price, size, ...], ...]
            asks = data.get('asks', [])  # [[price, size, ...], ...]
            
            # 计算前5档买卖总量
            bid_volume_top5 = sum(float(bid[1]) for bid in bids[:5])
            ask_volume_top5 = sum(float(ask[1]) for ask in asks[:5])
            
            # 计算订单簿不平衡度
            total_volume = bid_volume_top5 + ask_volume_top5
            order_book_imbalance = (bid_volume_top5 - ask_volume_top5) / (total_volume + 1e-10) if total_volume > 0 else 0.0
            
            return {
                'bids': bids,
                'asks': asks,
                'bid_volume_top5': bid_volume_top5,
                'ask_volume_top5': ask_volume_top5,
                'order_book_imbalance': order_book_imbalance,
                'timestamp': int(data.get('ts', time.time() * 1000))
            }
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX订单簿失败: {e}")
            return None
    
    def get_large_trades(self, symbol: str, min_amount: float = 100000.0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取大单交易数据（过滤小额成交）
        
        Args:
            symbol: 交易对符号
            min_amount: 最小金额阈值（USDT，默认10万）
            limit: 返回数量限制
        
        Returns:
            大单交易列表，每个元素包含：
            - price: 成交价格
            - size: 成交数量
            - side: 方向（buy/sell）
            - amount: 成交金额（USDT）
            - timestamp: 时间戳
        """
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 使用SDK的市场数据API获取成交数据
            # OKX API: GET /api/v5/market/trades
            response = self.market_api.get_trades(instId=okx_symbol, limit=str(limit))
            
            if response['code'] != '0':
                logger.error(f"获取成交数据失败: {response['msg']}")
                return []
            
            trades = response.get('data', [])
            
            # 过滤大单
            large_trades = []
            for trade in trades:
                price = self._safe_float(trade.get('px', 0.0))
                size = self._safe_float(trade.get('sz', 0.0))
                amount = price * size  # 成交金额
                
                if amount >= min_amount:
                    large_trades.append({
                        'price': price,
                        'size': size,
                        'side': trade.get('side', 'buy'),  # 'buy' 或 'sell'
                        'amount': amount,
                        'timestamp': int(trade.get('ts', time.time() * 1000))
                    })
            
            return large_trades
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ 获取OKX大单数据失败: {e}")
            return []
    
    def get_historical_funding_rate(
        self, 
        symbol: str, 
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取历史资金费率数据（SDK版，带自动分页）
        
        OKX API: GET /api/v5/public/funding-rate-history
        对应接口: PublicDataAPI.get_funding_rate_history
        
        Args:
            symbol: 交易对符号
            start_time: 开始时间戳（毫秒），可选
            end_time: 结束时间戳（毫秒），可选
            limit: API单次限制（默认100，最大100）
        
        Returns:
            历史资金费率列表，每个元素包含：
            - funding_rate: 资金费率
            - timestamp: 时间戳（毫秒）
        """
        try:
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            all_data = []
            
            # 时间处理
            current_end = end_time  # 如果为None，API默认返回最新数据
            target_start = start_time or (int(time.time() * 1000) - 90 * 24 * 60 * 60 * 1000)  # 默认90天
            
            logger.info(f"🔄 [SDK] 开始分页获取资金费率: {symbol}")
            
            # 分页游标 (after 参数用于获取更旧的数据)
            cursor_after = None
            max_iterations = 100  # 防止无限循环
            iteration = 0
            
            while iteration < max_iterations:
                # 构造参数
                # 注意：PublicData API 的分页参数通常是 'after' (请求此时间戳之前/更旧的数据)
                kwargs = {
                    'instId': okx_symbol,
                    'limit': str(limit)
                }
                
                # 如果有游标，传入 after
                if cursor_after:
                    kwargs['after'] = str(cursor_after)
                # 如果没有游标但有指定的结束时间，也可以作为起始点
                elif current_end:
                    kwargs['after'] = str(current_end)
                    # 注意：OKX API对于第一次请求，如果不传after默认返回最新。
                    # 如果我们想从特定的 end_time 开始往前查，应该把 end_time 传给 after
                    # 但需要注意 end_time 数据本身可能不包含在内，视具体API行为微调
                
                # ✅ SDK 调用
                # 检查方法名，通常是 get_funding_rate_history
                try:
                    if hasattr(self.public_api, 'get_funding_rate_history'):
                        response = self.public_api.get_funding_rate_history(**kwargs)
                    elif hasattr(self.public_api, 'funding_rate_history'):
                        response = self.public_api.funding_rate_history(**kwargs)
                    else:
                        logger.error("❌ SDK中未找到资金费率历史方法")
                        logger.error(f"   可用方法: {[m for m in dir(self.public_api) if not m.startswith('_')]}")
                        break
                except AttributeError as e:
                    logger.error(f"❌ SDK方法不存在: {e}")
                    logger.error(f"   可用方法: {[m for m in dir(self.public_api) if not m.startswith('_')]}")
                    break
                except Exception as e:
                    logger.error(f"❌ 调用SDK API失败: {e}")
                    break
                
                if response.get('code') != '0':
                    logger.warning(f"SDK API错误: {response.get('msg')}")
                    break
                
                data_list = response.get('data', [])
                if not data_list:
                    break
                
                # 数据转换
                batch_data = []
                min_ts_in_batch = float('inf')
                
                for item in data_list:
                    # OKX返回的字段: fundingRate, fundingTime (或 ts)
                    ts = int(item.get('ts') or item.get('fundingTime') or time.time() * 1000)
                    min_ts_in_batch = min(min_ts_in_batch, ts)
                    
                    if ts < target_start:
                        continue
                    
                    batch_data.append({
                        'funding_rate': self._safe_float(item.get('fundingRate'), 0.0),
                        'timestamp': ts
                    })
                
                all_data.extend(batch_data)
                logger.debug(f"   SDK已获取 {len(batch_data)} 条, 当前最早时间: {datetime.fromtimestamp(min_ts_in_batch/1000)}")
                
                # 分页终止条件
                if min_ts_in_batch <= target_start or len(data_list) < limit:
                    break
                
                # 更新游标：使用本批次最后一条数据的时间戳作为下一次请求的 'after'
                cursor_after = data_list[-1].get('ts') or data_list[-1].get('fundingTime')
                if not cursor_after:
                    break
                
                iteration += 1
                time.sleep(0.1)  # 限流保护
            
            # 去重并排序
            unique_data = {x['timestamp']: x for x in all_data}.values()
            sorted_data = sorted(unique_data, key=lambda x: x['timestamp'])
            
            logger.info(f"✅ [SDK] 历史资金费率获取完成: 共 {len(sorted_data)} 条 (分页{iteration}次)")
            return sorted_data
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ [SDK] 获取OKX资金费率失败: {e}")
            return []
    
    def get_historical_open_interest(
        self, 
        symbol: str, 
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None,
        period: str = "5m",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取历史持仓量数据（SDK版，带自动分页）
        
        OKX API: GET /api/v5/rubik/stat/contracts/open-interest-volume
        对应接口: TradingDataAPI.get_contracts_open_interest_volume
        
        Args:
            symbol: 交易对符号
            start_time: 开始时间戳（毫秒），可选
            end_time: 结束时间戳（毫秒），可选
            period: 时间周期（如5m, 15m, 1H, 4H, 1D），默认5m
            limit: 每次请求的数量限制（默认100）
        
        Returns:
            历史持仓量列表，每个元素包含：
            - open_interest: 持仓量（张数）
            - open_interest_usd: 持仓量（USDT价值）
            - timestamp: 时间戳（毫秒）
        """
        try:
            # 如果SDK未初始化，降级使用HTTP请求
            if self.trading_data_api is None:
                logger.warning("⚠️ TradingDataAPI未初始化，无法使用SDK")
                return []
            
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            all_data = []
            
            # 时间处理
            current_end = end_time or int(time.time() * 1000)
            target_start = start_time or (current_end - 24 * 60 * 60 * 1000)  # 默认24小时前
            
            logger.info(f"🔄 [SDK] 开始分页获取持仓量: {symbol}, 目标范围: {target_start} -> {current_end}")
            
            max_iterations = 100  # 防止无限循环
            iteration = 0
            
            while iteration < max_iterations:
                try:
                    # ✅ SDK 调用
                    # 注意：不同版本的 SDK 方法名可能略有不同
                    # 如果报错 AttributeError，请尝试 get_open_interest_volume
                    if hasattr(self.trading_data_api, 'get_contracts_open_interest_volume'):
                        response = self.trading_data_api.get_contracts_open_interest_volume(
                            instId=okx_symbol,
                            period=period,
                            limit=str(limit),
                            begin=str(current_end),  # 获取比这个时间更早的数据
                            end=str(target_start)   # (可选) 截止时间
                        )
                    elif hasattr(self.trading_data_api, 'get_open_interest_volume'):
                        response = self.trading_data_api.get_open_interest_volume(
                            instId=okx_symbol,
                            period=period,
                            limit=str(limit),
                            begin=str(current_end),
                            end=str(target_start)
                        )
                    else:
                        logger.error(f"❌ TradingDataAPI 不支持持仓量历史方法，可用方法: {[m for m in dir(self.trading_data_api) if not m.startswith('_')]}")
                        break
                    
                    if response.get('code') != '0':
                        logger.warning(f"SDK API错误: {response.get('msg')}")
                        break
                    
                    data_list = response.get('data', [])
                    if not data_list:
                        break
                    
                    # 数据转换
                    batch_data = []
                    min_ts_in_batch = current_end
                    
                    for item in data_list:
                        ts = int(item.get('ts', time.time() * 1000))
                        min_ts_in_batch = min(min_ts_in_batch, ts)
                        
                        if ts < target_start:
                            continue
                        
                        batch_data.append({
                            'open_interest': self._safe_float(item.get('oi'), 0.0),
                            'open_interest_usd': self._safe_float(item.get('oiCcy'), 0.0),
                            'timestamp': ts
                        })
                    
                    all_data.extend(batch_data)
                    logger.debug(f"   SDK已获取 {len(batch_data)} 条, 当前最早时间: {datetime.fromtimestamp(min_ts_in_batch/1000)}")
                    
                    # 分页终止条件
                    if min_ts_in_batch <= target_start or len(data_list) < limit:
                        break
                    
                    # 更新游标
                    current_end = min_ts_in_batch - 1
                    iteration += 1
                    time.sleep(0.1)  # 限流保护
                    
                except AttributeError as e:
                    logger.error(f"❌ SDK方法不存在: {e}")
                    logger.error(f"   可用方法: {[m for m in dir(self.trading_data_api) if not m.startswith('_')]}")
                    break
                except Exception as e:
                    logger.error(f"❌ 调用SDK API失败: {e}")
                    break
            
            all_data.sort(key=lambda x: x['timestamp'])
            logger.info(f"✅ [SDK] 历史持仓量获取完成: 共 {len(all_data)} 条 (分页{iteration}次)")
            return all_data
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ [SDK] 获取OKX历史持仓量失败: {e}")
            return []
    
    def get_historical_long_short_ratio(
        self, 
        symbol: str, 
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None,
        period: str = "5m",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取历史多空比数据（SDK版，带自动分页）
        
        OKX API: GET /api/v5/rubik/stat/contracts/long-short-account-ratio
        对应接口: TradingDataAPI.get_contracts_long_short_account_ratio
        
        Args:
            symbol: 交易对符号（如ETH-USDT，会提取基础货币ETH作为ccy）
            start_time: 开始时间戳（毫秒），可选
            end_time: 结束时间戳（毫秒），可选
            period: 时间周期（如5m, 15m, 1H, 4H, 1D），默认5m
            limit: 每次请求的数量限制（默认100）
        
        Returns:
            历史多空比列表，每个元素包含：
            - long_short_ratio: 多空比
            - long_account: 多头账户比例
            - short_account: 空头账户比例
            - timestamp: 时间戳（毫秒）
        """
        try:
            # 如果SDK未初始化，降级使用HTTP请求
            if self.trading_data_api is None:
                logger.warning("⚠️ TradingDataAPI未初始化，无法使用SDK")
                return []
            
            okx_symbol = SymbolMapper.to_exchange_format(symbol, "OKX")
            
            # 从symbol提取基础货币（ccy）
            # 例如：ETH-USDT-SWAP -> ETH, BTC-USDT-SWAP -> BTC
            if '-SWAP' in okx_symbol:
                ccy = okx_symbol.split('-')[0]  # ETH-USDT-SWAP -> ETH
            elif '-' in okx_symbol:
                ccy = okx_symbol.split('-')[0]  # ETH-USDT -> ETH
            else:
                ccy = okx_symbol  # 如果格式不对，使用原值
            
            all_data = []
            current_end = end_time or int(time.time() * 1000)
            target_start = start_time or (current_end - 24 * 60 * 60 * 1000)  # 默认24小时前
            
            logger.info(f"🔄 [SDK] 开始分页获取多空比: {ccy}, 目标范围: {target_start} -> {current_end}")
            
            max_iterations = 100  # 防止无限循环
            iteration = 0
            
            while iteration < max_iterations:
                try:
                    # ✅ SDK 调用
                    # 方法名通常是 get_contracts_long_short_account_ratio
                    if hasattr(self.trading_data_api, 'get_contracts_long_short_account_ratio'):
                        response = self.trading_data_api.get_contracts_long_short_account_ratio(
                            ccy=ccy,
                            period=period,
                            limit=str(limit),
                            begin=str(current_end),
                            end=str(target_start)
                        )
                    elif hasattr(self.trading_data_api, 'get_long_short_account_ratio'):
                        response = self.trading_data_api.get_long_short_account_ratio(
                            ccy=ccy,
                            period=period,
                            limit=str(limit),
                            begin=str(current_end),
                            end=str(target_start)
                        )
                    else:
                        logger.error(f"❌ TradingDataAPI 不支持多空比历史方法，可用方法: {[m for m in dir(self.trading_data_api) if not m.startswith('_')]}")
                        break
                    
                    if response.get('code') != '0':
                        logger.warning(f"SDK API错误: {response.get('msg')}")
                        break
                    
                    data_list = response.get('data', [])
                    if not data_list:
                        break
                    
                    batch_data = []
                    min_ts_in_batch = current_end
                    
                    for item in data_list:
                        ts = int(item.get('ts', time.time() * 1000))
                        min_ts_in_batch = min(min_ts_in_batch, ts)
                        
                        if ts < target_start:
                            continue
                        
                        long_account = self._safe_float(item.get('longRatio'), 0.0)
                        short_account = self._safe_float(item.get('shortRatio'), 0.0)
                        # 自动计算比率防止除零
                        ls_ratio = long_account / (short_account + 1e-10) if short_account > 0 else 0.0
                        
                        batch_data.append({
                            'long_short_ratio': ls_ratio,
                            'long_account': long_account,
                            'short_account': short_account,
                            'timestamp': ts
                        })
                    
                    all_data.extend(batch_data)
                    logger.debug(f"   SDK已获取 {len(batch_data)} 条, 当前最早时间: {datetime.fromtimestamp(min_ts_in_batch/1000)}")
                    
                    # 分页终止条件
                    if min_ts_in_batch <= target_start or len(data_list) < limit:
                        break
                    
                    # 更新游标
                    current_end = min_ts_in_batch - 1
                    iteration += 1
                    time.sleep(0.1)  # 限流保护
                    
                except AttributeError as e:
                    logger.error(f"❌ SDK方法不存在: {e}")
                    logger.error(f"   可用方法: {[m for m in dir(self.trading_data_api) if not m.startswith('_')]}")
                    break
                except Exception as e:
                    logger.error(f"❌ 调用SDK API失败: {e}")
                    break
            
            all_data.sort(key=lambda x: x['timestamp'])
            logger.info(f"✅ [SDK] 历史多空比获取完成: 共 {len(all_data)} 条 (分页{iteration}次)")
            return all_data
            
        except Exception as e:
            self._handle_sdk_exception(e)
            logger.error(f"❌ [SDK] 获取OKX历史多空比失败: {e}")
            return []



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
        # 🔥 根据OKX文档：
        # - K线频道（candle）需要使用 /ws/v5/business
        # - Tickers频道需要使用 /ws/v5/public
        # 当前统一使用business地址（主要用于K线），tickers订阅会失败但系统主要使用K线
        if settings.OKX_TESTNET:
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/business"  # 模拟盘
        else:
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/business"  # 实盘（K线频道使用business地址）
        
        # ⚠️ 注意：tickers频道需要使用 /ws/v5/public，但当前系统主要使用K线数据
        # 如果需要tickers，需要创建单独的WebSocket连接或使用不同的URL
        
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
            
            # 🔥 记录所有收到的WebSocket消息（用于调试）- 改为INFO级别以便观察

            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.error(f"❌ WebSocket消息JSON解析失败: {e}, 消息: {message[:200]}")
                return
            
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
                        if isinstance(first_item, list) and len(first_item) >= 9:
                            # 🔥 OKX文档：数组有9个元素，confirm是最后一个（索引8）
                            # [timestamp, open, high, low, close, volume, volCcyQuote, volCcy, confirm]
                            # confirm=1表示K线已完成，confirm=0表示K线未完成
                            confirm = first_item[8]
                            # 🔥 检查confirm字段类型和值（OKX可能返回字符串"1"或数字1）
                            # JSON解析后，confirm可能是字符串"1"或数字1
                            confirm_str = str(confirm).strip()
                            is_closed = (confirm_str == "1" or confirm == 1)
                            close_price = first_item[4]
                            # 🔥 记录所有K线的confirm值（用于诊断）
                            # 🔥 只记录已完成的K线
                            if is_closed:
                                logger.info(f"📊 收到已完成K线: {channel} {inst_id}, close={close_price}, confirm={confirm}")
                            else:
                                logger.debug(f"📥 收到进行中K线: {channel} {inst_id}, close={close_price}, confirm={confirm}")
                        else:
                            logger.warning(f"⚠️ K线数据格式异常: {channel} {inst_id}, first_item类型={type(first_item).__name__}, 长度={len(first_item) if isinstance(first_item, list) else 'N/A'}")
                    else:
                        logger.warning(f"⚠️ K线数据为空或格式错误: {channel} {inst_id}, data_list类型={type(data_list).__name__}, 长度={len(data_list) if isinstance(data_list, list) else 'N/A'}")
                else:
                    # tickers等其他数据：只记录DEBUG级别
                    logger.debug(f"📥 收到WebSocket数据: channel={channel}, instId={inst_id}, 数据长度={len(data['data']) if isinstance(data['data'], list) else 'N/A'}")
                
                if callback_key in self.callbacks:
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
