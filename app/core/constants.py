"""
全局常量定义

集中管理系统中使用的所有常量，便于维护和修改。
"""
from enum import Enum


# ==================== API限制 ====================

# Binance API限制
BINANCE_API_LIMIT_LARGE = 1000
BINANCE_API_LIMIT_MEDIUM = 500
BINANCE_RECV_WINDOW_MS = 5000
BINANCE_RATE_LIMIT_DELAY_SECONDS = 0.2

# 请求超时
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_WS_TIMEOUT_SECONDS = 30


# ==================== WebSocket配置 ====================

# WebSocket重连配置
WS_RECONNECT_INITIAL_DELAY = 1.0
WS_RECONNECT_MAX_DELAY = 60.0
WS_RECONNECT_BACKOFF_FACTOR = 2.0
WS_MAX_INITIAL_RETRIES = 3
WS_PERIODIC_RETRY_INTERVAL_SECONDS = 120
WS_MAX_WAIT_SECONDS = 30

# WebSocket心跳配置
WS_PING_INTERVAL = 30
WS_PONG_TIMEOUT = 10
WS_MESSAGE_TIMEOUT_SECONDS = 1200
WS_WARNING_TIMEOUT_SECONDS = 600


# ==================== 交易默认值 ====================

# 默认杠杆
DEFAULT_LEVERAGE = 20
MAX_LEVERAGE = 75
MIN_LEVERAGE = 1

# 默认仓位比例
DEFAULT_POSITION_RATIO = 0.5
MAX_POSITION_RATIO = 1.0
MIN_POSITION_RATIO = 0.1

# 默认止损止盈
DEFAULT_STOP_LOSS_PCT = 0.008  # 0.8%
DEFAULT_TAKE_PROFIT_PCT = 0.015  # 1.5%
MIN_STOP_LOSS_PCT = 0.005  # 0.5%
MAX_STOP_LOSS_PCT = 0.01  # 1%


# ==================== 风控限制 ====================

# 每日限制
DEFAULT_MAX_DAILY_TRADES = 50
DEFAULT_MAX_DAILY_LOSS_PCT = 0.15  # 15%

# 连续亏损限制
DEFAULT_MAX_CONSECUTIVE_LOSSES = 3
DEFAULT_COOLDOWN_MINUTES = 30

# 持仓限制
DEFAULT_MAX_POSITION_HOLD_MINUTES = 30


# ==================== 信号配置 ====================

# 动量阈值
DEFAULT_MOMENTUM_THRESHOLD = 0.005  # 0.5%
DEFAULT_VOLUME_MULTIPLIER = 2.0
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_FILTER_MULTIPLIER = 1.5

# 信号冷却
DEFAULT_SIGNAL_COOLDOWN_SECONDS = 60
DEFAULT_MIN_SIGNAL_SCORE = 0.5


# ==================== 数据配置 ====================

# K线数据
DEFAULT_KLINE_LIMIT = 500
MAX_KLINE_LIMIT = 1000

# 订单簿深度
DEFAULT_ORDERBOOK_DEPTH = 20

# 价格历史
DEFAULT_PRICE_HISTORY_SIZE = 60
DEFAULT_TRADE_HISTORY_SIZE = 1000


# ==================== 手续费 ====================

# Binance手续费率
BINANCE_MAKER_FEE_RATE = 0.0002  # 0.02%
BINANCE_TAKER_FEE_RATE = 0.0005  # 0.05%


# ==================== 枚举类型 ====================

class TradingDirection(str, Enum):
    """交易方向"""
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(str, Enum):
    """持仓方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


class ExitReason(str, Enum):
    """平仓原因"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN = "breakeven"
    TIMEOUT = "timeout"
    MANUAL = "manual"


class WebSocketErrorType(str, Enum):
    """WebSocket错误类型"""
    SSL_ERROR = "ssl_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    PROTOCOL_ERROR = "protocol_error"
    UNKNOWN_ERROR = "unknown_error"


# ==================== 交易所标识 ====================

class Exchange(str, Enum):
    """支持的交易所"""
    BINANCE = "BINANCE"
    # 预留扩展
    # OKX = "OKX"
    # BYBIT = "BYBIT"
