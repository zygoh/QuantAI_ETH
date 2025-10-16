"""
API数据模型
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# 基础响应模型
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseResponse):
    """错误响应模型"""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# 账户相关模型
class AccountInfo(BaseModel):
    """账户信息"""
    total_wallet_balance: float
    total_unrealized_pnl: float
    total_margin_balance: float
    available_balance: float
    max_withdraw_amount: float
    can_trade: bool
    can_deposit: bool
    can_withdraw: bool

class AccountResponse(BaseResponse):
    """账户响应"""
    data: Optional[AccountInfo] = None

# 持仓相关模型
class PositionInfo(BaseModel):
    """持仓信息"""
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    leverage: int
    margin_type: str

class PositionsResponse(BaseResponse):
    """持仓响应"""
    data: List[PositionInfo] = []

# 信号相关模型
class TradingSignal(BaseModel):
    """交易信号"""
    timestamp: datetime
    symbol: str
    signal_type: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float

class SignalRequest(BaseModel):
    """信号请求"""
    symbol: str
    force: bool = False

class SignalsResponse(BaseResponse):
    """信号响应"""
    data: List[TradingSignal] = []

# 交易相关模型
class TradeRequest(BaseModel):
    """交易请求"""
    symbol: str
    action: str  # LONG, SHORT, CLOSE
    quantity: Optional[float] = None

class TradingModeRequest(BaseModel):
    """交易模式请求"""
    mode: str  # AUTO, SIGNAL_ONLY

class TradingResponse(BaseResponse):
    """交易响应"""
    data: Optional[Dict[str, Any]] = None

# 训练相关模型
class TrainingRequest(BaseModel):
    """训练请求"""
    force_retrain: bool = False

class TrainingResponse(BaseResponse):
    """训练响应"""
    data: Optional[Dict[str, Any]] = None

# 绩效相关模型
class PerformanceMetrics(BaseModel):
    """绩效指标"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int

class PerformanceResponse(BaseResponse):
    """绩效响应"""
    data: Optional[PerformanceMetrics] = None

# 系统相关模型
class SystemStatus(BaseModel):
    """系统状态"""
    system_state: str
    auto_trading_enabled: bool
    trading_mode: str
    services: Dict[str, bool]

class SystemResponse(BaseResponse):
    """系统响应"""
    data: Optional[SystemStatus] = None

class SystemControlRequest(BaseModel):
    """系统控制请求"""
    action: str  # START, STOP, PAUSE, RESUME

# WebSocket消息模型
class WSMessage(BaseModel):
    """WebSocket消息"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class WSPriceUpdate(BaseModel):
    """价格更新消息"""
    symbol: str
    price: float
    change: float
    change_percent: float
    timestamp: datetime

class WSSignalUpdate(BaseModel):
    """信号更新消息"""
    signal: TradingSignal

class WSOrderUpdate(BaseModel):
    """订单更新消息"""
    order_id: str
    symbol: str
    side: str
    status: str
    filled_quantity: float
    avg_price: float

class WSRiskAlert(BaseModel):
    """风险警报消息"""
    alert_type: str
    message: str
    current_drawdown: float
    threshold: float
    recommended_action: str