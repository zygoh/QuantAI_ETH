"""
交易模块

职责：
1. 交易信号生成
2. 仓位管理
3. 订单执行
4. 交易控制

模块：
- signal_generator: 信号生成器
- position_manager: 仓位管理器
- trading_engine: 交易执行引擎
- trading_controller: 交易控制器
"""

from app.trading.signal_generator import SignalGenerator, TradingSignal
from app.trading.position_manager import PositionManager, position_manager
from app.trading.trading_engine import TradingEngine, TradingMode
from app.trading.trading_controller import TradingController

__all__ = [
    "SignalGenerator",
    "TradingSignal",
    "PositionManager",
    "position_manager",
    "TradingEngine",
    "TradingMode",
    "TradingController",
]

__version__ = "1.0.0"

