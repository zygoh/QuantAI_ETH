"""
30天百倍高频剥头皮交易系统

核心模块：
- multi_symbol_monitor: 多币种实时监控
- orderflow_analyzer: 订单流分析引擎
- signal_generator: 高频信号生成器
- position_manager: 复利滚仓仓位管理
- risk_controller: 风控系统
- scalping_engine: 剥头皮交易引擎
"""

from app.scalping.config import ScalpingConfig

__all__ = ['ScalpingConfig']
