"""
30天百倍系统配置

目标：5U → 500U（30天100倍）
策略：动量突破 + 波动率过滤 + 复利滚仓

核心理念：
- 跟随已发生的动量，而非预测市场方向
- 使用动量+成交量（简单有效），而非订单流分析（复杂且噪音大）
- 紧止损快认错，而非宽止损等回调
- 追求正期望值，而非高胜率
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TradingPhase(Enum):
    """交易阶段（根据资金量调整策略）"""
    PHASE_1 = "phase_1"  # 5U-50U: 高波动meme币，激进
    PHASE_2 = "phase_2"  # 50U-200U: 中等波动，平衡
    PHASE_3 = "phase_3"  # 200U+: 主流币，稳健


@dataclass
class SymbolConfig:
    """币种配置"""
    symbol: str                    # 交易对 (如 "PEPE/USDT")
    max_leverage: int              # 最大杠杆
    min_notional: float            # 最小下单金额
    tick_size: float               # 价格精度
    lot_size: float                # 数量精度
    volatility_rank: int           # 波动性排名 (1=最高)
    liquidity_rank: int            # 流动性排名 (1=最好)
    phase: TradingPhase            # 适用阶段


@dataclass
class ScalpingConfig:
    """剥头皮系统配置"""

    # ==================== 资金配置 ====================
    initial_balance: float = 5.0           # 初始资金 (USDT)
    target_balance: float = 500.0          # 目标资金 (100倍)
    target_days: int = 30                  # 目标天数

    # ==================== 币种配置 ====================
    # 是否自动扫描币种（True=自动获取高波动币种，False=使用默认列表）
    auto_scan_symbols: bool = True
    auto_scan_count: int = 5              # 自动扫描时选择的币种数量（减少以避免WebSocket限制）

    # 默认币种列表（auto_scan_symbols=False时使用）
    symbols: List[SymbolConfig] = field(default_factory=list)

    # 动态币种列表（运行时填充）
    _dynamic_symbols: Optional[List[SymbolConfig]] = field(default=None, repr=False)

    # ==================== 杠杆配置（按阶段） ====================
    leverage_by_phase: Dict[TradingPhase, int] = field(default_factory=lambda: {
        TradingPhase.PHASE_1: 20,   # 小资金用20倍
        TradingPhase.PHASE_2: 30,   # 中资金用30倍
        TradingPhase.PHASE_3: 50,   # 大资金用50倍
    })

    # ==================== 仓位配置 ====================
    base_position_ratio: float = 0.25      # 单仓位基础比例 (25%)
    max_total_position_ratio: float = 0.9  # 总仓位上限 (90%，保留10%作为缓冲)
    min_position_ratio: float = 0.10       # 单仓位最小比例 (10%)
    max_position_ratio: float = 0.40       # 单仓位最大比例 (40%)

    # 连胜/连亏仓位调整
    win_streak_position_boost: float = 0.05   # 每连胜1次，仓位+5%
    lose_streak_position_reduce: float = 0.05 # 每连亏1次，仓位-5%
    max_win_streak_boost: int = 4             # 最大连胜加仓次数
    max_lose_streak_reduce: int = 3           # 最大连亏减仓次数

    # ==================== 动量信号配置 ====================
    momentum_threshold: float = 0.006      # 动量阈值0.6%（提高：0.5% → 0.6%）
    volume_multiplier: float = 1.5         # 成交量放大倍数（降低：2.0 → 1.5，更容易满足）
    atr_period: int = 14                   # ATR计算周期
    atr_filter_multiplier: float = 1.2     # ATR过滤倍数（降低：1.5 → 1.2）
    trend_lookback_minutes: int = 5        # 趋势回看时间（分钟）
    min_signal_score: float = 0.65         # 最小信号得分（提高：0.50 → 0.65，过滤低质量信号）
    signal_cooldown_seconds: int = 30      # 同一币种信号冷却时间（增加：15 → 30秒，避免频繁交易）

    # ==================== 止损配置（基于ATR动态调整） ====================
    stop_loss_atr_multiplier: float = 1.2  # 止损 = 1.2倍ATR（收紧：1.5 → 1.2）
    max_stop_loss_pct: float = 0.006       # 最大止损0.6%（收紧：1% → 0.6%）
    min_stop_loss_pct: float = 0.003       # 最小止损0.3%（收紧：0.5% → 0.3%）
    stop_loss_pct: float = 0.005           # 默认止损0.5%（收紧：0.8% → 0.5%）

    # ==================== 追踪止盈配置（分级追踪） ====================
    trailing_stop_enabled: bool = True     # 启用追踪止盈
    trailing_stop_activation: float = 0.015  # 盈利1.5%后激活追踪（放宽：1% → 1.5%，让利润跑）
    trailing_stop_callback: float = 0.004  # 从最高点回撤0.4%触发止盈（收紧：0.5% → 0.4%）

    # 分级追踪止盈：(盈利阈值, 回撤阈值)
    trailing_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.015, 0.004),  # 盈利1.5%，回撤0.4%止盈
        (0.025, 0.003),  # 盈利2.5%，回撤0.3%止盈
        (0.035, 0.002),  # 盈利3.5%，回撤0.2%止盈
    ])

    # ==================== 移动保本配置 ====================
    breakeven_enabled: bool = True         # 启用移动保本
    breakeven_activation: float = 0.01     # 盈利1%后移动止损到保本（放宽：0.6% → 1%）
    breakeven_buffer: float = 0.001        # 保本缓冲（入场价+0.1%）（收紧：0.2% → 0.1%）

    # ==================== 金字塔加仓配置 ====================
    pyramid_enabled: bool = True           # 启用金字塔加仓
    pyramid_max_additions: int = 3         # 最大加仓次数

    # 加仓触发盈利阈值（价格波动，非杠杆后）
    pyramid_profit_trigger: float = 0.03   # 盈利3%触发加仓
    pyramid_spacing: float = 0.025         # 加仓间隔至少2.5%（防止回调同时扫损）

    # 加仓量递减因子（1.0=等量加仓，0.8=递减加仓）
    pyramid_scale_factor: float = 0.8      # 每次加仓量 = 底仓 × 此值

    # 加仓后止损设置
    pyramid_stop_buffer: float = 0.002     # 止损设在综合成本价上方0.2%

    # ==================== 手续费配置 ====================
    # Binance合约手续费（双边）
    taker_fee_rate: float = 0.0005         # Taker手续费 0.05%
    maker_fee_rate: float = 0.0002         # Maker手续费 0.02%
    use_taker_fee: bool = True             # 默认使用Taker费率（市价单）

    # ==================== 风控配置 ====================
    max_daily_loss_pct: float = 0.10       # 单日最大亏损 10%（收紧：15% → 10%）
    max_daily_trades: int = 50             # 单日最大交易次数（减少：100 → 50）
    max_consecutive_losses: int = 3        # 连续亏损暂停阈值
    cooldown_minutes: int = 15             # 连亏后冷却时间（增加：10 → 15分钟）
    max_position_hold_minutes: int = 15    # 最大持仓时间（缩短：30 → 15分钟）

    # ==================== 订单流分析配置（保留用于辅助过滤） ====================
    orderbook_depth: int = 20              # 订单簿深度
    volume_imbalance_threshold: float = 0.25  # 买卖量不平衡阈值
    large_order_threshold: float = 0.05    # 大单阈值（占总深度比例）
    momentum_lookback: int = 10            # 动量回看周期（秒）

    # ==================== 趋势确认配置 ====================
    trend_confirmation_enabled: bool = True  # 启用趋势确认
    min_trend_strength: float = 0.6          # 最小趋势强度

    # ==================== 阶段阈值 ====================
    phase_1_max_balance: float = 50.0      # 阶段1上限
    phase_2_max_balance: float = 200.0     # 阶段2上限

    def get_current_phase(self, balance: float) -> TradingPhase:
        """根据当前余额获取交易阶段"""
        if balance < self.phase_1_max_balance:
            return TradingPhase.PHASE_1
        elif balance < self.phase_2_max_balance:
            return TradingPhase.PHASE_2
        else:
            return TradingPhase.PHASE_3

    def get_symbols(self) -> List[SymbolConfig]:
        """获取币种列表（优先使用动态扫描的币种）"""
        if self._dynamic_symbols:
            return self._dynamic_symbols
        return self.symbols

    def set_dynamic_symbols(self, symbols: List[SymbolConfig]):
        """设置动态扫描的币种"""
        self._dynamic_symbols = symbols

    def get_active_symbols(self, balance: float) -> List[SymbolConfig]:
        """获取当前阶段可用的币种"""
        current_phase = self.get_current_phase(balance)
        all_symbols = self.get_symbols()

        if not all_symbols:
            return []

        # 当前阶段及之前阶段的币种都可用
        phase_order = [TradingPhase.PHASE_1, TradingPhase.PHASE_2, TradingPhase.PHASE_3]
        current_idx = phase_order.index(current_phase)
        allowed_phases = phase_order[:current_idx + 1]

        return [s for s in all_symbols if s.phase in allowed_phases]

    def get_leverage(self, balance: float) -> int:
        """获取当前阶段的杠杆倍数"""
        phase = self.get_current_phase(balance)
        return self.leverage_by_phase[phase]

    def get_max_positions(self, balance: float) -> int:
        """
        获取最大持仓数量

        - Phase 1 (5U-50U): 1个仓位（资金少，集中火力）
        - Phase 2 (50U-200U): 最多3个仓位
        - Phase 3 (200U+): 最大仓位数=监控币种数量（资金充足，不错过机会）
        """
        phase = self.get_current_phase(balance)
        symbols = self.get_symbols()
        symbol_count = len(symbols) if symbols else self.auto_scan_count

        if phase == TradingPhase.PHASE_1:
            return 1  # 小资金集中火力
        elif phase == TradingPhase.PHASE_2:
            return min(3, symbol_count)  # 中等资金，最多3个
        else:
            return symbol_count  # 大资金，全部币种都可开仓

    def get_position_ratio_per_symbol(self, balance: float, current_positions: int = 0) -> float:
        """
        获取单个仓位的资金比例

        动态计算：确保总仓位不超过 max_total_position_ratio
        """
        max_positions = self.get_max_positions(balance)

        # 单仓位比例 = 总仓位上限 / 最大仓位数
        ratio = self.max_total_position_ratio / max_positions

        # 确保不低于最小比例
        ratio = max(ratio, self.min_position_ratio)

        # 确保不超过基础比例
        ratio = min(ratio, self.base_position_ratio)

        return ratio

    def calculate_position_ratio(self, win_streak: int, lose_streak: int, balance: float = None) -> float:
        """计算当前仓位比例（考虑连胜连亏调整）"""
        # 获取基础比例
        if balance:
            base = self.get_position_ratio_per_symbol(balance)
        else:
            base = self.base_position_ratio

        ratio = base

        # 连胜加仓
        if win_streak > 0:
            boost = min(win_streak, self.max_win_streak_boost) * self.win_streak_position_boost
            ratio += boost

        # 连亏减仓
        if lose_streak > 0:
            reduce = min(lose_streak, self.max_lose_streak_reduce) * self.lose_streak_position_reduce
            ratio -= reduce

        # 限制范围
        return max(self.min_position_ratio, min(self.max_position_ratio, ratio))

    def get_daily_target_return(self) -> float:
        """计算每日目标收益率"""
        # 100倍 = (1 + r)^30，求r
        # r = 100^(1/30) - 1 ≈ 0.166 (16.6%)
        import math
        multiplier = self.target_balance / self.initial_balance
        return math.pow(multiplier, 1 / self.target_days) - 1


# 全局配置实例
scalping_config = ScalpingConfig()
