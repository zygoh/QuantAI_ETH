"""
高频信号生成器

功能：
- 整合动量分析器（替代订单流分析）
- 多币种信号筛选
- 信号优先级排序

核心策略：动量突破 + 波动率过滤
入场条件：
1. ATR过滤：当前ATR > 1小时均值ATR × 1.5（高波动环境）
2. 动量突破：3分钟价格变化 > 0.5%
3. 成交量确认：当前成交量 > 5分钟均量 × 2倍
4. 趋势一致：与5分钟趋势方向一致
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime

from app.scalping.config import scalping_config, SymbolConfig
from app.scalping.multi_symbol_monitor import MultiSymbolMonitor, SymbolData
from app.scalping.momentum_analyzer import MomentumAnalyzer, MomentumSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    direction: str                  # "LONG" or "SHORT"
    entry_price: float              # 入场价格
    take_profit: float              # 止盈价格（0表示使用追踪止盈）
    stop_loss: float                # 止损价格
    score: float                    # 信号强度
    timestamp: int
    momentum_signal: MomentumSignal  # 原始动量信号

    # 计算属性
    @property
    def risk_reward_ratio(self) -> float:
        """盈亏比"""
        if self.take_profit == 0:
            return 0  # 使用追踪止盈时不计算固定盈亏比
        if self.direction == "LONG":
            profit = self.take_profit - self.entry_price
            loss = self.entry_price - self.stop_loss
        else:
            profit = self.entry_price - self.take_profit
            loss = self.stop_loss - self.entry_price
        return profit / loss if loss > 0 else 0

    def __str__(self):
        return (f"TradingSignal({self.symbol}, {self.direction}, "
                f"entry={self.entry_price:.6f}, SL={self.stop_loss:.6f}, "
                f"score={self.score:.2f})")


class ScalpingSignalGenerator:
    """剥头皮信号生成器（动量突破策略）"""

    def __init__(self):
        self.monitor = MultiSymbolMonitor()
        self.analyzer = MomentumAnalyzer()
        self.is_running = False
        self.signal_callbacks: List[Callable[[TradingSignal], None]] = []

        # 最近生成的信号
        self.recent_signals: Dict[str, TradingSignal] = {}

        # 配置
        self.stop_loss_pct = scalping_config.stop_loss_pct

    async def start(self, balance: float = None):
        """
        启动信号生成器

        Args:
            balance: 当前余额（用于确定监控哪些币种）
        """
        if self.is_running:
            logger.warning("信号生成器已在运行")
            return

        # 根据余额获取可用币种
        if balance is None:
            balance = scalping_config.initial_balance

        # 监控所有扫描到的币种（不限制阶段，增加交易机会）
        active_symbols = scalping_config.get_symbols()
        if not active_symbols:
            # 如果没有动态币种，使用阶段性币种
            active_symbols = scalping_config.get_active_symbols(balance)

        logger.info(f"📊 当前余额: {balance}U, 监控 {len(active_symbols)} 个币种")

        # 启动监控
        await self.monitor.start(active_symbols)

        # 添加数据更新回调
        self.monitor.add_callback(self._on_data_update)

        self.is_running = True
        logger.info("✅ 信号生成器启动完成（动量突破策略）")

    def _on_data_update(self, symbol: str, data: SymbolData) -> None:
        """数据更新回调"""
        try:
            # 使用动量分析器分析
            momentum_signal = self.analyzer.analyze(symbol, data)

            if momentum_signal:
                # 生成交易信号
                trading_signal = self._create_trading_signal(momentum_signal, data)

                if trading_signal:
                    self.recent_signals[symbol] = trading_signal
                    self._notify_signal(trading_signal)

        except Exception as e:
            logger.error(f"处理数据更新失败 {symbol}: {e}")

    def _create_trading_signal(
        self,
        momentum_signal: MomentumSignal,
        data: SymbolData
    ) -> Optional[TradingSignal]:
        """创建交易信号"""
        if not data.orderbook:
            # 没有订单簿数据，使用最新价格
            entry_price = data.last_price
        else:
            # 入场价格（使用中间价）
            entry_price = data.orderbook.mid_price

        if not entry_price or entry_price <= 0:
            return None

        # 使用动量分析器建议的止损，或默认止损
        stop_loss_pct = momentum_signal.suggested_stop_loss_pct
        if stop_loss_pct <= 0:
            stop_loss_pct = self.stop_loss_pct

        # 计算止损价格
        if momentum_signal.direction == SignalDirection.LONG:
            stop_loss = entry_price * (1 - stop_loss_pct)
            direction = "LONG"
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            direction = "SHORT"

        return TradingSignal(
            symbol=momentum_signal.symbol,
            direction=direction,
            entry_price=entry_price,
            take_profit=0,  # 不使用固定止盈，由追踪止盈处理
            stop_loss=stop_loss,
            score=momentum_signal.score,
            timestamp=int(time.time() * 1000),
            momentum_signal=momentum_signal
        )

    def _notify_signal(self, signal: TradingSignal):
        """通知信号"""
        logger.info(f"🎯 新信号: {signal}")
        logger.info(f"   动量={signal.momentum_signal.price_momentum:.4f}, "
                   f"成交量比={signal.momentum_signal.volume_ratio:.2f}, "
                   f"ATR={signal.momentum_signal.current_atr:.6f}")

        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"信号回调执行失败: {e}")

    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """添加信号回调"""
        self.signal_callbacks.append(callback)

    def get_best_signal(self) -> Optional[TradingSignal]:
        """获取当前最佳信号"""
        if not self.recent_signals:
            return None

        # 按得分排序，返回最高分
        signals = list(self.recent_signals.values())
        signals.sort(key=lambda s: s.score, reverse=True)

        # 检查信号是否过期（超过30秒）
        best = signals[0]
        if time.time() * 1000 - best.timestamp > 30000:
            return None

        return best

    def get_all_signals(self) -> List[TradingSignal]:
        """获取所有有效信号"""
        current_time = time.time() * 1000
        valid_signals = [
            s for s in self.recent_signals.values()
            if current_time - s.timestamp <= 30000  # 30秒内有效
        ]
        return sorted(valid_signals, key=lambda s: s.score, reverse=True)

    def get_signal_for_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """获取指定币种的信号"""
        signal = self.recent_signals.get(symbol)
        if signal and time.time() * 1000 - signal.timestamp <= 30000:
            return signal
        return None

    def clear_signal(self, symbol: str):
        """清除指定币种的信号"""
        if symbol in self.recent_signals:
            del self.recent_signals[symbol]

    async def stop(self):
        """停止信号生成器"""
        self.is_running = False
        await self.monitor.stop()
        logger.info("🛑 信号生成器已停止")
