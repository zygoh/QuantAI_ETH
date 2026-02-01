"""
订单流分析引擎

功能：
- 买卖压力分析
- 大单追踪
- 成交量异动检测
- 动量分析
- 趋势确认（防止逆势交易）
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

from app.scalping.config import scalping_config
from app.scalping.multi_symbol_monitor import SymbolData, OrderBook, TradeData

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """信号方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class OrderFlowSignal:
    """订单流信号"""
    symbol: str
    direction: SignalDirection
    score: float                    # 信号强度 [0, 1]
    timestamp: int

    # 各因子得分
    volume_imbalance_score: float   # 买卖量不平衡得分
    large_order_score: float        # 大单得分
    momentum_score: float           # 动量得分
    trade_flow_score: float         # 成交流得分

    # 原始数据
    volume_imbalance: float         # 买卖量不平衡度 [-1, 1]
    bid_volume: float               # 买单量
    ask_volume: float               # 卖单量
    price_momentum: float           # 价格动量
    buy_trade_ratio: float          # 主动买入比例

    # 趋势得分（有默认值的放最后）
    trend_score: float = 0.0

    def __str__(self):
        return (f"OrderFlowSignal({self.symbol}, {self.direction.value}, "
                f"score={self.score:.2f}, imbalance={self.volume_imbalance:.2f})")


class OrderFlowAnalyzer:
    """订单流分析器"""

    def __init__(self):
        # 历史信号（用于冷却判断和趋势确认）
        self.signal_history: Dict[str, deque] = {}
        # 价格趋势缓存
        self.trend_cache: Dict[str, dict] = {}

        # 配置
        self.volume_imbalance_threshold = scalping_config.volume_imbalance_threshold
        self.large_order_threshold = scalping_config.large_order_threshold
        self.momentum_lookback = scalping_config.momentum_lookback
        self.min_signal_score = scalping_config.min_signal_score
        self.signal_cooldown = scalping_config.signal_cooldown_seconds

    def analyze(self, symbol: str, data: SymbolData) -> Optional[OrderFlowSignal]:
        """
        分析订单流，生成交易信号

        Args:
            symbol: 交易对
            data: 币种实时数据

        Returns:
            OrderFlowSignal 或 None
        """
        if not data.orderbook or not data.orderbook.bids or not data.orderbook.asks:
            return None

        # 检查信号冷却
        if self._is_in_cooldown(symbol):
            return None

        # 计算各因子
        volume_imbalance = data.orderbook.get_volume_imbalance(
            scalping_config.orderbook_depth
        )
        volume_imbalance_score = self._calc_volume_imbalance_score(volume_imbalance)

        large_order_score, large_order_direction = self._calc_large_order_score(
            data.orderbook
        )

        momentum_score, price_momentum = self._calc_momentum_score(data)

        trade_flow_score, buy_ratio = self._calc_trade_flow_score(data)

        # 计算趋势得分
        trend_score, trend_direction = self._calc_trend_score(data)

        # 判断方向
        direction = self._determine_direction(
            volume_imbalance,
            large_order_direction,
            price_momentum,
            buy_ratio,
            trend_direction
        )

        if direction == SignalDirection.NEUTRAL:
            return None

        # 趋势确认：如果信号方向与趋势相反，降低得分或拒绝
        if scalping_config.trend_confirmation_enabled:
            if trend_direction != SignalDirection.NEUTRAL and trend_direction != direction:
                # 逆势信号，大幅降低得分
                trend_penalty = 0.3
                logger.debug(f"⚠️ {symbol} 逆势信号，降低得分")
            else:
                trend_penalty = 0.0
        else:
            trend_penalty = 0.0

        # 综合评分（调整权重，加入趋势因子）
        # 权重分配：买卖压力35%，大单15%，动量20%，成交流15%，趋势15%
        weights = {
            'volume_imbalance': 0.35,
            'large_order': 0.15,
            'momentum': 0.20,
            'trade_flow': 0.15,
            'trend': 0.15
        }

        total_score = (
            weights['volume_imbalance'] * volume_imbalance_score +
            weights['large_order'] * large_order_score +
            weights['momentum'] * momentum_score +
            weights['trade_flow'] * trade_flow_score +
            weights['trend'] * trend_score
        ) - trend_penalty

        # 检查是否达到最小信号强度
        if total_score < self.min_signal_score:
            return None

        signal = OrderFlowSignal(
            symbol=symbol,
            direction=direction,
            score=total_score,
            timestamp=int(time.time() * 1000),
            volume_imbalance_score=volume_imbalance_score,
            large_order_score=large_order_score,
            momentum_score=momentum_score,
            trade_flow_score=trade_flow_score,
            trend_score=trend_score,
            volume_imbalance=volume_imbalance,
            bid_volume=data.orderbook.get_bid_volume(),
            ask_volume=data.orderbook.get_ask_volume(),
            price_momentum=price_momentum,
            buy_trade_ratio=buy_ratio
        )

        # 记录信号
        self._record_signal(symbol, signal)

        return signal

    def _calc_trend_score(self, data: SymbolData) -> Tuple[float, SignalDirection]:
        """
        计算趋势得分（基于更长时间窗口的价格走势）

        Returns:
            (得分, 趋势方向)
        """
        history = list(data.price_history)
        if len(history) < 10:
            return 0.0, SignalDirection.NEUTRAL

        current_time = int(time.time())

        # 获取最近30秒的价格
        prices_30s = [
            h['price'] for h in history
            if current_time - h['timestamp'] <= 30
        ]

        # 获取最近60秒的价格
        prices_60s = [
            h['price'] for h in history
            if current_time - h['timestamp'] <= 60
        ]

        if len(prices_30s) < 5 or len(prices_60s) < 10:
            return 0.0, SignalDirection.NEUTRAL

        # 计算短期和中期趋势
        short_trend = (prices_30s[-1] - prices_30s[0]) / prices_30s[0] if prices_30s[0] > 0 else 0
        mid_trend = (prices_60s[-1] - prices_60s[0]) / prices_60s[0] if prices_60s[0] > 0 else 0

        # 趋势一致性检查
        if short_trend > 0.001 and mid_trend > 0.001:
            # 上升趋势
            direction = SignalDirection.LONG
            strength = min(1.0, (abs(short_trend) + abs(mid_trend)) / 0.01)
        elif short_trend < -0.001 and mid_trend < -0.001:
            # 下降趋势
            direction = SignalDirection.SHORT
            strength = min(1.0, (abs(short_trend) + abs(mid_trend)) / 0.01)
        else:
            # 趋势不明确
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        return strength, direction

    def _calc_volume_imbalance_score(self, imbalance: float) -> float:
        """
        计算买卖量不平衡得分

        imbalance 范围 [-1, 1]
        返回得分 [0, 1]
        """
        # 取绝对值，越不平衡得分越高
        abs_imbalance = abs(imbalance)

        # 超过阈值才有效
        if abs_imbalance < self.volume_imbalance_threshold:
            return abs_imbalance / self.volume_imbalance_threshold * 0.5
        else:
            # 超过阈值，线性映射到 [0.5, 1]
            excess = abs_imbalance - self.volume_imbalance_threshold
            max_excess = 1 - self.volume_imbalance_threshold
            return 0.5 + 0.5 * (excess / max_excess)

    def _calc_large_order_score(self, orderbook: OrderBook) -> Tuple[float, SignalDirection]:
        """
        计算大单得分

        Returns:
            (得分, 大单方向)
        """
        bid_volume = orderbook.get_bid_volume()
        ask_volume = orderbook.get_ask_volume()
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return 0.0, SignalDirection.NEUTRAL

        # 检查买单中的大单
        large_bid_volume = 0
        for level in orderbook.bids[:10]:
            if level.quantity / total_volume > self.large_order_threshold:
                large_bid_volume += level.quantity

        # 检查卖单中的大单
        large_ask_volume = 0
        for level in orderbook.asks[:10]:
            if level.quantity / total_volume > self.large_order_threshold:
                large_ask_volume += level.quantity

        # 计算大单不平衡
        large_total = large_bid_volume + large_ask_volume
        if large_total == 0:
            return 0.0, SignalDirection.NEUTRAL

        large_imbalance = (large_bid_volume - large_ask_volume) / large_total

        # 确定方向
        if large_imbalance > 0.2:
            direction = SignalDirection.LONG
        elif large_imbalance < -0.2:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # 得分
        score = min(1.0, abs(large_imbalance))

        return score, direction

    def _calc_momentum_score(self, data: SymbolData) -> Tuple[float, float]:
        """
        计算动量得分

        Returns:
            (得分, 价格动量)
        """
        history = list(data.price_history)
        if len(history) < 2:
            return 0.0, 0.0

        current_time = int(time.time())
        recent_prices = [
            h['price'] for h in history
            if current_time - h['timestamp'] <= self.momentum_lookback
        ]

        if len(recent_prices) < 2:
            return 0.0, 0.0

        # 计算动量（价格变化百分比）
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # 动量得分：0.1%以上开始计分，0.5%满分
        abs_momentum = abs(momentum)
        if abs_momentum < 0.001:  # 0.1%
            score = 0.0
        elif abs_momentum > 0.005:  # 0.5%
            score = 1.0
        else:
            score = (abs_momentum - 0.001) / 0.004

        return score, momentum

    def _calc_trade_flow_score(self, data: SymbolData) -> Tuple[float, float]:
        """
        计算成交流得分

        Returns:
            (得分, 主动买入比例)
        """
        trades = list(data.trades)
        if len(trades) < 10:
            return 0.0, 0.5

        # 最近100笔成交
        recent_trades = trades[-100:]

        # 计算主动买入量和卖出量
        buy_volume = sum(t.quantity for t in recent_trades if not t.is_buyer_maker)
        sell_volume = sum(t.quantity for t in recent_trades if t.is_buyer_maker)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0, 0.5

        buy_ratio = buy_volume / total_volume

        # 得分：偏离0.5越多得分越高
        imbalance = abs(buy_ratio - 0.5) * 2  # 映射到 [0, 1]
        score = min(1.0, imbalance / 0.3)  # 30%偏离满分

        return score, buy_ratio

    def _determine_direction(
        self,
        volume_imbalance: float,
        large_order_direction: SignalDirection,
        price_momentum: float,
        buy_ratio: float,
        trend_direction: SignalDirection = SignalDirection.NEUTRAL
    ) -> SignalDirection:
        """
        综合判断信号方向（严格版本）

        要求多个因子方向一致才产生信号，并考虑趋势
        """
        long_votes = 0
        short_votes = 0
        strong_long = 0
        strong_short = 0

        # 买卖压力投票（核心因子，权重高）
        if volume_imbalance > self.volume_imbalance_threshold * 1.5:
            long_votes += 2
            strong_long += 1
        elif volume_imbalance > self.volume_imbalance_threshold:
            long_votes += 1
        elif volume_imbalance < -self.volume_imbalance_threshold * 1.5:
            short_votes += 2
            strong_short += 1
        elif volume_imbalance < -self.volume_imbalance_threshold:
            short_votes += 1

        # 大单方向投票
        if large_order_direction == SignalDirection.LONG:
            long_votes += 1
        elif large_order_direction == SignalDirection.SHORT:
            short_votes += 1

        # 动量投票（要求更强的动量）
        if price_momentum > 0.003:  # 0.3%以上（提高门槛）
            long_votes += 1
            if price_momentum > 0.005:  # 0.5%以上
                strong_long += 1
        elif price_momentum < -0.003:
            short_votes += 1
            if price_momentum < -0.005:
                strong_short += 1

        # 成交流投票（要求更明显的偏向）
        if buy_ratio > 0.68:
            long_votes += 1
        elif buy_ratio < 0.32:
            short_votes += 1

        # 趋势投票（重要因子）
        if trend_direction == SignalDirection.LONG:
            long_votes += 1
            strong_long += 1
        elif trend_direction == SignalDirection.SHORT:
            short_votes += 1
            strong_short += 1

        # 更严格的条件：需要至少4票且有2个强信号，或者5票以上
        if long_votes >= 5 and long_votes > short_votes + 2:
            return SignalDirection.LONG
        elif short_votes >= 5 and short_votes > long_votes + 2:
            return SignalDirection.SHORT
        elif long_votes >= 4 and strong_long >= 2 and long_votes > short_votes + 1:
            return SignalDirection.LONG
        elif short_votes >= 4 and strong_short >= 2 and short_votes > long_votes + 1:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL

    def _is_in_cooldown(self, symbol: str) -> bool:
        """检查是否在冷却期"""
        if symbol not in self.signal_history:
            return False

        history = self.signal_history[symbol]
        if not history:
            return False

        last_signal = history[-1]
        elapsed = (time.time() * 1000 - last_signal.timestamp) / 1000

        return elapsed < self.signal_cooldown

    def _record_signal(self, symbol: str, signal: OrderFlowSignal):
        """记录信号"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = deque(maxlen=100)
        self.signal_history[symbol].append(signal)

    def get_signal_stats(self, symbol: str) -> Dict:
        """获取信号统计"""
        if symbol not in self.signal_history:
            return {'total': 0, 'long': 0, 'short': 0}

        history = list(self.signal_history[symbol])
        return {
            'total': len(history),
            'long': sum(1 for s in history if s.direction == SignalDirection.LONG),
            'short': sum(1 for s in history if s.direction == SignalDirection.SHORT),
            'avg_score': sum(s.score for s in history) / len(history) if history else 0
        }
