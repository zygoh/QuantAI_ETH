"""
动量分析器

功能：
- ATR计算（真实波动幅度）
- 价格动量计算（3分钟变化率）
- 成交量放大检测
- 趋势方向判断

替代订单流分析，使用更简单有效的动量突破策略
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

from app.scalping.config import scalping_config
from app.scalping.multi_symbol_monitor import SymbolData

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """信号方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class MomentumSignal:
    """动量信号"""
    symbol: str
    direction: SignalDirection
    score: float                    # 信号强度 [0, 1]
    timestamp: int

    # 各因子得分
    momentum_score: float           # 动量得分
    volume_score: float             # 成交量得分
    atr_score: float                # ATR过滤得分
    trend_score: float              # 趋势一致性得分

    # 原始数据
    price_momentum: float           # 价格动量（3分钟变化率）
    current_atr: float              # 当前ATR
    avg_atr: float                  # 平均ATR
    volume_ratio: float             # 成交量比率（当前/均值）
    trend_direction: SignalDirection  # 趋势方向

    # 建议止损
    suggested_stop_loss_pct: float  # 基于ATR的建议止损百分比

    def __str__(self):
        return (f"MomentumSignal({self.symbol}, {self.direction.value}, "
                f"score={self.score:.2f}, momentum={self.price_momentum:.4f})")


@dataclass
class KlineData:
    """K线数据"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MomentumAnalyzer:
    """动量分析器"""

    def __init__(self):
        # 历史K线数据（用于ATR计算）
        self.kline_history: Dict[str, deque] = {}  # symbol -> deque of KlineData

        # 历史信号（用于冷却判断）
        self.signal_history: Dict[str, deque] = {}

        # ATR缓存
        self.atr_cache: Dict[str, deque] = {}  # symbol -> deque of ATR values

        # 成交量历史
        self.volume_history: Dict[str, deque] = {}  # symbol -> deque of volumes

        # 配置
        self.momentum_threshold = scalping_config.momentum_threshold
        self.volume_multiplier = scalping_config.volume_multiplier
        self.atr_period = scalping_config.atr_period
        self.atr_filter_multiplier = scalping_config.atr_filter_multiplier
        self.trend_lookback_minutes = scalping_config.trend_lookback_minutes
        self.min_signal_score = scalping_config.min_signal_score
        self.signal_cooldown = scalping_config.signal_cooldown_seconds

    def _update_from_symbol_data(self, symbol: str, data: SymbolData):
        """从SymbolData更新K线和ATR缓存"""
        # 初始化缓存
        if symbol not in self.kline_history:
            self.kline_history[symbol] = deque(maxlen=100)
            self.atr_cache[symbol] = deque(maxlen=60)
            self.volume_history[symbol] = deque(maxlen=30)

        # 从SymbolData获取K线历史
        if hasattr(data, 'kline_history') and data.kline_history:
            kline_list = list(data.kline_history)

            # 同步K线历史到本地缓存
            local_timestamps = {k.timestamp for k in self.kline_history[symbol]} if self.kline_history[symbol] else set()

            for kline in kline_list:
                if kline.timestamp not in local_timestamps:
                    # 转换为本地KlineData格式
                    local_kline = KlineData(
                        timestamp=kline.timestamp,
                        open=kline.open,
                        high=kline.high,
                        low=kline.low,
                        close=kline.close,
                        volume=kline.volume
                    )
                    self.kline_history[symbol].append(local_kline)
                    self.volume_history[symbol].append(kline.volume)

                    # 计算并缓存ATR
                    atr = self._calculate_current_atr(symbol)
                    if atr > 0:
                        self.atr_cache[symbol].append(atr)

    def update_kline(self, symbol: str, kline: KlineData):
        """更新K线数据"""
        if symbol not in self.kline_history:
            self.kline_history[symbol] = deque(maxlen=100)
            self.atr_cache[symbol] = deque(maxlen=60)
            self.volume_history[symbol] = deque(maxlen=30)

        self.kline_history[symbol].append(kline)
        self.volume_history[symbol].append(kline.volume)

        # 计算并缓存ATR
        atr = self._calculate_current_atr(symbol)
        if atr > 0:
            self.atr_cache[symbol].append(atr)

    def analyze(self, symbol: str, data: SymbolData) -> Optional[MomentumSignal]:
        """
        分析动量，生成交易信号

        入场条件（全部满足）：
        1. ATR过滤：当前ATR > 1小时均值ATR × 1.5（高波动环境）
        2. 动量突破：3分钟价格变化 > 0.5%
        3. 成交量确认：当前成交量 > 5分钟均量 × 2倍
        4. 趋势一致：与5分钟趋势方向一致

        Args:
            symbol: 交易对
            data: 币种实时数据

        Returns:
            MomentumSignal 或 None
        """
        # 检查信号冷却
        if self._is_in_cooldown(symbol):
            return None

        # 从SymbolData更新K线和ATR缓存
        self._update_from_symbol_data(symbol, data)

        # 获取价格历史
        price_history = list(data.price_history)
        if len(price_history) < 30:  # 至少需要30秒数据
            return None

        # 1. 计算价格动量（3分钟变化率）
        price_momentum = self._calculate_momentum(price_history, lookback_seconds=180)
        momentum_score = self._calc_momentum_score(price_momentum)

        # 2. 计算ATR过滤
        current_atr = self._get_current_atr(symbol)
        avg_atr = self._get_average_atr(symbol)
        atr_score = self._calc_atr_score(current_atr, avg_atr)

        # 3. 计算成交量放大
        volume_ratio = self._calculate_volume_ratio(symbol, data)
        volume_score = self._calc_volume_score(volume_ratio)

        # 4. 计算趋势方向
        trend_direction = self._determine_trend(price_history)
        trend_score = self._calc_trend_score(price_momentum, trend_direction)

        # 判断信号方向
        direction = self._determine_direction(
            price_momentum,
            trend_direction,
            momentum_score,
            volume_score,
            atr_score
        )

        if direction == SignalDirection.NEUTRAL:
            return None

        # 综合评分
        # 权重：动量40%，成交量25%，ATR20%，趋势15%
        total_score = (
            0.40 * momentum_score +
            0.25 * volume_score +
            0.20 * atr_score +
            0.15 * trend_score
        )

        # 检查是否达到最小信号强度
        if total_score < self.min_signal_score:
            return None

        # 计算建议止损（基于ATR）
        suggested_stop_loss = self._calculate_atr_stop_loss(current_atr, data.last_price)

        signal = MomentumSignal(
            symbol=symbol,
            direction=direction,
            score=total_score,
            timestamp=int(time.time() * 1000),
            momentum_score=momentum_score,
            volume_score=volume_score,
            atr_score=atr_score,
            trend_score=trend_score,
            price_momentum=price_momentum,
            current_atr=current_atr,
            avg_atr=avg_atr,
            volume_ratio=volume_ratio,
            trend_direction=trend_direction,
            suggested_stop_loss_pct=suggested_stop_loss
        )

        # 记录信号
        self._record_signal(symbol, signal)

        logger.info(f"📊 动量信号: {signal}")

        return signal

    def _calculate_momentum(self, price_history: List[Dict], lookback_seconds: int = 180) -> float:
        """
        计算价格动量

        Args:
            price_history: 价格历史 [{'timestamp': int, 'price': float}, ...]
            lookback_seconds: 回看秒数（默认3分钟=180秒）

        Returns:
            价格变化百分比
        """
        if len(price_history) < 2:
            return 0.0

        current_time = int(time.time())

        # 获取回看时间内的价格
        prices = [
            h['price'] for h in price_history
            if current_time - h['timestamp'] <= lookback_seconds
        ]

        if len(prices) < 2:
            return 0.0

        # 计算变化率
        return (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0

    def _calculate_current_atr(self, symbol: str) -> float:
        """
        计算当前ATR（真实波动幅度）

        ATR = Average(TR)
        TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
        """
        if symbol not in self.kline_history:
            return 0.0

        klines = list(self.kline_history[symbol])
        if len(klines) < 2:
            return 0.0

        # 计算最近一根K线的TR
        current = klines[-1]
        prev = klines[-2]

        tr = max(
            current.high - current.low,
            abs(current.high - prev.close),
            abs(current.low - prev.close)
        )

        return tr

    def _get_current_atr(self, symbol: str) -> float:
        """获取当前ATR"""
        if symbol not in self.atr_cache or not self.atr_cache[symbol]:
            return 0.0
        return self.atr_cache[symbol][-1]

    def _get_average_atr(self, symbol: str) -> float:
        """获取平均ATR（用于过滤）"""
        if symbol not in self.atr_cache or not self.atr_cache[symbol]:
            return 0.0

        atr_values = list(self.atr_cache[symbol])
        if not atr_values:
            return 0.0

        return sum(atr_values) / len(atr_values)

    def _calculate_volume_ratio(self, symbol: str, data: SymbolData) -> float:
        """
        计算成交量比率

        Returns:
            当前成交量 / 5分钟均量
        """
        # 优先使用K线的成交量数据
        if symbol in self.volume_history and len(self.volume_history[symbol]) >= 5:
            volumes = list(self.volume_history[symbol])
            avg_volume = sum(volumes[-5:]) / 5

            # 当前成交量（使用最新K线或SymbolData的volume_1m）
            current_volume = data.volume_1m if data.volume_1m > 0 else (volumes[-1] if volumes else 0)

            if avg_volume > 0 and current_volume > 0:
                return current_volume / avg_volume

        # 回退：从trades计算
        trades = list(data.trades)
        if len(trades) < 10:
            return 1.0

        # 最近1分钟的成交量
        current_time = int(time.time() * 1000)
        recent_volume = sum(
            t.quantity for t in trades
            if current_time - t.timestamp <= 60000
        )

        # 前5分钟的成交量
        older_volume = sum(
            t.quantity for t in trades
            if 60000 < current_time - t.timestamp <= 360000
        )

        if older_volume <= 0:
            return 1.0

        avg_volume = older_volume / 5
        if avg_volume <= 0:
            return 1.0

        return recent_volume / avg_volume

    def _determine_trend(self, price_history: List[Dict]) -> SignalDirection:
        """
        判断趋势方向（基于5分钟价格走势）
        """
        if len(price_history) < 10:
            return SignalDirection.NEUTRAL

        current_time = int(time.time())
        lookback = self.trend_lookback_minutes * 60

        # 获取回看时间内的价格
        prices = [
            h['price'] for h in price_history
            if current_time - h['timestamp'] <= lookback
        ]

        if len(prices) < 5:
            return SignalDirection.NEUTRAL

        # 计算趋势
        change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0

        # 使用更宽松的阈值判断趋势
        if change > 0.002:  # 0.2%以上认为是上升趋势
            return SignalDirection.LONG
        elif change < -0.002:  # -0.2%以下认为是下降趋势
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL

    def _calc_momentum_score(self, momentum: float) -> float:
        """
        计算动量得分

        动量阈值0.5%，超过阈值得分线性增加
        """
        abs_momentum = abs(momentum)

        if abs_momentum < self.momentum_threshold * 0.5:
            # 低于阈值一半，得分很低
            return abs_momentum / self.momentum_threshold * 0.3
        elif abs_momentum < self.momentum_threshold:
            # 接近阈值，得分中等
            return 0.3 + (abs_momentum - self.momentum_threshold * 0.5) / (self.momentum_threshold * 0.5) * 0.3
        else:
            # 超过阈值，得分高
            excess = abs_momentum - self.momentum_threshold
            return min(1.0, 0.6 + excess / self.momentum_threshold * 0.4)

    def _calc_atr_score(self, current_atr: float, avg_atr: float) -> float:
        """
        计算ATR过滤得分

        当前ATR > 均值ATR × 1.5 时得分高
        """
        if avg_atr <= 0:
            return 0.5  # 无数据时给中等分

        ratio = current_atr / avg_atr

        if ratio < 1.0:
            # 低波动环境，得分低
            return ratio * 0.4
        elif ratio < self.atr_filter_multiplier:
            # 中等波动，得分中等
            return 0.4 + (ratio - 1.0) / (self.atr_filter_multiplier - 1.0) * 0.3
        else:
            # 高波动环境，得分高
            return min(1.0, 0.7 + (ratio - self.atr_filter_multiplier) / self.atr_filter_multiplier * 0.3)

    def _calc_volume_score(self, volume_ratio: float) -> float:
        """
        计算成交量得分

        成交量 > 均量 × 2 时得分高
        """
        if volume_ratio < 1.0:
            # 成交量低于均值，得分低
            return volume_ratio * 0.3
        elif volume_ratio < self.volume_multiplier:
            # 成交量中等，得分中等
            return 0.3 + (volume_ratio - 1.0) / (self.volume_multiplier - 1.0) * 0.4
        else:
            # 成交量放大，得分高
            return min(1.0, 0.7 + (volume_ratio - self.volume_multiplier) / self.volume_multiplier * 0.3)

    def _calc_trend_score(self, momentum: float, trend_direction: SignalDirection) -> float:
        """
        计算趋势一致性得分

        动量方向与趋势方向一致时得分高
        """
        if trend_direction == SignalDirection.NEUTRAL:
            return 0.5  # 趋势不明确，中等分

        # 判断动量方向
        if momentum > 0:
            momentum_dir = SignalDirection.LONG
        elif momentum < 0:
            momentum_dir = SignalDirection.SHORT
        else:
            return 0.5

        # 方向一致得高分，不一致得低分
        if momentum_dir == trend_direction:
            return 0.9
        else:
            return 0.2

    def _determine_direction(
        self,
        momentum: float,
        trend_direction: SignalDirection,
        momentum_score: float,
        volume_score: float,
        atr_score: float
    ) -> SignalDirection:
        """
        综合判断信号方向

        简化条件：
        1. 动量方向明确
        2. 与趋势方向一致（或趋势中性）
        3. 各项得分达标
        """
        # 动量方向
        if momentum > self.momentum_threshold:
            momentum_dir = SignalDirection.LONG
        elif momentum < -self.momentum_threshold:
            momentum_dir = SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL

        # 趋势一致性检查
        if trend_direction != SignalDirection.NEUTRAL and trend_direction != momentum_dir:
            # 逆势，不产生信号
            return SignalDirection.NEUTRAL

        # 基本条件检查
        if momentum_score < 0.4:
            return SignalDirection.NEUTRAL

        if volume_score < 0.3:
            return SignalDirection.NEUTRAL

        if atr_score < 0.3:
            return SignalDirection.NEUTRAL

        return momentum_dir

    def _calculate_atr_stop_loss(self, current_atr: float, current_price: float) -> float:
        """
        基于ATR计算建议止损百分比

        止损 = 1.5倍ATR，但限制在[0.5%, 1%]范围内
        """
        if current_price <= 0 or current_atr <= 0:
            return scalping_config.stop_loss_pct

        # ATR止损
        atr_stop = (current_atr * scalping_config.stop_loss_atr_multiplier) / current_price

        # 限制范围
        return max(
            scalping_config.min_stop_loss_pct,
            min(scalping_config.max_stop_loss_pct, atr_stop)
        )

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

    def _record_signal(self, symbol: str, signal: MomentumSignal):
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
