"""
复利滚仓仓位管理器

功能：
- 根据余额动态调整仓位
- 连胜加仓，连亏减仓
- 阶段性杠杆调整
- 金字塔加仓法
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, date

from app.scalping.config import scalping_config, TradingPhase

logger = logging.getLogger(__name__)


@dataclass
class TradingStats:
    """交易统计"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    current_win_streak: int = 0
    current_lose_streak: int = 0
    max_win_streak: int = 0
    max_lose_streak: int = 0
    daily_trades: int = 0
    daily_profit: float = 0.0
    daily_start_balance: float = 0.0  # 当日起始余额（用于计算每日亏损百分比）
    last_trade_date: Optional[date] = None

    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        """利润因子"""
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0
        return self.total_profit / abs(self.total_loss)

    @property
    def net_profit(self) -> float:
        """净利润"""
        return self.total_profit + self.total_loss  # loss是负数


@dataclass
class PositionInfo:
    """仓位信息（支持金字塔加仓）"""
    symbol: str
    direction: str                  # "LONG" or "SHORT"
    entry_price: float              # 首次入场价格
    quantity: float                 # 总数量
    leverage: int
    take_profit: float
    stop_loss: float
    entry_time: datetime
    position_value: float           # 总仓位价值（USDT）

    # 金字塔加仓相关
    avg_entry_price: float = 0.0    # 综合成本价
    addition_count: int = 0         # 已加仓次数
    base_quantity: float = 0.0      # 底仓数量
    last_addition_price: float = 0.0  # 上次加仓价格
    entry_prices: List[float] = field(default_factory=list)  # 所有入场价格记录

    def __post_init__(self):
        """初始化后处理"""
        if self.avg_entry_price == 0.0:
            self.avg_entry_price = self.entry_price
        if self.base_quantity == 0.0:
            self.base_quantity = self.quantity
        if not self.entry_prices:
            self.entry_prices = [self.entry_price]

    @property
    def hold_duration_seconds(self) -> float:
        """持仓时长（秒）"""
        return (datetime.now() - self.entry_time).total_seconds()

    def add_position(self, price: float, quantity: float, position_value: float):
        """
        金字塔加仓

        Args:
            price: 加仓价格
            quantity: 加仓数量
            position_value: 加仓价值
        """
        # 计算新的综合成本价
        total_value = self.quantity * self.avg_entry_price + quantity * price
        self.quantity += quantity
        self.avg_entry_price = total_value / self.quantity
        self.position_value += position_value
        self.addition_count += 1
        self.last_addition_price = price
        self.entry_prices.append(price)


class PositionManager:
    """仓位管理器（支持多仓位）"""

    def __init__(self):
        self.balance: float = scalping_config.initial_balance
        self.stats = TradingStats()
        # 支持多仓位：从单个position改为字典
        self.positions: Dict[str, PositionInfo] = {}  # symbol -> PositionInfo
        self.is_in_cooldown = False
        self.cooldown_end_time: Optional[datetime] = None

        # 每日统计重置
        self._check_daily_reset()

    @property
    def current_position(self) -> Optional[PositionInfo]:
        """兼容旧代码：返回第一个持仓"""
        if self.positions:
            return list(self.positions.values())[0]
        return None

    def _check_daily_reset(self):
        """检查并重置每日统计"""
        today = date.today()
        if self.stats.last_trade_date != today:
            self.stats.daily_trades = 0
            self.stats.daily_profit = 0.0
            self.stats.daily_start_balance = self.balance  # 记录当日起始余额
            self.stats.last_trade_date = today
            logger.info(f"📅 新的一天，重置每日统计，起始余额: {self.balance:.4f}U")

    def get_current_phase(self) -> TradingPhase:
        """获取当前交易阶段"""
        return scalping_config.get_current_phase(self.balance)

    def get_leverage(self) -> int:
        """获取当前杠杆"""
        return scalping_config.get_leverage(self.balance)

    def get_position_ratio(self) -> float:
        """获取当前仓位比例"""
        return scalping_config.calculate_position_ratio(
            self.stats.current_win_streak,
            self.stats.current_lose_streak,
            self.balance
        )

    def calculate_position_size(self, entry_price: float) -> Optional[Dict[str, float]]:
        """
        计算仓位大小

        Args:
            entry_price: 入场价格

        Returns:
            {
                'quantity': 下单数量,
                'position_value': 仓位价值,
                'leverage': 杠杆倍数,
                'margin_required': 所需保证金
            }
        """
        self._check_daily_reset()

        # 验证价格
        if entry_price <= 0:
            logger.error(f"❌ 无效价格: {entry_price}")
            return None

        # 检查是否可以开仓
        if not self.can_open_position():
            return None

        # 获取仓位比例和杠杆
        position_ratio = self.get_position_ratio()
        leverage = self.get_leverage()

        # 计算仓位价值
        margin = self.balance * position_ratio
        position_value = margin * leverage

        # 计算数量
        quantity = position_value / entry_price

        logger.info(f"📊 仓位计算: 余额={self.balance:.2f}U, "
                   f"比例={position_ratio:.1%}, 杠杆={leverage}x, "
                   f"仓位价值={position_value:.2f}U, 数量={quantity:.6f}")

        return {
            'quantity': quantity,
            'position_value': position_value,
            'leverage': leverage,
            'margin_required': margin
        }

    def can_open_position(self, symbol: str = None) -> bool:
        """检查是否可以开仓"""
        self._check_daily_reset()

        # 检查该币种是否已有持仓
        if symbol and symbol in self.positions:
            logger.debug(f"已有 {symbol} 持仓，不能重复开仓")
            return False

        # 检查是否达到最大持仓数量
        max_positions = scalping_config.get_max_positions(self.balance)
        if len(self.positions) >= max_positions:
            logger.debug(f"已达最大持仓数 {max_positions}，不能开新仓")
            return False

        # 冷却期
        if self.is_in_cooldown:
            if datetime.now() < self.cooldown_end_time:
                remaining = (self.cooldown_end_time - datetime.now()).seconds
                logger.debug(f"冷却期中，剩余 {remaining} 秒")
                return False
            else:
                self.is_in_cooldown = False
                logger.info("✅ 冷却期结束")

        # 每日交易次数限制
        if self.stats.daily_trades >= scalping_config.max_daily_trades:
            logger.warning(f"⚠️ 达到每日交易上限 {scalping_config.max_daily_trades}")
            return False

        # 每日亏损限制（使用当日起始余额计算）
        base_balance = self.stats.daily_start_balance if self.stats.daily_start_balance > 0 else self.balance
        daily_loss_pct = abs(self.stats.daily_profit) / base_balance if self.stats.daily_profit < 0 else 0
        if daily_loss_pct >= scalping_config.max_daily_loss_pct:
            logger.warning(f"⚠️ 达到每日亏损上限 {scalping_config.max_daily_loss_pct:.1%}")
            return False

        return True

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        take_profit: float,
        stop_loss: float
    ) -> bool:
        """
        开仓

        Returns:
            是否成功
        """
        if symbol in self.positions:
            logger.error(f"已有 {symbol} 持仓，无法重复开仓")
            return False

        max_positions = scalping_config.get_max_positions(self.balance)
        if len(self.positions) >= max_positions:
            logger.error(f"已达最大持仓数 {max_positions}")
            return False

        position_value = quantity * entry_price / leverage

        position = PositionInfo(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            take_profit=take_profit,
            stop_loss=stop_loss,
            entry_time=datetime.now(),
            position_value=position_value
        )

        self.positions[symbol] = position

        logger.info(f"📈 开仓: {symbol} {direction} @ {entry_price:.6f}, "
                   f"数量={quantity:.6f}, 杠杆={leverage}x, "
                   f"TP={take_profit:.6f}, SL={stop_loss:.6f} "
                   f"[持仓数: {len(self.positions)}/{max_positions}]")

        return True

    def pyramid_add_position(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        金字塔加仓

        Args:
            symbol: 交易对
            current_price: 当前价格

        Returns:
            加仓结果，失败返回None
        """
        if not scalping_config.pyramid_enabled:
            return None

        pos = self.positions.get(symbol)
        if pos is None:
            return None

        # 检查是否达到最大加仓次数
        if pos.addition_count >= scalping_config.pyramid_max_additions:
            logger.debug(f"已达最大加仓次数 {scalping_config.pyramid_max_additions}")
            return None

        # 计算当前盈利百分比（价格波动，非杠杆后）
        if pos.direction == "LONG":
            pnl_pct = (current_price - pos.avg_entry_price) / pos.avg_entry_price
        else:
            pnl_pct = (pos.avg_entry_price - current_price) / pos.avg_entry_price

        # 检查是否达到加仓触发条件
        required_profit = scalping_config.pyramid_profit_trigger
        if pnl_pct < required_profit:
            return None

        # 检查与上次加仓的价格间隔
        if pos.last_addition_price > 0:
            if pos.direction == "LONG":
                spacing = (current_price - pos.last_addition_price) / pos.last_addition_price
            else:
                spacing = (pos.last_addition_price - current_price) / pos.last_addition_price

            if spacing < scalping_config.pyramid_spacing:
                logger.debug(f"加仓间隔不足: {spacing:.2%} < {scalping_config.pyramid_spacing:.2%}")
                return None

        # 计算加仓量（底仓 × 递减因子）
        add_quantity = pos.base_quantity * scalping_config.pyramid_scale_factor
        add_value = add_quantity * current_price / pos.leverage

        # 检查余额是否足够
        required_margin = add_value
        if required_margin > self.balance * 0.5:  # 不超过余额的50%
            logger.warning(f"余额不足以加仓: 需要 {required_margin:.4f}U")
            return None

        # 执行加仓
        old_avg = pos.avg_entry_price
        pos.add_position(current_price, add_quantity, add_value)

        # 计算新止损（综合成本价上方）
        if pos.direction == "LONG":
            new_stop_loss = pos.avg_entry_price * (1 + scalping_config.pyramid_stop_buffer)
        else:
            new_stop_loss = pos.avg_entry_price * (1 - scalping_config.pyramid_stop_buffer)

        pos.stop_loss = new_stop_loss

        result = {
            'symbol': symbol,
            'direction': pos.direction,
            'add_price': current_price,
            'add_quantity': add_quantity,
            'add_value': add_value,
            'total_quantity': pos.quantity,
            'old_avg_price': old_avg,
            'new_avg_price': pos.avg_entry_price,
            'new_stop_loss': new_stop_loss,
            'addition_count': pos.addition_count,
            'pnl_pct_at_add': pnl_pct
        }

        logger.info(f"🔺 金字塔加仓 #{pos.addition_count}: {symbol} {pos.direction} @ {current_price:.6f}")
        logger.info(f"   加仓量={add_quantity:.6f}, 总量={pos.quantity:.6f}")
        logger.info(f"   成本价: {old_avg:.6f} → {pos.avg_entry_price:.6f}")
        logger.info(f"   新止损: {new_stop_loss:.6f} (成本价+{scalping_config.pyramid_stop_buffer:.2%})")

        return result

    def can_pyramid_add(self, symbol: str, current_price: float) -> bool:
        """
        检查是否可以金字塔加仓

        Args:
            symbol: 交易对
            current_price: 当前价格

        Returns:
            是否可以加仓
        """
        if not scalping_config.pyramid_enabled:
            return False

        pos = self.positions.get(symbol)
        if pos is None:
            return False

        # 检查加仓次数
        if pos.addition_count >= scalping_config.pyramid_max_additions:
            return False

        # 计算盈利
        if pos.direction == "LONG":
            pnl_pct = (current_price - pos.avg_entry_price) / pos.avg_entry_price
        else:
            pnl_pct = (pos.avg_entry_price - current_price) / pos.avg_entry_price

        # 检查盈利阈值
        if pnl_pct < scalping_config.pyramid_profit_trigger:
            return False

        # 检查间隔
        if pos.last_addition_price > 0:
            if pos.direction == "LONG":
                spacing = (current_price - pos.last_addition_price) / pos.last_addition_price
            else:
                spacing = (pos.last_addition_price - current_price) / pos.last_addition_price

            if spacing < scalping_config.pyramid_spacing:
                return False

        return True

    def close_position(self, exit_price: float, reason: str = "manual", symbol: str = None) -> Optional[Dict[str, Any]]:
        """
        平仓

        Args:
            exit_price: 平仓价格
            reason: 平仓原因 (take_profit, stop_loss, timeout, manual)
            symbol: 要平仓的币种（如果不指定，平第一个）

        Returns:
            平仓结果
        """
        # 确定要平仓的持仓
        if symbol:
            pos = self.positions.get(symbol)
        else:
            pos = self.current_position  # 兼容旧代码

        if pos is None:
            logger.error("没有持仓")
            return None

        symbol = pos.symbol  # 确保有symbol

        # 计算手续费（开仓+平仓双边，使用综合成本价）
        fee_rate = scalping_config.taker_fee_rate if scalping_config.use_taker_fee else scalping_config.maker_fee_rate
        # 开仓手续费 = 仓位价值 * 费率（使用综合成本价）
        entry_fee = pos.quantity * pos.avg_entry_price * fee_rate
        # 平仓手续费 = 仓位价值 * 费率
        exit_fee = pos.quantity * exit_price * fee_rate
        total_fee = entry_fee + exit_fee

        # 计算盈亏（使用综合成本价）
        if pos.direction == "LONG":
            pnl_pct = (exit_price - pos.avg_entry_price) / pos.avg_entry_price
        else:
            pnl_pct = (pos.avg_entry_price - exit_price) / pos.avg_entry_price

        # 考虑杠杆
        pnl_pct_leveraged = pnl_pct * pos.leverage

        # 实际盈亏金额（扣除手续费）
        margin = pos.position_value / pos.leverage
        gross_pnl = margin * pnl_pct_leveraged  # 毛利润
        pnl_amount = gross_pnl - total_fee       # 净利润 = 毛利润 - 手续费

        # 更新余额
        old_balance = self.balance
        self.balance += pnl_amount

        # 更新统计
        self.stats.total_trades += 1
        self.stats.daily_trades += 1

        if pnl_amount > 0:
            self.stats.winning_trades += 1
            self.stats.total_profit += pnl_amount
            self.stats.current_win_streak += 1
            self.stats.current_lose_streak = 0
            self.stats.max_win_streak = max(self.stats.max_win_streak, self.stats.current_win_streak)
        else:
            self.stats.losing_trades += 1
            self.stats.total_loss += pnl_amount  # 负数
            self.stats.current_lose_streak += 1
            self.stats.current_win_streak = 0
            self.stats.max_lose_streak = max(self.stats.max_lose_streak, self.stats.current_lose_streak)

            # 检查连亏冷却
            if self.stats.current_lose_streak >= scalping_config.max_consecutive_losses:
                self._enter_cooldown()

        self.stats.daily_profit += pnl_amount

        result = {
            'symbol': pos.symbol,
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'quantity': pos.quantity,
            'leverage': pos.leverage,
            'pnl_pct': pnl_pct_leveraged,
            'gross_pnl': gross_pnl,
            'fee': total_fee,
            'pnl_amount': pnl_amount,
            'reason': reason,
            'hold_duration': pos.hold_duration_seconds,
            'old_balance': old_balance,
            'new_balance': self.balance
        }

        emoji = "✅" if pnl_amount > 0 else "❌"
        logger.info(f"{emoji} 平仓: {pos.symbol} {pos.direction} @ {exit_price:.6f}, "
                   f"毛利={gross_pnl:+.4f}U, 手续费={total_fee:.4f}U, 净利={pnl_amount:+.4f}U, "
                   f"原因={reason}, 余额: {old_balance:.4f} → {self.balance:.4f}U")

        # 清除持仓
        if symbol in self.positions:
            del self.positions[symbol]

        return result

    def _enter_cooldown(self):
        """进入冷却期"""
        from datetime import timedelta
        self.is_in_cooldown = True
        self.cooldown_end_time = datetime.now() + timedelta(minutes=scalping_config.cooldown_minutes)
        logger.warning(f"⏸️ 连续亏损 {self.stats.current_lose_streak} 次，"
                      f"进入冷却期 {scalping_config.cooldown_minutes} 分钟")

    def check_position_timeout(self, symbol: str = None) -> bool:
        """检查持仓是否超时"""
        if symbol:
            pos = self.positions.get(symbol)
        else:
            pos = self.current_position

        if pos is None:
            return False

        hold_seconds = pos.hold_duration_seconds
        max_seconds = scalping_config.max_position_hold_minutes * 60

        return hold_seconds >= max_seconds

    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """获取所有持仓"""
        return self.positions

    def reset_daily_stats(self):
        """手动重置每日统计（用于测试或紧急情况）"""
        self.stats.daily_trades = 0
        self.stats.daily_profit = 0.0
        self.stats.daily_start_balance = self.balance
        self.stats.last_trade_date = date.today()
        self.is_in_cooldown = False
        self.cooldown_end_time = None
        logger.info(f"🔄 手动重置每日统计，起始余额: {self.balance:.4f}U")

    def get_status(self) -> Dict:
        """获取状态"""
        max_positions = scalping_config.get_max_positions(self.balance)
        return {
            'balance': self.balance,
            'phase': self.get_current_phase().value,
            'leverage': self.get_leverage(),
            'position_ratio': self.get_position_ratio(),
            'has_position': len(self.positions) > 0,
            'position_count': len(self.positions),
            'max_positions': max_positions,
            'current_position': {
                'symbol': self.current_position.symbol,
                'direction': self.current_position.direction,
                'entry_price': self.current_position.entry_price,
                'avg_entry_price': self.current_position.avg_entry_price,
                'addition_count': self.current_position.addition_count,
                'hold_duration': self.current_position.hold_duration_seconds
            } if self.current_position else None,
            'all_positions': [
                {
                    'symbol': p.symbol,
                    'direction': p.direction,
                    'entry_price': p.entry_price,
                    'avg_entry_price': p.avg_entry_price,
                    'quantity': p.quantity,
                    'addition_count': p.addition_count,
                    'hold_duration': p.hold_duration_seconds
                } for p in self.positions.values()
            ],
            'pyramid_config': {
                'enabled': scalping_config.pyramid_enabled,
                'max_additions': scalping_config.pyramid_max_additions,
                'profit_trigger': scalping_config.pyramid_profit_trigger,
                'spacing': scalping_config.pyramid_spacing
            },
            'stats': {
                'total_trades': self.stats.total_trades,
                'win_rate': self.stats.win_rate,
                'profit_factor': self.stats.profit_factor,
                'net_profit': self.stats.net_profit,
                'current_win_streak': self.stats.current_win_streak,
                'current_lose_streak': self.stats.current_lose_streak,
                'daily_trades': self.stats.daily_trades,
                'daily_profit': self.stats.daily_profit
            },
            'is_in_cooldown': self.is_in_cooldown,
            'can_trade': self.can_open_position()
        }
