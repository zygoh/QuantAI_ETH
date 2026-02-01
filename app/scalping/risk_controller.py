"""
风控系统

功能：
- 动态止损（基于ATR）
- 分级追踪止盈
- 提前移动保本
- 持仓超时强制平仓
- 每日风控限制
- 平仓后通知引擎解锁币种

出场条件：
- 初始止损：0.8%（基于ATR动态调整）
- 移动保本：盈利0.6%后止损移至入场价+0.2%
- 追踪止盈：盈利1%后激活，分级回撤止盈
  - 盈利1%，回撤0.5%止盈
  - 盈利1.5%，回撤0.3%止盈
  - 盈利2%，回撤0.2%止盈
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Tuple
from datetime import datetime

from app.scalping.config import scalping_config
from app.scalping.position_manager import PositionManager, PositionInfo
from app.scalping.multi_symbol_monitor import MultiSymbolMonitor

logger = logging.getLogger(__name__)


@dataclass
class RiskEvent:
    """风控事件"""
    event_type: str                 # "take_profit", "stop_loss", "trailing_stop", "breakeven", "timeout"
    symbol: str
    trigger_price: float
    position_entry: float
    pnl_pct: float
    timestamp: int


class RiskController:
    """风控控制器（支持多仓位，分级追踪止盈）"""

    def __init__(
        self,
        position_manager: PositionManager,
        monitor: MultiSymbolMonitor
    ):
        self.position_manager = position_manager
        self.monitor = monitor
        self.is_running = False

        # 追踪状态（每个币种独立）
        # symbol -> {
        #   trailing_activated: bool,
        #   breakeven_activated: bool,
        #   highest: float,
        #   lowest: float,
        #   current_stop_loss: float,
        #   current_tier: int,  # 当前追踪止盈级别
        #   max_pnl_pct: float  # 最大盈利百分比
        # }
        self.trailing_states: Dict[str, dict] = {}

        # 回调
        self.on_risk_event: Optional[Callable[[RiskEvent], None]] = None
        self.on_position_closed: Optional[Callable[[str], None]] = None

        # 配置
        self.stop_loss_pct = scalping_config.stop_loss_pct
        self.trailing_enabled = scalping_config.trailing_stop_enabled
        self.trailing_activation = scalping_config.trailing_stop_activation
        self.trailing_callback = scalping_config.trailing_stop_callback
        self.trailing_tiers = scalping_config.trailing_tiers
        self.breakeven_enabled = scalping_config.breakeven_enabled
        self.breakeven_activation = scalping_config.breakeven_activation
        self.breakeven_buffer = scalping_config.breakeven_buffer

    async def start(self):
        """启动风控监控"""
        if self.is_running:
            return

        self.is_running = True
        asyncio.create_task(self._monitor_loop())
        logger.info("✅ 风控系统启动（分级追踪止盈）")

    async def stop(self):
        """停止风控监控"""
        self.is_running = False
        logger.info("🛑 风控系统停止")

    async def _monitor_loop(self):
        """风控监控循环"""
        while self.is_running:
            try:
                await self._check_all_positions()
                await asyncio.sleep(0.1)  # 100ms检查一次
            except Exception as e:
                logger.error(f"风控监控异常: {e}")
                await asyncio.sleep(1)

    async def _check_all_positions(self):
        """检查所有持仓的风控"""
        positions = self.position_manager.get_all_positions()

        # 清理已平仓的追踪状态
        for symbol in list(self.trailing_states.keys()):
            if symbol not in positions:
                del self.trailing_states[symbol]

        # 检查每个持仓
        for symbol, position in list(positions.items()):
            await self._check_position(position)

    async def _check_position(self, position: PositionInfo):
        """检查单个持仓风控"""
        symbol = position.symbol

        # 获取当前价格
        symbol_data = self.monitor.get_symbol_data(symbol)
        if not symbol_data or symbol_data.last_price <= 0:
            return

        current_price = symbol_data.last_price

        # 计算当前盈亏（使用综合成本价）
        if position.direction == "LONG":
            pnl_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
        else:
            pnl_pct = (position.avg_entry_price - current_price) / position.avg_entry_price

        # 更新移动保本和追踪止盈状态
        self._update_trailing_state(position, current_price, pnl_pct)

        # 1. 检查动态止损（含移动保本）
        if self._check_stop_loss(position, current_price, pnl_pct):
            return

        # 2. 检查金字塔加仓机会
        if self._check_pyramid_add(position, current_price, pnl_pct):
            # 加仓成功后继续监控，不return
            pass

        # 3. 检查分级追踪止盈
        if self.trailing_enabled:
            if self._check_trailing_stop(position, current_price, pnl_pct):
                return

        # 4. 检查持仓超时
        if self._check_timeout(position, current_price, pnl_pct):
            return

    def _check_pyramid_add(self, position: PositionInfo, current_price: float, pnl_pct: float) -> bool:
        """
        检查并执行金字塔加仓

        Returns:
            是否执行了加仓
        """
        if not scalping_config.pyramid_enabled:
            return False

        # 检查是否可以加仓
        if not self.position_manager.can_pyramid_add(position.symbol, current_price):
            return False

        # 执行加仓
        result = self.position_manager.pyramid_add_position(position.symbol, current_price)

        if result:
            logger.info(f"🔺 金字塔加仓成功: {position.symbol} 第{result['addition_count']}次加仓")
            return True

        return False

    def _get_trailing_state(self, symbol: str) -> dict:
        """获取或创建追踪状态"""
        if symbol not in self.trailing_states:
            self.trailing_states[symbol] = {
                'trailing_activated': False,
                'breakeven_activated': False,
                'highest': 0.0,
                'lowest': float('inf'),
                'current_stop_loss': None,  # None表示使用初始止损
                'current_tier': -1,  # -1表示未激活任何级别
                'max_pnl_pct': 0.0
            }
        return self.trailing_states[symbol]

    def _update_trailing_state(self, position: PositionInfo, current_price: float, pnl_pct: float):
        """更新追踪状态（移动保本+分级追踪止盈）"""
        symbol = position.symbol
        state = self._get_trailing_state(symbol)

        # 更新最高/最低价和最大盈利
        if position.direction == "LONG":
            if current_price > state['highest']:
                state['highest'] = current_price
            if pnl_pct > state['max_pnl_pct']:
                state['max_pnl_pct'] = pnl_pct
        else:
            if current_price < state['lowest']:
                state['lowest'] = current_price
            if pnl_pct > state['max_pnl_pct']:
                state['max_pnl_pct'] = pnl_pct

        # 检查移动保本激活（盈利0.6%后）
        if self.breakeven_enabled and not state['breakeven_activated']:
            if pnl_pct >= self.breakeven_activation:
                state['breakeven_activated'] = True
                # 计算保本止损价（综合成本价+缓冲）
                if position.direction == "LONG":
                    state['current_stop_loss'] = position.avg_entry_price * (1 + self.breakeven_buffer)
                else:
                    state['current_stop_loss'] = position.avg_entry_price * (1 - self.breakeven_buffer)
                logger.info(f"🛡️ 移动保本激活 {symbol}: 盈利{pnl_pct:.2%}, 止损移至 {state['current_stop_loss']:.6f}")

        # 检查追踪止盈激活（盈利1%后）
        if self.trailing_enabled and not state['trailing_activated']:
            if pnl_pct >= self.trailing_activation:
                state['trailing_activated'] = True
                state['current_tier'] = 0
                logger.info(f"📈 追踪止盈激活 {symbol}: 盈利{pnl_pct:.2%}")

        # 更新追踪止盈级别（分级追踪）
        if state['trailing_activated'] and self.trailing_tiers:
            self._update_trailing_tier(position, state, pnl_pct)

    def _update_trailing_tier(self, position: PositionInfo, state: dict, pnl_pct: float):
        """更新追踪止盈级别"""
        # 检查是否进入更高级别
        for i, (profit_threshold, _) in enumerate(self.trailing_tiers):
            if pnl_pct >= profit_threshold and i > state['current_tier']:
                state['current_tier'] = i
                logger.info(f"📊 追踪止盈升级 {position.symbol}: 级别{i+1}, 盈利{pnl_pct:.2%}")

    def _get_current_trailing_callback(self, state: dict) -> float:
        """获取当前级别的追踪回撤阈值"""
        if not state['trailing_activated'] or state['current_tier'] < 0:
            return self.trailing_callback

        if state['current_tier'] < len(self.trailing_tiers):
            _, callback = self.trailing_tiers[state['current_tier']]
            return callback

        # 超过最高级别，使用最后一级的回撤
        if self.trailing_tiers:
            _, callback = self.trailing_tiers[-1]
            return callback

        return self.trailing_callback

    def _reset_trailing_stop(self, symbol: str = None):
        """重置追踪状态"""
        if symbol:
            if symbol in self.trailing_states:
                del self.trailing_states[symbol]
        else:
            self.trailing_states.clear()

    def _check_stop_loss(
        self,
        position: PositionInfo,
        current_price: float,
        pnl_pct: float
    ) -> bool:
        """检查止损（支持移动保本）"""
        symbol = position.symbol
        state = self._get_trailing_state(symbol)

        # 确定当前止损价：优先使用移动后的止损，否则使用初始止损
        if state['current_stop_loss'] is not None:
            stop_price = state['current_stop_loss']
            reason = "breakeven"  # 保本止损
        else:
            stop_price = position.stop_loss
            reason = "stop_loss"  # 初始止损

        triggered = False
        if position.direction == "LONG":
            if current_price <= stop_price:
                triggered = True
        else:
            if current_price >= stop_price:
                triggered = True

        if triggered:
            self._trigger_risk_event(reason, position, current_price, pnl_pct)
            return True

        return False

    def _check_trailing_stop(
        self,
        position: PositionInfo,
        current_price: float,
        pnl_pct: float
    ) -> bool:
        """检查分级追踪止盈"""
        symbol = position.symbol
        state = self._get_trailing_state(symbol)

        # 未激活则不检查
        if not state['trailing_activated']:
            return False

        # 获取当前级别的回撤阈值
        callback_threshold = self._get_current_trailing_callback(state)

        # 检查从最高/最低点的回撤
        if position.direction == "LONG":
            if state['highest'] > 0:
                drawdown = (state['highest'] - current_price) / state['highest']
                if drawdown >= callback_threshold:
                    logger.info(f"📈 追踪止盈触发 {symbol}: 从最高点{state['highest']:.6f}回撤{drawdown:.2%}")
                    self._trigger_risk_event("trailing_stop", position, current_price, pnl_pct)
                    return True
        else:
            if state['lowest'] < float('inf'):
                drawdown = (current_price - state['lowest']) / state['lowest']
                if drawdown >= callback_threshold:
                    logger.info(f"📈 追踪止盈触发 {symbol}: 从最低点{state['lowest']:.6f}回撤{drawdown:.2%}")
                    self._trigger_risk_event("trailing_stop", position, current_price, pnl_pct)
                    return True

        return False

    def _check_timeout(
        self,
        position: PositionInfo,
        current_price: float,
        pnl_pct: float
    ) -> bool:
        """检查持仓超时"""
        if self.position_manager.check_position_timeout(position.symbol):
            logger.info(f"⏰ 持仓超时 {position.symbol}: 持仓{position.hold_duration_seconds/60:.1f}分钟")
            self._trigger_risk_event("timeout", position, current_price, pnl_pct)
            return True
        return False

    def _trigger_risk_event(
        self,
        event_type: str,
        position: PositionInfo,
        trigger_price: float,
        pnl_pct: float
    ):
        """触发风控事件"""
        symbol = position.symbol  # 保存symbol，因为平仓后position会被清除

        event = RiskEvent(
            event_type=event_type,
            symbol=symbol,
            trigger_price=trigger_price,
            position_entry=position.entry_price,
            pnl_pct=pnl_pct,
            timestamp=int(time.time() * 1000)
        )

        emoji_map = {
            "stop_loss": "🛑",
            "breakeven": "🛡️",
            "trailing_stop": "📈",
            "timeout": "⏰"
        }
        emoji = emoji_map.get(event_type, "⚠️")

        logger.info(f"{emoji} 风控触发: {event_type} | {symbol} | "
                   f"价格={trigger_price:.6f} | 盈亏={pnl_pct:.2%}")

        # 执行平仓（传入symbol）
        self.position_manager.close_position(trigger_price, event_type, symbol)

        # 重置该币种的追踪止盈
        self._reset_trailing_stop(symbol)

        # 通知引擎解锁币种
        if self.on_position_closed:
            try:
                self.on_position_closed(symbol)
            except Exception as e:
                logger.error(f"解锁币种回调执行失败: {e}")

        # 风控事件回调
        if self.on_risk_event:
            try:
                self.on_risk_event(event)
            except Exception as e:
                logger.error(f"风控回调执行失败: {e}")

    def update_stop_loss(self, symbol: str, new_stop_loss: float):
        """
        手动更新止损价格（用于基于ATR的动态止损）

        Args:
            symbol: 交易对
            new_stop_loss: 新的止损价格
        """
        state = self._get_trailing_state(symbol)
        state['current_stop_loss'] = new_stop_loss
        logger.info(f"🔧 更新止损 {symbol}: {new_stop_loss:.6f}")

    def get_status(self) -> Dict:
        """获取风控状态"""
        return {
            'is_running': self.is_running,
            'trailing_states': {
                symbol: {
                    'trailing_activated': state['trailing_activated'],
                    'breakeven_activated': state['breakeven_activated'],
                    'current_tier': state['current_tier'],
                    'max_pnl_pct': state['max_pnl_pct'],
                    'highest': state['highest'],
                    'lowest': state['lowest']
                }
                for symbol, state in self.trailing_states.items()
            },
            'active_trailing_count': sum(1 for s in self.trailing_states.values() if s.get('trailing_activated')),
            'active_breakeven_count': sum(1 for s in self.trailing_states.values() if s.get('breakeven_activated')),
            'config': {
                'stop_loss_pct': self.stop_loss_pct,
                'trailing_enabled': self.trailing_enabled,
                'trailing_activation': self.trailing_activation,
                'trailing_tiers': self.trailing_tiers,
                'breakeven_enabled': self.breakeven_enabled,
                'breakeven_activation': self.breakeven_activation,
                'breakeven_buffer': self.breakeven_buffer
            }
        }
