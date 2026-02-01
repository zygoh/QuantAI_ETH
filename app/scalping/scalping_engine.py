"""
剥头皮交易引擎

功能：
- 整合所有模块
- 自动交易执行
- 状态管理
- 自动扫描高波动币种
- 安全的币种刷新（保护持仓币种）
"""
import asyncio
import logging
import time
from typing import Optional, Dict, List, Set
from datetime import datetime

from app.scalping.config import scalping_config, SymbolConfig, TradingPhase
from app.scalping.signal_generator import ScalpingSignalGenerator, TradingSignal
from app.scalping.position_manager import PositionManager
from app.scalping.risk_controller import RiskController
from app.scalping.multi_symbol_monitor import MultiSymbolMonitor

logger = logging.getLogger(__name__)


class ScalpingEngine:
    """剥头皮交易引擎"""

    def __init__(self):
        self.signal_generator = ScalpingSignalGenerator()
        self.position_manager = PositionManager()
        self.risk_controller: Optional[RiskController] = None
        self.is_running = False

        # 统计
        self.start_time: Optional[datetime] = None
        self.signals_received = 0
        self.trades_executed = 0

        # 当前监控的币种
        self.active_symbols: List[SymbolConfig] = []

        # 锁定的币种（正在持仓，不能移除）
        self._locked_symbols: Set[str] = set()

        # 币种刷新锁（防止刷新时开仓）
        self._refresh_lock = asyncio.Lock()

    async def start(self, initial_balance: float = None):
        """
        启动交易引擎

        Args:
            initial_balance: 初始余额（可选，默认使用配置）
        """
        if self.is_running:
            logger.warning("交易引擎已在运行")
            return

        # 设置初始余额
        if initial_balance is not None:
            self.position_manager.balance = initial_balance
            logger.info(f"💰 设置初始余额: {initial_balance}U")

        self.start_time = datetime.now()

        # 自动扫描币种
        if scalping_config.auto_scan_symbols:
            await self._scan_and_select_symbols()
        else:
            self.active_symbols = scalping_config.get_symbols()
            logger.info(f"📋 使用预设币种: {[s.symbol for s in self.active_symbols]}")

        if not self.active_symbols:
            raise Exception("没有可用的交易币种")

        # 启动信号生成器
        await self.signal_generator.start(self.position_manager.balance)

        # 初始化风控
        self.risk_controller = RiskController(
            self.position_manager,
            self.signal_generator.monitor
        )
        # 注册平仓后解锁币种的回调
        self.risk_controller.on_position_closed = self._unlock_symbol
        await self.risk_controller.start()

        # 添加信号回调
        self.signal_generator.add_signal_callback(self._on_signal)

        self.is_running = True

        # 启动主循环
        asyncio.create_task(self._main_loop())

        # 启动定期币种刷新
        asyncio.create_task(self._periodic_symbol_refresh())

        logger.info("🚀 剥头皮交易引擎启动完成")
        logger.info(f"   初始余额: {self.position_manager.balance}U")
        logger.info(f"   目标余额: {scalping_config.target_balance}U")
        logger.info(f"   每日目标: {scalping_config.get_daily_target_return():.1%}")
        logger.info(f"   监控币种: {[s.symbol for s in self.active_symbols]}")

    async def _scan_and_select_symbols(self, preserve_locked: bool = True):
        """
        扫描并选择高波动币种

        Args:
            preserve_locked: 是否保留锁定的币种（持仓中的币种）
        """
        try:
            from app.scalping.symbol_scanner import symbol_scanner

            logger.info("🔍 自动扫描高波动币种...")

            await symbol_scanner.scan_all_symbols()
            new_symbols = symbol_scanner.get_top_symbols(
                scalping_config.auto_scan_count
            )

            # 如果有锁定的币种，确保它们被保留
            if preserve_locked and self._locked_symbols:
                new_symbol_names = {s.symbol for s in new_symbols}
                symbols_to_add = []

                for locked_symbol in self._locked_symbols:
                    if locked_symbol not in new_symbol_names:
                        # 锁定的币种不在新列表中，需要保留
                        # 从当前列表中找到该币种的配置
                        for existing in self.active_symbols:
                            if existing.symbol == locked_symbol:
                                symbols_to_add.append(existing)
                                logger.info(f"🔒 保留锁定币种: {locked_symbol} (持仓中)")
                                break

                # 将锁定的币种添加到新列表
                new_symbols.extend(symbols_to_add)

            self.active_symbols = new_symbols

            # 更新配置
            scalping_config.set_dynamic_symbols(self.active_symbols)

            logger.info(f"✅ 自动选择 {len(self.active_symbols)} 个币种:")
            for s in self.active_symbols:
                locked_mark = "🔒" if s.symbol in self._locked_symbols else "  "
                logger.info(f"   {locked_mark} {s.symbol} (阶段: {s.phase.value})")

        except Exception as e:
            logger.error(f"自动扫描币种失败: {e}")
            # 回退到默认币种（但保留锁定的）
            if not self.active_symbols:
                self.active_symbols = scalping_config.get_symbols()
            if not self.active_symbols:
                raise Exception("没有可用的交易币种")

    async def _periodic_symbol_refresh(self):
        """定期刷新币种列表（每小时）"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # 1小时

                if not scalping_config.auto_scan_symbols:
                    continue

                # 检查是否有持仓
                if self.position_manager.current_position is not None:
                    logger.info("🔄 定期刷新: 检测到持仓，等待平仓后刷新...")
                    # 等待平仓（最多等待30分钟）
                    wait_count = 0
                    while (self.position_manager.current_position is not None
                           and wait_count < 180  # 180 * 10秒 = 30分钟
                           and self.is_running):
                        await asyncio.sleep(10)
                        wait_count += 1

                    if self.position_manager.current_position is not None:
                        logger.warning("🔄 定期刷新: 等待超时，跳过本次刷新")
                        continue

                # 获取刷新锁
                async with self._refresh_lock:
                    logger.info("🔄 定期刷新币种列表...")
                    await self._scan_and_select_symbols(preserve_locked=True)

                    # 更新监控器的订阅
                    await self._update_monitor_subscriptions()

            except Exception as e:
                logger.error(f"定期刷新币种失败: {e}")

    async def _update_monitor_subscriptions(self):
        """更新监控器的订阅（添加新币种）"""
        try:
            current_monitored = set(self.signal_generator.monitor.symbol_data.keys())
            new_symbols = {s.symbol for s in self.active_symbols}

            # 找出需要新增订阅的币种
            to_add = new_symbols - current_monitored

            if to_add:
                logger.info(f"📡 新增订阅: {to_add}")
                for symbol in to_add:
                    # 找到对应的配置
                    config = next((s for s in self.active_symbols if s.symbol == symbol), None)
                    if config:
                        await self.signal_generator.monitor._subscribe_symbol(symbol)

            # 注意：不移除旧订阅，因为可能还有锁定的币种
            # 旧订阅会在下次重启时清理

        except Exception as e:
            logger.error(f"更新监控订阅失败: {e}")

    def _lock_symbol(self, symbol: str):
        """锁定币种（开仓时调用）"""
        self._locked_symbols.add(symbol)
        logger.debug(f"🔒 锁定币种: {symbol}")

    def _unlock_symbol(self, symbol: str):
        """解锁币种（平仓时调用）"""
        self._locked_symbols.discard(symbol)
        logger.debug(f"🔓 解锁币种: {symbol}")

    async def stop(self):
        """停止交易引擎"""
        self.is_running = False

        if self.risk_controller:
            await self.risk_controller.stop()

        await self.signal_generator.stop()

        logger.info("🛑 剥头皮交易引擎已停止")

    async def _main_loop(self):
        """主循环"""
        while self.is_running:
            try:
                # 检查是否达到目标
                if self.position_manager.balance >= scalping_config.target_balance:
                    logger.info(f"🎉🎉🎉 达到目标余额！当前: {self.position_manager.balance:.2f}U")
                    # 不自动停止，继续运行

                # 定期输出状态
                await asyncio.sleep(60)
                self._log_status()

            except Exception as e:
                logger.error(f"主循环异常: {e}")
                await asyncio.sleep(5)

    def _on_signal(self, signal: TradingSignal):
        """处理交易信号"""
        self.signals_received += 1

        # 检查刷新锁（刷新期间不开仓）
        if self._refresh_lock.locked():
            logger.debug(f"跳过信号: 币种刷新中")
            return

        # 检查是否可以开仓（传入symbol检查是否已有该币种持仓）
        if not self.position_manager.can_open_position(signal.symbol):
            logger.debug(f"跳过信号: 无法开仓 {signal.symbol}")
            return

        # 获取币种配置，确定杠杆
        symbol_config = next(
            (s for s in self.active_symbols if s.symbol == signal.symbol),
            None
        )

        # 计算杠杆（取配置杠杆和币种最大杠杆的较小值）
        config_leverage = self.position_manager.get_leverage()
        if symbol_config:
            leverage = min(config_leverage, symbol_config.max_leverage)
        else:
            leverage = config_leverage

        # 计算仓位
        position_info = self.position_manager.calculate_position_size(signal.entry_price)
        if position_info is None:
            logger.debug(f"跳过信号: 仓位计算失败")
            return

        # 使用调整后的杠杆
        position_info['leverage'] = leverage

        # 执行开仓
        success = self.position_manager.open_position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            quantity=position_info['quantity'],
            leverage=leverage,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss
        )

        if success:
            self.trades_executed += 1
            # 锁定该币种
            self._lock_symbol(signal.symbol)
            logger.info(f"🎯 执行交易 #{self.trades_executed}: {signal.symbol} {signal.direction} "
                       f"@ {signal.entry_price:.6f} | 杠杆={leverage}x")

    def _log_status(self):
        """输出状态日志"""
        status = self.get_status()
        stats = status['stats']

        logger.info(f"📊 状态更新 | "
                   f"余额: {status['balance']:.4f}U | "
                   f"阶段: {status['phase']} | "
                   f"交易: {stats['total_trades']} | "
                   f"胜率: {stats['win_rate']:.1%} | "
                   f"净利: {stats['net_profit']:+.4f}U")

    def get_status(self) -> Dict:
        """获取引擎状态"""
        pm_status = self.position_manager.get_status()

        # 计算运行时间
        runtime_seconds = 0
        if self.start_time:
            runtime_seconds = (datetime.now() - self.start_time).total_seconds()

        # 计算进度
        progress = (self.position_manager.balance / scalping_config.target_balance) * 100

        return {
            **pm_status,
            'is_running': self.is_running,
            'runtime_seconds': runtime_seconds,
            'runtime_hours': runtime_seconds / 3600,
            'signals_received': self.signals_received,
            'trades_executed': self.trades_executed,
            'target_balance': scalping_config.target_balance,
            'progress_pct': progress,
            'daily_target_return': scalping_config.get_daily_target_return(),
            'active_symbols': [s.symbol for s in self.active_symbols],
            'locked_symbols': list(self._locked_symbols),
            'auto_scan_enabled': scalping_config.auto_scan_symbols
        }

    def manual_close_position(self) -> Optional[Dict]:
        """手动平仓"""
        if self.position_manager.current_position is None:
            logger.warning("没有持仓")
            return None

        # 获取当前价格
        symbol = self.position_manager.current_position.symbol
        symbol_data = self.signal_generator.monitor.get_symbol_data(symbol)

        if not symbol_data or symbol_data.last_price <= 0:
            logger.error("无法获取当前价格")
            return None

        result = self.position_manager.close_position(
            symbol_data.last_price,
            "manual"
        )

        # 解锁币种
        if result:
            self._unlock_symbol(symbol)

        return result


# 全局引擎实例
scalping_engine = ScalpingEngine()


# 便捷函数
async def start_scalping(initial_balance: float = 5.0):
    """启动剥头皮交易"""
    await scalping_engine.start(initial_balance)


async def stop_scalping():
    """停止剥头皮交易"""
    await scalping_engine.stop()


def get_scalping_status() -> Dict:
    """获取交易状态"""
    return scalping_engine.get_status()
