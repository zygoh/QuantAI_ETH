"""
高频策略回测系统

功能：
- 基于历史K线数据模拟订单流
- 支持多币种回测
- 详细的回测报告
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import random

from app.scalping.config import scalping_config, SymbolConfig, TradingPhase
from app.exchange.clients.binance.binance_client import BinanceClient

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """回测交易记录"""
    trade_id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    entry_time: datetime
    exit_time: datetime
    pnl_pct: float
    pnl_amount: float
    exit_reason: str
    balance_before: float
    balance_after: float


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    symbol: str
    start_time: datetime
    end_time: datetime
    initial_balance: float
    final_balance: float

    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # 收益统计
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float

    # 详细数据
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

    def __str__(self):
        return (f"BacktestResult({self.symbol}, "
                f"trades={self.total_trades}, "
                f"win_rate={self.win_rate:.1%}, "
                f"return={self.total_return_pct:.1%}, "
                f"max_dd={self.max_drawdown_pct:.1%})")


class ScalpingBacktester:
    """剥头皮策略回测器"""

    def __init__(self):
        self.client = BinanceClient()

    async def run_backtest(
        self,
        symbol: str = "1000PEPE/USDT",
        days: int = 7,
        initial_balance: float = 5.0,
        leverage: int = 20
    ) -> BacktestResult:
        """
        运行回测

        Args:
            symbol: 交易对
            days: 回测天数
            initial_balance: 初始余额
            leverage: 杠杆倍数

        Returns:
            BacktestResult
        """
        logger.info(f"🔄 开始回测: {symbol} | {days}天 | 初始={initial_balance}U | 杠杆={leverage}x")

        # 获取历史数据
        klines = await self._fetch_klines(symbol, days)
        if not klines:
            raise Exception(f"无法获取 {symbol} 的历史数据")

        logger.info(f"📊 获取到 {len(klines)} 根K线")

        # 模拟交易
        result = await self._simulate_trading(
            symbol=symbol,
            klines=klines,
            initial_balance=initial_balance,
            leverage=leverage
        )

        logger.info(f"✅ 回测完成: {result}")

        return result

    async def _fetch_klines(self, symbol: str, days: int) -> List[Dict]:
        """获取历史K线数据"""
        # 使用1分钟K线
        interval = "1m"
        limit_per_request = 1000

        # 计算需要的K线数量
        total_minutes = days * 24 * 60
        klines_needed = min(total_minutes, 10000)  # 最多10000根

        logger.info(f"📥 获取 {symbol} {interval} K线，需要 {klines_needed} 根")

        # 分批获取
        all_klines = []
        end_time = int(time.time() * 1000)

        while len(all_klines) < klines_needed:
            batch = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit_per_request,
                end_time=end_time
            )

            if not batch:
                break

            # 转换为字典格式
            for k in batch:
                all_klines.append({
                    'timestamp': k.timestamp,
                    'open': k.open,
                    'high': k.high,
                    'low': k.low,
                    'close': k.close,
                    'volume': k.volume
                })

            # 更新end_time
            end_time = batch[0].timestamp - 1

            # 避免API限流
            await asyncio.sleep(0.2)

        # 按时间排序
        all_klines.sort(key=lambda x: x['timestamp'])

        return all_klines[-klines_needed:]

    async def _simulate_trading(
        self,
        symbol: str,
        klines: List[Dict],
        initial_balance: float,
        leverage: int
    ) -> BacktestResult:
        """模拟交易"""
        balance = initial_balance
        trades: List[BacktestTrade] = []
        equity_curve: List[Dict] = []
        trade_id = 0

        # 状态
        position = None  # {'direction', 'entry_price', 'quantity', 'entry_time', 'tp', 'sl'}
        win_streak = 0
        lose_streak = 0

        # 模拟订单流信号
        signal_generator = SimulatedSignalGenerator()

        # 遍历K线
        for i, kline in enumerate(klines):
            current_time = datetime.fromtimestamp(kline['timestamp'] / 1000)
            current_price = kline['close']

            # 记录权益曲线
            equity_curve.append({
                'timestamp': kline['timestamp'],
                'balance': balance,
                'price': current_price
            })

            # 如果有持仓，检查止盈止损
            if position:
                exit_price, exit_reason = self._check_exit(
                    position, kline
                )

                if exit_price:
                    # 平仓
                    pnl_pct, pnl_amount = self._calculate_pnl(
                        position, exit_price, leverage
                    )

                    balance_before = balance
                    balance += pnl_amount

                    trade = BacktestTrade(
                        trade_id=trade_id,
                        symbol=symbol,
                        direction=position['direction'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        leverage=leverage,
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        pnl_pct=pnl_pct,
                        pnl_amount=pnl_amount,
                        exit_reason=exit_reason,
                        balance_before=balance_before,
                        balance_after=balance
                    )
                    trades.append(trade)
                    trade_id += 1

                    # 更新连胜/连亏
                    if pnl_amount > 0:
                        win_streak += 1
                        lose_streak = 0
                    else:
                        lose_streak += 1
                        win_streak = 0

                    position = None

                    # 连亏冷却
                    if lose_streak >= scalping_config.max_consecutive_losses:
                        # 跳过一些K线
                        continue

            # 如果没有持仓，检查是否有信号
            if position is None and balance > 0:
                signal = signal_generator.generate_signal(kline, klines[max(0, i-30):i])

                if signal:
                    # 计算仓位
                    position_ratio = scalping_config.calculate_position_ratio(win_streak, lose_streak)
                    margin = balance * position_ratio
                    position_value = margin * leverage
                    quantity = position_value / current_price

                    # 计算止盈止损
                    if signal == "LONG":
                        tp = current_price * (1 + scalping_config.take_profit_pct)
                        sl = current_price * (1 - scalping_config.stop_loss_pct)
                    else:
                        tp = current_price * (1 - scalping_config.take_profit_pct)
                        sl = current_price * (1 + scalping_config.stop_loss_pct)

                    position = {
                        'direction': signal,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_time': current_time,
                        'tp': tp,
                        'sl': sl
                    }

        # 如果还有持仓，强制平仓
        if position:
            exit_price = klines[-1]['close']
            pnl_pct, pnl_amount = self._calculate_pnl(position, exit_price, leverage)
            balance += pnl_amount

            trade = BacktestTrade(
                trade_id=trade_id,
                symbol=symbol,
                direction=position['direction'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                quantity=position['quantity'],
                leverage=leverage,
                entry_time=position['entry_time'],
                exit_time=datetime.fromtimestamp(klines[-1]['timestamp'] / 1000),
                pnl_pct=pnl_pct,
                pnl_amount=pnl_amount,
                exit_reason="end_of_backtest",
                balance_before=balance - pnl_amount,
                balance_after=balance
            )
            trades.append(trade)

        # 计算统计数据
        return self._calculate_statistics(
            symbol=symbol,
            trades=trades,
            equity_curve=equity_curve,
            initial_balance=initial_balance,
            final_balance=balance,
            klines=klines
        )

    def _check_exit(self, position: Dict, kline: Dict) -> Tuple[Optional[float], str]:
        """检查是否触发平仓"""
        high = kline['high']
        low = kline['low']

        if position['direction'] == "LONG":
            # 先检查止损（假设止损先触发）
            if low <= position['sl']:
                return position['sl'], "stop_loss"
            # 再检查止盈
            if high >= position['tp']:
                return position['tp'], "take_profit"
        else:
            # 做空
            if high >= position['sl']:
                return position['sl'], "stop_loss"
            if low <= position['tp']:
                return position['tp'], "take_profit"

        return None, ""

    def _calculate_pnl(
        self,
        position: Dict,
        exit_price: float,
        leverage: int
    ) -> Tuple[float, float]:
        """计算盈亏（含手续费）"""
        # 计算手续费（开仓+平仓双边）
        fee_rate = scalping_config.taker_fee_rate if scalping_config.use_taker_fee else scalping_config.maker_fee_rate
        entry_fee = position['quantity'] * position['entry_price'] * fee_rate
        exit_fee = position['quantity'] * exit_price * fee_rate
        total_fee = entry_fee + exit_fee

        # 计算价格变动盈亏
        if position['direction'] == "LONG":
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

        # 考虑杠杆
        pnl_pct_leveraged = pnl_pct * leverage

        # 计算金额（基于仓位价值，扣除手续费）
        position_value = position['quantity'] * position['entry_price']
        margin = position_value / leverage
        gross_pnl = margin * pnl_pct_leveraged
        pnl_amount = gross_pnl - total_fee  # 净利润 = 毛利润 - 手续费

        return pnl_pct_leveraged, pnl_amount

    def _calculate_statistics(
        self,
        symbol: str,
        trades: List[BacktestTrade],
        equity_curve: List[Dict],
        initial_balance: float,
        final_balance: float,
        klines: List[Dict]
    ) -> BacktestResult:
        """计算统计数据"""
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl_amount > 0)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_profit = sum(t.pnl_amount for t in trades if t.pnl_amount > 0)
        total_loss = abs(sum(t.pnl_amount for t in trades if t.pnl_amount < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        total_return = final_balance - initial_balance
        total_return_pct = total_return / initial_balance if initial_balance > 0 else 0

        # 计算最大回撤
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(equity_curve)

        return BacktestResult(
            symbol=symbol,
            start_time=datetime.fromtimestamp(klines[0]['timestamp'] / 1000),
            end_time=datetime.fromtimestamp(klines[-1]['timestamp'] / 1000),
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            equity_curve=equity_curve
        )

    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> Tuple[float, float]:
        """计算最大回撤"""
        if not equity_curve:
            return 0, 0

        peak = equity_curve[0]['balance']
        max_dd = 0
        max_dd_pct = 0

        for point in equity_curve:
            balance = point['balance']
            if balance > peak:
                peak = balance
            else:
                dd = peak - balance
                dd_pct = dd / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct

        return max_dd, max_dd_pct


class SimulatedSignalGenerator:
    """模拟信号生成器（用于回测）"""

    def __init__(self):
        self.last_signal_time = 0
        self.signal_cooldown = 3 * 60 * 1000  # 3分钟冷却

    def generate_signal(self, current_kline: Dict, history: List[Dict]) -> Optional[str]:
        """
        基于K线数据模拟信号

        使用EMA交叉 + 动量确认
        """
        if len(history) < 15:
            return None

        # 冷却检查
        if current_kline['timestamp'] - self.last_signal_time < self.signal_cooldown:
            return None

        # 计算EMA
        closes = [k['close'] for k in history[-15:]] + [current_kline['close']]

        def ema(data, period):
            multiplier = 2 / (period + 1)
            result = data[0]
            for price in data[1:]:
                result = (price - result) * multiplier + result
            return result

        ema3 = ema(closes, 3)
        ema8 = ema(closes, 8)

        prev_closes = closes[:-1]
        prev_ema3 = ema(prev_closes, 3)
        prev_ema8 = ema(prev_closes, 8)

        # 动量
        momentum = (closes[-1] - closes[-3]) / closes[-3]

        # 成交量
        volumes = [k['volume'] for k in history[-5:]]
        avg_vol = sum(volumes) / len(volumes)
        vol_ok = current_kline['volume'] > avg_vol * 0.7

        # 金叉 + 动量向上
        if prev_ema3 <= prev_ema8 and ema3 > ema8 and momentum > 0 and vol_ok:
            self.last_signal_time = current_kline['timestamp']
            return "LONG"

        # 死叉 + 动量向下
        if prev_ema3 >= prev_ema8 and ema3 < ema8 and momentum < 0 and vol_ok:
            self.last_signal_time = current_kline['timestamp']
            return "SHORT"

        return None


# 便捷函数
async def run_scalping_backtest(
    symbol: str = "1000PEPE/USDT",
    days: int = 7,
    initial_balance: float = 5.0,
    leverage: int = 20
) -> BacktestResult:
    """运行剥头皮策略回测"""
    backtester = ScalpingBacktester()
    return await backtester.run_backtest(
        symbol=symbol,
        days=days,
        initial_balance=initial_balance,
        leverage=leverage
    )


def print_backtest_report(result: BacktestResult):
    """打印回测报告"""
    print("\n" + "=" * 60)
    print("📊 剥头皮策略回测报告")
    print("=" * 60)

    print(f"\n【基本信息】")
    print(f"  交易对: {result.symbol}")
    print(f"  回测期间: {result.start_time.strftime('%Y-%m-%d %H:%M')} ~ {result.end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  初始余额: {result.initial_balance:.4f} USDT")
    print(f"  最终余额: {result.final_balance:.4f} USDT")

    print(f"\n【收益统计】")
    print(f"  总收益: {result.total_return:+.4f} USDT ({result.total_return_pct:+.2%})")
    print(f"  最大回撤: {result.max_drawdown:.4f} USDT ({result.max_drawdown_pct:.2%})")

    print(f"\n【交易统计】")
    print(f"  总交易次数: {result.total_trades}")
    print(f"  盈利次数: {result.winning_trades}")
    print(f"  亏损次数: {result.losing_trades}")
    print(f"  胜率: {result.win_rate:.2%}")
    print(f"  利润因子: {result.profit_factor:.2f}")

    if result.trades:
        avg_win = sum(t.pnl_amount for t in result.trades if t.pnl_amount > 0) / max(1, result.winning_trades)
        avg_loss = sum(t.pnl_amount for t in result.trades if t.pnl_amount < 0) / max(1, result.losing_trades)
        print(f"  平均盈利: {avg_win:+.4f} USDT")
        print(f"  平均亏损: {avg_loss:+.4f} USDT")

        # 按退出原因统计
        exit_reasons = {}
        for t in result.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        print(f"\n【退出原因统计】")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} ({count/result.total_trades:.1%})")

    print("\n" + "=" * 60)
