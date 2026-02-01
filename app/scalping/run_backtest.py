"""
30天百倍系统启动脚本

使用方法：
    python -m app.scalping.run_backtest
"""
import asyncio
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    from app.scalping.backtest import run_scalping_backtest, print_backtest_report
    from app.scalping.config import scalping_config

    print("\n" + "=" * 60)
    print("🚀 30天百倍系统 - 回测模式")
    print("=" * 60)

    # 显示配置
    print(f"\n【系统配置】")
    print(f"  初始资金: {scalping_config.initial_balance}U")
    print(f"  目标资金: {scalping_config.target_balance}U (100倍)")
    print(f"  每日目标: {scalping_config.get_daily_target_return():.1%}")
    print(f"  止盈: {scalping_config.take_profit_pct:.1%}")
    print(f"  止损: {scalping_config.stop_loss_pct:.1%}")
    print(f"  盈亏比: {scalping_config.take_profit_pct / scalping_config.stop_loss_pct:.1f}:1")

    # 运行多币种回测
    symbols = ["1000PEPE/USDT", "DOGE/USDT", "SOL/USDT"]

    for symbol in symbols:
        try:
            print(f"\n{'=' * 60}")
            print(f"📊 回测 {symbol}")
            print("=" * 60)

            result = await run_scalping_backtest(
                symbol=symbol,
                days=7,
                initial_balance=5.0,
                leverage=20
            )

            print_backtest_report(result)

        except Exception as e:
            logger.error(f"回测 {symbol} 失败: {e}")
            continue

    print("\n✅ 回测完成")


if __name__ == "__main__":
    asyncio.run(main())
