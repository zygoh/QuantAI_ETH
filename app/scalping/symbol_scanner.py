"""
币种自动筛选器

功能：
- 从Binance获取所有USDT永续合约
- 按波动率、成交量、流动性排序
- 自动筛选最适合剥头皮的币种
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from app.exchange.clients.binance.binance_client import BinanceClient
from app.scalping.config import SymbolConfig, TradingPhase

logger = logging.getLogger(__name__)


@dataclass
class SymbolMetrics:
    """币种指标"""
    symbol: str                     # 交易对 (如 "BTCUSDT")
    standard_symbol: str            # 标准格式 (如 "BTC/USDT")
    price: float                    # 当前价格
    volume_24h: float               # 24小时成交量(USDT)
    price_change_24h: float         # 24小时价格变化百分比
    high_24h: float                 # 24小时最高价
    low_24h: float                  # 24小时最低价
    volatility: float               # 波动率 (high-low)/price
    max_leverage: int               # 最大杠杆
    min_notional: float             # 最小下单金额
    tick_size: float                # 价格精度
    lot_size: float                 # 数量精度

    # 评分
    volatility_score: float = 0.0  # 波动率得分
    volume_score: float = 0.0      # 成交量得分
    total_score: float = 0.0       # 综合得分


class SymbolScanner:
    """币种扫描器"""

    def __init__(self):
        self.client = BinanceClient()
        self.cache: Dict[str, SymbolMetrics] = {}
        self.last_scan_time: Optional[datetime] = None

        # 筛选条件（针对5U小资金优化）
        self.min_volume_24h = 30_000_000       # 最小24h成交量 3000万USDT
        self.min_volatility = 0.04             # 最小波动率 4%
        self.max_price = 10                    # 最大价格 10U（放宽，让更多币种参与）
        self.min_leverage = 20                 # 最小杠杆要求

        # 排除列表（只排除稳定币和指数）
        self.exclude_symbols = {
            # 稳定币
            'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'FDUSDUSDT', 'DAIUSDT', 'USDPUSDT',
            # 指数类
            'BTCDOMUSDT', 'DEFIUSDT',
        }

    async def scan_all_symbols(self) -> List[SymbolMetrics]:
        """
        扫描所有USDT永续合约

        Returns:
            按综合得分排序的币种列表
        """
        logger.info("🔍 开始扫描Binance USDT永续合约...")

        # 1. 获取交易所信息
        exchange_info = self.client.get_exchange_info()
        if not exchange_info:
            raise Exception("无法获取交易所信息")

        symbols_info = {
            s['symbol']: s for s in exchange_info.get('symbols', [])
            if s.get('contractType') == 'PERPETUAL'
            and s.get('quoteAsset') == 'USDT'
            and s.get('status') == 'TRADING'
        }

        logger.info(f"📊 找到 {len(symbols_info)} 个USDT永续合约")

        # 2. 获取24小时行情
        tickers = await self._get_24h_tickers()
        if not tickers:
            raise Exception("无法获取24小时行情")

        # 3. 解析并筛选
        all_metrics: List[SymbolMetrics] = []

        for symbol, info in symbols_info.items():
            # 排除特殊币种
            if symbol in self.exclude_symbols:
                continue

            # 获取行情数据
            ticker = tickers.get(symbol)
            if not ticker:
                continue

            try:
                metrics = self._parse_symbol_metrics(symbol, info, ticker)
                if metrics and self._filter_symbol(metrics):
                    all_metrics.append(metrics)
            except Exception as e:
                logger.debug(f"解析 {symbol} 失败: {e}")
                continue

        logger.info(f"✅ 筛选后剩余 {len(all_metrics)} 个币种")

        # 4. 计算得分并排序
        self._calculate_scores(all_metrics)
        all_metrics.sort(key=lambda x: x.total_score, reverse=True)

        # 缓存结果
        self.cache = {m.symbol: m for m in all_metrics}
        self.last_scan_time = datetime.now()

        return all_metrics

    async def _get_24h_tickers(self) -> Dict[str, Dict]:
        """获取所有币种24小时行情"""
        try:
            # 使用REST API获取
            import requests

            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"

            # 使用代理
            from app.core.config import settings
            proxies = None
            if settings.USE_PROXY:
                proxy_url = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
                proxies = {"http": proxy_url, "https": proxy_url}

            response = requests.get(url, proxies=proxies, timeout=30)
            response.raise_for_status()

            tickers = response.json()
            return {t['symbol']: t for t in tickers}

        except Exception as e:
            logger.error(f"获取24小时行情失败: {e}")
            return {}

    def _parse_symbol_metrics(
        self,
        symbol: str,
        info: Dict,
        ticker: Dict
    ) -> Optional[SymbolMetrics]:
        """解析币种指标"""
        try:
            # 解析价格精度和数量精度
            tick_size = 0.0001
            lot_size = 0.001
            min_notional = 5.0

            for f in info.get('filters', []):
                if f['filterType'] == 'PRICE_FILTER':
                    tick_size = float(f.get('tickSize', 0.0001))
                elif f['filterType'] == 'LOT_SIZE':
                    lot_size = float(f.get('stepSize', 0.001))
                elif f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(f.get('notional', 5.0))

            # 解析行情数据
            price = float(ticker.get('lastPrice', 0))
            volume_24h = float(ticker.get('quoteVolume', 0))
            price_change = float(ticker.get('priceChangePercent', 0)) / 100
            high_24h = float(ticker.get('highPrice', 0))
            low_24h = float(ticker.get('lowPrice', 0))

            # 计算波动率
            volatility = (high_24h - low_24h) / price if price > 0 else 0

            # 获取杠杆信息（从交易所信息中获取）
            # Binance的leverageBracket需要单独API调用，这里使用保守估计
            # 大多数主流币种支持20-75倍，小币种可能只有20倍
            # 为安全起见，根据成交量估算
            if volume_24h > 500_000_000:  # 5亿以上成交量
                max_leverage = 75
            elif volume_24h > 100_000_000:  # 1亿以上
                max_leverage = 50
            elif volume_24h > 50_000_000:  # 5000万以上
                max_leverage = 25
            else:
                max_leverage = 20  # 保守默认值

            # 转换为标准格式
            base_asset = info.get('baseAsset', symbol.replace('USDT', ''))
            standard_symbol = f"{base_asset}/USDT"

            return SymbolMetrics(
                symbol=symbol,
                standard_symbol=standard_symbol,
                price=price,
                volume_24h=volume_24h,
                price_change_24h=price_change,
                high_24h=high_24h,
                low_24h=low_24h,
                volatility=volatility,
                max_leverage=max_leverage,
                min_notional=min_notional,
                tick_size=tick_size,
                lot_size=lot_size
            )

        except Exception as e:
            logger.debug(f"解析 {symbol} 指标失败: {e}")
            return None

    def _filter_symbol(self, metrics: SymbolMetrics) -> bool:
        """筛选币种"""
        # 成交量筛选
        if metrics.volume_24h < self.min_volume_24h:
            return False

        # 波动率筛选
        if metrics.volatility < self.min_volatility:
            return False

        # 价格筛选（小资金友好）
        if metrics.price > self.max_price:
            return False

        # 最小下单金额筛选
        if metrics.min_notional > 10:  # 最小下单不能超过10U
            return False

        return True

    def _calculate_scores(self, metrics_list: List[SymbolMetrics]):
        """计算综合得分"""
        if not metrics_list:
            return

        import math

        # 波动率上限：超过15%不再加分（避免选择过于投机的新币）
        volatility_cap = 0.15
        max_volume = max(m.volume_24h for m in metrics_list)

        for m in metrics_list:
            # 波动率得分 (0-40分) - 波动率越高越好，但有上限
            capped_volatility = min(m.volatility, volatility_cap)
            if capped_volatility >= 0.04:
                m.volatility_score = ((capped_volatility - 0.04) / (volatility_cap - 0.04)) * 40
            else:
                m.volatility_score = 0

            # 成交量得分 (0-50分) - 流动性最重要
            volume_log = math.log10(m.volume_24h + 1)
            max_volume_log = math.log10(max_volume + 1)
            m.volume_score = (volume_log / max_volume_log) * 50 if max_volume_log > 0 else 0

            # 价格友好度得分 (0-10分) - 价格越低越好
            price_score = max(0, 10 - (m.price / 10) * 10)

            # 综合得分（纯粹按波动率+流动性排序）
            m.total_score = m.volatility_score + m.volume_score + price_score

    def get_top_symbols(
        self,
        count: int = 10,
        phase: Optional[TradingPhase] = None
    ) -> List[SymbolConfig]:
        """
        获取排名靠前的币种配置

        Args:
            count: 返回数量
            phase: 交易阶段（可选，用于分配阶段）

        Returns:
            SymbolConfig列表
        """
        if not self.cache:
            logger.warning("缓存为空，请先调用 scan_all_symbols()")
            return []

        # 按得分排序
        sorted_metrics = sorted(
            self.cache.values(),
            key=lambda x: x.total_score,
            reverse=True
        )[:count]

        # 转换为SymbolConfig
        configs = []
        for i, m in enumerate(sorted_metrics):
            # 根据排名分配阶段
            if i < 3:
                assigned_phase = TradingPhase.PHASE_1  # 前3名：高波动
            elif i < 6:
                assigned_phase = TradingPhase.PHASE_2  # 4-6名：中等
            else:
                assigned_phase = TradingPhase.PHASE_3  # 其他：稳健

            config = SymbolConfig(
                symbol=m.standard_symbol,
                max_leverage=m.max_leverage,
                min_notional=m.min_notional,
                tick_size=m.tick_size,
                lot_size=m.lot_size,
                volatility_rank=i + 1,
                liquidity_rank=i + 1,  # 简化处理
                phase=assigned_phase
            )
            configs.append(config)

        return configs

    def print_scan_report(self, top_n: int = 20):
        """打印扫描报告"""
        if not self.cache:
            print("缓存为空，请先调用 scan_all_symbols()")
            return

        sorted_metrics = sorted(
            self.cache.values(),
            key=lambda x: x.total_score,
            reverse=True
        )[:top_n]

        print("\n" + "=" * 80)
        print("📊 币种扫描报告")
        print("=" * 80)
        print(f"扫描时间: {self.last_scan_time}")
        print(f"符合条件币种: {len(self.cache)}")
        print()

        print(f"{'排名':<4} {'币种':<15} {'价格':<12} {'24h波动':<10} {'24h成交量':<15} {'得分':<8}")
        print("-" * 80)

        for i, m in enumerate(sorted_metrics, 1):
            volume_str = f"${m.volume_24h/1_000_000:.1f}M"
            print(f"{i:<4} {m.standard_symbol:<15} ${m.price:<11.4f} {m.volatility:>8.2%} {volume_str:<15} {m.total_score:.1f}")

        print("=" * 80)


# 全局扫描器实例
symbol_scanner = SymbolScanner()


async def auto_select_symbols(count: int = 10) -> List[SymbolConfig]:
    """
    自动选择最佳交易币种

    Args:
        count: 选择数量

    Returns:
        SymbolConfig列表
    """
    await symbol_scanner.scan_all_symbols()
    return symbol_scanner.get_top_symbols(count)


# 测试脚本
async def main():
    """测试币种扫描"""
    scanner = SymbolScanner()

    print("🔍 扫描中...")
    await scanner.scan_all_symbols()

    scanner.print_scan_report(20)

    print("\n📋 推荐交易币种:")
    configs = scanner.get_top_symbols(10)
    for c in configs:
        print(f"  - {c.symbol} (阶段: {c.phase.value})")


if __name__ == "__main__":
    asyncio.run(main())
