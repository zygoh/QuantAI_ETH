# -*- coding: utf-8 -*-
"""
市场微观数据：订单簿、近期成交、成交分布

将订单簿与逐笔/聚合成交汇总为一段文本，供 AI 分析时参考（买卖压力、大单、挂单墙等）。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_orderbook_summary(
    depth: Dict[str, Any],
    current_price: float,
    top_n: int = 5,
) -> str:
    """
    从 depth 接口返回的 {bids, asks} 生成订单簿摘要。
    bids/asks: [[price, qty], ...]，按价格排序。
    """
    lines = []
    bids = depth.get("bids") or []
    asks = depth.get("asks") or []

    def fmt_side(rows: List[List[Any]], label: str) -> str:
        if not rows:
            return f"{label}: 无数据"
        part = []
        for i, (p, q) in enumerate(rows[:top_n]):
            try:
                price = float(p)
                qty = float(q)
                part.append(f"  {price:.6g} x {qty:.4g}")
            except (TypeError, ValueError):
                continue
        return f"{label}:\n" + "\n".join(part) if part else f"{label}: 无"

    bid_str = fmt_side(bids, "买盘")
    ask_str = fmt_side(asks, "卖盘")
    lines.append(bid_str)
    lines.append(ask_str)

    # 买卖量合计（前 top_n 档）
    try:
        bid_vol = sum(float(q) for _, q in bids[:top_n])
        ask_vol = sum(float(q) for _, q in asks[:top_n])
        lines.append(f"前{top_n}档买量合计: {bid_vol:.4g}, 卖量合计: {ask_vol:.4g}")
        if ask_vol > 0:
            ratio = bid_vol / ask_vol
            lines.append(f"买卖比(买/卖): {ratio:.2f}")
    except (TypeError, ValueError):
        pass

    if bids and asks:
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_pct = (spread / current_price) * 100 if current_price > 0 else 0
            lines.append(f"买一/卖一: {best_bid:.6g} / {best_ask:.6g}, 价差: {spread:.6g} ({spread_pct:.3f}%)")
        except (TypeError, ValueError, IndexError):
            pass

    return "\n".join(lines)


def build_trades_summary(
    agg_trades: List[Dict[str, Any]],
    current_price: float,
) -> str:
    """
    从 aggTrades 生成近期成交分布与买卖压力。
    单条: p=price, q=quantity, m=was buyer maker (True=卖方主动/卖压, False=买方主动/买压)
    """
    if not agg_trades:
        return "近期成交: 无数据"

    buy_vol = 0.0
    sell_vol = 0.0
    total_qty = 0.0
    by_price: Dict[str, float] = {}  # 按价格档位汇总量（用字符串 key 避免浮点 key）

    for t in agg_trades:
        try:
            p = float(t.get("p", 0))
            q = float(t.get("q", 0))
            m = t.get("m", False)  # True = 卖方主动
            if p <= 0 or q <= 0:
                continue
            total_qty += q
            if m:
                sell_vol += q
            else:
                buy_vol += q
            # 成交分布：按当前价的 0.5% 为一档
            if current_price > 0:
                bucket = round((p - current_price) / current_price * 200) / 200  # 0.5% step
                key = f"{bucket:+.2%}"
                by_price[key] = by_price.get(key, 0) + q
        except (TypeError, ValueError):
            continue

    lines = [
        f"近期{len(agg_trades)}笔聚合成交:",
        f"  主动买量(吃单): {buy_vol:.4g}",
        f"  主动卖量(吃单): {sell_vol:.4g}",
        f"  总成交量: {total_qty:.4g}",
    ]
    if total_qty > 0:
        buy_ratio = buy_vol / total_qty * 100
        lines.append(f"  买压占比: {buy_ratio:.1f}% (买>50% 偏多，卖>50% 偏空)")

    # 成交分布：相对当前价的档位（0.5% 一档）
    if by_price and current_price > 0:
        def _bucket_sort_key(k: str) -> float:
            s = k.strip().replace("%", "").replace("+", "")
            try:
                return float(s)
            except ValueError:
                return 0.0
        sorted_buckets = sorted(by_price.keys(), key=_bucket_sort_key)
        dist_lines = ["  成交分布(相对当前价):"]
        for k in sorted_buckets[:11]:  # 约 ±2.5%
            dist_lines.append(f"    {k}: 量 {by_price[k]:.4g}")
        lines.append("\n".join(dist_lines))

    return "\n".join(lines)


def build_market_context(
    symbol: str,
    current_price: float,
    orderbook: Optional[Dict[str, Any]] = None,
    agg_trades: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    组装订单簿 + 近期成交 + 成交分布为一段「市场微观」文本，供 prompt 使用。
    任一数据缺失时只输出已有部分。
    """
    parts = [f"### 市场微观数据（{symbol}）", ""]

    if orderbook:
        parts.append("【订单簿】")
        parts.append(build_orderbook_summary(orderbook, current_price))
        parts.append("")
    else:
        parts.append("【订单簿】无数据")
        parts.append("")

    if agg_trades is not None:
        parts.append("【近期成交与买卖压力】")
        parts.append(build_trades_summary(agg_trades, current_price))
    else:
        parts.append("【近期成交】无数据")

    return "\n".join(parts)
