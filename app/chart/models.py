# -*- coding: utf-8 -*-
"""
图表数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class ChartResult:
    """
    图表生成结果

    Attributes:
        symbol: 交易对
        chart_5m: 5m 图表路径
        chart_15m: 15m 图表路径
        indicators_5m: 5m 指标快照（最新值）
        indicators_15m: 15m 指标快照（最新值）
        generated_at: 生成时间戳 (ms)
    """
    symbol: str
    chart_5m: str = ""
    chart_15m: str = ""
    indicators_5m: Dict = field(default_factory=dict)
    indicators_15m: Dict = field(default_factory=dict)
    generated_at: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
