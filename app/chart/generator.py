# -*- coding: utf-8 -*-
"""
图表生成器

生成 K 线图表，包括：
- 15m 和 2h 两个时间周期
- 技术指标叠加（EMA、布林带、VWAP）
- 副图指标（RSI、KDJ、ADX、MACD、成交量、OBV）
- 旧图表清理

需求: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
"""

import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 强制使用 Agg 后端（无 GUI），避免多线程 Tkinter 问题
import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from app.core.config import settings
from app.exchange.clients.binance.binance_client import binance_client, UnifiedKlineData
from app.chart.indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from app.chart.models import ChartResult


logger = logging.getLogger(__name__)

# 设置 matplotlib 字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 图表路径生成函数（纯函数，用于属性测试）
# =============================================================================

def generate_chart_path(symbol: str, interval: str, base_dir: str = "image") -> str:
    """
    生成图表保存路径
    
    路径格式: image/{symbol}/{symbol}_{interval}.png
    
    这是一个纯函数，用于属性测试。
    
    Args:
        symbol: 交易对（不含斜杠，如 "BTCUSDT"）
        interval: 时间周期（如 "15m", "2h"）
        base_dir: 基础目录
        
    Returns:
        图表文件路径
    """
    # 移除斜杠，确保路径安全
    safe_symbol = symbol.replace("/", "").replace("\\", "")
    return os.path.join(base_dir, safe_symbol, f"{safe_symbol}_{interval}.png")


def validate_chart_path(path: str) -> bool:
    """
    验证图表路径格式是否正确
    
    路径应符合格式: image/{symbol}/{symbol}_{interval}.png
    
    Args:
        path: 图表路径
        
    Returns:
        True 如果路径格式正确
    """
    # 检查是否以 .png 结尾
    if not path.endswith(".png"):
        return False
    
    # 分解路径
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 3:
        return False
    
    # 检查目录名和文件名是否匹配
    dir_name = parts[-2]
    file_name = parts[-1]
    
    # 文件名应为 {symbol}_{interval}.png
    if not file_name.startswith(dir_name + "_"):
        return False
    
    # 检查 interval 是否有效
    interval_part = file_name[len(dir_name) + 1:-4]  # 去掉 symbol_ 和 .png
    valid_intervals = ["5m", "15m", "2h"]
    if interval_part not in valid_intervals:
        return False
    
    return True


# =============================================================================
# 技术指标计算辅助函数
# =============================================================================

def calculate_kdj(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 KDJ 指标
    
    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        period: RSV 周期
        k_smooth: K 值平滑周期
        d_smooth: D 值平滑周期
        
    Returns:
        (K, D, J) 三元组
    """
    n = len(closes)
    rsv = np.zeros(n)
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(period - 1, n):
        highest = np.max(highs[i - period + 1:i + 1])
        lowest = np.min(lows[i - period + 1:i + 1])
        
        if highest == lowest:
            rsv[i] = 50.0
        else:
            rsv[i] = (closes[i] - lowest) / (highest - lowest) * 100
    
    # 计算 K 值（RSV 的 EMA）
    k[period - 1] = rsv[period - 1]
    for i in range(period, n):
        k[i] = (k[i - 1] * (k_smooth - 1) + rsv[i]) / k_smooth
    
    # 计算 D 值（K 的 EMA）
    d[period - 1] = k[period - 1]
    for i in range(period, n):
        d[i] = (d[i - 1] * (d_smooth - 1) + k[i]) / d_smooth
    
    # 计算 J 值
    j = 3 * k - 2 * d
    
    return k, d, j


def calculate_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    计算 ADX 指标
    
    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        period: 周期
        
    Returns:
        ADX 序列
    """
    n = len(closes)
    if n < period + 1:
        return np.full(n, 25.0)
    
    # 计算 TR, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # 平滑 TR, +DM, -DM
    atr = calculate_ema(tr, period)
    plus_di = calculate_ema(plus_dm, period) / np.maximum(atr, 1e-10) * 100
    minus_di = calculate_ema(minus_dm, period) / np.maximum(atr, 1e-10) * 100
    
    # 计算 DX
    dx = np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10) * 100
    
    # 计算 ADX
    adx = calculate_ema(dx, period)
    
    return adx


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    计算 ATR（真实波幅均值）

    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        period: 周期

    Returns:
        ATR 序列
    """
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    return calculate_ema(tr, period)


def calculate_vwap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray
) -> np.ndarray:
    """
    计算 VWAP（成交量加权平均价格）
    
    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        volumes: 成交量序列
        
    Returns:
        VWAP 序列
    """
    typical_price = (highs + lows + closes) / 3
    cumulative_tp_vol = np.cumsum(typical_price * volumes)
    cumulative_vol = np.cumsum(volumes)
    
    vwap = cumulative_tp_vol / np.maximum(cumulative_vol, 1e-10)
    return vwap


def calculate_obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    计算 OBV（能量潮指标）
    
    Args:
        closes: 收盘价序列
        volumes: 成交量序列
        
    Returns:
        OBV 序列
    """
    n = len(closes)
    obv = np.zeros(n)
    
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    
    return obv


# =============================================================================
# 图表生成器
# =============================================================================

class ChartGenerator:
    """
    图表生成器
    
    生成 K 线图表，包括技术指标和副图。
    
    Attributes:
        intervals: 图表时间周期列表
        base_dir: 图表保存基础目录
        watermark: 水印文字
    """
    
    def __init__(self) -> None:
        """初始化图表生成器"""
        self.intervals: List[str] = settings.CHART_INTERVALS
        self.base_dir: str = "image"
        self.watermark: str = "@Three_Dog_z"
        self.kline_limit: int = 50
        
        logger.info(
            f"📊 图表生成器初始化完成 - "
            f"周期: {self.intervals}, "
            f"保存目录: {self.base_dir}"
        )
    
    def generate_charts(self, symbol: str, current_price: float = 0.0) -> ChartResult:
        """
        生成 5m 和 15m 图表

        Args:
            symbol: 交易对（支持标准格式和交易所格式）
            current_price: 外部传入的实时价格，确保两张图显示一致

        Returns:
            ChartResult: 包含图表路径
        """
        # 标准化交易对格式
        exchange_symbol = symbol.replace("/", "")

        logger.info(f"📊 开始生成 {exchange_symbol} 图表")

        # 清理旧图表
        self.cleanup_old_charts(exchange_symbol)

        chart_paths: Dict[str, str] = {}

        for interval in self.intervals:
            try:
                path = self._generate_single_chart(exchange_symbol, interval, current_price)
                chart_paths[interval] = path
                logger.debug(f"✅ {exchange_symbol} {interval} 图表生成成功: {path}")
            except Exception as e:
                logger.error(f"❌ {exchange_symbol} {interval} 图表生成失败: {e}")
                chart_paths[interval] = ""

        return ChartResult(
            symbol=exchange_symbol,
            chart_5m=chart_paths.get("5m", ""),
            chart_15m=chart_paths.get("15m", "")
        )
    
    def cleanup_old_charts(self, symbol: str) -> None:
        """
        删除旧图表
        
        Args:
            symbol: 交易对
        """
        safe_symbol = symbol.replace("/", "").replace("\\", "")
        chart_dir = os.path.join(self.base_dir, safe_symbol)
        
        if os.path.exists(chart_dir):
            try:
                shutil.rmtree(chart_dir)
                logger.debug(f"🗑️ 已删除旧图表目录: {chart_dir}")
            except Exception as e:
                logger.warning(f"⚠️ 删除旧图表目录失败: {e}")
    
    def _generate_single_chart(self, symbol: str, interval: str, current_price: float = 0.0) -> str:
        """
        生成单个时间周期的图表

        Args:
            symbol: 交易对
            interval: 时间周期
            current_price: 外部传入的实时价格

        Returns:
            图表文件路径
        """
        # 获取 K 线数据
        klines = binance_client.get_klines(symbol, interval, self.kline_limit)

        if not klines or len(klines) < 2:
            raise ValueError(f"K 线数据不足: {len(klines) if klines else 0}")

        # 提取数据
        timestamps = [k.timestamp for k in klines]
        opens = np.array([k.open for k in klines])
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])
        closes = np.array([k.close for k in klines])
        volumes = np.array([k.volume for k in klines])
        quote_volumes = np.array([k.quote_volume for k in klines])

        # 转换时间戳
        dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]

        # 计算技术指标
        indicators = self._calculate_indicators(opens, highs, lows, closes, volumes)

        # 生成图表
        filepath = self._create_chart(
            symbol, interval, dates,
            opens, highs, lows, closes, volumes, quote_volumes,
            indicators, current_price
        )

        return filepath
    
    def _calculate_indicators(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        计算所有技术指标
        
        Args:
            opens: 开盘价序列
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列
            volumes: 成交量序列
            
        Returns:
            指标字典
        """
        indicators = {}
        
        # EMA
        indicators['ema9'] = calculate_ema(closes, 9)
        indicators['ema21'] = calculate_ema(closes, 21)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes, 20, 2.0)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # VWAP
        indicators['vwap'] = calculate_vwap(highs, lows, closes, volumes)
        
        # RSI
        indicators['rsi'] = calculate_rsi(closes, 14)
        
        # KDJ
        k, d, j = calculate_kdj(highs, lows, closes)
        indicators['kdj_k'] = k
        indicators['kdj_d'] = d
        
        # ADX
        indicators['adx'] = calculate_adx(highs, lows, closes)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(closes)
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_hist'] = histogram

        # OBV
        indicators['obv'] = calculate_obv(closes, volumes)

        # ATR（用于 AI 判断波动幅度和止损距离）
        indicators['atr'] = calculate_atr(highs, lows, closes, 14)

        # 成交量 SMA（用于 AI 判断放量/缩量）
        indicators['vol_sma20'] = calculate_sma(volumes, 20)

        return indicators
    
    def _create_chart(
        self,
        symbol: str,
        interval: str,
        dates: List[datetime],
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        quote_volumes: np.ndarray,
        indicators: Dict[str, np.ndarray],
        current_price: float = 0.0
    ) -> str:
        """
        创建图表

        Args:
            symbol: 交易对
            interval: 时间周期
            dates: 日期列表
            opens: 开盘价
            highs: 最高价
            lows: 最低价
            closes: 收盘价
            volumes: 成交量
            quote_volumes: 成交额
            indicators: 技术指标

        Returns:
            图表文件路径
        """
        # 深色主题
        bg_color = '#1a1a2e'
        panel_color = '#16213e'
        grid_color = '#2a2a4a'
        text_color = '#e0e0e0'
        up_color = '#26a69a'
        down_color = '#ef5350'

        fig = plt.figure(figsize=(16, 14), facecolor=bg_color)
        gs = fig.add_gridspec(4, 1, height_ratios=[4, 1.5, 1.5, 1.2], hspace=0.15)

        axes = []
        for i in range(4):
            ax = fig.add_subplot(gs[i])
            ax.set_facecolor(panel_color)
            ax.tick_params(colors=text_color, labelsize=9)
            ax.spines['top'].set_color(grid_color)
            ax.spines['bottom'].set_color(grid_color)
            ax.spines['left'].set_color(grid_color)
            ax.spines['right'].set_color(grid_color)
            ax.yaxis.label.set_color(text_color)
            axes.append(ax)

        ax1, ax2, ax3, ax4 = axes

        # 共享 X 轴
        ax2.sharex(ax1)
        ax3.sharex(ax1)
        ax4.sharex(ax1)

        # 主图：K 线 + 指标
        self._draw_candlesticks(ax1, dates, opens, highs, lows, closes, interval)
        self._draw_main_indicators(ax1, dates, closes, indicators)
        ax1.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{symbol} - {interval.upper()} Chart', fontsize=14, fontweight='bold', color=text_color, pad=10)
        ax1.legend(loc='upper left', fontsize=8, ncol=2, facecolor=panel_color, edgecolor=grid_color, labelcolor=text_color)
        ax1.grid(True, alpha=0.2, color=grid_color)
        ax1.tick_params(axis='x', labelbottom=False)

        # ATR 标注（右上角，供 AI 参考波动幅度）
        if 'atr' in indicators:
            atr_val = indicators['atr'][-1]
            atr_pct = (atr_val / closes[-1] * 100) if closes[-1] > 0 else 0
            ax1.text(0.98, 0.95, f'ATR(14): {atr_val:.4f} ({atr_pct:.2f}%)',
                     transform=ax1.transAxes, fontsize=9, color='#ffd54f',
                     ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=panel_color, edgecolor=grid_color, alpha=0.8))

        # RSI/KDJ/ADX 子图
        self._draw_rsi_kdj_adx(ax2, dates, indicators)
        ax2.tick_params(axis='x', labelbottom=False)

        # MACD 子图
        self._draw_macd(ax3, dates, indicators, interval)
        ax3.tick_params(axis='x', labelbottom=False)

        # 成交量子图
        self._draw_volume(ax4, dates, opens, closes, volumes, indicators, interval)

        # 价格信息（优先使用外部传入的实时价格，确保多图一致）
        display_price = current_price if current_price > 0 else closes[-1]
        price_change = display_price - opens[0]
        price_change_pct = (price_change / opens[0] * 100) if opens[0] > 0 else 0
        change_color = up_color if price_change >= 0 else down_color
        info_text = f'Current: {display_price:,.2f}|Change: {price_change:+,.2f} ({price_change_pct:+.2f}%)'
        fig.text(0.5, 0.02, info_text, fontsize=11, ha='center', color=change_color, fontweight='bold')

        # 水印
        fig.text(0.99, 0.01, self.watermark, fontsize=9, color='gray', alpha=0.3,
                 ha='right', va='bottom', fontstyle='italic')

        # 保存图表
        filepath = generate_chart_path(symbol, interval, self.base_dir)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=bg_color)
        plt.close()

        return filepath
    
    def _draw_candlesticks(
        self,
        ax: plt.Axes,
        dates: List[datetime],
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        interval: str
    ) -> None:
        """绘制 K 线"""
        up_color = '#26a69a'
        down_color = '#ef5350'

        for i, date in enumerate(dates):
            is_up = closes[i] >= opens[i]
            color = up_color if is_up else down_color

            # 影线（颜色跟随涨跌）
            ax.plot([date, date], [lows[i], highs[i]], color=color, linewidth=0.8)

            # 实体
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])

            width_hours = 0.055 if interval == '5m' else (0.17 if interval == '15m' else 1.5)
            width_days = width_hours / 24.0

            rect = Rectangle(
                (mdates.date2num(date) - width_days * 0.4, body_bottom),
                width_days * 0.8,
                body_height if body_height > 0 else (highs[i] - lows[i]) * 0.05,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5
            )
            ax.add_patch(rect)
    
    def _draw_main_indicators(
        self,
        ax: plt.Axes,
        dates: List[datetime],
        closes: np.ndarray,
        indicators: Dict[str, np.ndarray]
    ) -> None:
        """绘制主图指标"""
        n = len(dates)

        # EMA
        if 'ema9' in indicators:
            ax.plot(dates, indicators['ema9'][-n:], label='EMA(9)', color='#ffb74d', linewidth=1.2)
        if 'ema21' in indicators:
            ax.plot(dates, indicators['ema21'][-n:], label='EMA(21)', color='#42a5f5', linewidth=1.2)

        # VWAP
        if 'vwap' in indicators:
            ax.plot(dates, indicators['vwap'][-n:], label='VWAP', color='#ce93d8', linewidth=1.2, linestyle=':')

        # 布林带（轻量填充，不遮挡 K 线）
        if all(k in indicators for k in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_u = indicators['bb_upper'][-n:]
            bb_m = indicators['bb_middle'][-n:]
            bb_l = indicators['bb_lower'][-n:]

            valid_mask = ~np.isnan(bb_u)
            if np.any(valid_mask):
                valid_dates = [d for d, v in zip(dates, valid_mask) if v]
                ax.fill_between(valid_dates, bb_l[valid_mask], bb_u[valid_mask],
                               alpha=0.06, color='#90caf9', label='Bollinger Bands')
                ax.plot(valid_dates, bb_u[valid_mask], color='#64b5f6', linewidth=0.6, alpha=0.5)
                ax.plot(valid_dates, bb_l[valid_mask], color='#64b5f6', linewidth=0.6, alpha=0.5)
    
    def _draw_rsi_kdj_adx(
        self,
        ax: plt.Axes,
        dates: List[datetime],
        indicators: Dict[str, np.ndarray]
    ) -> None:
        """绘制 RSI/KDJ/ADX 子图"""
        n = len(dates)
        panel_color = '#16213e'
        grid_color = '#2a2a4a'
        text_color = '#e0e0e0'

        # RSI
        if 'rsi' in indicators:
            ax.plot(dates, indicators['rsi'][-n:], label='RSI', color='#f06292', linewidth=1.5)
            ax.axhline(y=70, color='#ef5350', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.axhline(y=30, color='#26a69a', linestyle='--', linewidth=0.7, alpha=0.6)
            # 超买超卖区域填充
            ax.axhspan(70, 100, alpha=0.05, color='#ef5350')
            ax.axhspan(0, 30, alpha=0.05, color='#26a69a')

        # KDJ（只画 K 和 D，去掉 J 减少杂乱）
        if 'kdj_k' in indicators and 'kdj_d' in indicators:
            ax.plot(dates, indicators['kdj_k'][-n:], label='K', color='#4dd0e1', linewidth=1, alpha=0.8)
            ax.plot(dates, indicators['kdj_d'][-n:], label='D', color='#ffb74d', linewidth=1, alpha=0.8)

        # ADX（右轴）
        if 'adx' in indicators:
            ax2_twin = ax.twinx()
            ax2_twin.plot(dates, indicators['adx'][-n:], label='ADX', color='#a1887f', linewidth=1.2, linestyle='-.')
            ax2_twin.set_ylabel('ADX', fontsize=9, color='#a1887f')
            ax2_twin.tick_params(axis='y', labelcolor='#a1887f', labelsize=8)
            ax2_twin.set_ylim(0, 80)
            ax2_twin.spines['right'].set_color(grid_color)

        ax.set_ylabel('RSI / KDJ', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=7, ncol=3, facecolor=panel_color, edgecolor=grid_color, labelcolor=text_color)
        ax.grid(True, alpha=0.15, color=grid_color)
    
    def _draw_macd(
        self,
        ax: plt.Axes,
        dates: List[datetime],
        indicators: Dict[str, np.ndarray],
        interval: str
    ) -> None:
        """绘制 MACD 子图"""
        n = len(dates)
        panel_color = '#16213e'
        grid_color = '#2a2a4a'
        text_color = '#e0e0e0'

        if 'macd' in indicators and 'macd_signal' in indicators:
            ax.plot(dates, indicators['macd'][-n:], label='MACD', color='#42a5f5', linewidth=1.5)
            ax.plot(dates, indicators['macd_signal'][-n:], label='Signal', color='#ffb74d', linewidth=1.5)

        if 'macd_hist' in indicators:
            hist = indicators['macd_hist'][-n:]
            # 根据周期计算合适的柱宽
            if len(dates) >= 2:
                avg_gap = (mdates.date2num(dates[-1]) - mdates.date2num(dates[0])) / max(len(dates) - 1, 1)
                bar_width = avg_gap * 0.7
            else:
                bar_width = 0.003

            pos_hist = np.where(hist >= 0, hist, 0)
            neg_hist = np.where(hist < 0, hist, 0)
            ax.bar(dates, pos_hist, color='#26a69a', alpha=0.7, width=bar_width, edgecolor='none')
            ax.bar(dates, neg_hist, color='#ef5350', alpha=0.7, width=bar_width, edgecolor='none')

        ax.axhline(y=0, color='#555555', linestyle='-', linewidth=0.5)
        ax.set_ylabel('MACD', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, facecolor=panel_color, edgecolor=grid_color, labelcolor=text_color)
        ax.grid(True, alpha=0.15, color=grid_color)
    
    def _draw_volume(
        self,
        ax: plt.Axes,
        dates: List[datetime],
        opens: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        indicators: Dict[str, np.ndarray],
        interval: str
    ) -> None:
        """绘制成交量子图"""
        n = len(dates)
        grid_color = '#2a2a4a'
        text_color = '#e0e0e0'

        # 根据实际数据间距计算柱宽
        if len(dates) >= 2:
            avg_gap = (mdates.date2num(dates[-1]) - mdates.date2num(dates[0])) / max(len(dates) - 1, 1)
            bar_width = avg_gap * 0.7
        else:
            bar_width = 0.003

        # 成交量柱（不加边框，干净）
        up_mask = closes >= opens
        down_mask = ~up_mask
        ax.bar([d for d, m in zip(dates, up_mask) if m], volumes[up_mask],
               color='#26a69a', alpha=0.8, width=bar_width, edgecolor='none')
        ax.bar([d for d, m in zip(dates, down_mask) if m], volumes[down_mask],
               color='#ef5350', alpha=0.8, width=bar_width, edgecolor='none')

        # OBV（右轴，细线低透明度）
        if 'obv' in indicators:
            ax_obv = ax.twinx()
            ax_obv.plot(dates, indicators['obv'][-n:], color='#ce93d8', linewidth=1, alpha=0.5)
            ax_obv.set_ylabel('OBV', fontsize=8, color='#ce93d8')
            ax_obv.tick_params(axis='y', labelcolor='#ce93d8', labelsize=7)
            ax_obv.spines['right'].set_color(grid_color)

        # 成交量 SMA(20)
        if 'vol_sma20' in indicators:
            vol_sma = indicators['vol_sma20'][-n:]
            valid_mask = ~np.isnan(vol_sma)
            if np.any(valid_mask):
                valid_dates = [d for d, v in zip(dates, valid_mask) if v]
                ax.plot(valid_dates, vol_sma[valid_mask], color='#ffd54f', linewidth=1.2, alpha=0.8, linestyle='--')

        # 成交量格式化
        def format_volume(value: float, pos: int) -> str:
            if value >= 1_000_000:
                return f'{value / 1_000_000:.1f}M'
            elif value >= 1_000:
                return f'{value / 1_000:.1f}K'
            return f'{value:.0f}'

        ax.yaxis.set_major_formatter(FuncFormatter(format_volume))
        ax.set_ylabel('Volume', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.15, color=grid_color)
        ax.tick_params(axis='y', labelsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.tick_params(axis='x', labelsize=9, rotation=30, colors=text_color)

        # 强制在最后一根 K 线位置添加标签，确保时间范围清晰
        if len(dates) >= 2:
            ax.set_xlim(
                mdates.date2num(dates[0]) - 0.01,
                mdates.date2num(dates[-1]) + 0.01
            )


# 全局图表生成器实例
chart_generator = ChartGenerator()
