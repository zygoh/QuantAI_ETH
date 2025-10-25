"""
交易频率自适应控制模块
用于优化手续费影响，实现智能交易频率控制

核心功能：
1. 动态调整交易频率
2. 手续费影响评估
3. 市场状态感知
4. 盈亏比优化

作者: QuantAI-ETH Team
版本: v3.0
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FrequencyControl:
    """频率控制结果"""
    allow_trade: bool
    reason: str
    frequency_score: float
    fee_impact: float
    optimal_frequency: float
    current_frequency: float


class AdaptiveFrequencyController:
    """
    自适应交易频率控制器
    
    核心功能：
    1. 动态调整交易频率
    2. 手续费影响评估
    3. 市场状态感知
    4. 盈亏比优化
    """
    
    def __init__(self):
        # 频率控制参数
        self.base_frequency_limit = 0.3  # 基础频率限制（30%）
        self.max_daily_trades = 20       # 最大日交易次数
        self.min_trade_interval = 15     # 最小交易间隔（分钟）
        
        # 手续费参数
        self.fee_rate = 0.0007           # 手续费率（0.07%）
        self.target_fee_impact = 0.05    # 目标手续费影响（5%/日）
        
        # 市场状态参数
        self.volatility_threshold = 0.02  # 波动率阈值
        self.trend_strength_threshold = 0.6  # 趋势强度阈值
        
        # 历史记录
        self.trade_history: List[Dict] = []
        self.frequency_history: List[float] = []
        
        logger.info("✅ 自适应交易频率控制器初始化完成")
    
    def calculate_market_state(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算市场状态指标
        
        Args:
            price_data: 价格数据（包含OHLCV）
        
        Returns:
            Dict[str, float]: 市场状态指标
        """
        try:
            if len(price_data) < 20:
                return {
                    'volatility': 0.01,
                    'trend_strength': 0.5,
                    'volume_ratio': 1.0,
                    'price_momentum': 0.0
                }
            
            # 1. 波动率计算（ATR标准化）
            high_low = price_data['high'] - price_data['low']
            high_close = np.abs(price_data['high'] - price_data['close'].shift(1))
            low_close = np.abs(price_data['low'] - price_data['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean()
            volatility = (atr / price_data['close']).mean()
            
            # 2. 趋势强度计算（ADX）
            high_diff = price_data['high'].diff()
            low_diff = price_data['low'].diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), -low_diff, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            trend_strength = adx.mean() / 100
            
            # 3. 成交量比率
            volume_ma = price_data['volume'].rolling(20).mean()
            volume_ratio = price_data['volume'].iloc[-1] / volume_ma.iloc[-1]
            
            # 4. 价格动量
            price_momentum = (price_data['close'].iloc[-1] - price_data['close'].iloc[-20]) / price_data['close'].iloc[-20]
            
            return {
                'volatility': float(volatility),
                'trend_strength': float(trend_strength),
                'volume_ratio': float(volume_ratio),
                'price_momentum': float(price_momentum)
            }
            
        except Exception as e:
            logger.error(f"❌ 市场状态计算失败: {e}")
            return {
                'volatility': 0.01,
                'trend_strength': 0.5,
                'volume_ratio': 1.0,
                'price_momentum': 0.0
            }
    
    def calculate_fee_impact(self, frequency: float) -> float:
        """
        计算手续费影响
        
        Args:
            frequency: 交易频率
        
        Returns:
            float: 手续费影响（%/日）
        """
        return frequency * self.fee_rate * 100 * 2  # 开仓+平仓
    
    def calculate_optimal_frequency(
        self, 
        market_state: Dict[str, float],
        recent_performance: Dict[str, float]
    ) -> float:
        """
        计算最优交易频率
        
        Args:
            market_state: 市场状态
            recent_performance: 近期表现
        
        Returns:
            float: 最优交易频率
        """
        try:
            # 基础频率
            base_freq = self.base_frequency_limit
            
            # 市场状态调整
            volatility = market_state['volatility']
            trend_strength = market_state['trend_strength']
            volume_ratio = market_state['volume_ratio']
            
            # 1. 波动率调整（高波动率降低频率）
            volatility_factor = max(0.5, 1.0 - (volatility - 0.01) * 10)
            
            # 2. 趋势强度调整（强趋势增加频率）
            trend_factor = 0.5 + trend_strength * 0.5
            
            # 3. 成交量调整（高成交量增加频率）
            volume_factor = min(1.5, max(0.5, volume_ratio))
            
            # 4. 近期表现调整
            win_rate = recent_performance.get('win_rate', 0.5)
            avg_profit = recent_performance.get('avg_profit', 0.0)
            
            performance_factor = 0.5 + win_rate * 0.5
            if avg_profit > 0:
                performance_factor = min(1.2, performance_factor * 1.1)
            
            # 综合计算
            optimal_freq = base_freq * volatility_factor * trend_factor * volume_factor * performance_factor
            
            # 限制范围
            optimal_freq = max(0.05, min(0.8, optimal_freq))
            
            logger.debug(f"🔍 最优频率计算: 基础={base_freq:.3f}, "
                        f"波动率因子={volatility_factor:.3f}, "
                        f"趋势因子={trend_factor:.3f}, "
                        f"成交量因子={volume_factor:.3f}, "
                        f"表现因子={performance_factor:.3f}, "
                        f"最优频率={optimal_freq:.3f}")
            
            return optimal_freq
            
        except Exception as e:
            logger.error(f"❌ 最优频率计算失败: {e}")
            return self.base_frequency_limit
    
    def check_trade_frequency(
        self,
        current_time: datetime,
        signal_confidence: float,
        market_state: Dict[str, float],
        recent_performance: Dict[str, float]
    ) -> FrequencyControl:
        """
        检查交易频率限制
        
        Args:
            current_time: 当前时间
            signal_confidence: 信号置信度
            market_state: 市场状态
            recent_performance: 近期表现
        
        Returns:
            FrequencyControl: 频率控制结果
        """
        try:
            # 1. 计算当前交易频率
            current_frequency = self._calculate_current_frequency(current_time)
            
            # 2. 计算最优频率
            optimal_frequency = self.calculate_optimal_frequency(market_state, recent_performance)
            
            # 3. 计算手续费影响
            fee_impact = self.calculate_fee_impact(current_frequency)
            
            # 4. 频率评分
            frequency_score = self._calculate_frequency_score(
                current_frequency, optimal_frequency, signal_confidence
            )
            
            # 5. 判断是否允许交易
            allow_trade, reason = self._should_allow_trade(
                current_frequency, optimal_frequency, fee_impact, signal_confidence
            )
            
            # 6. 记录历史
            self._record_trade_attempt(current_time, allow_trade, frequency_score)
            
            logger.debug(f"🔍 频率检查: 当前频率={current_frequency:.3f}, "
                        f"最优频率={optimal_frequency:.3f}, "
                        f"手续费影响={fee_impact:.3f}%, "
                        f"频率评分={frequency_score:.3f}, "
                        f"允许交易={allow_trade}")
            
            return FrequencyControl(
                allow_trade=allow_trade,
                reason=reason,
                frequency_score=frequency_score,
                fee_impact=fee_impact,
                optimal_frequency=optimal_frequency,
                current_frequency=current_frequency
            )
            
        except Exception as e:
            logger.error(f"❌ 频率检查失败: {e}")
            return FrequencyControl(
                allow_trade=False,
                reason=f"频率检查异常: {e}",
                frequency_score=0.0,
                fee_impact=0.0,
                optimal_frequency=self.base_frequency_limit,
                current_frequency=0.0
            )
    
    def _calculate_current_frequency(self, current_time: datetime) -> float:
        """计算当前交易频率"""
        try:
            # 计算过去24小时的交易次数
            cutoff_time = current_time - timedelta(hours=24)
            recent_trades = [
                trade for trade in self.trade_history
                if trade['timestamp'] >= cutoff_time
            ]
            
            # 计算频率（交易次数/24小时）
            frequency = len(recent_trades) / 24.0
            
            return min(1.0, frequency)
            
        except Exception as e:
            logger.error(f"❌ 当前频率计算失败: {e}")
            return 0.0
    
    def _calculate_frequency_score(
        self, 
        current_freq: float, 
        optimal_freq: float, 
        confidence: float
    ) -> float:
        """计算频率评分"""
        try:
            # 频率匹配度
            freq_match = 1.0 - abs(current_freq - optimal_freq) / optimal_freq
            
            # 置信度加权
            score = freq_match * confidence
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ 频率评分计算失败: {e}")
            return 0.0
    
    def _should_allow_trade(
        self, 
        current_freq: float, 
        optimal_freq: float, 
        fee_impact: float, 
        confidence: float
    ) -> Tuple[bool, str]:
        """判断是否允许交易"""
        try:
            # 1. 频率限制检查
            if current_freq >= optimal_freq * 1.2:  # 超过最优频率20%
                return False, f"交易频率过高 ({current_freq:.3f} >= {optimal_freq:.3f})"
            
            # 2. 手续费影响检查
            if fee_impact > self.target_fee_impact * 1.5:  # 超过目标手续费1.5倍
                return False, f"手续费影响过大 ({fee_impact:.3f}% > {self.target_fee_impact:.3f}%)"
            
            # 3. 置信度检查
            if confidence < 0.4:  # 置信度过低
                return False, f"信号置信度过低 ({confidence:.3f} < 0.4)"
            
            # 4. 最小间隔检查
            if self.trade_history:
                last_trade_time = self.trade_history[-1]['timestamp']
                time_diff = (datetime.now() - last_trade_time).total_seconds() / 60
                if time_diff < self.min_trade_interval:
                    return False, f"交易间隔过短 ({time_diff:.1f}分钟 < {self.min_trade_interval}分钟)"
            
            return True, "通过所有频率检查"
            
        except Exception as e:
            logger.error(f"❌ 交易允许判断失败: {e}")
            return False, f"判断异常: {e}"
    
    def _record_trade_attempt(self, timestamp: datetime, allowed: bool, score: float):
        """记录交易尝试"""
        self.trade_history.append({
            'timestamp': timestamp,
            'allowed': allowed,
            'score': score
        })
        
        # 保持历史记录在合理范围内
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
    
    def get_frequency_statistics(self) -> Dict[str, float]:
        """获取频率统计信息"""
        try:
            if not self.trade_history:
                return {
                    'total_attempts': 0,
                    'allowed_trades': 0,
                    'blocked_trades': 0,
                    'avg_frequency_score': 0.0,
                    'avg_fee_impact': 0.0
                }
            
            total_attempts = len(self.trade_history)
            allowed_trades = sum(1 for trade in self.trade_history if trade['allowed'])
            blocked_trades = total_attempts - allowed_trades
            avg_score = np.mean([trade['score'] for trade in self.trade_history])
            
            # 计算平均手续费影响
            recent_freq = self._calculate_current_frequency(datetime.now())
            avg_fee_impact = self.calculate_fee_impact(recent_freq)
            
            return {
                'total_attempts': total_attempts,
                'allowed_trades': allowed_trades,
                'blocked_trades': blocked_trades,
                'avg_frequency_score': avg_score,
                'avg_fee_impact': avg_fee_impact
            }
            
        except Exception as e:
            logger.error(f"❌ 频率统计计算失败: {e}")
            return {
                'total_attempts': 0,
                'allowed_trades': 0,
                'blocked_trades': 0,
                'avg_frequency_score': 0.0,
                'avg_fee_impact': 0.0
            }
