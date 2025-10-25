"""
交易方向一致性检查模块
用于降低致命错误率（LONG↔SHORT反向交易）

核心功能：
1. 多时间框架方向一致性检查
2. 模型预测一致性验证
3. 置信度阈值过滤
4. 致命错误预防

作者: QuantAI-ETH Team
版本: v3.0
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型枚举"""
    LONG = 2
    HOLD = 1
    SHORT = 0


@dataclass
class ConsistencyCheck:
    """一致性检查结果"""
    is_consistent: bool
    confidence_score: float
    direction_strength: float
    timeframe_agreement: float
    risk_level: str  # LOW, MEDIUM, HIGH


class TradingDirectionConsistencyChecker:
    """
    交易方向一致性检查器
    
    核心功能：
    1. 多时间框架方向一致性检查
    2. 模型预测一致性验证
    3. 置信度阈值过滤
    4. 致命错误预防
    """
    
    def __init__(self):
        self.consistency_threshold = 0.7  # 一致性阈值
        self.confidence_threshold = 0.6   # 置信度阈值
        self.direction_strength_threshold = 0.5  # 方向强度阈值
        
        logger.info("✅ 交易方向一致性检查器初始化完成")
    
    def check_multi_timeframe_consistency(
        self, 
        predictions: Dict[str, int], 
        probabilities: Dict[str, np.ndarray]
    ) -> ConsistencyCheck:
        """
        检查多时间框架预测一致性
        
        Args:
            predictions: {timeframe: prediction} 预测结果
            probabilities: {timeframe: probabilities} 预测概率
        
        Returns:
            ConsistencyCheck: 一致性检查结果
        """
        try:
            timeframes = list(predictions.keys())
            if len(timeframes) < 2:
                return ConsistencyCheck(
                    is_consistent=True,
                    confidence_score=1.0,
                    direction_strength=1.0,
                    timeframe_agreement=1.0,
                    risk_level="LOW"
                )
            
            # 1. 计算方向一致性
            directions = [predictions[tf] for tf in timeframes]
            non_hold_directions = [d for d in directions if d != SignalType.HOLD.value]
            
            if not non_hold_directions:
                # 全部是HOLD，认为一致
                return ConsistencyCheck(
                    is_consistent=True,
                    confidence_score=1.0,
                    direction_strength=0.0,
                    timeframe_agreement=1.0,
                    risk_level="LOW"
                )
            
            # 计算方向一致性比例
            direction_agreement = len(set(non_hold_directions)) == 1
            timeframe_agreement = sum(1 for d in directions if d == non_hold_directions[0]) / len(directions)
            
            # 2. 计算平均置信度
            avg_confidence = np.mean([
                np.max(probabilities[tf]) for tf in timeframes
            ])
            
            # 3. 计算方向强度（非HOLD预测的比例）
            direction_strength = len(non_hold_directions) / len(directions)
            
            # 4. 综合判断
            is_consistent = (
                direction_agreement and 
                timeframe_agreement >= self.consistency_threshold and
                avg_confidence >= self.confidence_threshold
            )
            
            # 5. 风险评估
            if timeframe_agreement >= 0.8 and avg_confidence >= 0.7:
                risk_level = "LOW"
            elif timeframe_agreement >= 0.6 and avg_confidence >= 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            logger.debug(f"🔍 一致性检查: 方向一致={direction_agreement}, "
                        f"时间框架一致性={timeframe_agreement:.3f}, "
                        f"平均置信度={avg_confidence:.3f}, "
                        f"风险等级={risk_level}")
            
            return ConsistencyCheck(
                is_consistent=is_consistent,
                confidence_score=avg_confidence,
                direction_strength=direction_strength,
                timeframe_agreement=timeframe_agreement,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"❌ 一致性检查失败: {e}")
            return ConsistencyCheck(
                is_consistent=False,
                confidence_score=0.0,
                direction_strength=0.0,
                timeframe_agreement=0.0,
                risk_level="HIGH"
            )
    
    def check_model_prediction_consistency(
        self,
        base_predictions: Dict[str, int],
        meta_prediction: int,
        meta_confidence: float
    ) -> ConsistencyCheck:
        """
        检查基础模型与元学习器预测一致性
        
        Args:
            base_predictions: {model_name: prediction} 基础模型预测
            meta_prediction: 元学习器预测
            meta_confidence: 元学习器置信度
        
        Returns:
            ConsistencyCheck: 一致性检查结果
        """
        try:
            # 1. 基础模型一致性
            base_values = list(base_predictions.values())
            non_hold_base = [v for v in base_values if v != SignalType.HOLD.value]
            
            if not non_hold_base:
                # 全部是HOLD
                base_agreement = 1.0
                base_direction = SignalType.HOLD.value
            else:
                base_agreement = sum(1 for v in base_values if v == non_hold_base[0]) / len(base_values)
                base_direction = non_hold_base[0]
            
            # 2. 基础模型与元学习器一致性
            if meta_prediction == SignalType.HOLD.value:
                meta_base_consistency = True  # HOLD总是与基础模型一致
            else:
                meta_base_consistency = meta_prediction == base_direction
            
            # 3. 综合判断
            is_consistent = (
                base_agreement >= self.consistency_threshold and
                meta_base_consistency and
                meta_confidence >= self.confidence_threshold
            )
            
            # 4. 风险评估
            if base_agreement >= 0.8 and meta_confidence >= 0.7:
                risk_level = "LOW"
            elif base_agreement >= 0.6 and meta_confidence >= 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            logger.debug(f"🔍 模型一致性检查: 基础模型一致性={base_agreement:.3f}, "
                        f"元学习器一致性={meta_base_consistency}, "
                        f"元学习器置信度={meta_confidence:.3f}, "
                        f"风险等级={risk_level}")
            
            return ConsistencyCheck(
                is_consistent=is_consistent,
                confidence_score=meta_confidence,
                direction_strength=base_agreement,
                timeframe_agreement=base_agreement,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"❌ 模型一致性检查失败: {e}")
            return ConsistencyCheck(
                is_consistent=False,
                confidence_score=0.0,
                direction_strength=0.0,
                timeframe_agreement=0.0,
                risk_level="HIGH"
            )
    
    def filter_fatal_error_signals(
        self,
        signal: int,
        consistency_check: ConsistencyCheck,
        previous_signal: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        过滤致命错误信号
        
        Args:
            signal: 当前信号
            consistency_check: 一致性检查结果
            previous_signal: 前一个信号
        
        Returns:
            Tuple[bool, str]: (是否通过过滤, 过滤原因)
        """
        try:
            # 1. 一致性检查
            if not consistency_check.is_consistent:
                return False, f"多时间框架不一致 (一致性={consistency_check.timeframe_agreement:.3f})"
            
            # 2. 置信度检查
            if consistency_check.confidence_score < self.confidence_threshold:
                return False, f"置信度过低 ({consistency_check.confidence_score:.3f} < {self.confidence_threshold})"
            
            # 3. 致命错误检查（LONG↔SHORT）
            if previous_signal is not None:
                if (previous_signal == SignalType.LONG.value and signal == SignalType.SHORT.value) or \
                   (previous_signal == SignalType.SHORT.value and signal == SignalType.LONG.value):
                    return False, f"致命错误: {SignalType(previous_signal).name}→{SignalType(signal).name}"
            
            # 4. 风险等级检查
            if consistency_check.risk_level == "HIGH":
                return False, f"风险等级过高: {consistency_check.risk_level}"
            
            return True, "通过所有检查"
            
        except Exception as e:
            logger.error(f"❌ 致命错误过滤失败: {e}")
            return False, f"过滤异常: {e}"
    
    def calculate_consistency_metrics(
        self,
        predictions_history: List[Dict[str, int]],
        probabilities_history: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        计算一致性指标
        
        Args:
            predictions_history: 历史预测列表
            probabilities_history: 历史概率列表
        
        Returns:
            Dict[str, float]: 一致性指标
        """
        try:
            if not predictions_history:
                return {
                    'consistency_rate': 0.0,
                    'avg_confidence': 0.0,
                    'direction_stability': 0.0,
                    'fatal_error_rate': 0.0
                }
            
            # 1. 一致性率
            consistent_count = 0
            total_confidence = 0.0
            direction_changes = 0
            
            for i, (preds, probs) in enumerate(zip(predictions_history, probabilities_history)):
                check = self.check_multi_timeframe_consistency(preds, probs)
                if check.is_consistent:
                    consistent_count += 1
                total_confidence += check.confidence_score
                
                # 计算方向变化
                if i > 0:
                    prev_preds = predictions_history[i-1]
                    current_preds = predictions_history[i]
                    
                    # 检查是否有致命错误
                    for tf in preds.keys():
                        if tf in prev_preds:
                            prev_signal = prev_preds[tf]
                            curr_signal = current_preds[tf]
                            if ((prev_signal == SignalType.LONG.value and curr_signal == SignalType.SHORT.value) or
                                (prev_signal == SignalType.SHORT.value and curr_signal == SignalType.LONG.value)):
                                direction_changes += 1
                                break
            
            consistency_rate = consistent_count / len(predictions_history)
            avg_confidence = total_confidence / len(predictions_history)
            fatal_error_rate = direction_changes / len(predictions_history)
            
            # 方向稳定性（1 - 致命错误率）
            direction_stability = 1.0 - fatal_error_rate
            
            logger.info(f"📊 一致性指标: 一致性率={consistency_rate:.3f}, "
                       f"平均置信度={avg_confidence:.3f}, "
                       f"方向稳定性={direction_stability:.3f}, "
                       f"致命错误率={fatal_error_rate:.3f}")
            
            return {
                'consistency_rate': consistency_rate,
                'avg_confidence': avg_confidence,
                'direction_stability': direction_stability,
                'fatal_error_rate': fatal_error_rate
            }
            
        except Exception as e:
            logger.error(f"❌ 一致性指标计算失败: {e}")
            return {
                'consistency_rate': 0.0,
                'avg_confidence': 0.0,
                'direction_stability': 0.0,
                'fatal_error_rate': 0.0
            }
