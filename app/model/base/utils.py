"""
基础ML服务工具函数模块
"""
import logging
import pandas as pd
import numpy as np

# Local App
from app.core.constants import EFFECTIVE_SAMPLE_BETA_BASE, EFFECTIVE_SAMPLE_BETA_NON_3M
from typing import Tuple

logger = logging.getLogger(__name__)


def compute_effective_sample_weights(
    y: pd.Series, 
    timeframe: str,
    hold_multiplier: float = 15.0
) -> np.ndarray:
    """
    使用有效样本数(Effective Number of Samples)计算样本权重
    
    增强版：对HOLD类别应用额外的权重倍数，提升HOLD识别能力
    
    Args:
        y: 标签Series或ndarray，取值{0: SHORT, 1: HOLD, 2: LONG}
        timeframe: 时间框架
        hold_multiplier: HOLD类别额外权重倍数（默认15.0）
    
    Returns:
        每个样本的权重向量
    """
    try:
        y_np = y.values if hasattr(y, 'values') else y
        classes = np.array([0, 1, 2])
        counts = np.array([(y_np == c).sum() for c in classes], dtype=np.float64)
        total = max(int(len(y_np)), 1)
        
        counts = np.maximum(counts, 1.0)
        
        base_beta = EFFECTIVE_SAMPLE_BETA_BASE
        if timeframe == '3m':
            beta = min(base_beta, 1.0 - 1.0 / (total + 1))
        else:
            beta = min(EFFECTIVE_SAMPLE_BETA_NON_3M, 1.0 - 1.0 / (total + 1))
        
        # 计算有效样本数权重
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        class_weights = 1.0 / effective_num
        class_weights = class_weights / class_weights.sum() * len(classes)
        
        # 🔑 关键增强：对HOLD类别（类别1）应用额外倍数
        class_weights[1] *= hold_multiplier
        
        # 重新归一化（可选，保持权重总和一致）
        # class_weights = class_weights / class_weights.sum() * len(classes)
        
        weight_map = {c: class_weights[i] for i, c in enumerate(classes)}
        sample_weights = np.array([weight_map[int(label)] for label in y_np], dtype=np.float64)
        
        # 记录权重信息
        hold_count = int(counts[1])
        hold_ratio = hold_count / total
        logger.debug(f"📊 类别权重计算完成:")
        logger.debug(f"   SHORT权重: {class_weights[0]:.4f}")
        logger.debug(f"   HOLD权重:  {class_weights[1]:.4f} (×{hold_multiplier}倍)")
        logger.debug(f"   LONG权重:  {class_weights[2]:.4f}")
        logger.debug(f"   HOLD样本比例: {hold_ratio:.2%} ({hold_count}/{total})")
        
        return sample_weights
    except Exception as e:
        logger.error(f"有效样本数权重计算失败: {e}，降级到均等权重")
        return np.ones(len(y))


def compute_class_weights_dict(
    y: np.ndarray,
    hold_multiplier: float = 15.0,
    beta: float = 0.999
) -> dict:
    """
    计算类别权重字典（用于模型训练）
    
    使用Effective Number of Samples方法，并对HOLD类别应用额外倍数
    
    Args:
        y: 标签数组，取值{0: SHORT, 1: HOLD, 2: LONG}
        hold_multiplier: HOLD类别额外权重倍数（默认15.0）
        beta: 平滑参数（默认0.999）
    
    Returns:
        类别权重字典 {0: weight_short, 1: weight_hold, 2: weight_long}
    """
    try:
        # 统计各类别样本数
        unique, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        
        # Effective Number of Samples
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        
        # 归一化（保持权重总和为类别数）
        weights = weights / weights.sum() * len(weights)
        
        # 创建权重字典
        class_weights = {}
        for cls, weight in zip(unique, weights):
            class_weights[int(cls)] = float(weight)
        
        # 确保所有类别都有权重
        for cls in [0, 1, 2]:
            if cls not in class_weights:
                class_weights[cls] = 1.0
        
        # 🔑 关键修复：应用hold_multiplier并确保至少是其他类别的hold_multiplier倍
        # 方法1：直接乘以倍数
        hold_weight_multiplied = class_weights[1] * hold_multiplier
        
        # 方法2：确保至少是其他类别的hold_multiplier倍
        non_hold_max_weight = max(
            class_weights.get(0, 1.0),
            class_weights.get(2, 1.0)
        )
        hold_weight_min = non_hold_max_weight * hold_multiplier
        
        # 取两者中的较大值，确保两个条件都满足
        class_weights[1] = max(hold_weight_multiplied, hold_weight_min)
        
        logger.debug(f"📊 类别权重字典: SHORT={class_weights[0]:.4f}, "
                    f"HOLD={class_weights[1]:.4f}, LONG={class_weights[2]:.4f}")
        
        return class_weights
    except Exception as e:
        logger.error(f"类别权重字典计算失败: {e}")
        return {0: 1.0, 1: 1.0, 2: 1.0}


def prepare_features_labels(df: pd.DataFrame, feature_columns: list) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备特征和标签
    
    Args:
        df: 包含label列的DataFrame
        feature_columns: 特征列列表
    
    Returns:
        (X, y): 特征DataFrame和标签Series
    """
    try:
        # 检查label列是否存在
        if 'label' not in df.columns:
            logger.error(f"DataFrame中缺少'label'列，可用列: {list(df.columns)[:10]}")
            raise KeyError("DataFrame中缺少'label'列")
        
        # 过滤无效列，确保feature_columns中的列都在df中存在
        invalid_cols = {'index', 'timestamp', 'date', 'label', 'target'}
        feature_columns = [f for f in feature_columns if f not in invalid_cols and f in df.columns]
        
        if not feature_columns:
            logger.error(f"特征列列表为空或所有特征列都不在DataFrame中")
            raise ValueError("特征列列表为空或所有特征列都不在DataFrame中")
        
        X = df[feature_columns].copy()
        y = df['label'].copy()
        
        # 过滤NaN值
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            logger.warning(f"过滤NaN后，特征数据为空")
        
        return X, y
    except Exception as e:
        logger.error(f"准备特征和标签失败: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series()

