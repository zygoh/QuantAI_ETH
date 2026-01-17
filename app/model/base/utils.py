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


def compute_effective_sample_weights(y: pd.Series, timeframe: str) -> np.ndarray:
    """
    使用有效样本数(Effective Number of Samples)计算样本权重
    
    Args:
        y: 标签Series或ndarray，取值{0: SHORT, 1: HOLD, 2: LONG}
        timeframe: 时间框架
    
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
        
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        class_weights = 1.0 / effective_num
        class_weights = class_weights / class_weights.sum() * len(classes)
        
        weight_map = {c: class_weights[i] for i, c in enumerate(classes)}
        sample_weights = np.array([weight_map[int(label)] for label in y_np], dtype=np.float64)
        
        return sample_weights
    except Exception:
        logger.error("有效样本数权重计算失败，降级到均等权重")
        return np.ones(len(y))


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

