"""
SMOTE过采样模块

用于处理类别不平衡问题，特别是提升HOLD类别的样本数量
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

# 可选依赖：imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("⚠️ imbalanced-learn未安装，SMOTE过采样将不可用。安装命令: pip install imbalanced-learn")


def apply_smote_sampling(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.3,
    method: str = 'smote',
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用SMOTE或ADASYN过采样
    
    Args:
        X: 特征数据 (n_samples, n_features)
        y: 标签数据 (n_samples,)
        target_ratio: HOLD类别目标比例（相对于多数类，默认0.3即30%）
        method: 'smote' 或 'adasyn'
        k_neighbors: SMOTE的k近邻数量（默认5）
        random_state: 随机种子
    
    Returns:
        (X_resampled, y_resampled): 重采样后的数据
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("⚠️ imbalanced-learn未安装，跳过SMOTE过采样")
        return X, y
    
    try:
        # 统计当前类别分布
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"📊 原始类别分布: {class_dist}")
        
        # 计算采样策略
        # 假设0=SHORT, 1=HOLD, 2=LONG
        max_count = counts.max()
        target_hold_count = int(max_count * target_ratio)
        
        # 确保HOLD类别至少有k_neighbors+1个样本（SMOTE要求）
        current_hold_count = class_dist.get(1, 0)
        if current_hold_count < k_neighbors + 1:
            logger.warning(f"⚠️ HOLD样本数({current_hold_count})少于k_neighbors+1({k_neighbors+1})，"
                          f"无法应用SMOTE，跳过过采样")
            return X, y
        
        # 只对HOLD类别进行过采样
        sampling_strategy = {
            1: max(target_hold_count, current_hold_count)  # HOLD类别
        }
        
        logger.info(f"🎯 SMOTE采样策略: HOLD类别目标数量={sampling_strategy[1]} "
                   f"(当前{current_hold_count} → 目标{target_hold_count})")
        
        # 应用SMOTE或ADASYN
        if method == 'smote':
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=min(k_neighbors, current_hold_count - 1)  # 确保不超过样本数
            )
        else:  # adasyn
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                n_neighbors=min(k_neighbors, current_hold_count - 1)
            )
        
        # 执行重采样
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # 统计重采样后的类别分布
        unique_new, counts_new = np.unique(y_resampled, return_counts=True)
        class_dist_new = dict(zip(unique_new, counts_new))
        logger.info(f"✅ 重采样后类别分布: {class_dist_new}")
        
        # 计算HOLD类别增加的样本数
        hold_increase = class_dist_new.get(1, 0) - class_dist.get(1, 0)
        logger.info(f"📈 HOLD类别增加: {hold_increase}个样本 "
                   f"({class_dist.get(1, 0)} → {class_dist_new.get(1, 0)})")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"❌ SMOTE过采样失败: {e}")
        logger.error(f"   原因可能是：HOLD样本过少、特征维度过高或数据质量问题")
        logger.error(f"   回退到原始数据")
        return X, y


def apply_smote_to_dataframe(
    df: pd.DataFrame,
    feature_columns: list,
    label_column: str = 'label',
    target_ratio: float = 0.3,
    method: str = 'smote'
) -> pd.DataFrame:
    """
    对DataFrame应用SMOTE过采样
    
    Args:
        df: 包含特征和标签的DataFrame
        feature_columns: 特征列名列表
        label_column: 标签列名（默认'label'）
        target_ratio: HOLD类别目标比例
        method: 'smote' 或 'adasyn'
    
    Returns:
        重采样后的DataFrame
    """
    try:
        # 提取特征和标签
        X = df[feature_columns].values
        y = df[label_column].values
        
        # 应用SMOTE
        X_resampled, y_resampled = apply_smote_sampling(
            X, y,
            target_ratio=target_ratio,
            method=method
        )
        
        # 如果没有变化，直接返回原DataFrame
        if len(X_resampled) == len(X):
            return df
        
        # 创建新的DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        df_resampled[label_column] = y_resampled
        
        # 复制其他列（如果有）
        other_columns = [col for col in df.columns if col not in feature_columns and col != label_column]
        if other_columns:
            # 对于新增的样本，使用默认值或插值
            for col in other_columns:
                if col in df.columns:
                    # 使用原始数据的最后一个值填充新增样本
                    default_value = df[col].iloc[-1] if len(df) > 0 else None
                    df_resampled[col] = default_value
        
        return df_resampled
        
    except Exception as e:
        logger.error(f"❌ DataFrame SMOTE过采样失败: {e}")
        return df
