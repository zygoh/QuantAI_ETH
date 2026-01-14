"""
集成模型工具函数模块
"""
# StdLib
import gc
import logging
import os
from typing import Tuple

# Third-Party
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
# 深度学习模型（PyTorch）- 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Local App
from app.core.config import settings

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """清理GPU内存"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_used = torch.cuda.memory_allocated(0)
        gpu_free = gpu_memory - gpu_used
        logger.info(f"GPU内存已清理 (使用: {gpu_used/1024**3:.1f}GB, 可用: {gpu_free/1024**3:.1f}GB)")
    else:
        logger.info("CPU模式，无需清理GPU内存")


def monitor_gpu_memory():
    """监控GPU内存使用情况"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_used = torch.cuda.memory_allocated(0)
        gpu_free = gpu_memory - gpu_used
        gpu_reserved = torch.cuda.memory_reserved(0)
        
        return {
            'total': gpu_memory,
            'used': gpu_used,
            'free': gpu_free,
            'reserved': gpu_reserved,
            'usage_percent': (gpu_used / gpu_memory) * 100
        }
    else:
        return {'error': 'GPU不可用'}


def prepare_features_labels_reuse(
    df: pd.DataFrame,
    timeframe: str,
    feature_columns_dict: dict
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备特征和标签（复用已选择的特征列）
    
    Args:
        df: 包含label列的DataFrame
        timeframe: 时间框架
        feature_columns_dict: 特征列字典 {timeframe: [columns]}
    
    Returns:
        (X, y): 特征DataFrame和标签Series
    """
    try:
        feature_columns = feature_columns_dict.get(timeframe, [])
        
        if not feature_columns:
            logger.error(f"{timeframe} 特征列未找到，无法复用")
            return pd.DataFrame(), pd.Series()
        
        invalid_cols = {'index', 'timestamp', 'date', 'label', 'target'}
        feature_columns = [f for f in feature_columns if f not in invalid_cols]
        
        X = df[feature_columns].copy()
        y = df['label'].copy()
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        return X, y
        
    except Exception as e:
        logger.error(f"准备特征和标签（复用）失败: {e}")
        return pd.DataFrame(), pd.Series()


def create_sequence_input(
    df: pd.DataFrame,
    seq_len: int,
    timeframe: str,
    feature_columns_dict: dict,
    model_dir: str = "models",
    use_sequence_memmap: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造序列输入（用于Informer-2模型）- 内存优化版
    
    Args:
        df: 特征工程后的DataFrame（包含label列）
        seq_len: 序列长度
        timeframe: 时间框架
        feature_columns_dict: 特征列字典
        model_dir: 模型目录
        use_sequence_memmap: 是否使用内存映射
    
    Returns:
        X_seq: (n_samples, seq_len, n_features) - 序列特征
        y: (n_samples,) - 标签
    """
    try:
        feature_columns = feature_columns_dict.get(timeframe, [])
        
        if not feature_columns:
            logger.error(f"{timeframe} 特征列未找到，无法构造序列输入")
            return np.array([]), np.array([])
        
        invalid_cols = {'index', 'timestamp', 'date', 'label', 'target'}
        feature_columns = [f for f in feature_columns if f not in invalid_cols]
        
        logger.debug(f"{timeframe} 开始构造序列输入（seq_len={seq_len}）...")
        X_all = df[feature_columns].values.astype(np.float32)
        y_all = df['label'].values.astype(np.int8)
        
        n_total = len(df)
        n_features = len(feature_columns)
        max_samples = n_total - seq_len
        
        if max_samples <= 0:
            logger.warning(f"{timeframe} 数据量不足，无法构造序列（需要>{seq_len}条）")
            return np.array([]), np.array([])
        
        X_seq = np.empty((max_samples, seq_len, n_features), dtype=np.float32)
        y = np.empty(max_samples, dtype=np.int8)
        
        valid_count = 0
        for i in range(seq_len, n_total):
            idx = i - seq_len
            X_window = X_all[idx:i]
            y_label = y_all[i]
            
            if not np.isnan(X_window).any() and not np.isnan(y_label):
                X_seq[valid_count] = X_window
                y[valid_count] = y_label
                valid_count += 1
        
        X_seq = X_seq[:valid_count]
        y = y[:valid_count]
        
        memory_mb = (X_seq.nbytes + y.nbytes) / (1024 ** 2)
        
        if use_sequence_memmap:
            try:
                os.makedirs(model_dir, exist_ok=True)
                safe_symbol = settings.SYMBOL.replace('/', '_')
                seq_path = os.path.join(model_dir, f"{safe_symbol}_{timeframe}_Xseq.npy")
                y_path = os.path.join(model_dir, f"{safe_symbol}_{timeframe}_Yseq.npy")
                
                mm_x = open_memmap(seq_path, mode='w+', dtype=np.float32, shape=X_seq.shape)
                mm_x[:] = X_seq
                del mm_x
                mm_y = open_memmap(y_path, mode='w+', dtype=np.int8, shape=y.shape)
                mm_y[:] = y
                del mm_y
                
                del X_seq, y
                gc.collect()
                
                X_seq = np.load(seq_path, mmap_mode='r')
                y = np.load(y_path, mmap_mode='r')
                
                logger.info(f"已启用内存映射: {seq_path} ({memory_mb:.1f} MB)")
            except Exception:
                logger.warning("序列内存映射失败，回退为内存数组")
        
        logger.info(f"{timeframe} 序列输入构造完成: {X_seq.shape} (样本数={valid_count}, 序列长度={seq_len}, 特征数={n_features})")
        logger.info(f"内存占用: {memory_mb:.1f} MB (float32优化)")
        
        del X_all, y_all
        gc.collect()
        
        return X_seq, y
        
    except Exception as e:
        logger.error(f"构造序列输入失败: {e}")
        return np.array([]), np.array([])

