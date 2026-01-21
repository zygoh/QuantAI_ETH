"""
自定义目标函数和评估指标

用于LightGBM、XGBoost等模型的自定义损失函数
"""
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Softmax函数（数值稳定版本）
    
    Args:
        x: 输入数组
        axis: 计算轴
    
    Returns:
        Softmax概率
    """
    # 减去最大值以提高数值稳定性
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def lgb_fatal_error_objective(
    fatal_error_weight: float = 5.0,
    hold_weight: float = 15.0
):
    """
    LightGBM自定义目标函数：致命错误惩罚 + HOLD权重
    
    Args:
        fatal_error_weight: 致命错误权重（LONG↔SHORT）
        hold_weight: HOLD类别权重
    
    Returns:
        (objective_function, eval_function)
    """
    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算梯度和Hessian
        
        Args:
            y_true: 真实标签 (n_samples,)
            y_pred: 预测值 (n_samples * n_classes,)
        
        Returns:
            (grad, hess): 梯度和Hessian矩阵
        """
        n_samples = len(y_true)
        n_classes = 3
        
        # Reshape预测值
        y_pred = y_pred.reshape(n_samples, n_classes)
        
        # Softmax
        y_pred_prob = softmax(y_pred, axis=1)
        
        # One-hot编码真实标签
        y_true_onehot = np.zeros((n_samples, n_classes))
        y_true_onehot[np.arange(n_samples), y_true.astype(int)] = 1
        
        # 计算预测类别
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        
        # 计算权重
        weights = np.ones(n_samples, dtype=np.float64)
        
        # 致命错误权重：LONG(2) → SHORT(0) 或 SHORT(0) → LONG(2)
        fatal_mask = ((y_true == 2) & (y_pred_class == 0)) | \
                     ((y_true == 0) & (y_pred_class == 2))
        weights[fatal_mask] *= fatal_error_weight
        
        # HOLD类别权重
        hold_mask = (y_true == 1)
        weights[hold_mask] *= hold_weight
        
        # 计算梯度：grad = (pred_prob - true_onehot) * weight
        grad = (y_pred_prob - y_true_onehot) * weights[:, np.newaxis]
        
        # 计算Hessian（对角近似）：hess = pred_prob * (1 - pred_prob) * weight
        hess = y_pred_prob * (1 - y_pred_prob) * weights[:, np.newaxis]
        
        # 确保Hessian为正（数值稳定性）
        hess = np.maximum(hess, 1e-8)
        
        return grad.flatten(), hess.flatten()
    
    def eval_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """
        评估指标：致命错误率
        
        Args:
            y_true: 真实标签 (n_samples,)
            y_pred: 预测值 (n_samples * n_classes,)
        
        Returns:
            (metric_name, metric_value, is_higher_better)
        """
        n_samples = len(y_true)
        n_classes = 3
        y_pred = y_pred.reshape(n_samples, n_classes)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 计算致命错误率
        fatal_errors = np.sum(
            ((y_true == 2) & (y_pred_class == 0)) |
            ((y_true == 0) & (y_pred_class == 2))
        )
        fatal_error_rate = fatal_errors / n_samples
        
        return 'fatal_error_rate', fatal_error_rate, False
    
    return objective, eval_metric


def xgb_fatal_error_objective(
    fatal_error_weight: float = 5.0,
    hold_weight: float = 15.0
):
    """
    XGBoost自定义目标函数：致命错误惩罚 + HOLD权重
    
    注意：XGBoost的自定义目标函数格式与LightGBM类似
    
    Args:
        fatal_error_weight: 致命错误权重（LONG↔SHORT）
        hold_weight: HOLD类别权重
    
    Returns:
        objective_function
    """
    def objective(y_pred: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算梯度和Hessian
        
        Args:
            y_pred: 预测值 (n_samples * n_classes,)
            dtrain: DMatrix对象
        
        Returns:
            (grad, hess): 梯度和Hessian矩阵
        """
        y_true = dtrain.get_label()
        n_samples = len(y_true)
        n_classes = 3
        
        # Reshape预测值
        y_pred = y_pred.reshape(n_samples, n_classes)
        
        # Softmax
        y_pred_prob = softmax(y_pred, axis=1)
        
        # One-hot编码真实标签
        y_true_onehot = np.zeros((n_samples, n_classes))
        y_true_onehot[np.arange(n_samples), y_true.astype(int)] = 1
        
        # 计算预测类别
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        
        # 计算权重
        weights = np.ones(n_samples, dtype=np.float64)
        
        # 致命错误权重
        fatal_mask = ((y_true == 2) & (y_pred_class == 0)) | \
                     ((y_true == 0) & (y_pred_class == 2))
        weights[fatal_mask] *= fatal_error_weight
        
        # HOLD类别权重
        hold_mask = (y_true == 1)
        weights[hold_mask] *= hold_weight
        
        # 计算梯度和Hessian
        grad = (y_pred_prob - y_true_onehot) * weights[:, np.newaxis]
        hess = y_pred_prob * (1 - y_pred_prob) * weights[:, np.newaxis]
        hess = np.maximum(hess, 1e-8)
        
        return grad.flatten(), hess.flatten()
    
    return objective


def compute_fatal_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算致命错误率
    
    Args:
        y_true: 真实标签
        y_pred: 预测类别
    
    Returns:
        致命错误率
    """
    fatal_errors = np.sum(
        ((y_true == 2) & (y_pred == 0)) |
        ((y_true == 0) & (y_pred == 2))
    )
    return fatal_errors / len(y_true)


def compute_hold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算HOLD类别的性能指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测类别
    
    Returns:
        HOLD类别指标字典
    """
    # HOLD类别掩码
    hold_true_mask = (y_true == 1)
    hold_pred_mask = (y_pred == 1)
    
    # True Positives, False Positives, False Negatives
    tp = np.sum(hold_true_mask & hold_pred_mask)
    fp = np.sum(~hold_true_mask & hold_pred_mask)
    fn = np.sum(hold_true_mask & ~hold_pred_mask)
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'hold_precision': precision,
        'hold_recall': recall,
        'hold_f1': f1,
        'hold_support': int(np.sum(hold_true_mask))
    }
