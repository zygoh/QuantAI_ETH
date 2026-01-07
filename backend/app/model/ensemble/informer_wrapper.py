"""
Informer-2模型包装器
"""
# StdLib
import logging

# Third-Party
import numpy as np
# 深度学习模型（PyTorch）- 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class InformerWrapper:
    """
    包装Informer-2模型，提供predict_proba接口（支持序列输入）
    
    将类移到模块级别以支持pickle序列化
    """
    
    def __init__(self, model, device):
        """
        初始化包装器
        
        Args:
            model: Informer2ForClassification模型实例
            device: PyTorch设备（'cuda'或'cpu'）
        """
        self.model = model
        self.device = device
    
    def predict_proba(self, X_seq):
        """
        预测概率（兼容scikit-learn，支持序列输入）
        
        Args:
            X_seq: NumPy数组 (n_samples, seq_len, n_features)
        
        Returns:
            概率数组 (n_samples, n_classes)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用Informer-2模型")
        
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X_seq, torch.Tensor):
                if not isinstance(X_seq, np.ndarray):
                    X_seq = np.asarray(X_seq, dtype=np.float32)
                elif X_seq.dtype != np.float32:
                    X_seq = X_seq.astype(np.float32)
                
                if not X_seq.flags['C_CONTIGUOUS']:
                    X_seq = np.ascontiguousarray(X_seq)
                    X_tensor = torch.from_numpy(X_seq).to(self.device)
                else:
                    X_tensor = torch.from_numpy(X_seq).to(self.device)
            else:
                X_tensor = X_seq.to(self.device)
            
            probs = self.model.predict_proba(X_tensor)
            return probs.cpu().numpy()
    
    def predict(self, X_seq):
        """
        预测类别（兼容scikit-learn，支持序列输入）
        
        Args:
            X_seq: NumPy数组 (n_samples, seq_len, n_features)
        
        Returns:
            预测类别数组
        """
        probs = self.predict_proba(X_seq)
        return np.argmax(probs, axis=1)

