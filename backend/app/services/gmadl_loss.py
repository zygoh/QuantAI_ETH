"""
GMADL损失函数 - Generalized Mean Absolute Deviation Loss
论文: "GMADL: Generalized Mean Absolute Deviation Loss for Time Series Forecasting" (2024)

核心思想:
- 对异常值更鲁棒（比RMSE）
- 对分位数更灵活（比Quantile Loss）
- 自适应权重机制

实证效果（30min比特币数据）:
- RMSE Informer: 年化94.09%, 夏普2.26
- Quantile Informer: 年化98.74%, 夏普2.68
- GMADL Informer: 年化183.71%, 夏普4.42 ⭐⭐⭐
"""

import torch
import torch.nn as nn
import numpy as np


class GMADLoss(nn.Module):
    """
    GMADL损失函数（用于分类任务）
    
    原始GMADL是回归损失，这里改编为分类损失
    核心思想：对难分样本给予更高权重
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        初始化GMADL损失
        
        Args:
            alpha: 控制对异常值的敏感度（越大越鲁棒）
            beta: 控制损失的凸性（0.5-1.0，越小越关注难样本）
            reduction: 'mean' 或 'sum'
        """
        super(GMADLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算GMADL损失
        
        Args:
            logits: 模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
        
        Returns:
            损失值
        """
        # 1. 计算softmax概率
        probs = torch.softmax(logits, dim=1)
        
        # 2. 获取正确类别的概率
        batch_size = targets.size(0)
        target_probs = probs[torch.arange(batch_size), targets]
        
        # 3. 计算预测误差（1 - 正确概率）
        errors = 1.0 - target_probs
        
        # 4. GMADL核心公式：loss = (|error|^beta) / (alpha + |error|^(1-beta))
        # 这个公式对难分样本（error大）给予更高权重
        abs_errors = torch.abs(errors)
        numerator = torch.pow(abs_errors, self.beta)
        denominator = self.alpha + torch.pow(abs_errors, 1.0 - self.beta)
        
        loss = numerator / (denominator + 1e-8)
        
        # 5. 聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GMADLossWithClassWeight(nn.Module):
    """
    GMADL损失 + 类别权重（用于处理不平衡数据）
    """
    
    def __init__(
        self,
        class_weights: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        初始化带类别权重的GMADL损失
        
        Args:
            class_weights: 类别权重 (num_classes,)
            alpha: GMADL参数
            beta: GMADL参数
            reduction: 'mean' 或 'sum'
        """
        super(GMADLossWithClassWeight, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算带权重的GMADL损失
        
        Args:
            logits: 模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
        
        Returns:
            损失值
        """
        # 1. 计算softmax概率
        probs = torch.softmax(logits, dim=1)
        
        # 2. 获取正确类别的概率
        batch_size = targets.size(0)
        target_probs = probs[torch.arange(batch_size), targets]
        
        # 3. 计算预测误差
        errors = 1.0 - target_probs
        
        # 4. GMADL损失
        abs_errors = torch.abs(errors)
        numerator = torch.pow(abs_errors, self.beta)
        denominator = self.alpha + torch.pow(abs_errors, 1.0 - self.beta)
        
        loss = numerator / (denominator + 1e-8)
        
        # 5. 应用类别权重
        weights = self.class_weights[targets]
        weighted_loss = loss * weights
        
        # 6. 聚合损失
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class GMADLossWithHOLDPenalty(nn.Module):
    """
    GMADL损失 + HOLD惩罚（专为交易系统设计）
    
    核心思想：
    - GMADL对难分样本给予更高关注
    - HOLD惩罚减少过度预测HOLD
    - 组合使用，既提升准确率又增加信号频率
    """
    
    def __init__(
        self,
        hold_penalty: float = 0.65,
        alpha: float = 1.0,
        beta: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        初始化GMADL + HOLD惩罚损失
        
        Args:
            hold_penalty: HOLD类别权重（0-1，越小惩罚越重）
            alpha: GMADL参数
            beta: GMADL参数
            reduction: 'mean' 或 'sum'
        """
        super(GMADLossWithHOLDPenalty, self).__init__()
        self.hold_penalty = hold_penalty
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算GMADL + HOLD惩罚损失
        
        Args:
            logits: 模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,) 其中1=HOLD
        
        Returns:
            损失值
        """
        # 1. 计算softmax概率
        probs = torch.softmax(logits, dim=1)
        
        # 2. 获取正确类别的概率
        batch_size = targets.size(0)
        target_probs = probs[torch.arange(batch_size), targets]
        
        # 3. 计算预测误差
        errors = 1.0 - target_probs
        
        # 4. GMADL损失
        abs_errors = torch.abs(errors)
        numerator = torch.pow(abs_errors, self.beta)
        denominator = self.alpha + torch.pow(abs_errors, 1.0 - self.beta)
        
        loss = numerator / (denominator + 1e-8)
        
        # 5. 应用HOLD惩罚（targets==1为HOLD类别）
        hold_weights = torch.where(
            targets == 1,
            torch.tensor(self.hold_penalty, device=targets.device),
            torch.tensor(1.0, device=targets.device)
        )
        
        weighted_loss = loss * hold_weights
        
        # 6. 聚合损失
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


def create_gmadl_loss(
    num_classes: int = 3,
    use_class_weights: bool = True,
    use_hold_penalty: bool = True,
    hold_penalty: float = 0.65,
    alpha: float = 1.0,
    beta: float = 0.5
) -> nn.Module:
    """
    工厂函数：创建合适的GMADL损失
    
    Args:
        num_classes: 类别数量
        use_class_weights: 是否使用类别权重
        use_hold_penalty: 是否使用HOLD惩罚
        hold_penalty: HOLD惩罚系数
        alpha: GMADL鲁棒性参数
        beta: GMADL凸性参数
    
    Returns:
        GMADL损失函数实例
    """
    if use_hold_penalty:
        # 推荐：GMADL + HOLD惩罚
        return GMADLossWithHOLDPenalty(
            hold_penalty=hold_penalty,
            alpha=alpha,
            beta=beta
        )
    elif use_class_weights:
        # 备选：GMADL + 类别权重
        # 注意：需要在训练时传入实际的class_weights
        class_weights = torch.ones(num_classes)
        return GMADLossWithClassWeight(
            class_weights=class_weights,
            alpha=alpha,
            beta=beta
        )
    else:
        # 基础版：纯GMADL
        return GMADLoss(alpha=alpha, beta=beta)


# 损失函数配置（用于实验对比）
LOSS_CONFIGS = {
    'gmadl_default': {
        'alpha': 1.0,
        'beta': 0.5,
        'description': 'GMADL默认参数（论文推荐）'
    },
    'gmadl_robust': {
        'alpha': 2.0,
        'beta': 0.6,
        'description': 'GMADL鲁棒版（更关注异常值）'
    },
    'gmadl_sensitive': {
        'alpha': 0.5,
        'beta': 0.4,
        'description': 'GMADL敏感版（更关注难分样本）'
    },
    'gmadl_trading': {
        'alpha': 1.0,
        'beta': 0.5,
        'hold_penalty': 0.65,
        'description': 'GMADL交易版（+HOLD惩罚）'
    }
}

