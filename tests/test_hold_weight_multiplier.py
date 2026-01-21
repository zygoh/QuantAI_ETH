"""
属性测试：HOLD类别权重倍数

Feature: trading-system-prediction-fix
Property 4: HOLD类别权重倍数
验证需求：2.7, 4.3
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays


# 生成随机标签分布的策略
@st.composite
def label_distribution(draw):
    """生成随机标签分布"""
    n_samples = draw(st.integers(min_value=100, max_value=1000))
    
    # 生成标签（0=SHORT, 1=HOLD, 2=LONG）
    # 确保每个类别至少有一些样本
    labels = []
    
    # 至少10%的样本是HOLD
    n_hold = max(10, int(n_samples * 0.1))
    labels.extend([1] * n_hold)
    
    # 剩余样本随机分配给SHORT和LONG
    remaining = n_samples - n_hold
    n_short = draw(st.integers(min_value=1, max_value=remaining - 1))
    n_long = remaining - n_short
    
    labels.extend([0] * n_short)
    labels.extend([2] * n_long)
    
    # 打乱顺序
    np.random.shuffle(labels)
    
    return np.array(labels, dtype=np.int32)


class TestHoldWeightMultiplier:
    """测试HOLD类别权重倍数"""
    
    @given(labels=label_distribution())
    @settings(max_examples=100, deadline=None)
    def test_hold_weight_at_least_10x(self, labels):
        """
        属性4：HOLD类别权重至少是LONG/SHORT类别的10倍
        
        对于任意标签分布，计算类别权重后，HOLD类别（类别1）的权重
        应该至少是LONG类别（类别2）和SHORT类别（类别0）权重的10倍
        """
        from app.model.base.utils import compute_class_weights_dict
        
        # 计算类别权重
        class_weights = compute_class_weights_dict(
            labels,
            hold_multiplier=15.0,
            beta=0.999
        )
        
        # 获取各类别权重
        short_weight = class_weights[0]
        hold_weight = class_weights[1]
        long_weight = class_weights[2]
        
        # 验证HOLD权重至少是SHORT权重的10倍
        assert hold_weight >= short_weight * 10.0, \
            f"HOLD权重({hold_weight:.4f})应该至少是SHORT权重({short_weight:.4f})的10倍"
        
        # 验证HOLD权重至少是LONG权重的10倍
        assert hold_weight >= long_weight * 10.0, \
            f"HOLD权重({hold_weight:.4f})应该至少是LONG权重({long_weight:.4f})的10倍"
    
    @given(labels=label_distribution())
    @settings(max_examples=100, deadline=None)
    def test_sample_weights_reflect_hold_multiplier(self, labels):
        """
        属性4：样本权重正确反映HOLD倍数
        
        对于任意标签分布，HOLD样本的平均权重应该显著高于非HOLD样本
        """
        from app.model.base.utils import compute_effective_sample_weights
        import pandas as pd
        
        # 计算样本权重
        labels_series = pd.Series(labels)
        sample_weights = compute_effective_sample_weights(
            labels_series,
            timeframe='5m',
            hold_multiplier=15.0
        )
        
        # 计算HOLD和非HOLD样本的平均权重
        hold_mask = (labels == 1)
        non_hold_mask = (labels != 1)
        
        if np.any(hold_mask) and np.any(non_hold_mask):
            avg_hold_weight = np.mean(sample_weights[hold_mask])
            avg_non_hold_weight = np.mean(sample_weights[non_hold_mask])
            
            # HOLD样本的平均权重应该显著高于非HOLD样本
            # 考虑到时间衰减和其他因素，放宽到至少5倍
            if avg_non_hold_weight > 1e-6:
                ratio = avg_hold_weight / avg_non_hold_weight
                assert ratio >= 5.0, \
                    f"HOLD样本平均权重应该至少是非HOLD样本的5倍，实际比例: {ratio:.2f}"
    
    def test_hold_weight_with_specific_distribution(self):
        """
        属性4：特定分布测试
        
        使用具体的标签分布验证HOLD权重倍数
        """
        from app.model.base.utils import compute_class_weights_dict
        
        # 案例1：平衡分布
        labels_balanced = np.array([0]*100 + [1]*100 + [2]*100)
        weights_balanced = compute_class_weights_dict(
            labels_balanced,
            hold_multiplier=15.0
        )
        
        assert weights_balanced[1] >= weights_balanced[0] * 10.0, \
            "平衡分布：HOLD权重应该至少是SHORT权重的10倍"
        assert weights_balanced[1] >= weights_balanced[2] * 10.0, \
            "平衡分布：HOLD权重应该至少是LONG权重的10倍"
        
        # 案例2：HOLD稀少分布（更接近实际情况）
        labels_sparse = np.array([0]*450 + [1]*10 + [2]*450)
        weights_sparse = compute_class_weights_dict(
            labels_sparse,
            hold_multiplier=15.0
        )
        
        assert weights_sparse[1] >= weights_sparse[0] * 10.0, \
            "稀少分布：HOLD权重应该至少是SHORT权重的10倍"
        assert weights_sparse[1] >= weights_sparse[2] * 10.0, \
            "稀少分布：HOLD权重应该至少是LONG权重的10倍"
        
        # 案例3：极端不平衡（HOLD只有1%）
        labels_extreme = np.array([0]*495 + [1]*10 + [2]*495)
        weights_extreme = compute_class_weights_dict(
            labels_extreme,
            hold_multiplier=15.0
        )
        
        assert weights_extreme[1] >= weights_extreme[0] * 10.0, \
            "极端不平衡：HOLD权重应该至少是SHORT权重的10倍"
        assert weights_extreme[1] >= weights_extreme[2] * 10.0, \
            "极端不平衡：HOLD权重应该至少是LONG权重的10倍"
    
    def test_hold_multiplier_parameter_effect(self):
        """
        属性4：hold_multiplier参数效果测试
        
        验证不同的hold_multiplier值产生不同的权重比例
        """
        from app.model.base.utils import compute_class_weights_dict
        
        labels = np.array([0]*400 + [1]*20 + [2]*400)
        
        # 测试不同的倍数
        multipliers = [1.0, 5.0, 10.0, 15.0, 20.0]
        prev_hold_weight = 0.0
        
        for multiplier in multipliers:
            weights = compute_class_weights_dict(
                labels,
                hold_multiplier=multiplier
            )
            
            hold_weight = weights[1]
            
            # HOLD权重应该随着倍数增加而增加
            if prev_hold_weight > 0:
                assert hold_weight > prev_hold_weight, \
                    f"HOLD权重应该随着倍数增加而增加（{multiplier}倍）"
            
            prev_hold_weight = hold_weight


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
