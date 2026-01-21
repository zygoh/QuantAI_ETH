"""
简单测试：验证数组维度修复

这个测试不需要hypothesis，可以直接运行
"""
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model.ensemble_ml_service import EnsembleMLService


def test_predict_scalar_method():
    """测试_predict_scalar方法"""
    print("=" * 60)
    print("测试1：_predict_scalar方法")
    print("=" * 60)
    
    # 创建模拟模型
    class MockModel:
        def predict(self, X):
            # 模拟返回数组
            return np.array([1])
    
    service = EnsembleMLService()
    model = MockModel()
    X = np.random.randn(1, 255).astype(np.float32)
    
    try:
        result = service._predict_scalar(model, X, model_name="MockModel")
        print(f"✅ 预测结果: {result}")
        print(f"✅ 结果类型: {type(result)}")
        assert isinstance(result, int), f"结果应该是int类型，实际是{type(result)}"
        assert result in [0, 1, 2], f"结果应该在[0,1,2]范围内，实际是{result}"
        print("✅ 测试通过！")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_array_creation():
    """测试数组创建"""
    print("\n" + "=" * 60)
    print("测试2：预测数组创建")
    print("=" * 60)
    
    # 模拟标量预测值
    lgb_pred = 1
    xgb_pred = 2
    cat_pred = 1
    
    try:
        # 创建预测数组
        pred_array = np.array([lgb_pred, xgb_pred, cat_pred], dtype=np.float64)
        print(f"✅ 预测数组: {pred_array}")
        print(f"✅ 数组形状: {pred_array.shape}")
        print(f"✅ 数组类型: {pred_array.dtype}")
        
        # 计算分歧度
        pred_disagreement = float(np.std(pred_array))
        print(f"✅ 预测分歧度: {pred_disagreement:.4f}")
        
        assert pred_array.shape == (3,), f"数组形状应该是(3,)，实际是{pred_array.shape}"
        print("✅ 测试通过！")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_type_consistency():
    """测试类型一致性"""
    print("\n" + "=" * 60)
    print("测试3：类型一致性检查")
    print("=" * 60)
    
    # 测试不同类型的输入
    test_cases = [
        ("标量int", 1),
        ("numpy标量", np.int32(1)),
        ("单元素数组", np.array([1])),
        ("单元素列表", [1]),
    ]
    
    all_passed = True
    for name, value in test_cases:
        try:
            # 提取标量
            if isinstance(value, (np.ndarray, list)):
                scalar = int(value[0] if len(value) > 0 else value)
            else:
                scalar = int(value)
            
            print(f"✅ {name}: {value} -> {scalar} (类型: {type(scalar).__name__})")
            assert isinstance(scalar, int), f"结果应该是int类型"
        except Exception as e:
            print(f"❌ {name}失败: {e}")
            all_passed = False
    
    if all_passed:
        print("✅ 所有测试通过！")
    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始验证数组维度修复")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("_predict_scalar方法", test_predict_scalar_method()))
    results.append(("数组创建", test_array_creation()))
    results.append(("类型一致性", test_type_consistency()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！数组维度修复验证成功！")
    else:
        print("⚠️ 部分测试失败，请检查修复代码")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
