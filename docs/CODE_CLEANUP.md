# 🧹 代码清理记录

**清理时间**: 2025-10-16  
**原则**: 遵循项目规则"禁止冗余代码"

---

## ❌ 已删除的冗余代码

### 1. `calculate_confidence`方法

**位置**: ~~`ml_service.py:1020-1026`~~ （已删除）

**原代码**:
```python
def calculate_confidence(self, prediction: float) -> float:
    """计算预测置信度"""
    try:
        return min(max(prediction, 0.0), 1.0)  # 简单裁剪
    except:
        return 0.0
```

**删除原因**：
1. ❌ **完全未被调用**（搜索整个项目无调用）
2. ❌ **功能冗余**（实际置信度来自`predict_proba`）
3. ❌ **实现过于简单**（仅做0-1裁剪）
4. ❌ **违反项目规则**（禁止冗余方法）

**实际置信度来源**：
```python
# ml_service.py - predict方法
probabilities = model.predict_proba(X)
confidence = np.max(probabilities, axis=1)[0]  # 最大概率值
# 直接使用，不需要额外计算
```

---

## 📊 代码清理效果

**删除前**:
- 方法数: 1个（未使用）
- 代码行数: 8行
- 维护成本: 存在

**删除后**:
- 冗余代码: -8行 ✅
- 代码清晰度: 提升 ✅
- 维护成本: 降低 ✅

---

## ✅ 其他检查

### 检查其他潜在冗余

**已检查区域**：
- ✅ `ml_service.py` - 无其他冗余方法
- ✅ `signal_generator.py` - 仓位计算已统一到`position_manager`
- ✅ `trading_engine.py` - 职责清晰

**状态**: ✅ 无其他明显冗余

---

## 📝 总结

**删除理由**：`calculate_confidence`方法
1. 未被任何地方调用
2. 功能可由现有代码替代
3. 违反"禁止冗余"原则

**清理效果**：
- 代码更简洁 ✅
- 维护成本降低 ✅
- 遵守项目规则 ✅

---

**清理完成**: 2025-10-16  
**清理项数**: 1个

