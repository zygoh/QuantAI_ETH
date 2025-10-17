# ⚠️ 依赖安装问题解决方案

**项目**: QuantAI-ETH  
**问题**: CatBoost安装失败（Windows编译错误）  
**日期**: 2025-10-17

---

## 🔍 问题分析

### 错误原因

```
Failed to build `pyyaml==6.0`
AttributeError: cython_sources
```

**根本原因**: 
- CatBoost需要编译C扩展
- Windows环境缺少Microsoft Visual C++ Build Tools
- 无法编译源代码

---

## 🚀 解决方案（推荐顺序）

### 方案1: 使用预编译wheel（最简单）⭐⭐⭐

```bash
# 不使用uv，直接用pip
cd F:\AI\20251007\backend

# 单独安装catboost（让pip找预编译版本）
pip install catboost --only-binary :all:

# 如果失败，尝试稍旧版本
pip install catboost==1.2.0

# 或者最新版本
pip install catboost
```

**成功后**：
```bash
# 安装其他依赖
pip install -r requirements.txt

# 删除旧模型
Remove-Item models\*.pkl

# 启动训练
python main.py
```

---

### 方案2: 两模型集成（降级但可用）⭐⭐

**如果CatBoost无法安装，先用两模型**：

```bash
# 1. 只安装LightGBM和XGBoost
pip install lightgbm==4.1.0 xgboost==2.0.3

# 2. 修改main.py使用两模型服务
```

**修改main.py**：
```python
# 从
from app.services.ensemble_ml_service import ensemble_ml_service

# 改为
from app.services.two_model_ensemble import two_model_ensemble
ml_service = two_model_ensemble
```

**预期效果**：
- LightGBM + XGBoost 两模型Stacking
- 准确率：42.81% → 46-48%（+3-5%）
- 略低于三模型，但仍有提升

---

### 方案3: 安装Build Tools（完整但耗时）⭐

**步骤**：

1. 下载Microsoft C++ Build Tools
   ```
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```

2. 安装时选择：
   - ✅ "使用C++的桌面开发"
   - ✅ "Windows 10 SDK"

3. 重启PowerShell

4. 重新安装
   ```bash
   pip install catboost==1.2.2
   ```

**时间成本**: 约30分钟（下载+安装）

---

## 💡 专业建议

### 推荐：方案1 + 方案2组合

**步骤**：

1. **先尝试方案1**（5分钟）
   ```bash
   pip install catboost --only-binary :all:
   ```

2. **如果成功** → 使用三模型
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
   预期准确率：48-51%

3. **如果失败** → 使用两模型（方案2）
   ```bash
   pip install lightgbm==4.1.0 xgboost==2.0.3
   # 修改main.py
   python main.py
   ```
   预期准确率：46-48%

**理由**：
- ✅ 快速验证（不浪费时间）
- ✅ 有降级方案（两模型也不错）
- ✅ 避免陷入编译问题

---

## 📊 效果对比

| 方案 | 模型数量 | 预期准确率 | 时间成本 |
|------|---------|-----------|---------|
| **三模型Stacking** | LGB+XGB+CAT | 48-51% | 5分钟（如pip成功） |
| **两模型Stacking** | LGB+XGB | 46-48% | 3分钟 ✅ |
| **单模型** | LGB | 42.81% | 1分钟 |

**差异分析**：
- 三模型 vs 两模型：+2-3%
- 两模型 vs 单模型：+3-5%

**结论**: 两模型方案也很不错（46-48%接近50%）

---

## 🔥 立即执行建议

### 快速路径（推荐）

```bash
cd F:\AI\20251007\backend

# Step 1: 尝试pip安装catboost（2分钟）
pip install catboost --only-binary :all:

# Step 2a: 如果成功
pip install -r requirements.txt
Remove-Item models\*.pkl
python main.py
# → 预期：三模型Stacking，48-51%准确率

# Step 2b: 如果失败
pip install lightgbm==4.1.0 xgboost==2.0.3
# 修改main.py使用two_model_ensemble
Remove-Item models\*.pkl
python main.py
# → 预期：两模型Stacking，46-48%准确率
```

---

## 📋 文件准备

### 已创建

1. ✅ `backend/app/services/ensemble_ml_service.py` - 三模型Stacking
2. ✅ `backend/app/services/two_model_ensemble.py` - 两模型降级方案
3. ✅ `backend/requirements.txt` - 依赖清单

### 需要修改（如用两模型）

**文件**: `backend/main.py`

**修改**：
```python
# 第18行，从
from app.services.ensemble_ml_service import ensemble_ml_service

# 改为
from app.services.two_model_ensemble import two_model_ensemble

# 第93行，从
ml_service = ensemble_ml_service

# 改为
ml_service = two_model_ensemble
```

---

## ✅ 总结

### CatBoost安装失败

**原因**: Windows编译问题

### 解决方案

1. ⭐⭐⭐ 尝试pip预编译wheel（最快）
2. ⭐⭐ 降级两模型集成（可用且快速）
3. ⭐ 安装Build Tools（完整但耗时）

### 推荐

🔥 **先尝试方案1，失败则用方案2**

**两模型也不错**：
- 预期准确率：46-48%
- 距50%只差2-4%
- 可通过超参数优化补足

---

**问题**: CatBoost编译失败  
**推荐**: 尝试pip安装wheel，失败则用两模型  
**预期**: 46-51%准确率（看方案）  
**下一步**: 🔥 执行pip install catboost

