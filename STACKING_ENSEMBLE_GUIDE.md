# 🚀 Stacking模型集成实施指南

**项目**: QuantAI-ETH  
**优化**: Phase 2.3 - 三模型Stacking融合  
**当前准确率**: 42.81%  
**目标准确率**: ≥50%  
**预期提升**: +5-8%  
**实施时间**: 2025-10-17

---

## 🎯 Stacking集成方案

### 为什么用Stacking？

**Stacking优势**：
- ✅ 使用元学习器（meta-learner）学习如何组合
- ✅ 比简单加权更智能
- ✅ 可以学习每个模型的强项
- ✅ 更好的泛化能力

**vs简单加权**：
```python
# 简单加权（固定权重）
ensemble = lgb * 0.4 + xgb * 0.3 + cat * 0.3  ❌ 不够智能

# Stacking（学习权重）
meta_learner.fit(base_predictions, y)  ✅ 自动学习最优组合
```

---

## 📋 实施步骤

### Step 1: 安装依赖（立即）

```bash
cd F:\AI\20251007\backend
pip install xgboost==2.0.3
pip install catboost==1.2.2
```

**验证安装**：
```bash
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
python -c "import catboost; print(f'CatBoost: {catboost.__version__}')"
```

---

### Step 2: 删除旧模型（立即）

```powershell
Remove-Item backend\models\*.pkl
```

**原因**: 
- 旧模型是单一LightGBM
- 需要用Stacking集成重新训练

---

### Step 3: 启动Stacking训练（立即）

```bash
cd backend
python main.py
```

**预计时间**: 1-2分钟（训练3×3=9个模型）

---

## 🏗️ Stacking架构

### 三层结构

```
第0层: 原始特征（186个）
    ↓
第1层: 基础模型（3个）
    ├─ LightGBM → 预测概率 [p1_SHORT, p1_HOLD, p1_LONG]
    ├─ XGBoost  → 预测概率 [p2_SHORT, p2_HOLD, p2_LONG]
    └─ CatBoost → 预测概率 [p3_SHORT, p3_HOLD, p3_LONG]
    ↓
第2层: 元特征（9维）
    [p1_SHORT, p1_HOLD, p1_LONG, p2_SHORT, p2_HOLD, p2_LONG, p3_SHORT, p3_HOLD, p3_LONG]
    ↓
第3层: 元学习器（LogisticRegression）
    学习如何组合基础模型
    ↓
最终预测: 信号类型 + 置信度
```

---

## 🤖 三个基础模型

### 1. LightGBM（主力）

**参数**：
```python
{
    'num_leaves': 95,        # 15m（已优化）
    'learning_rate': 0.03,
    'n_estimators': 300,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5
}
```

**特点**: 快速、准确、梯度提升

---

### 2. XGBoost（辅助）

**参数**：
```python
{
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'subsample': 0.8
}
```

**特点**: 强大的正则化、稳定性好

---

### 3. CatBoost（辅助）

**参数**：
```python
{
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8
}
```

**特点**: 处理类别特征、抗过拟合

---

## 🎯 元学习器（Stacking核心）

### LogisticRegression

**为什么选择LR**：
- ✅ 简单高效
- ✅ 不易过拟合
- ✅ 可解释性好
- ✅ 适合组合概率

**参数**：
```python
{
    'multi_class': 'multinomial',
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': 42
}
```

**作用**：
- 学习每个模型的权重
- 学习模型之间的互补性
- 自动找到最优组合

---

## 📊 训练流程

### 1. 准备数据（与单模型相同）

```
数据获取 → 特征工程（186特征） → 标签创建 → 特征选择（141/38/39）
→ 特征缩放 → 数据分割（80%训练，20%验证）
```

### 2. 训练基础模型（新增）

```python
# 训练3个模型（使用相同的训练数据）
lgb_model = train_lightgbm(X_train, y_train)
xgb_model = train_xgboost(X_train, y_train)
cat_model = train_catboost(X_train, y_train)
```

### 3. 生成元特征（Stacking核心）

```python
# 训练集上生成元特征
lgb_proba = lgb_model.predict_proba(X_train)  # (N, 3)
xgb_proba = xgb_model.predict_proba(X_train)  # (N, 3)
cat_proba = cat_model.predict_proba(X_train)  # (N, 3)

# 合并为9维元特征
meta_features = np.hstack([lgb_proba, xgb_proba, cat_proba])  # (N, 9)
```

### 4. 训练元学习器

```python
# 元学习器学习如何组合基础模型
meta_learner = LogisticRegression(multi_class='multinomial')
meta_learner.fit(meta_features, y_train)
```

### 5. 验证集评估

```python
# 验证集上测试Stacking效果
meta_features_val = np.hstack([
    lgb_model.predict_proba(X_val),
    xgb_model.predict_proba(X_val),
    cat_model.predict_proba(X_val)
])

stacking_pred = meta_learner.predict(meta_features_val)
accuracy = accuracy_score(y_val, stacking_pred)
```

---

## 📈 预期效果

### 准确率提升预测

| 模型 | 预期准确率 | 说明 |
|------|-----------|------|
| LightGBM单独 | 45.79% | 当前最好 |
| XGBoost单独 | 43-45% | 预估 |
| CatBoost单独 | 43-45% | 预估 |
| **Stacking集成** | **48-51%** | **目标** ✅ |

**提升幅度**: +5-12%

**关键**：
- 每个模型有不同的强项
- Stacking学习如何互补
- 综合效果优于单模型

---

### 置信度分布改善

**当前问题**：
- 单模型置信度集中在0.34-0.45
- 10小时内预测完全相同

**Stacking改善**：
- 置信度分布更分散
- 预测更加动态
- 对市场变化更敏感

---

## 🔧 期待的训练日志

### Stage 1: 基础模型训练

```log
🎯 Stage 1: 训练3个基础模型...
  📊 训练LightGBM...
  ✅ 样本加权已启用
  
  📊 训练XGBoost...
  ✅ 样本加权已启用
  
  📊 训练CatBoost...
  ✅ 样本加权已启用

✅ 3个基础模型训练完成
```

### Stage 2-3: 元学习器训练

```log
🎯 Stage 2: 生成元特征...
🎯 Stage 3: 训练元学习器（Stacking）...
✅ 元学习器训练完成
```

### Stage 4: 集成评估

```log
🎯 Stage 4: 验证集评估...
📊 15m Stacking集成评估:
  基础模型准确率:
    LightGBM: 0.4579
    XGBoost:  0.4420
    CatBoost: 0.4395
  Stacking准确率: 0.4850  ← 关键！
  提升: +2.71%
```

**期待平均准确率**: 0.48-0.51

---

## ⏰ 时间估算

### 训练时间

| 时间框架 | LightGBM | XGBoost | CatBoost | 元学习器 | 总计 |
|---------|---------|---------|---------|---------|------|
| **15m** | 5秒 | 8秒 | 10秒 | 1秒 | 24秒 |
| **2h** | 2秒 | 3秒 | 4秒 | 1秒 | 10秒 |
| **4h** | 1秒 | 2秒 | 3秒 | 1秒 | 7秒 |
| **总计** | 8秒 | 13秒 | 17秒 | 3秒 | **41秒** |

**预计总训练时间**: 1分钟

---

## 🔍 预测流程（实时）

### Stacking预测

```python
# 1. 获取最新数据
data = get_latest_klines()

# 2. 特征工程
X = create_features(data)

# 3. 三模型预测
lgb_proba = lgb_model.predict_proba(X)  # [0.35, 0.30, 0.35]
xgb_proba = xgb_model.predict_proba(X)  # [0.33, 0.32, 0.35]
cat_proba = cat_model.predict_proba(X)  # [0.34, 0.31, 0.35]

# 4. 合并元特征（9维）
meta_features = [0.35, 0.30, 0.35, 0.33, 0.32, 0.35, 0.34, 0.31, 0.35]

# 5. 元学习器预测
stacking_proba = meta_learner.predict_proba([meta_features])
# 输出: [0.30, 0.25, 0.45]  ← LONG置信度最高

# 6. 最终结果
signal = 'LONG'
confidence = 0.45
```

**特点**：
- 综合3个模型的意见
- 置信度更可靠
- 减少单模型偏差

---

## 📦 文件清单

### 新增文件

1. ✅ `backend/app/services/ensemble_ml_service.py`（545行）
   - Stacking集成实现
   - 三模型训练
   - 元学习器

2. ✅ `backend/requirements.txt`
   - 所有Python依赖
   - xgboost==2.0.3
   - catboost==1.2.2

### 修改文件

3. ✅ `backend/main.py`
   - 使用ensemble_ml_service
   - 导入修改

### 将产生的模型文件

4. `backend/models/ETHUSDT_15m_lgb_model.pkl`（LightGBM）
5. `backend/models/ETHUSDT_15m_xgb_model.pkl`（XGBoost）
6. `backend/models/ETHUSDT_15m_cat_model.pkl`（CatBoost）
7. `backend/models/ETHUSDT_15m_meta_model.pkl`（元学习器）
8. ... 2h和4h各4个模型

**总计**: 12个模型文件（3个时间框架 × 4个模型）

---

## 🚀 立即执行

### 命令序列

```bash
# 1. 安装新依赖
cd F:\AI\20251007\backend
pip install xgboost==2.0.3
pip install catboost==1.2.2

# 2. 删除旧模型
Remove-Item models\*.pkl

# 3. 启动Stacking训练
python main.py
```

**预计总时间**: 5分钟（安装2分钟 + 训练1分钟 + 启动2分钟）

---

## 📊 期待的结果

### 训练日志

```log
🚀 开始Stacking集成模型训练...
三模型融合: LightGBM + XGBoost + CatBoost

📊 训练 15m 集成模型...
🎯 Stage 1: 训练3个基础模型...
  📊 训练LightGBM...
  📊 训练XGBoost...
  📊 训练CatBoost...
✅ 3个基础模型训练完成

🎯 Stage 2: 生成元特征...
🎯 Stage 3: 训练元学习器（Stacking）...
✅ 元学习器训练完成

🎯 Stage 4: 验证集评估...
📊 15m Stacking集成评估:
  基础模型准确率:
    LightGBM: 0.4579
    XGBoost:  0.4420
    CatBoost: 0.4395
  Stacking准确率: 0.4850
  提升: +2.71%
  
平均准确率: 0.48-0.51  ← 关键！
```

---

### 准确率目标

| 时间框架 | LightGBM单独 | Stacking集成 | 提升 |
|---------|------------|-------------|------|
| **15m** | 45.79% | **48-50%** | +5-9% |
| **2h** | 42.60% | **46-48%** | +8-13% |
| **4h** | 40.05% | **44-46%** | +10-15% |
| **平均** | 42.81% | **46-48%** | **+7-12%** |

**如果达到48%+**：
- 距50%只差2%
- 可能需要Phase 2.4（超参数优化）
- 或者已经达标 ✅

---

## 🎯 成功标准

### 基本要求

- ✅ Stacking训练成功（无错误）
- ✅ 12个模型全部保存
- ✅ 平均准确率≥46%
- ✅ 比单模型提升≥3%

### 理想目标

- ⭐ 平均准确率≥48%
- ⭐ 15m准确率≥48%
- ⭐ 比单模型提升≥5%
- ⭐ **达到50%目标** ✅

---

## ⚠️ 可能的问题

### 问题1: 依赖安装失败

**症状**：
```
ERROR: No matching distribution found for xgboost
```

**解决**：
```bash
# 升级pip
python -m pip install --upgrade pip

# 重试安装
pip install xgboost catboost
```

---

### 问题2: GPU支持

**XGBoost GPU**：
```python
# 如果有GPU，使用GPU加速
'tree_method': 'gpu_hist'  # 替代 'hist'
```

**CatBoost GPU**：
```python
# 如果有GPU
'task_type': 'GPU'
```

**当前配置**: CPU模式（通用性更好）

---

### 问题3: 内存不足

**症状**: 训练时内存占用过高

**解决**：
```python
# 减少n_estimators
'n_estimators': 200  # 从300减少
```

---

## 📋 验证清单

### 安装验证

- [ ] pip install xgboost成功
- [ ] pip install catboost成功
- [ ] import测试通过

### 训练验证

- [ ] 3×3=9个基础模型训练成功
- [ ] 3个元学习器训练成功
- [ ] 12个模型文件保存成功
- [ ] 无错误和警告

### 性能验证

- [ ] 平均准确率≥46%
- [ ] Stacking优于单模型
- [ ] 置信度分布改善
- [ ] 预测有动态变化

---

## 🎓 技术细节

### Stacking vs Bagging vs Boosting

| 方法 | 特点 | 适用场景 |
|------|------|---------|
| **Bagging** | 并行训练，简单平均 | 减少方差 |
| **Boosting** | 串行训练，加权组合 | 减少偏差 |
| **Stacking** | 元学习器学习组合 | **最智能** ✅ |

**为什么选Stacking**：
- LightGBM, XGBoost, CatBoost都是Boosting
- 使用Bagging效果不明显
- **Stacking可以学习不同Boosting模型的互补性**

---

### 元特征的意义

**9维元特征** = 3个模型 × 3个类别概率

```
[LGB_SHORT, LGB_HOLD, LGB_LONG,
 XGB_SHORT, XGB_HOLD, XGB_LONG,
 CAT_SHORT, CAT_HOLD, CAT_LONG]
```

**元学习器学习的问题**：
- 什么时候相信LightGBM？
- 什么时候相信XGBoost？
- 什么时候相信CatBoost？
- 如何组合他们的意见？

**自动学习 > 手工设定权重**

---

## ✅ 总结

### 已完成

- ✅ 创建ensemble_ml_service.py（545行）
- ✅ 实现Stacking三模型融合
- ✅ 实现元学习器
- ✅ 集成到系统
- ✅ 创建requirements.txt
- ✅ 通过语法检查

### 待执行

1. 🔥 **安装依赖**（xgboost, catboost）
2. 🔥 **删除旧模型**
3. 🔥 **重启训练**
4. 📊 **验证准确率≥48%**

### 预期

**准确率**: 42.81% → **48-51%**  
**提升**: +12-19%  
**vs目标**: 接近或达到50% ✅

---

**实施方案**: Stacking三模型融合  
**预期准确率**: 48-51%  
**下一步**: 🔥 安装依赖，删除旧模型，重启训练

