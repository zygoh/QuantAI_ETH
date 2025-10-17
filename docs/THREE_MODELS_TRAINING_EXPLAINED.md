# 📍 三个模型的训练位置详解

**项目**: QuantAI-ETH  
**架构**: Stacking三模型融合  
**创建时间**: 2025-10-17

---

## 🎯 三个模型的训练位置

### 模型1: LightGBM

**训练方法位置**: 
- **文件**: `backend/app/services/ml_service.py`
- **行数**: 第806行
- **方法**: `_train_lightgbm()`

**调用位置**:
- **文件**: `backend/app/services/ensemble_ml_service.py`
- **行数**: 第175行
- **代码**: `lgb_model = self._train_lightgbm(X_train, y_train, timeframe)`

**调用方式**: ✅ **继承自父类**（复用代码）

---

### 模型2: XGBoost

**训练方法位置**:
- **文件**: `backend/app/services/ensemble_ml_service.py`
- **行数**: 第284行
- **方法**: `_train_xgboost()`

**调用位置**:
- **文件**: `backend/app/services/ensemble_ml_service.py`
- **行数**: 第179行
- **代码**: `xgb_model = self._train_xgboost(X_train, y_train, timeframe)`

**调用方式**: ✅ **在子类中新增**

---

### 模型3: CatBoost

**训练方法位置**:
- **文件**: `backend/app/services/ensemble_ml_service.py`
- **行数**: 第319行
- **方法**: `_train_catboost()`

**调用位置**:
- **文件**: `backend/app/services/ensemble_ml_service.py`
- **行数**: 第183行
- **代码**: `cat_model = self._train_catboost(X_train, y_train, timeframe)`

**调用方式**: ✅ **在子类中新增**

---

## 🏗️ 完整调用链

### Stacking训练流程

```
main.py
    ↓
scheduler.py (定时任务)
    ↓
ensemble_ml_service.train_model()
    ↓
ensemble_ml_service.train_all_timeframes()
    ↓
ensemble_ml_service._train_ensemble_single_timeframe(timeframe)
    ↓
【准备数据】（复用父类方法）
    self._prepare_training_data_for_timeframe()  ← ml_service.py
    self.feature_engineer.create_features()      ← ml_service.py
    self._create_labels()                        ← ml_service.py
    self._prepare_features_labels()              ← ml_service.py
    self._scale_features()                       ← ml_service.py
    ↓
【训练三个基础模型】
    ├─ self._train_lightgbm()   ← ml_service.py:806 ✅
    ├─ self._train_xgboost()    ← ensemble_ml_service.py:284 ✅
    └─ self._train_catboost()   ← ensemble_ml_service.py:319 ✅
    ↓
【生成元特征】
    lgb_proba = lgb_model.predict_proba(X_train)
    xgb_proba = xgb_model.predict_proba(X_train)
    cat_proba = cat_model.predict_proba(X_train)
    meta_features = np.hstack([lgb_proba, xgb_proba, cat_proba])
    ↓
【训练元学习器】
    meta_learner = LogisticRegression()
    meta_learner.fit(meta_features, y_train)
    ↓
【保存4个模型】
    self._save_ensemble_models()
```

---

## 📊 代码分布

### ml_service.py（基类）

**提供的训练相关方法**：

| 方法 | 行数 | 作用 |
|------|------|------|
| `_prepare_training_data_for_timeframe()` | ~400行 | 数据准备 |
| `_create_labels()` | ~80行 | 标签创建 |
| `_prepare_features_labels()` | ~100行 | 特征准备 |
| `_select_features_intelligent()` | ~200行 | 智能特征选择 |
| `_scale_features()` | ~50行 | 特征缩放 |
| **`_train_lightgbm()`** | **~150行** | **LightGBM训练** ⭐ |
| `_evaluate_model()` | ~50行 | 模型评估 |
| `_save_model()` | ~30行 | 模型保存 |

**总计**: 约1060行（提供所有基础功能）

---

### ensemble_ml_service.py（子类）

**新增的训练方法**：

| 方法 | 行数 | 作用 |
|------|------|------|
| `train_all_timeframes()` | ~70行 | 协调所有时间框架训练 |
| `_train_ensemble_single_timeframe()` | ~60行 | 单时间框架Stacking |
| `_train_stacking_ensemble()` | ~150行 | Stacking核心逻辑 |
| **`_train_xgboost()`** | **~70行** | **XGBoost训练** ⭐ |
| **`_train_catboost()`** | **~70行** | **CatBoost训练** ⭐ |
| `_save_ensemble_models()` | ~30行 | 保存4个模型 |
| `_load_ensemble_models()` | ~30行 | 加载4个模型 |
| `predict()` | ~80行 | Stacking预测 |
| `train_model()` | ~30行 | 覆盖父类方法 |

**总计**: 约545行（新增集成逻辑）

---

## 📋 训练方法对比

### 1. _train_lightgbm()

**位置**: `ml_service.py:806-955`

**参数**:
```python
def _train_lightgbm(
    self, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    timeframe: str
) -> lgb.LGBMClassifier:
```

**核心代码**:
```python
# 样本加权
class_weights = compute_sample_weight('balanced', y_train)
time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
sample_weights = class_weights * time_decay

# 差异化参数
params = self.lgb_params_by_timeframe[timeframe]

# 训练
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train, sample_weight=sample_weights)

return model
```

**特点**:
- 使用差异化参数（15m: 95叶子，2h: 63，4h: 47）
- 样本加权（类别平衡 × 时间衰减）
- GPU支持

---

### 2. _train_xgboost()

**位置**: `ensemble_ml_service.py:284-316`

**参数**:
```python
def _train_xgboost(
    self, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    timeframe: str
):
```

**核心代码**:
```python
import xgboost as xgb

# 样本加权（与LightGBM一致）
class_weights = compute_sample_weight('balanced', y_train)
time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
sample_weights = class_weights * time_decay

# 参数配置
params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'tree_method': 'hist',
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# 训练
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

return model
```

**特点**:
- 统一参数（所有时间框架相同）
- 样本加权（与LightGBM一致）
- 强正则化

---

### 3. _train_catboost()

**位置**: `ensemble_ml_service.py:319-354`

**参数**:
```python
def _train_catboost(
    self, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    timeframe: str
):
```

**核心代码**:
```python
from catboost import CatBoostClassifier

# 样本加权（与LightGBM一致）
class_weights = compute_sample_weight('balanced', y_train)
time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
sample_weights = class_weights * time_decay

# 参数配置
params = {
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MultiClass',
    'random_seed': 42,
    'verbose': False,
    'l2_leaf_reg': 3.0,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1.0,
    'subsample': 0.8
}

# 训练
model = CatBoostClassifier(**params)
model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

return model
```

**特点**:
- 统一参数（所有时间框架相同）
- 样本加权（与LightGBM一致）
- 贝叶斯自助法

---

## 🔄 训练调用流程

### 在_train_stacking_ensemble()中调用

**位置**: `ensemble_ml_service.py:136-228`

```python
def _train_stacking_ensemble(
    self, 
    X_train, y_train, X_val, y_val, 
    timeframe
):
    """训练Stacking集成"""
    
    logger.info(f"🎯 Stage 1: 训练3个基础模型...")
    
    # 1️⃣ 训练LightGBM
    logger.info(f"  📊 训练LightGBM...")
    lgb_model = self._train_lightgbm(X_train, y_train, timeframe)
    # ↑ 调用父类方法（ml_service.py:806）
    
    # 2️⃣ 训练XGBoost
    logger.info(f"  📊 训练XGBoost...")
    xgb_model = self._train_xgboost(X_train, y_train, timeframe)
    # ↑ 调用本类方法（ensemble_ml_service.py:284）
    
    # 3️⃣ 训练CatBoost
    logger.info(f"  📊 训练CatBoost...")
    cat_model = self._train_catboost(X_train, y_train, timeframe)
    # ↑ 调用本类方法（ensemble_ml_service.py:319）
    
    logger.info(f"✅ 3个基础模型训练完成")
    
    # 4️⃣ 生成元特征
    logger.info(f"🎯 Stage 2: 生成元特征...")
    lgb_pred_train = lgb_model.predict_proba(X_train)
    xgb_pred_train = xgb_model.predict_proba(X_train)
    cat_pred_train = cat_model.predict_proba(X_train)
    
    meta_features_train = np.hstack([
        lgb_pred_train,
        xgb_pred_train,
        cat_pred_train
    ])
    
    # 5️⃣ 训练元学习器
    logger.info(f"🎯 Stage 3: 训练元学习器（Stacking）...")
    from sklearn.linear_model import LogisticRegression
    
    meta_learner = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    meta_learner.fit(meta_features_train, y_train)
    
    # 6️⃣ 保存到字典
    self.ensemble_models[timeframe] = {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'cat': cat_model,
        'meta': meta_learner
    }
```

---

## 📊 文件依赖关系

### 文件结构

```
ml_service.py（基类，1063行）
    ├── 提供通用功能（95%）
    ├── _prepare_training_data()
    ├── _create_labels()
    ├── _prepare_features_labels()
    ├── _scale_features()
    ├── _train_lightgbm() ⭐
    └── ... 30+ 方法
        ↑ 继承
ensemble_ml_service.py（子类，545行）
    ├── 继承所有父类方法 ✅
    ├── 新增 _train_xgboost() ⭐
    ├── 新增 _train_catboost() ⭐
    ├── 新增 _train_stacking_ensemble()
    │   ├─ 调用 self._train_lightgbm() ← 父类方法
    │   ├─ 调用 self._train_xgboost() ← 本类方法
    │   └─ 调用 self._train_catboost() ← 本类方法
    └── 覆盖 train_model(), predict()
```

---

## 🔍 代码位置速查表

| 模型 | 训练方法定义 | 调用位置 | 来源 |
|------|------------|---------|------|
| **LightGBM** | ml_service.py:806 | ensemble_ml_service.py:175 | 父类继承 ✅ |
| **XGBoost** | ensemble_ml_service.py:284 | ensemble_ml_service.py:179 | 子类新增 ✅ |
| **CatBoost** | ensemble_ml_service.py:319 | ensemble_ml_service.py:183 | 子类新增 ✅ |
| **元学习器** | ensemble_ml_service.py:195 | ensemble_ml_service.py:195 | 子类新增 ✅ |

---

## 💻 训练参数对比

### LightGBM参数

**文件**: `ml_service.py`（通过self.lgb_params_by_timeframe）

```python
# 15m时间框架
{
    'num_leaves': 95,
    'learning_rate': 0.03,
    'n_estimators': 300,
    'max_depth': 7,
    'min_child_samples': 50,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**特点**: 差异化配置（每个timeframe不同）

---

### XGBoost参数

**文件**: `ensemble_ml_service.py:291-303`

```python
{
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'tree_method': 'hist',
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**特点**: 统一配置（所有timeframe相同）

---

### CatBoost参数

**文件**: `ensemble_ml_service.py:329-340`

```python
{
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MultiClass',
    'random_seed': 42,
    'verbose': False,
    'l2_leaf_reg': 3.0,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1.0,
    'subsample': 0.8
}
```

**特点**: 统一配置（所有timeframe相同）

---

## 🎯 为什么这样设计？

### 设计原则

1. **继承复用**：
   - LightGBM训练逻辑已经很成熟（ml_service.py）
   - 直接继承使用，避免重复代码
   - 保持一致性

2. **新增扩展**：
   - XGBoost和CatBoost是新增功能
   - 在子类中实现，不影响基类
   - 便于维护和测试

3. **模块分离**：
   - 基础功能在ml_service.py
   - 集成功能在ensemble_ml_service.py
   - 职责清晰

---

## 🚀 训练执行顺序

### 单个时间框架（例如15m）

```
Step 1: 准备数据
    ↓ self._prepare_training_data_for_timeframe('15m')
    34560条K线

Step 2: 特征工程
    ↓ self.feature_engineer.create_features(data)
    186个特征

Step 3: 创建标签
    ↓ self._create_labels(data, '15m')
    阈值±0.15%

Step 4: 特征选择
    ↓ self._prepare_features_labels(data, '15m')
    智能选择141个特征

Step 5: 特征缩放
    ↓ self._scale_features(X, '15m', fit=True)
    StandardScaler

Step 6: 数据分割
    ↓ split_idx = int(len(X) * 0.8)
    训练27488，验证6872

Step 7: 训练基础模型1
    ↓ self._train_lightgbm(X_train, y_train, '15m')
    【ml_service.py:806】
    LightGBM模型

Step 8: 训练基础模型2
    ↓ self._train_xgboost(X_train, y_train, '15m')
    【ensemble_ml_service.py:284】
    XGBoost模型

Step 9: 训练基础模型3
    ↓ self._train_catboost(X_train, y_train, '15m')
    【ensemble_ml_service.py:319】
    CatBoost模型

Step 10: 生成元特征
    ↓ meta_features = [lgb概率, xgb概率, cat概率]
    9维元特征

Step 11: 训练元学习器
    ↓ meta_learner = LogisticRegression()
    ↓ meta_learner.fit(meta_features, y_train)
    元学习器（Stacking核心）

Step 12: 保存4个模型
    ↓ ETHUSDT_15m_lgb_model.pkl
    ↓ ETHUSDT_15m_xgb_model.pkl
    ↓ ETHUSDT_15m_cat_model.pkl
    ↓ ETHUSDT_15m_meta_model.pkl
```

---

## ✅ 总结

### 三个模型训练位置

| 模型 | 训练方法位置 | 调用位置 |
|------|------------|---------|
| **LightGBM** | `ml_service.py:806` | `ensemble_ml_service.py:175` |
| **XGBoost** | `ensemble_ml_service.py:284` | `ensemble_ml_service.py:179` |
| **CatBoost** | `ensemble_ml_service.py:319` | `ensemble_ml_service.py:183` |

### 为什么ml_service.py不能删除？

1. ✅ 提供LightGBM训练方法
2. ✅ 提供所有数据准备方法
3. ✅ 是ensemble_ml_service的父类
4. ✅ 复用率68%
5. ✅ **删除会导致系统崩溃**

### 文件职责

- **ml_service.py**: 基类，提供通用ML功能
- **ensemble_ml_service.py**: 子类，实现Stacking集成

---

**LightGBM训练**: ml_service.py:806 ✅  
**XGBoost训练**: ensemble_ml_service.py:284 ✅  
**CatBoost训练**: ensemble_ml_service.py:319 ✅  
**元学习器**: ensemble_ml_service.py:195 ✅

