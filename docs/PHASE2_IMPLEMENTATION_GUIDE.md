# 🚀 Phase 2优化实施指南

**项目**: QuantAI-ETH  
**当前准确率**: 38.92%  
**目标准确率**: ≥50%（最低要求，不可妥协）  
**态度**: 🔴 **准确率<50%没有任何意义**  
**实施时间**: 2025-10-17 - 2025-10-24（5-7天）

---

## 🎯 核心理念

### 用户明确要求

✅ **"准确率就是不能低于0.5不然没有任何意义"**

**完全正确！**

### 现实情况

```
38.92%准确率 = 每10个信号有6个错误 ❌
→ 胜率<40%
→ 即使有止损也难盈利
→ 完全不具备交易价值
```

### 行动方针

🔥 **立即全面优化，不达50%不停止**

---

## ✅ 已完成的优化（Phase 2.1-2.2）

### 优化2.1: 调整15m标签阈值 ✅

**文件**: `backend/app/services/ml_service.py:542-544`

**修改**:
```python
'15m': {
    'up': 0.0015,     # ±0.15%（从±0.1%放宽50%）
    'down': -0.0015
}
```

**预期**: 
- HOLD: 29% → 36-40%
- 准确率: 38.45% → 42-45%
- 提升: **+8-17%**

---

### 优化2.2: 降低15m模型复杂度 ✅

**文件**: `backend/app/services/ml_service.py:66-72`

**修改**:
```python
'15m': {
    'num_leaves': 95,        # 从127降低25%
    'min_child_samples': 50,  # 从30提高67%
    'max_depth': 7,          # 从10降低30%
    'reg_alpha': 0.5,        # 新增L1正则化
    'reg_lambda': 0.5        # 新增L2正则化
}
```

**预期**:
- 防止过拟合
- 泛化能力提升
- 准确率: +2-3%
- 提升: **+5-8%**

---

### 优化2.0: 修复置信度阈值 ✅

**文件**: `backend/app/core/config.py:27`

**修改**:
```python
CONFIDENCE_THRESHOLD: float = 0.40  # 从0.5修复
```

**影响**: 允许更多信号通过

---

## 🚀 待实施的优化（Phase 2.3-2.4）

### Phase 2.3: 模型集成（Day 2-3）⭐⭐⭐

**方案**: LightGBM + XGBoost + CatBoost

**预期**: +5-8%，达到48-51%

**详细实施**:

#### 1. 安装依赖

```bash
pip install xgboost catboost
```

#### 2. 创建集成服务

**新文件**: `backend/app/services/ensemble_ml_service.py`

**核心代码**:
```python
class EnsembleMLService(MLService):
    """集成机器学习服务"""
    
    def __init__(self):
        super().__init__()
        self.ensemble_models = {}  # {timeframe: {lgb, xgb, cat}}
    
    async def _train_ensemble_single_timeframe(self, timeframe: str):
        """训练单个时间框架的集成模型"""
        
        # 获取训练数据
        data = await self._prepare_training_data_for_timeframe(timeframe)
        data = self.feature_engineer.create_features(data)
        data = self._create_labels(data, timeframe)
        X, y = self._prepare_features_labels(data, timeframe)
        X_scaled = self._scale_features(X, timeframe, fit=True)
        
        # 时间序列分割
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 训练三个模型
        logger.info(f"🚂 训练 {timeframe} 集成模型（3个模型）...")
        
        # 1. LightGBM
        lgb_model = self._train_lightgbm(X_train, y_train, timeframe)
        
        # 2. XGBoost
        xgb_model = self._train_xgboost(X_train, y_train, timeframe)
        
        # 3. CatBoost
        cat_model = self._train_catboost(X_train, y_train, timeframe)
        
        # 保存集成模型
        self.ensemble_models[timeframe] = {
            'lgb': lgb_model,
            'xgb': xgb_model,
            'cat': cat_model
        }
        
        # 评估集成效果
        ensemble_acc = self._evaluate_ensemble(
            X_val, y_val, timeframe
        )
        
        logger.info(f"✅ {timeframe} 集成准确率: {ensemble_acc:.4f}")
        
        return ensemble_acc
    
    def _train_xgboost(self, X_train, y_train, timeframe):
        """训练XGBoost"""
        import xgboost as xgb
        
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'tree_method': 'hist'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_catboost(self, X_train, y_train, timeframe):
        """训练CatBoost"""
        from catboost import CatBoostClassifier
        
        params = {
            'iterations': 300,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'MultiClass',
            'random_seed': 42,
            'verbose': False
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    async def predict(self, symbol: str, timeframe: str):
        """集成预测"""
        # 获取数据
        data = await self._prepare_prediction_data(symbol, timeframe)
        X = self._prepare_features_for_prediction(data, timeframe)
        
        # 获取集成模型
        models = self.ensemble_models[timeframe]
        
        # 三模型预测
        lgb_prob = models['lgb'].predict_proba(X)[-1]
        xgb_prob = models['xgb'].predict_proba(X)[-1]
        cat_prob = models['cat'].predict_proba(X)[-1]
        
        # 加权平均
        ensemble_prob = lgb_prob * 0.4 + xgb_prob * 0.3 + cat_prob * 0.3
        
        # 最终预测
        prediction = ensemble_prob.argmax()
        confidence = ensemble_prob[prediction]
        
        return prediction, confidence, ensemble_prob
```

#### 3. 集成到系统

**修改**: `main.py`
```python
# 使用集成ML服务替代单模型
from app.services.ensemble_ml_service import EnsembleMLService
ml_service = EnsembleMLService()
```

---

### Phase 2.4: 超参数优化（Day 4-7）⭐⭐

**方案**: Optuna自动调参

**预期**: +2-4%，达到50-54%

**实施代码**:

```python
# ml_service.py - 新增方法
def optimize_hyperparameters_for_timeframe(self, timeframe: str):
    """为特定时间框架优化超参数"""
    import optuna
    
    # 准备数据
    data = await self._prepare_training_data_for_timeframe(timeframe)
    data = self.feature_engineer.create_features(data)
    data = self._create_labels(data, timeframe)
    X, y = self._prepare_features_labels(data, timeframe)
    X_scaled = self._scale_features(X, timeframe, fit=True)
    
    def objective(trial):
        """优化目标函数"""
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0)
        }
        
        # 合并基础参数
        final_params = {**self.base_lgb_params, **params}
        
        # 创建模型
        model = lgb.LGBMClassifier(**final_params, random_state=42)
        
        # 5折时间序列交叉验证
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return np.mean(scores)
    
    # 创建优化研究
    logger.info(f"🔍 开始 {timeframe} 超参数优化（100次试验）...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    logger.info(f"✅ {timeframe} 最优参数:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"✅ {timeframe} 最优准确率: {study.best_value:.4f}")
    
    return study.best_params
```

---

## 📊 预期效果汇总

### 累积提升预测

| 优化阶段 | 准确率 | 提升 | 累计提升 |
|---------|--------|------|---------|
| 当前 | 38.92% | - | - |
| Phase 2.1+2.2 | 42-45% | +8-15% | +8-15% |
| Phase 2.3 | 48-51% | +5-8% | +23-31% |
| Phase 2.4 | **50-55%** | +2-4% | **+29-42%** ✅ |

---

## 🔧 实施步骤详解

### Step 1: 立即优化（已完成）✅

- [x] 调整15m标签阈值：0.001 → 0.0015
- [x] 降低15m模型复杂度：num_leaves 127→95
- [x] 修复置信度阈值：0.5 → 0.40
- [x] 更新项目规则

---

### Step 2: 删除旧模型并重启（现在）

```powershell
# Windows PowerShell
Remove-Item backend\models\*.pkl

# 重启训练
cd backend
python main.py
```

**验证要点**：
- [ ] 15m HOLD: 36-40%（从29%提升）
- [ ] 15m准确率: ≥42%
- [ ] 平均准确率: ≥42%
- [ ] 预测有变化（不再固定）

---

### Step 3: 准备模型集成（Day 2）

```bash
# 安装依赖
pip install xgboost catboost
```

**创建文件**：
- `backend/app/services/ensemble_ml_service.py`（集成服务）

**实现功能**：
- 三模型训练
- 加权投票预测
- 集成评估

---

### Step 4: 实施集成（Day 2-3）

**修改文件**：
- `main.py`：使用EnsembleMLService
- `signal_generator.py`：调用集成预测

**重新训练**：
```bash
# 删除旧模型
Remove-Item backend\models\*.pkl

# 重启训练（会自动使用集成）
python main.py
```

**验证**: 准确率≥48%

---

### Step 5: 超参数优化（Day 4-7）

```bash
# 安装Optuna
pip install optuna
```

**实现**：
- 为每个时间框架单独调参
- 运行100次试验
- 应用最优参数
- 重新训练验证

**目标**: **准确率≥50%** ✅

---

## 📈 时间表

### 激进方案（5天）

```
Day 0 (今天):
  - 调整参数 ✅
  - 重启训练
  - 验证42-45%
  
Day 1:
  - 准备模型集成代码
  
Day 2-3:
  - 实施模型集成
  - 验证48-51%
  
Day 4-5:
  - 超参数优化
  - 验证≥50% ✅
```

---

### 保守方案（7天）

```
Day 0: 重启验证（42-44%）
Day 1: 准备集成代码
Day 2-4: 实施模型集成（47-49%）
Day 5-7: 超参数优化（50-53%） ✅
```

---

## 🎯 成功标准

### 不可妥协的要求

- 🔴 **准确率: ≥50%**（三分类）
- 🔴 **不允许低于50%就停止优化**
- 🔴 **不允许低于50%就实盘**

### 达标后

- ✅ 运行虚拟交易3-7天
- ✅ 验证实际胜率≥55%
- ✅ 验证盈亏比≥1.8:1
- ✅ 准备切换实盘

---

## ⚠️ 风险预案

### 如果7天后仍<50%

**Plan B1**: 改为二分类
```python
# 去掉HOLD类别，只预测UP/DOWN
labels = [0, 1]  # DOWN, UP
# 预期准确率可能更高（更简单）
```

**Plan B2**: 增加训练数据
```python
training_days = {
    '15m': 540,  # 从360天增加
    '2h': 540,
    '4h': 720
}
```

**Plan B3**: 尝试深度学习
```python
# LSTM或Transformer
# 可能更适合时间序列预测
```

---

## 📋 检查清单

### Phase 2.1-2.2（已完成）

- [x] ✅ 调整15m标签阈值
- [x] ✅ 降低15m模型复杂度
- [x] ✅ 修复置信度阈值
- [x] ✅ 更新项目规则
- [ ] 🔄 删除旧模型
- [ ] 🔄 重新训练验证

### Phase 2.3（待实施）

- [ ] 安装xgboost, catboost
- [ ] 创建ensemble_ml_service.py
- [ ] 实现三模型训练
- [ ] 实现集成预测
- [ ] 集成到系统
- [ ] 验证准确率≥48%

### Phase 2.4（待实施）

- [ ] 安装Optuna
- [ ] 实现自动调参
- [ ] 运行优化
- [ ] 应用最优参数
- [ ] 验证准确率≥50%

---

## 🔥 立即执行

### 现在就做

```powershell
# 1. 删除旧模型
Remove-Item backend\models\*.pkl

# 2. 重启训练
cd backend
python main.py

# 3. 观察日志，验证准确率≥42%
```

**预计时间**: 1小时（30秒训练 + 30分钟观察）

---

## ✅ 总结

### 用户要求

🔴 **准确率不能低于50%，否则没有任何意义**

### 当前状态

❌ 38.92%（不可用）

### 已完成

✅ 3项立即优化（阈值、复杂度、置信度）

### 待完成

🔄 重新训练验证（预期42-45%）  
🔄 模型集成（预期48-51%）  
🔄 超参数优化（预期50-55%）✅

### 承诺

**不达50%，不停止优化！**

---

**实施开始**: 2025-10-17  
**预计完成**: 2025-10-24（5-7天）  
**目标**: 准确率≥50%  
**下一步**: 🔥 删除旧模型，重启训练

