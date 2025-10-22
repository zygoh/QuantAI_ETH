# Informer-2 模型错误深度分析

## 🚨 核心错误

```
ERROR - Informer-2训练失败: max(): Expected reduction dim 3 to have non-zero size.
```

**错误位置**: `ProbSparseSelfAttention._prob_QK()` 方法中的 `M = Q_K.max(dim=-1)[0]`

---

## 🔍 根本原因分析

### 问题1: **输入数据维度不匹配**

#### 当前实现的问题：

**Informer-2模型期望的输入**:
```python
# informer2_model.py line 447
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (batch, n_features) - 单个样本的特征向量  ❌ 错误的注释
    """
```

**实际的Informer架构需求**:
- Informer是为**时间序列**设计的，需要 `(batch, seq_len, features)` 三维输入
- 但当前实现接收的是 `(batch, features)` 二维输入（单个时间点的特征）

#### 代码中的"临时修复"：

```python
# line 451-452
x = self.input_projection(x)  # (batch, d_model)
x = x.unsqueeze(1)  # (batch, 1, d_model) ⚠️ 强行添加seq_len=1
```

**这导致了致命问题**：
- `seq_len = 1` 意味着只有1个时间步
- 在ProbSparse注意力中，需要计算 `sample_k = factor * ceil(log(L_K))`
- 当 `L_K = 1` 时，`log(1) = 0`，导致 `sample_k = 0`
- 然后 `K_sample = K[:, :, torch.randperm(L_K)[:sample_k], :]` 变成空张量
- 最终 `Q_K.max(dim=-1)` 在空维度上操作，触发错误

---

### 问题2: **ProbSparse注意力的采样逻辑缺陷**

```python
# informer2_model.py line 73-76
def _prob_QK(self, Q, K, sample_k, n_top):
    # ...
    K_sample = K[:, :, torch.randperm(L_K)[:sample_k], :]  # ❌ 当sample_k=0时为空
    Q_K = torch.matmul(Q, K_sample.transpose(-2, -1))  # (B, H, L_Q, 0)
    M = Q_K.max(dim=-1)[0]  # 💥 在维度3(size=0)上求max，报错！
```

**数学推导**:
- `L_K = 1` (序列长度)
- `sample_k = factor * ceil(log(1)) = 5 * 0 = 0`
- `K_sample.shape = (B, H, 0, d)` ← 第3维为0
- `Q_K.shape = (B, H, L_Q, 0)` ← 第4维为0
- `max(dim=-1)` 在空维度上操作 → **ERROR**

---

## 📊 配置合理性分析

### 当前配置：

```python
# ensemble_ml_service.py
self.informer_d_model = 128      # 模型维度
self.informer_n_heads = 8        # 注意力头数
self.informer_n_layers = 3       # Encoder层数
self.informer_epochs = 50        # 训练轮数
self.informer_batch_size = 256   # 批次大小
```

### 超参数搜索空间（15m）：

```python
'd_model': [64, 128, 256]
'n_heads': [4, 8, 16]
'n_layers': [2, 3, 4]
'epochs': [20, 40]
'batch_size': [128, 256, 512]
```

---

## ⚖️ 配置合理性评估

### ✅ 合理的部分：

1. **d_model = 128**: 适中，平衡性能和计算成本
2. **n_heads = 8**: 标准配置（Transformer论文推荐）
3. **batch_size = 256**: 对于27,025个样本合理
4. **epochs = 50**: GPU加速下可接受

### ❌ 不合理的部分：

#### 1. **架构设计根本性错误**

**问题**: Informer是为**长序列时间序列预测**设计的，但你的任务是**单时间点分类**

| 维度 | Informer原始设计 | 你的实际需求 |
|------|-----------------|-------------|
| 输入 | (batch, seq_len=96, features) | (batch, features) |
| 任务 | 预测未来24步 | 分类当前时刻(LONG/HOLD/SHORT) |
| 优势 | 长序列建模 | 特征提取 |

**结论**: **Informer不适合你的任务！**

#### 2. **seq_len = 1 的致命缺陷**

```python
x = x.unsqueeze(1)  # (batch, 1, d_model)
```

- ProbSparse注意力需要 `seq_len >= 10` 才有意义
- `log(1) = 0` 导致采样失败
- 失去了Informer的核心优势（长序列建模）

#### 3. **n_layers = 3 过深**

- 对于 `seq_len = 1`，多层Encoder毫无意义
- 每层蒸馏会减半序列长度：`1 → 0.5 → 0.25` (不可行)
- 增加计算成本但无性能提升

#### 4. **use_distilling = True 不适用**

```python
# informer2_model.py line 177
class DistillingLayer:
    def forward(self, x):
        # 对每个特征维度分别进行蒸馏
        # 使用MaxPool1d(kernel_size=3, stride=2)
        # 输入: (B, L, D) → 输出: (B, L//4, D)
```

- 蒸馏层设计用于减少序列长度
- 当 `L = 1` 时，`L//4 = 0` → 序列消失！

---

## 💡 解决方案

### 方案1: **修复Informer-2（不推荐）**

#### 步骤：
1. 移除ProbSparse注意力，使用标准注意力
2. 移除蒸馏层
3. 设置 `n_layers = 1`
4. 简化为普通Transformer Encoder

#### 代码修改：
```python
class Informer2ForClassification(nn.Module):
    def __init__(self, ...):
        # 使用标准MultiHeadAttention替代ProbSparse
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        # 移除蒸馏层
        # 单层Encoder
```

**缺点**: 失去Informer的所有优势，变成普通Transformer

---

### 方案2: **使用更适合的模型（强烈推荐）**

#### 推荐模型：

##### A. **TabNet** (Google 2019)
- 专为表格数据设计
- 可解释性强（特征重要性）
- 性能优于传统GBDT

```python
from pytorch_tabnet.tab_model import TabNetClassifier

model = TabNetClassifier(
    n_d=64,  # 决策层维度
    n_a=64,  # 注意力层维度
    n_steps=5,  # 决策步数
    gamma=1.5,  # 稀疏性系数
    n_independent=2,
    n_shared=2
)
```

##### B. **FT-Transformer** (2021)
- Feature Tokenizer + Transformer
- 专为表格数据优化
- SOTA性能

```python
class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, n_heads=8, n_layers=3):
        # 每个特征独立嵌入
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        # 标准Transformer Encoder
        self.transformer = nn.TransformerEncoder(...)
```

##### C. **简化版Transformer**
- 移除时间序列相关组件
- 保留自注意力机制
- 轻量级

```python
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, n_features, d_model=128, n_heads=8):
        self.embedding = nn.Linear(n_features, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.classifier = nn.Linear(d_model, 3)
```

---

### 方案3: **重新设计输入（如果坚持用Informer）**

#### 构造时间序列输入：

```python
def create_sequence_input(df, seq_len=96):
    """
    将单时间点特征转换为时间序列
    
    Args:
        df: 原始数据 (n_samples, n_features)
        seq_len: 序列长度（如96个15分钟 = 24小时）
    
    Returns:
        X_seq: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    X_seq = []
    y_seq = []
    
    for i in range(seq_len, len(df)):
        # 取过去seq_len个时间步的特征
        X_seq.append(df.iloc[i-seq_len:i].values)
        y_seq.append(df.iloc[i]['label'])
    
    return np.array(X_seq), np.array(y_seq)
```

**优点**: 充分利用Informer的长序列建模能力
**缺点**: 
- 需要重新设计数据管道
- 训练样本减少（前96个样本无法使用）
- 推理时需要96个历史时间步

---

## 📋 推荐配置

### 如果使用方案2A (TabNet):

```python
# ensemble_ml_service.py
self.enable_tabnet = True
self.tabnet_n_d = 64
self.tabnet_n_a = 64
self.tabnet_n_steps = 5
self.tabnet_gamma = 1.5
self.tabnet_epochs = 50
self.tabnet_batch_size = 256
self.tabnet_lr = 0.02
```

### 如果使用方案2C (简化Transformer):

```python
self.enable_simple_transformer = True
self.st_d_model = 128
self.st_n_heads = 8
self.st_n_layers = 2  # 减少层数
self.st_epochs = 50
self.st_batch_size = 256
self.st_lr = 0.001
```

### 如果使用方案3 (重新设计Informer):

```python
self.enable_informer2 = True
self.informer_seq_len = 96  # 新增：序列长度
self.informer_d_model = 128
self.informer_n_heads = 8
self.informer_n_layers = 2  # 减少到2层
self.informer_factor = 5  # ProbSparse采样因子
self.informer_epochs = 30  # 减少轮数（序列输入训练更慢）
self.informer_batch_size = 64  # 减小批次（序列输入占用更多内存）
self.informer_lr = 0.0005
```

---

## 🎯 最终建议

### 短期（立即修复）：
1. **禁用Informer-2**: `self.enable_informer2 = False`
2. 继续使用3个GBDT模型的Stacking集成
3. 系统已经有47%的准确率，可以正常运行

### 中期（1-2周）：
1. **实现TabNet**: 最适合表格数据的深度学习模型
2. 替换Informer-2为TabNet
3. 重新训练4模型集成（LGB + XGB + CAT + TabNet）

### 长期（1个月+）：
1. **重新设计数据管道**: 构造时间序列输入
2. 实现真正的Informer-2（用于长序列预测）
3. 探索多任务学习（分类 + 价格预测）

---

## 📊 性能预期

| 模型组合 | 预期准确率 | 训练时间 | 推理速度 |
|---------|-----------|---------|---------|
| 当前(3 GBDT) | 47% | 1.5h | 快 |
| + TabNet | 49-51% | 2h | 中 |
| + 序列Informer | 52-55% | 3h | 慢 |

---

## 🔧 立即可执行的修复

```python
# backend/app/services/ensemble_ml_service.py
# Line 70: 临时禁用Informer-2
self.enable_informer2 = False  # ❌ 暂时禁用，等待修复

# 或者添加序列长度检查
def _train_informer2(self, ...):
    # 在训练前检查
    if X_train.shape[0] < 96:
        logger.warning("⚠️ 样本数不足，跳过Informer-2训练")
        return None
    
    # 构造序列输入
    X_seq, y_seq = self._create_sequence_input(X_train, y_train, seq_len=96)
    # ... 继续训练
```

---

## 总结

**核心问题**: Informer-2是为长序列时间序列预测设计的，但你的任务是单时间点特征分类，架构不匹配。

**最佳方案**: 使用TabNet或FT-Transformer替代Informer-2。

**临时方案**: 禁用Informer-2，使用3模型集成（已经有47%准确率）。
