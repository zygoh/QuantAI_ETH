# 设计文档 - Informer-2 + GMADL 量化模型优化

## 概述

本设计文档针对加密货币自动交易系统中Informer-2模型的维度错误问题，提供详细的技术分析和解决方案。系统使用四模型Stacking集成（LightGBM + XGBoost + CatBoost + Informer-2）进行交易信号预测。

### 当前问题诊断

根据错误日志分析，核心问题是：

```
ERROR - Informer-2训练失败: max(): Expected reduction dim 3 to have non-zero size.
错误位置: ProbSparseSelfAttention._prob_QK() 方法中的 M = Q_K.max(dim=-1)[0]
```

**数据流程分析**：
1. **原始数据**：K线序列（timestamp, open, high, low, close, volume）
2. **特征工程**：每根K线 → 82个技术指标（RSI、MACD、均线等）
3. **模型输入**：每个样本是单根K线的特征向量 `(batch, 82_features)`
4. **问题**：Informer-2期望序列输入 `(batch, seq_len, features)`，但收到的是 `(batch, features)`

**根本原因**：
1. Informer-2是为长序列时间序列预测设计的，期望输入形状为 `(batch, seq_len, features)`
2. 当前实现接收的是单时间点特征 `(batch, 82)`，通过 `unsqueeze(1)` 强行添加 `seq_len=1`
3. 当 `seq_len=1` 时，ProbSparse注意力的采样参数 `sample_k = factor * ceil(log(1)) = 5 * 0 = 0`
4. 导致 `K_sample` 为空张量，`Q_K.max(dim=-1)` 在空维度上操作触发错误

**为什么不是序列输入**：
- 虽然原始数据是K线序列，但特征工程将每根K线转换成独立的特征向量
- 技术指标（如RSI_14、SMA_20）已经包含了历史信息（通过滚动窗口计算）
- 模型训练时，每个样本是一个独立的特征向量，不是序列

## 架构设计

### 方案选择

经过分析，提供三种解决方案：

#### 方案A：简化Informer-2（推荐 - 短期修复）

**设计思路**：
- 移除ProbSparse注意力，使用标准MultiHeadAttention
- 移除蒸馏层（Distilling Layer）
- 简化为单层Transformer Encoder
- 保留GMADL损失函数

**优点**：
- 快速修复，无需重构数据管道
- 保留Transformer的特征提取能力
- 继续使用GMADL损失函数的优势

**缺点**：
- 失去Informer的核心创新（ProbSparse、蒸馏）
- 本质上变成普通Transformer

**实现复杂度**：低

#### 方案B：构造序列输入（推荐 - 中期优化）

**设计思路**：
- 重新设计数据管道，构造时间序列输入
- 使用滑动窗口，将过去N个时间步的特征组合成序列
- 充分利用Informer-2的长序列建模能力

**优点**：
- 充分发挥Informer-2的设计优势
- 利用历史时间序列信息提升预测
- 保留所有创新组件（ProbSparse、蒸馏）

**缺点**：
- 需要重构数据管道
- 训练样本减少（前N个样本无法使用）
- 推理时需要N个历史时间步

**实现复杂度**：中

#### 方案C：替换为TabNet（推荐 - 长期优化）

**设计思路**：
- 使用Google的TabNet模型替代Informer-2
- TabNet专为表格数据设计，天然适合单时间点特征分类
- 提供特征重要性解释能力

**优点**：
- 架构完全匹配任务需求
- 性能优于传统GBDT
- 可解释性强

**缺点**：
- 需要引入新依赖（pytorch-tabnet）
- 需要重新训练和调优

**实现复杂度**：中

### 推荐实施路线

**用户选择：方案B - 构造序列输入**

这是最能发挥Informer-2优势的方案，充分利用历史时间序列信息提升预测准确率。

## 组件设计

### 1. 序列输入构造（方案B - 用户选择）

#### 1.1 设计思路

**核心概念**：
- 使用滑动窗口，将过去N根K线的特征组合成序列
- 每个样本从 `(82_features)` 变成 `(seq_len, 82_features)`
- 充分利用Informer-2的长序列建模能力

**序列长度选择**：
```python
seq_len_config = {
    '15m': 96,   # 96 × 15分钟 = 24小时
    '2h': 48,    # 48 × 2小时 = 4天
    '4h': 24     # 24 × 4小时 = 4天
}
```

**优点**：
- 保留Informer-2的所有创新组件（ProbSparse、蒸馏）
- 利用历史时间序列信息，提升预测准确率
- 符合Informer-2的设计初衷

**缺点**：
- 训练样本减少（前N个样本无法使用）
- 推理时需要N个历史时间步
- 需要重构数据管道

#### 1.2 数据管道重构

**新增方法：构造序列输入**

```python
def _create_sequence_input(
    self,
    df: pd.DataFrame,
    seq_len: int,
    timeframe: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造序列输入
    
    Args:
        df: 特征工程后的DataFrame（包含label列）
        seq_len: 序列长度
        timeframe: 时间框架
    
    Returns:
        X_seq: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    feature_columns = self.feature_columns_dict.get(timeframe, [])
    
    X_list = []
    y_list = []
    
    # 滑动窗口
    for i in range(seq_len, len(df)):
        # 取过去seq_len个时间步的特征
        X_window = df.iloc[i-seq_len:i][feature_columns].values
        y_label = df.iloc[i]['label']
        
        X_list.append(X_window)
        y_list.append(y_label)
    
    X_seq = np.array(X_list)  # (n_samples, seq_len, n_features)
    y = np.array(y_list)      # (n_samples,)
    
    logger.info(f"✅ 序列输入构造完成: {X_seq.shape}")
    return X_seq, y
```

**修改训练流程**：

```python
async def _train_ensemble_single_timeframe(self, timeframe: str):
    # ... 前面的数据准备代码 ...
    
    # 特征工程
    data_lgb = self.feature_engineer.create_features(data_lgb)
    data_lgb = self._create_labels(data_lgb, timeframe=timeframe)
    
    # 🆕 构造序列输入（仅用于Informer-2）
    seq_len = self.seq_len_config[timeframe]
    X_seq, y_seq = self._create_sequence_input(data_lgb, seq_len, timeframe)
    
    # 时间序列分割
    split_idx = int(len(X_seq) * 0.8)
    X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
    y_seq_train, y_seq_val = y_seq[:split_idx], y_seq[split_idx:]
    
    # 训练Informer-2（使用序列输入）
    inf_model = self._train_informer2(X_seq_train, y_seq_train, timeframe)
    
    # ... 后续代码 ...
```

#### 1.3 Informer-2模型适配

**修改forward方法**：

```python
class Informer2ForClassification(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, n_features) - 序列输入
        
        Returns:
            logits: (batch, n_classes) - 分类logits
        """
        # 1. 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 2. Encoder处理（保留ProbSparse和蒸馏）
        for encoder_layer in self.encoder_layers:
            x, _ = encoder_layer(x)  # (batch, seq_len, d_model)
            
            # 蒸馏层（如果启用）
            if self.use_distilling:
                x = self.distilling_layer(x)  # (batch, seq_len//2, d_model)
        
        # 3. 全局池化（聚合序列信息）
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 4. 分类
        logits = self.classifier(x)  # (batch, n_classes)
        
        return logits
```

**关键修改**：
- 移除 `unsqueeze(1)`，直接接收序列输入
- ProbSparse注意力正常工作（seq_len >= 24）
- 蒸馏层正常工作（seq_len逐层减半）

#### 1.4 推理流程适配

**修改predict方法**：

```python
async def predict(self, data: pd.DataFrame, timeframe: str):
    # 特征工程
    processed_data = feature_engineer.create_features(data.copy())
    
    # 🆕 构造序列输入（取最新seq_len个时间步）
    seq_len = self.seq_len_config[timeframe]
    
    if len(processed_data) < seq_len:
        raise Exception(f"数据不足：需要{seq_len}个时间步，实际{len(processed_data)}个")
    
    # 取最新seq_len个时间步
    latest_seq = processed_data.iloc[-seq_len:][feature_columns].values
    latest_seq = latest_seq.reshape(1, seq_len, -1)  # (1, seq_len, n_features)
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(latest_seq).to(device)
    
    # 预测
    with torch.no_grad():
        logits = inf_model(X_tensor)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    
    # ... 后续代码 ...
```

### 2. 简化Informer-2模型（方案A - 备选）

#### 1.1 移除ProbSparse注意力

**当前实现**：
```python
class ProbSparseSelfAttention(nn.Module):
    def _prob_QK(self, Q, K, sample_k, n_top):
        # 问题：当sample_k=0时，K_sample为空
        K_sample = K[:, :, torch.randperm(L_K)[:sample_k], :]
        Q_K = torch.matmul(Q, K_sample.transpose(-2, -1))
        M = Q_K.max(dim=-1)[0]  # 错误发生在这里
```

**修复方案**：
```python
# 使用PyTorch标准MultiHeadAttention替代
self.attention = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=n_heads,
    dropout=dropout,
    batch_first=True
)
```

#### 1.2 移除蒸馏层

**当前实现**：
```python
class DistillingLayer(nn.Module):
    def forward(self, x):
        # 问题：当L=1时，L//4=0，序列消失
        output = self.pooling(feature)  # MaxPool1d(kernel_size=3, stride=2)
```

**修复方案**：
```python
# 直接移除蒸馏层，不进行序列长度压缩
# 在Informer2ForClassification中设置 use_distilling=False
```

#### 1.3 简化Encoder层

**修复后的架构**：
```python
class SimplifiedInformer2(nn.Module):
    def __init__(self, n_features, n_classes=3, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(n_features, d_model)
        
        # 标准Transformer Encoder（单层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, n_features)
        x = self.input_projection(x)  # (batch, d_model)
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        x = self.encoder(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        logits = self.classifier(x)  # (batch, n_classes)
        return logits
```

### 2. GMADL损失函数优化

#### 2.1 当前实现分析

**数学公式**：
```
loss = (|error|^beta) / (alpha + |error|^(1-beta))
其中 error = 1 - P(correct_class)
```

**参数分析**：
- `alpha=1.0`: 控制对异常值的鲁棒性（越大越鲁棒）
- `beta=0.5`: 控制损失的凸性（0.5-1.0，越小越关注难分样本）

**评估结论**：实现正确，参数合理

#### 2.2 HOLD惩罚机制

**当前实现**：
```python
hold_weights = torch.where(
    targets == 1,  # HOLD类别
    torch.tensor(0.65),  # 惩罚系数
    torch.tensor(1.0)
)
weighted_loss = loss * hold_weights
```

**评估结论**：
- 惩罚系数0.65合理（降低HOLD类别的损失权重）
- 有效减少过度预测HOLD信号
- 与Optuna优化中的HOLD惩罚保持一致

### 3. Optuna超参数优化

#### 3.1 搜索空间设计

**当前配置分析**：

| 时间框架 | 样本数 | 模型复杂度 | 正则化强度 | 评估 |
|---------|--------|-----------|-----------|------|
| 15m | 多（~27k） | 高（depth=6-12） | 低（reg=0-0.5） | ✅ 合理 |
| 2h | 中（~6k） | 中（depth=3-6） | 中（reg=0.5-1.2） | ✅ 合理 |
| 4h | 少（~3k） | 低（depth=2-5） | 高（reg=0.8-1.5） | ✅ 合理 |

**设计原则**：
1. 样本越少，模型越简单，正则化越强
2. 防止过拟合：2h/4h使用更强的正则化
3. 差异化配置：不同时间框架使用不同搜索空间

#### 3.2 Informer-2搜索空间

**当前配置**：
```python
# 15m时间框架
'd_model': [64, 128, 256]
'n_heads': [4, 8, 16]
'n_layers': [2, 3, 4]
'epochs': [20, 40]
'batch_size': [128, 256, 512]
'lr': [0.0005, 0.005]
'dropout': [0.05, 0.2]
'alpha': [0.5, 2.0]  # GMADL参数
'beta': [0.3, 0.7]   # GMADL参数
```

**评估结论**：
- ✅ d_model范围合理（64-256）
- ✅ n_heads与d_model匹配（d_model必须能被n_heads整除）
- ⚠️ n_layers可能过多（简化后应设为1）
- ✅ epochs合理（20-40轮，GPU加速下可接受）
- ✅ batch_size合理（128-512）
- ✅ 学习率范围合理（0.0005-0.005）
- ✅ dropout范围合理（0.05-0.2）
- ✅ GMADL参数范围合理

#### 3.3 试验次数和超时配置

**当前配置**：
```python
# 传统模型（LightGBM/XGBoost/CatBoost）
optuna_n_trials = 100
optuna_timeout = 1800  # 30分钟

# Informer-2
informer_n_trials = 50
informer_timeout = 1200  # 20分钟
```

**评估结论**：
- ✅ 传统模型：100次试验 + 30分钟超时，平衡效率和效果
- ✅ Informer-2：50次试验 + 20分钟超时，考虑深度学习训练时间
- ✅ GPU加速下，时间配置合理
- ✅ 使用TimeSeriesSplit 5折交叉验证，评估可靠

### 4. GPU配置验证

#### 4.1 当前配置

```python
# config.py
USE_GPU = True
GPU_DEVICE = "cuda:0"

# ensemble_ml_service.py
self.use_gpu = settings.USE_GPU
self.gpu_device = settings.GPU_DEVICE
```

#### 4.2 各模型GPU配置

**LightGBM**：
```python
if self.use_gpu:
    base_params['device'] = 'gpu'
    base_params['gpu_platform_id'] = 0
    base_params['gpu_device_id'] = 0
```
✅ 配置正确

**XGBoost**：
```python
if self.use_gpu:
    base_params['tree_method'] = 'gpu_hist'
    base_params['gpu_id'] = 0
```
✅ 配置正确

**CatBoost**：
```python
if self.use_gpu:
    base_params['task_type'] = 'GPU'
    base_params['devices'] = '0'
```
✅ 配置正确

**Informer-2**：
```python
device = torch.device('cuda:0' if self.use_gpu and torch.cuda.is_available() else 'cpu')
model = Informer2ForClassification(...).to(device)
X_tensor = torch.FloatTensor(X).to(device)
```
✅ 配置正确

### 5. 其他模型参数优化

#### 5.1 元学习器配置

**当前配置**：
```python
meta_learner = lgb.LGBMClassifier(
    n_estimators=50,     # 树数量
    max_depth=3,         # 树深度
    learning_rate=0.15,  # 学习率
    num_leaves=7,        # 叶子数
    min_child_samples=30,  # 最小样本数
    subsample=0.7,       # 行采样
    colsample_bytree=0.7,  # 列采样
    reg_alpha=0.3,       # L1正则
    reg_lambda=0.3,      # L2正则
)
```

**评估结论**：
- ✅ 极简配置，有效防止过拟合
- ✅ 元学习器只需学习如何组合基础模型，不需要复杂模型
- ✅ 强正则化（reg_alpha=0.3, reg_lambda=0.3）
- ✅ 低采样率（subsample=0.7, colsample_bytree=0.7）

#### 5.2 动态HOLD惩罚

**当前实现**：
```python
hold_ratio = (meta_labels_val == 1).sum() / len(meta_labels_val)

if hold_ratio > 0.60:
    meta_hold_penalty_weight = 0.45  # 重惩罚
elif hold_ratio > 0.50:
    meta_hold_penalty_weight = 0.55  # 中等
elif hold_ratio > 0.40:
    meta_hold_penalty_weight = 0.65  # 轻度
else:
    meta_hold_penalty_weight = 0.75  # 正常
```

**评估结论**：
- ✅ 根据HOLD占比动态调整惩罚系数
- ✅ HOLD占比越高，惩罚越重
- ✅ 有效平衡类别分布

## 数据模型

### 输入数据格式

**当前格式**：
```python
X: (batch_size, n_features)  # 单时间点特征
y: (batch_size,)  # 标签（0=SHORT, 1=HOLD, 2=LONG）
```

**特征数量**：82个高级技术指标

### 模型输出格式

**Informer-2输出**：
```python
logits: (batch_size, 3)  # 原始分数
probs: (batch_size, 3)   # softmax概率
```

### 元特征格式（Stacking）

**不含Informer-2**：
```python
meta_features: (batch_size, 20)
# 包含：
# - lgb_proba (3) + xgb_proba (3) + cat_proba (3)
# - agreement (1) + max_prob (3) + entropy (3)
# - avg_proba (3) + prob_std_max (1)
```

**含Informer-2**：
```python
meta_features: (batch_size, 23)
# 额外增加：
# - inf_proba (3) + inf_max_prob (1) + inf_entropy (1)
```

## 错误处理

### 1. Informer-2训练失败

**错误类型**：维度错误、GPU内存不足

**处理策略**：
```python
try:
    inf_model = self._train_informer2(X_train, y_train, timeframe)
except Exception as e:
    logger.error(f"Informer-2训练失败: {e}")
    inf_model = None  # 降级到三模型集成
```

### 2. GPU不可用

**处理策略**：
```python
if not torch.cuda.is_available():
    logger.warning("GPU不可用，降级到CPU训练")
    device = torch.device('cpu')
```

### 3. Optuna优化超时

**处理策略**：
```python
try:
    study.optimize(objective, n_trials=100, timeout=1800)
except KeyboardInterrupt:
    logger.warning("优化被用户中断")
# 使用已完成的试验中的最佳参数
best_params = study.best_params
```

## 测试策略

### 1. 单元测试

- 测试简化Informer-2的forward方法
- 测试GMADL损失函数的梯度计算
- 测试GPU设备分配逻辑

### 2. 集成测试

- 测试完整的训练流程（数据加载→训练→评估）
- 测试Stacking集成（基础模型→元特征→元学习器）
- 测试Optuna优化流程

### 3. 性能测试

- 对比简化前后的训练时间
- 对比GPU加速前后的训练时间
- 对比不同超参数配置的准确率

### 4. 回归测试

- 确保修复后准确率不低于当前水平（47%）
- 确保三模型集成仍然正常工作
- 确保元学习器正常组合预测

## 性能优化

### 1. 训练速度优化

- ✅ 使用GPU加速（LightGBM/XGBoost/CatBoost/Informer-2）
- ✅ 使用批处理（batch_size=256）
- ✅ 减少Informer-2训练轮数（50轮）
- ✅ 使用早停（如果验证损失不再下降）

### 2. 内存优化

- ✅ 使用float32而非float64
- ✅ 及时释放中间变量
- ✅ 使用梯度累积（如果GPU内存不足）

### 3. 准确率优化

- ✅ 使用GMADL损失函数（关注难分样本）
- ✅ 使用HOLD惩罚（减少过度预测HOLD）
- ✅ 使用Optuna自动优化超参数
- ✅ 使用Stacking集成（组合多个模型）

## 部署考虑

### 1. 模型保存

```python
# 保存Informer-2模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {...}
}, f'models/{symbol}_{timeframe}_informer2.pth')
```

### 2. 模型加载

```python
# 加载Informer-2模型
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 3. 推理优化

```python
# 使用torch.no_grad()加速推理
with torch.no_grad():
    logits = model(X_tensor)
    probs = F.softmax(logits, dim=-1)
```

## 监控指标

### 1. 训练指标

- 训练损失（GMADL Loss）
- 验证准确率
- 各类别的精确率/召回率/F1
- HOLD信号占比

### 2. 性能指标

- 训练时间（总时间、每轮时间）
- GPU利用率
- 内存使用量

### 3. 业务指标

- 交易信号准确率
- 年化收益率
- 夏普比率
- 最大回撤
