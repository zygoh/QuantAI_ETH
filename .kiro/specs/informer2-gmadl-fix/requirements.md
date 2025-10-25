# 需求文档 - Informer-2 + GMADL 量化模型优化

## 简介

本项目是一个加密货币自动交易系统，使用Stacking集成学习（LightGBM + XGBoost + CatBoost + Informer-2）进行交易信号预测（LONG/HOLD/SHORT）。当前Informer-2神经网络模型在训练时出现维度错误，需要修复并优化整体配置。

## 术语表

- **System**: 量化交易系统（Quantitative Trading System）
- **Informer-2**: 改进版Informer时间序列预测模型，使用ProbSparse自注意力机制
- **GMADL**: Generalized Mean Absolute Deviation Loss，广义平均绝对偏差损失函数
- **Stacking**: 集成学习方法，使用元学习器组合多个基础模型的预测
- **GPU**: 图形处理单元，用于加速深度学习训练
- **Optuna**: 超参数自动优化框架，使用TPE算法
- **ProbSparse Attention**: 概率稀疏注意力机制，Informer的核心创新
- **HOLD惩罚**: 对HOLD类别施加的权重惩罚，减少过度预测HOLD信号
- **TimeSeriesSplit**: 时间序列交叉验证，保持数据时间顺序

## 需求

### 需求 1: 修复Informer-2 + GMADL模型

**用户故事**: 作为量化模型开发工程师，我希望Informer-2模型能够正常训练并使用GMADL损失函数，以便利用深度学习提升交易信号预测准确率

#### 验收标准

1. WHEN System训练Informer-2模型时，THE System SHALL成功完成训练而不出现"max(): Expected reduction dim 3 to have non-zero size"错误
2. WHEN Informer-2模型接收输入数据时，THE System SHALL正确处理单时间点特征向量（batch, n_features）的输入格式
3. WHEN ProbSparse注意力计算采样参数时，THE System SHALL确保sample_k大于0以避免空张量错误
4. THE System SHALL使用GMADL损失函数训练Informer-2模型，并正确应用HOLD惩罚
5. WHEN Informer-2模型训练完成时，THE System SHALL能够生成预测概率用于Stacking集成

### 需求 2: 优化Optuna超参数配置

**用户故事**: 作为量化模型开发工程师，我希望Optuna超参数优化配置合理高效，以便在有限时间内找到最佳模型参数

#### 验收标准

1. THE System SHALL为不同时间框架（15m/2h/4h）配置差异化的超参数搜索空间
2. THE System SHALL使用TimeSeriesSplit进行5折交叉验证以评估超参数组合
3. THE System SHALL在超参数优化中应用与训练一致的HOLD惩罚系数（0.65）
4. THE System SHALL为传统模型（LightGBM/XGBoost/CatBoost）设置合理的试验次数（100次）和超时时间（30分钟）
5. THE System SHALL为Informer-2模型设置适合深度学习的试验次数（50次）和超时时间（20分钟）

### 需求 3: 优化其他模型配置

**用户故事**: 作为量化模型开发工程师，我希望所有模型参数配置合理，以便在不同时间框架下达到最佳性能并防止过拟合

#### 验收标准

1. THE System SHALL根据时间框架（15m/2h/4h）差异化配置模型复杂度（树深度、层数、正则化强度）
2. THE System SHALL确保训练数据量与模型复杂度匹配（15m用360天，2h用540天，4h用720天）
3. THE System SHALL为Informer-2配置合理的学习率（0.001）、批次大小（256）、训练轮数（50）
4. THE System SHALL验证GPU加速配置正确（LightGBM device='gpu'，XGBoost tree_method='gpu_hist'，CatBoost task_type='GPU'）
5. THE System SHALL确保元学习器使用极简配置防止过拟合（n_estimators=50, max_depth=3, num_leaves=7）
