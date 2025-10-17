# QuantAI-ETH

**ETH量化智能交易系统** - Quantitative AI for Ethereum Trading

<div align="center">

[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](PROJECT_NAME.md)
[![Python](https://img.shields.io/badge/python-3.12+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-yellow.svg)](https://lightgbm.readthedocs.io)

**智能 • 自适应 • 盈利**

</div>

---

## 📋 项目简介

**QuantAI-ETH** 是一个生产级的以太坊合约量化交易系统，使用**多时间框架机器学习模型**进行中频智能交易。

### 核心特性

- 🤖 **机器学习驱动** - LightGBM多时间框架独立模型
- ⏰ **多时间框架** - 15m/2h/4h独立训练与信号合成
- 🎯 **中频交易** - 日均5-10个高质量信号
- 🛡️ **智能风险管理** - 动态ATR止损，全仓策略
- 📊 **实时数据流** - WebSocket实时推送 + Binance API
- 🔧 **工业级代码** - 两阶段特征选择，样本加权训练

---

## 📊 性能指标

### 当前状态（v2.0）

| 指标 | 当前值 | Stacking后（预期） | 基本要求 |
|------|--------|-------------------|---------|
| **模型准确率** | 42.81% | **48-51%** | **≥50%** |
| **特征数量** | 186个 | 186个 | - |
| **模型架构** | 单模型 | **3模型Stacking** | - |
| **HOLD比例** | 36-43% ✅ | 36-43% | 28-45% |
| **日均信号** | 待验证 | 5-10个 | - |

### 性能基准（三分类）

- 随机猜测: 33.3%
- **基本可用**: **50%** ← 最低要求
- 良好: 55%
- 优秀: 60%+

---

## 🏗️ 技术架构

### 技术栈

**后端**:
- Python 3.12+
- FastAPI（异步Web框架）
- LightGBM（机器学习）
- PostgreSQL + TimescaleDB（时序数据库）
- Redis（缓存）

**前端**:
- React 18
- TypeScript
- Tailwind CSS

**数据源**:
- Binance Futures API（唯一可信数据源）
- WebSocket实时推送

---

### 核心模块

```
backend/
├── app/
│   ├── services/          # 核心业务
│   │   ├── ml_service.py           # 机器学习（模型训练/预测）
│   │   ├── signal_generator.py    # 信号生成（多时间框架合成）
│   │   ├── trading_engine.py      # 交易执行
│   │   ├── position_manager.py    # 仓位管理（全仓+动态）
│   │   ├── risk_service.py        # 风险管理（动态ATR止损）
│   │   ├── data_service.py        # WebSocket数据流
│   │   └── feature_engineering.py # 特征工程（195个特征）
│   ├── api/               # REST API
│   └── core/              # 配置/数据库
└── models/                # 训练好的模型
```

---

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PostgreSQL 14+（带TimescaleDB扩展）
- Redis 7+
- GPU（可选，加速训练）

### 安装

```bash
# 1. 克隆项目
cd F:\AI\20251007

# 2. 安装依赖
cd backend
pip install -r requirements.txt

# 3. 配置环境变量
# 编辑 .env 文件，填入Binance API密钥

# 4. 初始化数据库
# PostgreSQL会自动初始化

# 5. 启动系统
python main.py
```

### 首次启动

系统会自动：
1. 初始化WebSocket连接
2. 下载60天历史数据作为缓冲区
3. 训练3个时间框架的模型（约30-60秒）
4. 开始实时信号生成

---

## 🎯 核心功能

### 1. 多时间框架模型

**独立训练**：
- 15m模型（短期）
- 2h模型（中期）
- 4h模型（长期）

**智能合成**：
```python
weights = {
    '15m': 0.60,  # 主导
    '2h': 0.25,   # 辅助
    '4h': 0.15    # 确认
}
final_signal = weighted_synthesis(predictions, weights)
```

---

### 2. 智能特征工程

**195个高质量特征**：
- 基础特征: 50个
- 技术指标派生: 99个
- 市场微观结构: 31个 🆕
- 市场情绪: 15个 🆕

**两阶段智能选择**：
- 阶段1: Filter过滤低重要性特征
- 阶段2: 嵌入式选择（动态预算）
- 结果: 15m用150个，2h用38个，4h用39个

---

### 3. 样本加权训练

```python
# 解决类别不平衡 + 重视最新数据
class_weights = compute_sample_weight('balanced', y)
time_decay = exp(-arange(N) / (N * 0.1))[::-1]
sample_weights = class_weights * time_decay
```

---

### 4. 动态ATR止损

**基于市场波动自适应**：
- 止损: 1.5×ATR
- 止盈: 3-4×ATR（根据置信度）
- 盈亏比: 1.8:1 - 2.67:1
- 跟踪止损: 1×ATR距离

---

### 5. 全仓策略

**默认策略**：
```
仓位价值 = 全部可用余额 × 杠杆倍数
示例: 10,000 USDT × 20x = 200,000 USDT
```

**备用**: 动态仓位调整（波动率/持仓/连亏保护）

---

## 📈 优化历史

### Phase 1: 基础优化（已完成）✅

1. ✅ 标签阈值优化（15m: ±0.1%→±0.15%）
2. ✅ 置信度调整（0.5→0.40）
3. ✅ 样本加权训练
4. ✅ 市场微观结构特征（31个）
5. ✅ 市场情绪特征（15个）
6. ✅ 智能两阶段特征选择
7. ✅ Kim工业级改进（5项）
8. ✅ 动态ATR止损
9. ✅ 全仓策略
10. ✅ 代码冗余清理
11. ✅ 降低15m模型复杂度

**实际效果**: 准确率 34% → **42.81%** (+26%)

---

### Phase 2: Stacking模型集成（已实施）✅

1. ✅ **Stacking三模型融合**
   - LightGBM + XGBoost + CatBoost
   - LogisticRegression元学习器
   - 代码已完成，待训练验证

**预期效果**: 准确率 42.81% → **48-51%** (+12-19%)

### Phase 2.4: 超参数优化（待定）

2. ⚪ Optuna超参数自动优化（如需要）

**预期效果**: 准确率 48-51% → **50-55%** ✅

---

## 🎯 性能目标

### 基本要求（必达）

- ✅ 模型准确率: **≥50%**
- ✅ 实际胜率: ≥55%
- ✅ 盈亏比: ≥1.8:1
- ✅ 最大回撤: <10%

### 优秀目标

- ⭐ 模型准确率: ≥55%
- ⭐ 实际胜率: ≥60%
- ⭐ 盈亏比: ≥2.0:1
- ⭐ 夏普比率: ≥2.0

---

## 📚 文档索引

### 核心文档

1. **项目名称**: `PROJECT_NAME.md`
2. **优化路线图**: `OPTIMIZATION_ROADMAP.md`（90KB）
3. **50%达标计划**: `docs/ACCURACY_50_PLAN.md`
4. **部署指南**: `DEPLOYMENT.md`

### 技术文档

1. **智能特征选择**: `docs/INTELLIGENT_FEATURE_SELECTION.md`
2. **Kim建议实施**: `docs/KIM_SUGGESTIONS_IMPLEMENTED.md`
3. **代码审计**: `docs/COMPLETE_CODE_AUDIT.md`
4. **仓位策略**: `docs/POSITION_STRATEGY.md`

### 规则文档

1. **项目规则**: `.cursor/rules/general.mdc` (v6.0)
2. **后端规则**: `backend/.cursor/rules/backend.mdc` (v5.0)
3. **前端规则**: `frontend/.cursor/rules/frontend.mdc`

---

## 🔐 安全说明

### 数据源原则

**唯一可信数据源**: Binance API

- ✅ 训练数据: `binance_client.get_klines()`
- ✅ 预测数据: WebSocket缓冲区（Binance推送）
- ✅ ATR计算: `binance_client.get_klines()`
- ❌ 数据库: **仅用于前端展示和缓存**

### 交易模式

- **SIGNAL_ONLY**（默认）: 虚拟交易，无资金风险
- **AUTO**: 实盘交易，需谨慎

---

## ⚙️ 配置

### 核心配置（config.py）

```python
# 交易配置
SYMBOL = "ETHUSDT"
LEVERAGE = 20  # 杠杆倍数
CONFIDENCE_THRESHOLD = 0.40  # 最低置信度
TRADING_MODE = "SIGNAL_ONLY"  # 交易模式

# 时间框架
TIMEFRAMES = ["15m", "2h", "4h"]

# 性能目标
TARGET_ACCURACY = 0.50  # 基本要求：50%
```

---

## 📊 当前进度

### Phase 1 优化进度

- [x] ✅ 标签阈值修复
- [x] ✅ 智能特征选择（195个→150/38/39个）
- [x] ✅ 样本加权训练
- [x] ✅ Kim工业级改进
- [x] ✅ 动态风险管理
- [x] ✅ 代码质量优化
- [ ] 🔄 验证准确率达到40%+
- [ ] 🔄 实战测试（3-7天）

### Phase 2 准备中

- [ ] ⚪ 模型集成
- [ ] ⚪ 超参数优化  
- [ ] ⚪ 达到50%准确率目标

---

## 🎯 下一步

### 立即执行

```bash
# 1. 重启系统验证所有优化
cd backend
python main.py

# 2. 观察训练日志
# 期待：准确率40-45%

# 3. 等待首个信号（约30分钟）
# 验证：动态止损、全仓策略

# 4. 运行虚拟交易3-7天
# 收集：实际信号表现数据
```

### 根据结果决定

**If 准确率≥42%**:
- 观察实战表现
- 如表现良好 → Phase 2
- 目标：50%+

**If 准确率<42%**:
- 立即Phase 2（模型集成）
- 快速达到50%

---

## 🙏 致谢

特别感谢：
- **Kim** - 提供工业级优化建议
- **项目规则** - 严格的代码质量要求
- **Binance API** - 可靠的数据源

---

## 📄 许可

内部项目，未开源

---

## 📞 联系方式

项目维护者: QuantAI-ETH Team

---

**项目名称**: QuantAI-ETH  
**版本**: v2.0  
**状态**: Phase 1完成，Phase 2准备中  
**性能目标**: 准确率≥50%（基本要求）

---

<div align="center">

**Built with ❤️ for Ethereum Trading**

</div>

