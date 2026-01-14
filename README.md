# QuantAI - 量化AI交易系统

<p align="center">
  <strong>基于机器学习的合约中频智能交易系统</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.5+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.1-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-Proprietary-yellow.svg" alt="License">
</p>

---

## 📖 项目概述

QuantAI 是一个生产级的量化交易系统，采用 **Stacking 集成学习** 策略，结合多个机器学习模型（LightGBM、XGBoost、CatBoost）和深度学习模型（Informer-2），实现高精度的交易信号生成。

### 核心特性

- 🚀 **四模型集成**: LightGBM + XGBoost + CatBoost + Meta Learner
- 🧠 **深度学习增强**: Informer-2 时序预测模型
- ⚡ **GPU 加速**: 完整支持 CUDA 12.1，训练速度提升 4x
- 📊 **14 维特征工程**: 价格、成交量、动量、波动率等多维度特征
- 🔄 **多时间框架**: 3m/5m/15m 多周期信号合成
- 🛡️ **风险管理**: 实时止盈止损、回撤监控、Kelly 仓位
- 🌐 **多交易所支持**: Binance、OKX 统一接口

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         QuantAI System                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   FastAPI   │  │  WebSocket  │  │    REST API Endpoints   │  │
│  │   Server    │  │   Server    │  │  /signals /trading ...  │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
│         │                │                      │               │
│  ┌──────┴────────────────┴──────────────────────┴──────┐        │
│  │                   Trading Layer                      │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │        │
│  │  │SignalGenerat│  │TradingEngine │  │PositionMgr │ │        │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │        │
│  └─────────┴─────────────────┴────────────────┴────────┘        │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                    Model Layer                         │      │
│  │  ┌────────────────────────────────────────────────┐   │      │
│  │  │           Ensemble ML Service                   │   │      │
│  │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌──────┐  ┌─────┐  │   │      │
│  │  │  │ LGB │  │ XGB │  │ CAT │  │ META │  │ INF │  │   │      │
│  │  │  └─────┘  └─────┘  └─────┘  └──────┘  └─────┘  │   │      │
│  │  └────────────────────────────────────────────────┘   │      │
│  │  ┌────────────────────────────────────────────────┐   │      │
│  │  │           Feature Engineering                   │   │      │
│  │  │  14 Feature Modules (Price, Volume, Momentum..)│   │      │
│  │  └────────────────────────────────────────────────┘   │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                  Exchange Layer                        │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐│      │
│  │  │   Binance   │  │     OKX     │  │   SymbolMapper  ││      │
│  │  │   Client    │  │   Client    │  │ IntervalMapper  ││      │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘│      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                  Data Layer                            │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐│      │
│  │  │ PostgreSQL  │  │    Redis    │  │   TimescaleDB   ││      │
│  │  │  (Storage)  │  │   (Cache)   │  │  (Time Series)  ││      │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘│      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
backend/
├── main.py                    # 应用入口
├── requirements.txt           # 依赖包
├── init_timescaledb.sql      # 数据库初始化脚本
│
├── app/
│   ├── api/                   # API 层
│   │   ├── routes.py          # 路由注册
│   │   ├── middleware.py      # 中间件
│   │   ├── models.py          # API 数据模型
│   │   └── endpoints/         # API 端点
│   │       ├── signals.py     # 信号接口
│   │       ├── trading.py     # 交易接口
│   │       ├── training.py    # 训练接口
│   │       ├── positions.py   # 持仓接口
│   │       ├── performance.py # 绩效接口
│   │       ├── system.py      # 系统接口
│   │       └── websocket.py   # WebSocket 接口
│   │
│   ├── core/                  # 核心配置
│   │   ├── config.py          # 系统配置
│   │   ├── database.py        # 数据库连接
│   │   ├── cache.py           # 缓存管理
│   │   └── constants.py       # 常量定义
│   │
│   ├── model/                 # 模型层 (Layer 2)
│   │   ├── base/              # 基础 ML 服务
│   │   │   ├── ml_service.py  # ML 服务基类
│   │   │   └── utils.py       # 工具函数
│   │   │
│   │   ├── ensemble/          # 集成学习
│   │   │   ├── trainers.py    # 模型训练器
│   │   │   ├── predictors.py  # 模型预测器
│   │   │   ├── model_managers.py  # 模型管理
│   │   │   ├── informer_wrapper.py  # Informer 封装
│   │   │   └── utils.py       # 集成工具
│   │   │
│   │   ├── optimizers/        # 超参数优化
│   │   │   └── hyperparameter_optimizer.py
│   │   │
│   │   ├── features/          # 特征工程层 (Layer 1)
│   │   │   ├── price_features.py      # 价格特征
│   │   │   ├── volume_features.py     # 成交量特征
│   │   │   ├── momentum_features.py   # 动量特征
│   │   │   ├── volatility_features.py # 波动率特征
│   │   │   ├── trend_features.py      # 趋势特征
│   │   │   ├── pattern_features.py    # 形态特征
│   │   │   ├── technical_indicators.py# 技术指标
│   │   │   ├── time_features.py       # 时间特征
│   │   │   ├── swing_features.py      # 摆动特征
│   │   │   ├── order_flow_features.py # 订单流特征
│   │   │   ├── microstructure_features.py  # 微观结构
│   │   │   ├── sentiment_features.py  # 情绪特征
│   │   │   ├── multi_timeframe_features.py # 多周期特征
│   │   │   └── utils.py               # 特征工具
│   │   │
│   │   ├── ensemble_ml_service.py  # 集成 ML 服务入口
│   │   ├── informer2_model.py      # Informer-2 模型
│   │   ├── gmadl_loss.py           # GMADL 损失函数
│   │   └── model_stability_enhancer.py  # 模型稳定性
│   │
│   ├── trading/               # 交易层 (Layer 3)
│   │   ├── trading_engine.py      # 交易引擎
│   │   ├── signal_generator.py    # 信号生成器
│   │   ├── trading_controller.py  # 交易控制器
│   │   └── position_manager.py    # 仓位管理器
│   │
│   ├── services/              # 服务层 (Layer 3)
│   │   ├── data_service.py        # 数据服务
│   │   ├── risk_service.py        # 风险服务
│   │   ├── scheduler.py           # 任务调度
│   │   ├── health_monitor.py      # 健康监控
│   │   ├── drawdown_monitor.py    # 回撤监控
│   │   ├── historical_data.py     # 历史数据
│   │   ├── direction_consistency_checker.py  # 方向一致性
│   │   └── adaptive_frequency_controller.py  # 自适应频率
│   │
│   ├── exchange/              # 交易所层
│   │   ├── base_exchange_client.py  # 基础客户端
│   │   ├── exchange_factory.py      # 工厂模式
│   │   ├── mappers.py               # 格式映射器
│   │   ├── exceptions.py            # 异常定义
│   │   └── clients/                 # 交易所客户端
│   │       ├── binance/             # Binance
│   │       │   └── binance_client.py
│   │       └── okx/                 # OKX
│   │           └── okx_client.py
│   │
│   └── utils/                 # 工具层
│       └── helpers.py         # 辅助函数
│
├── models/                    # 模型文件存储
└── logs/                      # 日志文件
```

---

## 🚀 快速开始

### 环境要求

- Python 3.12+
- CUDA 12.1+ (GPU 加速)
- PostgreSQL 15+ (TimescaleDB)
- Redis 7+

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd 20251007

# 2. 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r backend/requirements.txt

# 4. 安装 PyTorch GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 配置数据库
psql -U postgres -f backend/init_timescaledb.sql

# 6. 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 文件

# 7. 启动服务
cd backend
python main.py
```

### 验证安装

```bash
# 检查 GPU 支持
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 健康检查
curl http://localhost:8000/health
```

---

## 📊 API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/signals` | GET | 获取交易信号 |
| `/api/signals/predict` | POST | 触发预测 |
| `/api/trading/status` | GET | 交易状态 |
| `/api/trading/mode` | POST | 切换模式 |
| `/api/positions` | GET | 持仓信息 |
| `/api/training/start` | POST | 开始训练 |
| `/api/training/status` | GET | 训练状态 |
| `/api/performance/metrics` | GET | 绩效指标 |
| `/api/system/status` | GET | 系统状态 |
| `/api/ws/market` | WS | 实时行情 |

---

## ⚙️ 配置说明

### 核心配置 (`app/core/config.py`)

```python
# 交易配置
SYMBOL: str = "BTC/USDT"      # 交易对
LEVERAGE: int = 50            # 杠杆倍数
CONFIDENCE_THRESHOLD: float = 0.45  # 信号阈值

# 时间框架
TIMEFRAMES: list = ["3m", "5m", "15m"]

# GPU 配置
USE_GPU: bool = True
GPU_DEVICE: str = "cuda:0"

# 风险管理
MAX_DRAWDOWN_LIMIT: float = 0.15  # 最大回撤 15%
KELLY_MULTIPLIER: float = 0.25   # Kelly 系数
```

---

## 🧠 模型架构

### Stacking 集成学习

```
           ┌─────────────────────────────────────────┐
           │            原始特征 (200+)               │
           └───────────────────┬─────────────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌───────────┐            ┌───────────┐            ┌───────────┐
│  LightGBM │            │  XGBoost  │            │  CatBoost │
│   (lgb)   │            │   (xgb)   │            │   (cat)   │
└─────┬─────┘            └─────┬─────┘            └─────┬─────┘
      │                        │                        │
      │         ┌──────────────┼──────────────┐         │
      │         │              │              │         │
      └─────────┴──────────────┴──────────────┴─────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Meta Learner     │
                    │   (Ridge/Logistic)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Final Signal     │
                    │  (LONG/SHORT/HOLD)  │
                    └─────────────────────┘
```

### 特征工程模块

| 模块 | 特征数 | 描述 |
|------|--------|------|
| price_features | 15+ | 价格变化、收益率 |
| volume_features | 12+ | 成交量分析 |
| momentum_features | 20+ | RSI、MACD、动量 |
| volatility_features | 15+ | ATR、布林带 |
| trend_features | 10+ | 均线、趋势强度 |
| pattern_features | 8+ | K线形态识别 |
| technical_indicators | 25+ | 综合技术指标 |
| time_features | 5+ | 时间周期特征 |
| order_flow_features | 10+ | 买卖压力 |

---

## 📈 性能指标

### 训练性能 (RTX 4060 Ti)

| 指标 | CPU | GPU | 加速比 |
|------|-----|-----|--------|
| Optuna 优化 | 45min | 10min | 4.5x |
| 模型训练 | 60min | 15min | 4.0x |
| 总时间 | 130min | 30min | 4.3x |

### 预测性能

| 指标 | 数值 |
|------|------|
| 预测延迟 | <100ms |
| 信号频率 | 5min |
| 内存占用 | <4GB |

---

## 🛡️ 风险管理

- **实时止盈止损**: 每次 WebSocket 消息触发检查
- **最大回撤限制**: 15% 自动暂停
- **Kelly 仓位管理**: 动态调整仓位大小
- **方向一致性检查**: 防止信号震荡

---

## 📝 开发规范

详见 [.cursor/rules/general.mdc](.cursor/rules/general.mdc)

### 核心原则

1. **Alpha First**: ROI > 稳定性
2. **Signal Fidelity**: 精度 > 优化
3. **Environmental Hygiene**: 零冗余
4. **Architectural Integrity**: 根因 > 补丁

---

## 📄 License

Proprietary - All Rights Reserved

---

## 🤝 贡献

本项目为私有项目，暂不接受外部贡献。

---

<p align="center">
  <strong>QuantAI</strong> - Built for Alpha Generation 🚀
</p>
