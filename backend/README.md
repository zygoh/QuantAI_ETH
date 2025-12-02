# QuantAI-ETH 量化交易系统

基于机器学习的以太坊合约中频智能交易系统，采用Stacking集成学习策略，支持多时间框架信号生成和自动化交易。

## 📋 目录

- [系统特性](#系统特性)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [API文档](#api文档)
- [开发规范](#开发规范)
- [常见问题](#常见问题)

## ✨ 系统特性

### 核心功能

- **多交易所支持** - 支持Binance和OKX交易所，可灵活切换 🆕
- **多时间框架分析** - 支持3m/5m/15m多周期信号合成
- **Stacking集成学习** - LightGBM + XGBoost + CatBoost + Informer2四模型融合
- **实时信号生成** - WebSocket实时数据处理，毫秒级响应
- **智能风险控制** - 动态止损止盈、回撤监控、仓位管理
- **虚拟/实盘模式** - 支持信号模式和自动交易模式切换
- **自动化训练** - 定时模型训练和超参数优化

### 技术亮点

- **生产级代码质量** - 严格遵循专业开发规范
- **模块化架构** - 清晰的职责划分和依赖管理
- **高性能设计** - 异步处理、缓存优化、GPU加速
- **完整监控体系** - 健康检查、性能监控、日志追踪
- **数据持久化** - PostgreSQL + TimescaleDB时序数据库

## 🏗️ 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
├─────────────────────────────────────────────────────────────┤
│  API Layer          │  WebSocket Layer  │  Middleware       │
├─────────────────────────────────────────────────────────────┤
│                     Trading Controller                       │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ Signal Gen   │ Trading Eng  │ Position Mgr │ Risk Service  │
├──────────────┴──────────────┴──────────────┴───────────────┤
│                      ML Model Layer                          │
│  Ensemble ML Service (Stacking)                             │
│  ├─ LightGBM  ├─ XGBoost  ├─ CatBoost  ├─ Informer2       │
├─────────────────────────────────────────────────────────────┤
│  Data Layer  │  Feature Eng │  Cache      │  Database      │
├─────────────────────────────────────────────────────────────┤
│              Binance Exchange (WebSocket + REST)            │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

**后端框架**
- Python 3.12+
- FastAPI - 高性能异步Web框架
- Uvicorn - ASGI服务器

**机器学习**
- LightGBM - 梯度提升决策树
- XGBoost - 极端梯度提升
- CatBoost - 类别特征优化
- PyTorch - 深度学习框架（Informer2）
- Scikit-learn - 机器学习工具库

**数据存储**
- PostgreSQL 14+ - 关系型数据库
- TimescaleDB - 时序数据扩展
- Redis - 缓存和消息队列

**数据处理**
- Pandas - 数据分析
- NumPy - 数值计算
- TA-Lib - 技术指标库

**交易接口**
- Binance Futures API - 合约交易
- WebSocket - 实时数据流

## 📁 项目结构

```
QuantAI-ETH/
├── app/                          # 应用主目录
│   ├── api/                      # API接口层
│   │   ├── endpoints/            # API端点
│   │   │   ├── account.py        # 账户管理
│   │   │   ├── positions.py      # 持仓查询
│   │   │   ├── signals.py        # 信号管理
│   │   │   ├── trading.py        # 交易操作
│   │   │   ├── training.py       # 模型训练
│   │   │   ├── performance.py    # 性能分析
│   │   │   ├── system.py         # 系统控制
│   │   │   └── websocket.py      # WebSocket推送
│   │   ├── dependencies.py       # 依赖注入
│   │   ├── middleware.py         # 中间件
│   │   ├── models.py             # API数据模型
│   │   └── routes.py             # 路由配置
│   │
│   ├── core/                     # 核心配置
│   │   ├── config.py             # 系统配置
│   │   ├── database.py           # 数据库连接
│   │   └── cache.py              # 缓存管理
│   │
│   ├── exchange/                 # 交易所接口 🆕
│   │   ├── base_exchange_client.py  # 统一接口定义
│   │   ├── exchange_factory.py      # 工厂模式管理
│   │   ├── binance_client.py        # Binance客户端
│   │   ├── okx_client.py            # OKX客户端
│   │   ├── mock_client.py           # Mock客户端（测试）
│   │   ├── exceptions.py            # 统一异常类
│   │   └── mappers.py               # 数据格式映射
│   │
│   ├── model/                    # ML模型模块 ⭐
│   │   ├── ml_service.py         # 基础ML服务
│   │   ├── ensemble_ml_service.py # 集成学习服务
│   │   ├── feature_engineering.py # 特征工程
│   │   ├── hyperparameter_optimizer.py # 超参数优化
│   │   ├── informer2_model.py    # Informer2深度学习模型
│   │   ├── gmadl_loss.py         # GMADL损失函数
│   │   ├── model_stability_enhancer.py # 模型稳定性增强
│   │
│   ├── services/                 # 业务服务层
│   │   ├── data_service.py       # 数据服务
│   │   ├── risk_service.py       # 风险管理
│   │   ├── health_monitor.py     # 健康监控
│   │   ├── drawdown_monitor.py   # 回撤监控
│   │   ├── scheduler.py          # 任务调度
│   │   ├── historical_data.py    # 历史数据管理
│   │   ├── adaptive_frequency_controller.py # 频率控制
│   │   └── direction_consistency_checker.py # 方向一致性
│   │
│   ├── trading/                  # 交易模块 ⭐
│   │   ├── signal_generator.py   # 信号生成器
│   │   ├── position_manager.py   # 仓位管理器
│   │   ├── trading_engine.py     # 交易执行引擎
│   │   ├── trading_controller.py # 交易控制器
│   │
│   └── utils/                    # 工具函数
│       └── helpers.py            # 辅助函数
│
├── models/                       # 模型文件存储 ⭐
│   ├── lgb_3m_v1.pkl            # LightGBM 3分钟模型
│   ├── lgb_5m_v1.pkl            # LightGBM 5分钟模型
│   ├── lgb_15m_v1.pkl           # LightGBM 15分钟模型
│   └── README.md                 # 模型说明
│
├── logs/                         # 日志文件
│   └── trading_system.log       # 系统日志
│
├── scripts/                      # 自动化脚本
│   ├── refactor_project.ps1     # 项目重构脚本
│   └── complete_ml_migration.py # ML模块迁移脚本
│
├── .cursor/                      # 开发规范
│   └── rules/
│       └── general.mdc          # 代码质量规范
│
├── main.py                       # 应用入口
├── requirements.txt              # Python依赖
├── init_timescaledb.sql         # 数据库初始化脚本
├── README.md                     # 项目说明（本文件）
├── REFACTORING.md               # 重构文档
└── PROJECT_RESTRUCTURE_FINAL.md # 重构总结
```

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PostgreSQL 14+ (with TimescaleDB extension)
- Redis 6+
- Windows 11 (推荐) / Linux / macOS

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd QuantAI-ETH
```

2. **创建虚拟环境**

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置数据库**

```sql
-- 创建数据库
CREATE DATABASE quantai_eth;

-- 启用TimescaleDB扩展
\c quantai_eth
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 执行初始化脚本
\i init_timescaledb.sql
```

5. **配置环境变量**

创建 `.env` 文件：

```env
# 交易所选择 🆕
EXCHANGE_TYPE=BINANCE  # 支持: BINANCE, OKX, MOCK

# Binance API配置
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
BINANCE_TESTNET=True

# OKX API配置 🆕
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_PASSPHRASE=your_okx_passphrase
OKX_TESTNET=False

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/quantai_eth

# Redis配置
REDIS_URL=redis://localhost:6379/0

# 系统配置
SYMBOL=ETHUSDT
TIMEFRAMES=3m,5m,15m
LEVERAGE=10
TRADING_MODE=SIGNAL_ONLY
```

6. **启动系统**

```bash
python main.py
```

系统将在 `http://localhost:8000` 启动。

### 验证安装

访问以下端点验证系统运行：

- 健康检查: `http://localhost:8000/health`
- API文档: `http://localhost:8000/docs`
- 系统状态: `http://localhost:8000/api/system/status`

## ⚙️ 配置说明

### 核心配置 (app/core/config.py)

```python
# 交易配置
SYMBOL = "ETHUSDT"              # 交易对
TIMEFRAMES = ["3m", "5m", "15m"] # 时间框架
LEVERAGE = 50                    # 杠杆倍数
CONFIDENCE_THRESHOLD = 0.35      # 信号置信度阈值

# 交易模式
TRADING_MODE = "SIGNAL_ONLY"     # SIGNAL_ONLY: 仅信号 | AUTO: 自动交易

# 风险控制
MAX_POSITION_SIZE = 1000         # 最大持仓数量
MAX_DAILY_TRADES = 50            # 每日最大交易次数
STOP_LOSS_PCT = 0.015           # 止损百分比 (1.5%)
TAKE_PROFIT_PCT = 0.04          # 止盈百分比 (4%)
```

### WebSocket连接配置

```python
# 重连策略
WS_RECONNECT_INITIAL_DELAY = 1.0    # 初始重连延迟（秒）
WS_RECONNECT_MAX_DELAY = 60.0       # 最大重连延迟（秒）
WS_RECONNECT_BACKOFF_FACTOR = 2.0   # 指数退避因子
WS_RECONNECT_MAX_RETRIES = 10       # 最大重试次数

# 心跳保活
WS_PING_INTERVAL = 30               # Ping间隔（秒）
WS_PONG_TIMEOUT = 10                # Pong超时（秒）
WS_SSL_TIMEOUT = 30                 # SSL握手超时（秒）
WS_MESSAGE_TIMEOUT = 1200           # 消息超时（秒，20分钟）
```

### 模型训练配置

```python
# GradScaler配置（混合精度训练）
GRAD_SCALER_GROWTH_FACTOR = 1.2         # 缩放增长因子（保守）
GRAD_SCALER_GROWTH_INTERVAL = 2000      # 缩放增长间隔
GRAD_SCALER_MAX_SCALE = 100000.0        # 最大scale阈值
GRAD_SCALER_AUTO_RESET = True           # 启用自动重置
GRAD_SCALER_RESET_THRESHOLD_EPOCHS = 3  # 重置阈值epoch数
GRAD_SCALER_MAX_CONSECUTIVE_OVERFLOW = 5 # 最大连续溢出次数

# LightGBM参数
LGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}

# 训练配置
TRAINING_DAYS = {
    '3m': 120,   # 120天训练数据
    '5m': 120,
    '15m': 120
}
```

### 配置说明

**WebSocket重连策略**:
- 采用指数退避算法，避免频繁重连
- 重连延迟: 1秒 → 2秒 → 4秒 → 8秒 → 16秒 → 32秒 → 60秒
- 心跳保活机制防止长时间无数据导致断连
- SSL配置优化，禁用旧协议，增强安全性

**GradScaler配置**:
- 动态初始化：根据模型规模自动选择初始scale
- 保守增长策略：防止scale值无限增长
- 自动监控和重置：检测异常自动恢复
- 适用于Informer2等深度学习模型的混合精度训练

## 📖 使用指南

### 启动系统

```bash
# 启动主服务
python main.py

# 系统会自动启动以下服务：
# - FastAPI Web服务器
# - WebSocket数据流
# - ML模型服务
# - 信号生成器
# - 交易引擎
# - 监控服务
```

### 模型训练

**通过API触发训练**

```bash
# 训练所有时间框架的模型
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": false}'

# 强制重新训练
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true}'
```

**自动定时训练**

系统会在每天 00:00 自动触发模型训练（由scheduler管理）。

### 交易模式切换

**信号模式（默认）**

```bash
# 切换到信号模式（仅生成信号，不执行实盘交易）
curl -X POST http://localhost:8000/api/trading/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "SIGNAL_ONLY"}'
```

**自动交易模式**

```bash
# 切换到自动交易模式（执行实盘交易）
curl -X POST http://localhost:8000/api/trading/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "AUTO"}'
```

### 查看信号

```bash
# 获取最新信号
curl http://localhost:8000/api/signals/latest

# 获取信号列表
curl http://localhost:8000/api/signals/?limit=20

# 获取信号性能
curl http://localhost:8000/api/signals/performance?days=7

# 获取信号统计
curl http://localhost:8000/api/signals/statistics?days=30
```

### 查看持仓

```bash
# 获取当前持仓
curl http://localhost:8000/api/positions/current

# 获取持仓摘要
curl http://localhost:8000/api/positions/summary
```

### 性能分析

```bash
# 获取交易性能
curl http://localhost:8000/api/performance/trading

# 获取信号性能
curl http://localhost:8000/api/performance/signals
```

## 🔌 API文档

### 主要端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/system/status` | GET | 系统状态 |
| `/api/system/control` | POST | 控制系统（START/STOP/PAUSE/RESUME） |
| `/api/trading/mode` | POST | 切换交易模式 |
| `/api/signals/` | GET | 获取信号列表 |
| `/api/signals/latest` | GET | 最新信号 |
| `/api/signals/generate` | POST | 强制生成信号 |
| `/api/signals/performance` | GET | 信号性能 |
| `/api/signals/statistics` | GET | 信号统计 |
| `/api/trading/manual` | POST | 手动交易 |
| `/api/positions/current` | GET | 当前持仓 |
| `/api/positions/summary` | GET | 持仓摘要 |
| `/api/training/start` | POST | 开始训练模型 |
| `/api/training/status` | GET | 训练状态 |
| `/api/training/metrics` | GET | 训练指标 |
| `/api/training/history` | GET | 训练历史 |
| `/api/performance/trading` | GET | 交易性能 |
| `/api/performance/signals` | GET | 信号性能 |

### WebSocket端点

```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8000/api/ws');

// 接收实时信号
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Signal:', data);
};
```

完整API文档: `http://localhost:8000/docs`

## 📐 开发规范

### 代码质量标准

本项目严格遵循 `.cursor/rules/general.mdc` 中定义的开发规范：

- **零容忍低级错误** - 无重复定义、无未定义变量、无逻辑错误
- **完整类型提示** - 所有函数参数和返回值必须有类型注解
- **全面错误处理** - 所有I/O操作必须有try-except
- **统一导入管理** - 所有导入必须在文件顶部
- **一致命名规范** - snake_case函数、PascalCase类、UPPERCASE常量
- **4空格缩进** - 严格的Python缩进规范

### 模块职责

- **app/exchange/** - 交易所API接口封装
- **app/model/** - ML模型训练、预测、特征工程
- **app/services/** - 数据服务、风险管理、监控
- **app/trading/** - 信号生成、仓位管理、订单执行
- **app/api/** - REST API和WebSocket接口

### 提交规范

```bash
# 功能开发
git commit -m "feat: 添加新功能描述"

# Bug修复
git commit -m "fix: 修复问题描述"

# 文档更新
git commit -m "docs: 更新文档说明"

# 代码重构
git commit -m "refactor: 重构模块说明"
```

## 🧪 测试

```bash
# 运行单元测试
pytest tests/

# 运行特定测试
pytest tests/test_ml_service.py

# 生成覆盖率报告
pytest --cov=app tests/
```

## 📊 监控和日志

### 日志位置

- 系统日志: `logs/trading_system.log`
- 日志轮转: 单文件最大10MB，保留5个备份

### 日志级别

```python
# 配置日志级别
LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR
```

### 监控指标

- 系统健康状态
- 模型准确率
- 信号胜率
- 交易盈亏
- API响应时间
- 数据库连接状态

## ❓ 常见问题

### Q: 如何切换测试网和主网？

A: 修改 `.env` 文件中的 `BINANCE_TESTNET` 配置：
```env
BINANCE_TESTNET=True   # 测试网
BINANCE_TESTNET=False  # 主网（谨慎使用）
```

### Q: 模型文件存储在哪里？

A: 训练好的模型文件存储在项目根目录的 `models/` 文件夹，命名格式为 `{model_type}_{timeframe}_v{version}.pkl`

### Q: 如何添加新的时间框架？

A: 修改 `app/core/config.py` 中的 `TIMEFRAMES` 配置，然后重新训练模型。

### Q: 系统支持哪些交易所？

A: 当前支持以下交易所：
- **Binance Futures** - 币安合约交易
- **OKX Futures** - OKX合约交易 🆕
- **Mock模式** - 用于测试的模拟交易所

通过修改 `.env` 文件中的 `EXCHANGE_TYPE` 配置即可切换。详见 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)。

### Q: 如何切换交易所？

A: 修改 `.env` 文件中的 `EXCHANGE_TYPE` 配置：
```env
EXCHANGE_TYPE=OKX  # 切换到OKX
```
然后重启系统。系统会自动使用新的交易所，无需修改代码。

### Q: 如何备份和恢复模型？

A: 模型文件位于 `models/` 目录，直接复制该目录即可备份。恢复时将文件放回即可。

### Q: 虚拟交易和实盘交易有什么区别？

A: 
- **SIGNAL_ONLY模式**: 仅生成信号并记录虚拟交易，不执行实盘订单
- **AUTO模式**: 自动执行实盘交易订单


## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

- 项目维护者: [Your Name]
- Email: [your.email@example.com]
- 项目地址: [GitHub Repository URL]

## 🙏 致谢

- Binance API
- FastAPI Framework
- LightGBM, XGBoost, CatBoost
- TimescaleDB
- 所有开源贡献者

---

**版本**: 2.0.0  
**最后更新**: 2025-01-15  
**状态**: 生产就绪 ✅
