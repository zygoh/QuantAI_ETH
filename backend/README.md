# 量化交易系统

基于机器学习的合约中频智能交易系统，采用Stacking集成学习策略，支持多时间框架信号生成和模拟交易。

## ✨ 核心特性

- **动态币种配置** - 只需在config中配置SYMBOL，系统自动适配
- **Binance公共接口** - 仅使用公共API获取市场数据，无需API Key
- **模拟交易** - 完整的虚拟交易系统，支持回测和策略验证
- **Stacking集成学习** - LightGBM + XGBoost + CatBoost + Informer2四模型融合
- **多时间框架** - 支持3m/5m/15m多周期信号合成
- **实时数据处理** - WebSocket实时K线数据，毫秒级响应
- **自动化训练** - 定时模型训练和超参数优化

## 🏗️ 技术架构

```
┌─────────────────────────────────────────┐
│         FastAPI Server                   │
├─────────────────────────────────────────┤
│  API Layer  │  WebSocket  │  Middleware│
├─────────────────────────────────────────┤
│         Trading Controller               │
├──────────┬──────────┬──────────┬─────────┤
│ Signal   │ Trading  │ Position │ Risk   │
│ Generator│ Engine   │ Manager  │ Service │
├──────────┴──────────┴──────────┴─────────┤
│         ML Model Layer                   │
│  Ensemble ML Service (Stacking)          │
│  ├─ LightGBM  ├─ XGBoost                │
│  ├─ CatBoost  ├─ Informer2              │
├─────────────────────────────────────────┤
│  Data │ Feature │ Cache │ Database      │
├─────────────────────────────────────────┤
│  Binance Public API (WebSocket + REST)  │
└─────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PostgreSQL 14+ (with TimescaleDB)
- Redis 6+

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd backend
```

2. **创建虚拟环境**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置数据库**
```sql
CREATE DATABASE trading_data;
\c trading_data
CREATE EXTENSION IF NOT EXISTS timescaledb;
\i init_timescaledb.sql
```

5. **配置系统** (`app/core/config.py`)

```python
# 交易配置（只需修改这里即可切换币种）
SYMBOL: str = "BTC/USDT"  # 支持任意币种：BTC/USDT, ETH/USDT, SOL/USDT等
TIMEFRAMES: list = ["3m", "5m", "15m"]
LEVERAGE: int = 20
TRADING_MODE: str = "SIGNAL_ONLY"  # 模拟交易模式

# 数据库配置
PG_HOST: str = "localhost"
PG_PORT: int = 5432
PG_USER: str = "postgres"
PG_PASSWORD: str = "your_password"
PG_DATABASE: str = "trading_data"

# Redis配置
REDIS_URL: str = "redis://localhost:6379/0"
```

6. **启动系统**
```bash
python main.py
```

系统将在 `http://localhost:8000` 启动。

## ⚙️ 配置说明

### 核心配置

所有配置都在 `app/core/config.py` 中，**只需修改此文件即可**：

```python
# 交易对配置（动态，支持任意币种）
SYMBOL: str = "BTC/USDT"  # 标准格式，系统自动转换为交易所格式

# 时间框架
TIMEFRAMES: list = ["3m", "5m", "15m"]

# 交易模式
TRADING_MODE: str = "SIGNAL_ONLY"  # 模拟交易（默认）

# 机器学习
CONFIDENCE_THRESHOLD: float = 0.35
USE_GPU: bool = True
```

### 币种切换

**只需修改 `SYMBOL` 配置即可**，系统会自动：
- 转换交易对格式（Binance/OKX）
- 更新所有相关引用
- 无需修改任何其他代码

示例：
```python
SYMBOL: str = "ETH/USDT"  # 切换到ETH
SYMBOL: str = "SOL/USDT"  # 切换到SOL
```

## 📖 使用指南

### 模型训练

```bash
# 通过API触发训练
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": false}'
```

系统会在每天 00:01 自动训练模型。

### 查看信号

```bash
# 获取最新信号
curl http://localhost:8000/api/signals/latest

# 获取信号列表
curl http://localhost:8000/api/signals/?limit=20
```

### 查看持仓（模拟）

```bash
# 获取虚拟持仓
curl http://localhost:8000/api/positions/

# 获取持仓摘要
curl http://localhost:8000/api/positions/summary
```

## 🔌 API文档

主要端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/system/status` | GET | 系统状态 |
| `/api/signals/latest` | GET | 最新信号 |
| `/api/signals/` | GET | 信号列表 |
| `/api/training/start` | POST | 开始训练 |
| `/api/training/status` | GET | 训练状态 |
| `/api/positions/` | GET | 虚拟持仓 |
| `/api/performance/trading` | GET | 交易性能 |

完整API文档: `http://localhost:8000/docs`

## 🎯 系统特点

### 简洁架构

- **单一配置源** - 所有配置集中在 `config.py`
- **动态币种** - 无需修改代码，只需改配置
- **公共接口** - 仅使用Binance公共API，无需API Key
- **模拟交易** - 完整的虚拟交易系统

### 核心功能

1. **数据获取** - 从Binance公共接口获取K线数据
2. **特征工程** - 自动生成技术指标和特征
3. **模型训练** - Stacking集成学习，多模型融合
4. **信号生成** - 多时间框架信号合成
5. **模拟交易** - 虚拟交易执行和持仓管理

## 📐 开发规范

- **4空格缩进** - 严格Python规范
- **类型提示** - 100%类型覆盖
- **错误处理** - 完整的异常处理
- **代码简洁** - 无冗余代码和注释

## ❓ 常见问题

### Q: 如何切换币种？

A: 只需修改 `app/core/config.py` 中的 `SYMBOL` 配置：
```python
SYMBOL: str = "ETH/USDT"  # 切换到ETH
```
无需修改任何其他代码。

### Q: 需要Binance API Key吗？

A: **不需要**。系统仅使用Binance公共接口获取市场数据，无需API Key。

### Q: 支持实盘交易吗？

A: 当前版本仅支持模拟交易。所有交易都在虚拟环境中执行。

### Q: 如何查看交易历史？

A: 通过 `/api/positions/` 和 `/api/performance/trading` 端点查看虚拟交易记录。

---

**版本**: 2.0.0  
**最后更新**: 2025-01-15  
**状态**: 生产就绪 ✅
