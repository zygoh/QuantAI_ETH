# QuantAI-ETH

> **简介**：**Quantitative AI for Ethereum Trading** — 以太坊量化智能交易系统。FastAPI + 多时间框架信号 + Stacking 集成模型（LGB/XGB/Cat）+ 严格模式（训练/回测/实盘同一路径）。

QuantAI 量化交易系统（Strict Mode）

## 项目简介

本项目是一个基于 **FastAPI** 的量化交易/信号系统，核心能力：

- **多时间框架信号生成**：默认 `["3m", "5m", "15m"]`，以 5m 为主框架合成信号。
- **虚拟交易（默认）**：默认模式为 `SIGNAL_ONLY`，走完整的“信号→下单→成交→仓位更新”流程，但不依赖交易所下单权限。
- **模型体系**：以 Stacking 集成（`lgb/xgb/cat/meta`）为主，支持 GPU 加速（LightGBM/XGBoost/CatBoost/PyTorch）。
- **数据与存储**：WebSocket 实时行情 + PostgreSQL/TimescaleDB（K线/信号/订单/仓位）+ Redis（缓存/状态）。
- **严格模式（Strict Mode）**：训练、回测、实时预测共享同一条特征工程与预测路径，避免回测“特供逻辑”带来的偏差。

## 严格模式原则（必须读）

- **训练 / 回测 / 实盘预测**：必须走同一套特征工程与预测逻辑（禁止分支逻辑）。
- **防未来函数**：特征只允许使用 \(t-1\) 及更早数据（详见项目规则与测试）。
- 关键对比文档：
  - `docs/backtest_vs_realtime_config.md`
  - `docs/config_unification_and_cumulative_backtest.md`

## 重要行为（⚠️ 启动即清理）

启动 `main.py` 时会进行 **系统启动清理**（用于保证“干净启动”）：

- **清空数据库交易相关表**（如 `klines / virtual_positions / orders / trading_signals` 等，含序列重置）
- **清理 Redis 缓存**（多个 `pattern`）
- **重置虚拟账户余额**
- **重置回测累积余额（内存）**

因此：

- **不要把 `PG_* / REDIS_*` 指向你想保留数据的生产库**。
- 回测结果写库后也会在下一次回测/启动时被清理（回测表会被 TRUNCATE）。

## 环境要求

- **Windows 10/11 + PowerShell**
- Python **3.12**（推荐与当前工程一致）
- （可选）NVIDIA GPU（CUDA 环境用于加速训练/预测）
- PostgreSQL（建议启用 TimescaleDB 扩展）+ Redis

## 安装依赖

```powershell
cd "f:\AI\20251007"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### GPU（可选）

`requirements.txt` 已包含 `torch/cupy-cuda12x` 等依赖；如你的环境需要使用官方 CUDA 轮子，可参考 `requirements.txt` 注释里的 PyTorch 安装方式。

## 配置（推荐使用 .env 覆盖）

项目使用 `pydantic-settings` 读取 `.env`，环境变量会覆盖默认值（见 `app/core/config.py`）。

建议在项目根目录创建 `.env`（**不要提交到 git**），示例：

```dotenv
# 服务
HOST=0.0.0.0
PORT=8000

# 交易对/信号
SYMBOL=BTC/USDT
LEVERAGE=20
CONFIDENCE_THRESHOLD=0.6
TIMEFRAMES=["3m","5m","15m"]

# 数据库（⚠️ 请使用测试库/本地库，避免被启动清理误删）
PG_HOST=127.0.0.1
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=your_password
PG_DATABASE=trading-data

# Redis
REDIS_URL=redis://127.0.0.1:6379
REDIS_DB=0

# GPU
USE_GPU=true
GPU_DEVICE=cuda:0

# 代理（可选）
USE_PROXY=false
USE_PROXY_WS=false
PROXY_HOST=127.0.0.1
PROXY_PORT=10808
PROXY_TYPE=socks5
```

## 数据库初始化

- 系统启动会自动初始化表结构（见 `app/core/database.py`、`app/core/database_schema.py`）。
- 也可以手动执行 `init_timescaledb.sql`：

```powershell
psql -U postgres -d trading-data -f .\init_timescaledb.sql
```

## 启动服务

```powershell
python .\main.py
```

启动后可访问：

- `GET /health`（基础健康检查）
- `GET /api/system/info`（环境与配置）

日志默认写入 `.\logs\trading_system.log`。

## API 快速参考

说明：接口依赖 `HTTP Bearer`，但当前鉴权为简化实现（`app/api/dependencies.py`），**Authorization 可不传**。

### 训练

- `POST /api/training/start`：训练模型
  - body：`{"force_retrain": false}`
- `GET /api/training/status`：模型状态
- `GET /api/training/metrics`：训练指标
- `GET /api/training/features`：特征重要性（训练后）
- `GET /api/training/schedule`：调度状态
- `POST /api/training/schedule/run`：手动触发训练任务

### 回测（异步任务）

- `POST /api/training/backtest`：创建回测任务（立即返回 `task_id`）
- `GET /api/training/backtest/{task_id}`：查询任务状态/结果
- `GET /api/training/backtest`：列出任务

回测请求体（核心字段）：

```json
{
  "symbol": "BTC/USDT",
  "days": 60,
  "initial_balance": 20.0,
  "leverage": 20.0,
  "primary_timeframe": "5m",
  "timeframes": ["3m", "5m", "15m"],
  "include_trades": false,
  "cumulative_mode": false
}
```

说明：

- `cumulative_mode=true` 时，回测会以 **累积模式**运行：接口侧把 `initial_balance` 置为 `null`，由回测服务使用“内存累积余额”继续跑；独立模式会在任务完成后自动重置累积余额。

### 信号

- `GET /api/signals`：信号历史（默认查最近 24h）
- `GET /api/signals/latest`：最新信号
- `POST /api/signals/generate`：手动生成信号（`{"symbol":"BTC/USDT","force":false}`）
- `GET /api/signals/performance`：信号表现统计
- `GET /api/signals/statistics`：信号统计摘要
- `GET /api/signals/model/prediction`：取最新数据做一次模型预测（用于调试）

### 仓位（虚拟仓位）

- `GET /api/positions`
- `GET /api/positions/summary`
- `GET /api/positions/risk`
- `GET /api/positions/{symbol}`
- `GET /api/positions/{symbol}/value`

### 交易（手动/模式切换）

- `POST /api/trading/execute`：手动下单（`LONG/SHORT/CLOSE`）
- `POST /api/trading/mode`：切换交易模式（`AUTO` / `SIGNAL_ONLY`）
- `GET /api/trading/status`
- `GET /api/trading/performance`
- `POST /api/trading/close/{symbol}`
- `GET /api/trading/limits`

⚠️ 注意：工程默认以“信号/虚拟交易”为主；若切换 `AUTO`，请先确认你的交易所客户端与密钥配置，避免产生真实下单风险。

### 绩效/风险

- `GET /api/performance`
- `GET /api/performance/risk`
- `GET /api/performance/drawdown`
- `GET /api/performance/returns`
- `GET /api/performance/ratios`
- `GET /api/performance/summary`

### 系统

- `GET /api/system/status`
- `POST /api/system/control`（`START/STOP/PAUSE/RESUME`）
- `GET /api/system/health`（详细健康检查）
- `GET /api/system/info`
- `GET /api/system/tasks` / `POST /api/system/tasks/{task_name}/run`
- `GET /api/system/cache/stats`

### WebSocket

- 连接：`/api/ws/connect`
- 订阅消息示例：

```json
{"type":"subscribe","channel":"signals"}
```

可用频道（服务端使用频道名过滤广播）：

- `price`、`signals`、`orders`、`risk`、`system`

查看连接统计：

- `GET /api/ws/stats`

## 回测分析脚本

优先推荐“无需数据库”的简化分析：

```powershell
python .\test\analyze_backtest_simple.py
```

如果你已跑过回测并写入数据库，可运行完整版分析：

```powershell
python .\test\analyze_backtest_trades.py
```

更多说明见：`docs/how_to_analyze_backtest.md`

## 测试

```powershell
pytest -q
```

## 代码结构（3 层架构）

- **Features（纯函数）**：`app/model/features/`
- **Model（有状态）**：`app/model/`
- **External（服务/交易）**：`app/services/`、`app/trading/`
- **API**：`app/api/`
- **Core（配置/数据库/缓存）**：`app/core/`

依赖方向：`Features → Model → External`（禁止反向依赖）。

## 风险声明

本项目用于研究与工程实现参考，不构成投资建议。实盘交易存在巨大风险（特别是高杠杆与高频策略），请在隔离环境中验证并自行承担后果。

---

## GitHub About 简介

`QuantAI-ETH：以太坊量化智能交易系统，FastAPI + Stacking 集成模型 + 严格模式回测/信号。`
