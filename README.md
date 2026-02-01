# 30天百倍剥头皮交易系统

## 项目简介

本项目是一个基于 **FastAPI** 的高频剥头皮交易系统，核心目标：

- **目标**：5U → 500U（30天100倍）
- **策略**：高频剥头皮 + 复利滚仓
- **核心能力**：
  - 订单流分析（买卖压力、大单追踪、成交量异动、动量分析）
  - 自动扫描高波动币种
  - 多仓位管理
  - 追踪止盈 + 移动保本
  - 阶段性杠杆调整（根据资金量）

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
│                    (FastAPI 应用入口)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ScalpingEngine                            │
│                    (剥头皮交易引擎)                            │
│  - 整合所有模块                                               │
│  - 自动交易执行                                               │
│  - 币种刷新管理                                               │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Signal    │ │  Position   │ │    Risk     │ │   Symbol    │
│  Generator  │ │   Manager   │ │ Controller  │ │   Scanner   │
│ (信号生成器) │ │ (仓位管理器) │ │ (风控系统)  │ │ (币种扫描器) │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
         │              │              │
         ▼              │              │
┌─────────────┐         │              │
│  OrderFlow  │         │              │
│  Analyzer   │         │              │
│(订单流分析器)│         │              │
└─────────────┘         │              │
         │              │              │
         ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                  MultiSymbolMonitor                          │
│                  (多币种实时监控)                              │
│  - WebSocket 订单簿                                          │
│  - WebSocket 成交流                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BinanceClient                             │
│                   (币安交易所客户端)                           │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 交易引擎 | `app/scalping/scalping_engine.py` | 整合所有模块，自动交易执行 |
| 信号生成器 | `app/scalping/signal_generator.py` | 整合订单流分析，生成交易信号 |
| 订单流分析 | `app/scalping/orderflow_analyzer.py` | 买卖压力、大单追踪、动量分析 |
| 仓位管理 | `app/scalping/position_manager.py` | 复利滚仓，连胜加仓/连亏减仓 |
| 风控系统 | `app/scalping/risk_controller.py` | 止盈止损、追踪止盈、持仓超时 |
| 币种扫描 | `app/scalping/symbol_scanner.py` | 自动扫描高波动币种 |
| 多币种监控 | `app/scalping/multi_symbol_monitor.py` | WebSocket 实时数据订阅 |
| 回测系统 | `app/scalping/backtest.py` | 历史数据回测 |
| 配置 | `app/scalping/config.py` | 系统配置参数 |

## 交易阶段

系统根据资金量自动调整策略：

| 阶段 | 资金范围 | 杠杆 | 最大持仓数 | 策略特点 |
|------|----------|------|-----------|----------|
| Phase 1 | 5U - 50U | 20x | 1 | 高波动meme币，激进，集中火力 |
| Phase 2 | 50U - 200U | 30x | 3 | 中等波动，平衡 |
| Phase 3 | 200U+ | 50x | 全部 | 主流币，稳健，不错过机会 |

## 风控机制

### 止盈止损
- **初始止损**：1.5%（价格波动，不含杠杆）
- **追踪止盈**：盈利2.0%后激活，从最高点回撤0.8%触发
- **移动保本**：盈利1.2%后止损移至入场价+0.4%

### 信号质量控制
- **最小信号得分**：0.72（只做高确定性交易）
- **信号冷却时间**：30秒（避免频繁切换方向）
- **趋势确认**：启用（防止逆势交易）

### 风控限制
- 单日最大亏损：15%
- 单日最大交易次数：100
- 连续亏损暂停阈值：3次
- 连亏后冷却时间：10分钟
- 最大持仓时间：30分钟

## 环境要求

- **Windows 10/11 + PowerShell**
- Python **3.12**
- （可选）NVIDIA GPU（CUDA 环境用于加速）

## 安装依赖

```powershell
cd "F:\AI\20251007_1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置

项目使用 `pydantic-settings` 读取 `.env`，环境变量会覆盖默认值。

建议在项目根目录创建 `.env`（**不要提交到 git**），示例：

```dotenv
# 服务
HOST=0.0.0.0
PORT=8001

# 日志
LOG_LEVEL=INFO
LOG_FILE=trading_system.log

# 代理（可选）
USE_PROXY=false
USE_PROXY_WS=false
PROXY_HOST=127.0.0.1
PROXY_PORT=10808
PROXY_TYPE=socks5
```

## 启动服务

```powershell
python .\main.py
```

启动后可访问：

- `GET /health`（健康检查）
- `GET /api/scalping/status`（交易状态）

日志默认写入 `.\logs\trading_system.log`。

## API 快速参考

### 剥头皮交易

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/scalping/start` | 启动交易引擎 |
| POST | `/api/scalping/stop` | 停止交易引擎 |
| GET | `/api/scalping/status` | 获取交易状态 |
| POST | `/api/scalping/close-position` | 手动平仓 |
| GET | `/api/scalping/scan-symbols` | 扫描高波动币种 |
| GET | `/api/scalping/debug` | 调试信息 |

### 回测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/scalping/backtest` | 创建回测任务 |
| GET | `/api/scalping/backtest/{task_id}` | 查询回测结果 |

回测请求体：

```json
{
  "symbol": "1000PEPE/USDT",
  "days": 7,
  "initial_balance": 5.0,
  "leverage": 20
}
```

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/system/info` | 系统信息 |
| GET | `/health` | 健康检查 |

## 代码结构

```
app/
├── api/                    # API层
│   ├── endpoints/
│   │   ├── scalping.py     # 剥头皮交易端点
│   │   └── system.py       # 系统端点
│   ├── routes.py           # 路由注册
│   ├── models.py           # 请求/响应模型
│   └── dependencies.py     # 依赖注入
├── scalping/               # 剥头皮交易核心
│   ├── scalping_engine.py  # 交易引擎
│   ├── signal_generator.py # 信号生成器
│   ├── orderflow_analyzer.py # 订单流分析
│   ├── position_manager.py # 仓位管理
│   ├── risk_controller.py  # 风控系统
│   ├── symbol_scanner.py   # 币种扫描
│   ├── multi_symbol_monitor.py # 多币种监控
│   ├── backtest.py         # 回测系统
│   └── config.py           # 配置
├── exchange/               # 交易所客户端
│   ├── clients/
│   │   └── binance/
│   │       └── binance_client.py
│   └── mappers.py          # 格式映射
├── core/                   # 核心配置
│   └── config.py           # 全局配置
└── utils/                  # 工具函数
    └── helpers.py
```

## 测试

```powershell
pytest -q
```

## 风险声明

本项目用于研究与工程实现参考，不构成投资建议。高频交易和高杠杆策略存在巨大风险，请在隔离环境中验证并自行承担后果。

**目标收益率计算**：
- 100倍 = (1 + r)^30
- 每日目标收益率 r ≈ 16.6%

这是一个极具挑战性的目标，需要高胜率和严格的风控配合。
