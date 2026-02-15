# AI 模拟交易系统

## 项目简介

基于 **FastAPI** 的加密货币 **AI 模拟交易** 系统，核心流程：

- **云端选币**：从云端接口获取当日推荐交易对
- **K 线图表**：本地生成 5 分钟、15 分钟 K 线图（含均线等指标）
- **Claude 分析**：调用 Claude API 看图分析，输出开多/开空/观望/平仓/调仓等信号
- **模拟交易**：根据 AI 信号在本地模拟开平仓，支持杠杆、止损、止盈
- **实时监控**：WebSocket 订阅币安合约价格，实时检查止盈止损与爆仓

本系统为**模拟盘**，不连接实盘下单；适合验证策略与 AI 提示词效果。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
│              (FastAPI 应用 + 选币/图表/交易循环)               │
└─────────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    Chart     │  │   Trading    │  │   Exchange   │
│  (图表生成)   │  │  (交易逻辑)   │  │ (币安数据)   │
├──────────────┤  ├──────────────┤  └──────────────┘
│ generator    │  │ ai_analyzer  │
│ indicators   │  │ simulator    │
│ models       │  │ price_monitor│
└──────────────┘  │ models       │
                  └──────────────┘
```

- **Chart**：根据 Binance K 线数据生成 5m/15m PNG 图表，供 AI 分析。
- **Trading**：AI 分析结果 → 信号解析 → 模拟开平仓、止盈止损；WebSocket 价格回调驱动检查。
- **Exchange**：仅拉取行情与选币所需数据（REST），不实盘下单。

## 核心模块

| 模块       | 路径                         | 功能说明 |
|------------|------------------------------|----------|
| 图表生成   | `app/chart/generator.py`     | 请求 K 线、绘制 5m/15m 图并落盘 |
| 图表指标   | `app/chart/indicators.py`    | 均线等指标计算 |
| AI 分析器  | `app/trading/ai_analyzer.py` | 调用 Claude 分析图表，返回交易信号 |
| 模拟器     | `app/trading/simulator.py`   | 模拟账户、开平仓、止盈止损、爆仓检查 |
| 价格监控   | `app/trading/price_monitor.py` | WebSocket 订阅实时价格并触发止盈止损回调 |
| 交易模型   | `app/trading/models.py`     | 信号、持仓、账户等数据结构 |
| 币安客户端 | `app/exchange/clients/binance/binance_client.py` | K 线、ticker 等 REST 请求 |
| 全局配置   | `app/core/config.py`        | 服务端口、代理等配置 |

## 运行流程概要

1. **定时选币**：按 4H 周期（如 0:05, 4:05, …）请求云端选币接口，取推荐交易对。
2. **定时出图**：每 5 分钟生成当前选中币种的 5m/15m 图。
3. **AI 分析**：有图后调用 Claude，传入两张图 + 当前价（及持仓信息），得到 JSON 信号（open_long / open_short / wait / close_position / hold / adjust_stops）。
4. **执行信号**：模拟器根据信号开仓、平仓或调整止损止盈；开仓后启动价格监控。
5. **实时风控**：WebSocket 收到最新价后检查止盈、止损、爆仓，触发则平仓并停止监控。

## 风控与规则

- **止盈 / 止损**：由 AI 在信号中给出具体价格，或由模拟器默认逻辑约束；多仓止损 < 入场价、止盈 > 入场价，空仓相反。
- **爆仓**：浮亏 ≥ 保证金时强制平仓。
- **开仓门槛**：AI 提示词要求 15m 与 5m 方向一致、置信度 ≥ 70，解析层对低于 70 的开仓信号会强制改为观望。

## 环境要求

- **Windows 10/11 + PowerShell**（或兼容 Python 3.12 的环境）
- **Python 3.12**
- 可访问币安 API（若需代理，在 `.env` 中配置）

## 安装依赖

```powershell
cd "F:\AI\20251007_1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置

使用 `pydantic-settings` 读取 `.env`，环境变量会覆盖默认值。在项目根目录创建 `.env`（**勿提交到 git**），示例：

```dotenv
# 服务
HOST=0.0.0.0
PORT=8001

# 日志（如项目内使用）
LOG_LEVEL=INFO

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

- **首页**：`http://localhost:8001/`
- **API**：`http://localhost:8001/api/status` 等（见下）

日志按日写入 `logs/selector_YYYYMMDD.log`。

## API 快速参考

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端首页 |
| GET | `/api/status` | 系统状态（运行中、当前币种、当前价、是否持仓、余额、总盈亏、交易次数） |
| GET | `/api/account` | 账户详情（初始资金、余额、总盈亏、收益率、胜率、手续费等） |
| GET | `/api/position` | 当前持仓详情（方向、入场价、止损止盈、浮盈、爆仓价等） |
| GET | `/api/trades` | 交易历史（支持 `?limit=50`） |
| GET | `/api/chat` | AI 分析对话历史（支持 `?limit=20`） |

静态资源：`/static`、`/image`（图表等）。

## 代码结构

```
app/
├── api/                  # API 层
│   ├── models.py         # 请求/响应模型
│   └── routes.py         # /api 路由
├── chart/                # 图表
│   ├── generator.py     # K 线图生成
│   ├── indicators.py    # 技术指标
│   └── models.py        # 图表数据模型
├── core/
│   └── config.py        # 全局配置
├── exchange/             # 交易所
│   ├── clients/binance/
│   │   └── binance_client.py
│   └── mappers.py
└── trading/              # 交易逻辑
    ├── ai_analyzer.py   # Claude 图表分析
    ├── models.py        # 信号、持仓、账户模型
    ├── price_monitor.py # WebSocket 价格监控
    └── simulator.py     # 模拟交易引擎
```

## 测试

```powershell
pytest -q
```

## 风险声明

本项目仅供学习与策略研究，不构成任何投资建议。加密货币与杠杆交易风险极高，请勿将模拟逻辑直接等同于实盘，请在隔离环境中验证并自行承担使用后果。
