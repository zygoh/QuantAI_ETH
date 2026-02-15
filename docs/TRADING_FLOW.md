# 模拟交易逻辑梳理

本文档对照 `main.py` 与 `simulator` 梳理：云端选币、持仓判断、有/无仓时的图表、AI 分析、模拟交易与实时订阅。

---

## 1. 获取云端币种逻辑

| 项目 | 位置 | 说明 |
|------|------|------|
| **接口** | `CLOUD_COIN_SELECT_URL` | `https://n8n.do2ge.com/tail/tro`，GET 请求 |
| **调用时机** | 仅**无仓位**时 | `do_select_coin()` 只在「无仓位 + 每 5 分钟到点」时被调用 |
| **流程** | `do_select_coin()` | 请求 → 解析 `data.symbol` → 校验无 `detail`、symbol 存在 → 设置 `_current_selected_symbol`、`trading_simulator.current_symbol` → **立即**对该 symbol 调用 `do_chart_and_analyze(symbol)` |
| **失败** | 有 `detail` 或 无 symbol | 直接 return，不更新 symbol，本轮回不生成图表、不分析 |

**结论**：云端选币只在无仓位时执行；有仓位时不会调云端，也不会用云端结果做分析。

---

## 2. 当前是否持仓

| 项目 | 说明 |
|------|------|
| **判断** | `trading_simulator.has_position()` |
| **依据** | `self.position is not None`（simulator 内） |
| **使用处** | 主循环开头（WebSocket 订阅）、每 5 分钟分支（选币 vs 分析持仓）、`do_chart_and_analyze` 内（position_info、各信号分支） |

**结论**：全链路统一用 `has_position()` 判断，无多处不一致。

---

## 3. 无持仓时

| 步骤 | 逻辑 | 代码位置 |
|------|------|----------|
| **选币** | 每 5 分钟到点 → 调用 `do_select_coin()`，从云端取 symbol | 主循环 `else: await do_select_coin()` |
| **生成图表** | 对**云端返回的 symbol** 拉价、生成 5m/15m 图 | `do_select_coin()` 内 `do_chart_and_analyze(_current_selected_symbol)` |
| **AI 分析** | 对**同一 symbol** 调用 `ai_analyzer.analyze_charts(symbol, ...)`，`position_info=None` | `do_chart_and_analyze` 内 |
| **模拟交易** | 若信号为 open_long/open_short → 执行开仓；close/hold/adjust 忽略 | 同函数内各 `elif` 分支 |
| **实时订阅** | 无仓时主循环开头执行 `price_monitor.unsubscribe()`；**开仓成功后**在信号分支里 `price_monitor.subscribe(新持仓.symbol, on_price_update)` | 主循环 237–239 行；开仓分支 441–445 行 |

**结论**：无仓时「选币 → 图表 → AI 分析 → 开仓 → 订阅新持仓 symbol」链路一致，分析对象与交易、订阅均为同一 symbol。

---

## 4. 有持仓时

| 步骤 | 逻辑 | 代码位置 |
|------|------|----------|
| **选币** | **不调用** `do_select_coin()`，不请求云端 | 主循环 `if has_position(): ... do_chart_and_analyze(pos.symbol)` |
| **生成图表** | 对**持仓币种 `pos.symbol`** 拉价、生成 5m/15m 图 | 主循环内 `await do_chart_and_analyze(pos.symbol)` |
| **AI 分析** | 对**同一 pos.symbol** 调用 `analyze_charts(symbol, ...)`，并传入 `position_info`（持仓 + 用持仓币种价格算的浮盈） | `do_chart_and_analyze` 内 |
| **模拟交易** | close_position → 用持仓币种价格平仓并 unsubscribe；adjust_stops → 执行；open_long/open_short 仅处理**同币种反向**（先平再开并重新 subscribe），同向或新币种开仓信号忽略 | 同函数内各分支 |
| **实时订阅** | 主循环开头若 `price_monitor.current_symbol != pos.symbol` 则 `subscribe(pos.symbol, on_price_update)`；平仓时在对应分支内 `unsubscribe()`；反向开仓后 `subscribe(新持仓.symbol, ...)` | 主循环 232–236；平仓/反向分支 428、431–434；开仓分支 441–445 |

**结论**：有仓时「不选币、只分析持仓 symbol、图表与 AI 均为持仓币种、只对同币种反向开仓执行、WebSocket 始终盯持仓 symbol」逻辑一致。

---

## 5. 状态与 API 一致性

| 项目 | 说明 |
|------|------|
| **current_symbol** | 无仓：由 `do_select_coin()` 设为云端 symbol；有仓：在主循环 5 分钟分支内设为 `pos.symbol`，再调用 `do_chart_and_analyze(pos.symbol)`，保证 API/status 的「当前分析/选中币种」与真实一致。 |
| **current_price** | 来自 `price_monitor.current_price`；有仓时订阅的是持仓 symbol，故为持仓币种价格；无仓时可能未订阅或为上次残留，API 仅作展示。 |

---

## 6. 流程简图

```
主循环 (每轮)
├── 有仓?
│   ├── 是 → WebSocket 订阅 pos.symbol（若未订阅）
│   └── 否 → WebSocket 取消订阅
├── 是否 5 分钟到点?
│   └── 是
│       ├── 有仓? → current_symbol=pos.symbol → do_chart_and_analyze(pos.symbol)
│       └── 无仓? → do_select_coin() → [云端取 symbol → do_chart_and_analyze(symbol)]
│                    ↑ 内部会 set current_symbol 并立即分析
└── do_chart_and_analyze(symbol)
    ├── 拉 symbol 价格、生成图表
    ├── 有仓? → position_info = 持仓 + 持仓币种价格算浮盈
    ├── 并行拉取：订单簿(depth) + 近期成交(aggTrades) → build_market_context → market_context 文本
    ├── AI 分析(symbol, position_info, market_context)：K 线图 + 市场微观一并注入 prompt
    └── 执行信号
        ├── open → 无仓则开仓并 subscribe(新仓.symbol)
        ├── 有仓且同币种反向 → 平仓 unsubscribe、开新仓、subscribe(新仓.symbol)
        └── close → 平仓并 unsubscribe
```

---

## 7. 市场微观数据（订单簿、成交分布、逐笔）

在保持「每 5 分钟 K 线图 + 当前价 + 持仓信息」分析模式不变的前提下，**额外**为 AI 注入订单簿与近期成交数据，供判断买卖压力、挂单墙、成交分布等。

| 项目 | 说明 |
|------|------|
| **数据来源** | Binance：`/fapi/v1/depth`（订单簿）、`/fapi/v1/aggTrades`（近期聚合成交，limit=100） |
| **拉取时机** | `do_chart_and_analyze` 内，在生成图表、构建 `position_info` 之后，**调用 AI 之前** |
| **拉取方式** | `asyncio.gather` 并行请求；任一侧失败仅打 debug 日志，不阻断分析 |
| **汇总模块** | `app/trading/market_context.py`：`build_orderbook_summary`（买卖盘前 5 档、买卖比、价差）、`build_trades_summary`（主动买/卖量、买压占比、按相对当前价 0.5% 档位的成交分布） |
| **注入方式** | `build_market_context(symbol, current_price, orderbook, agg_trades)` 生成一段文本，作为 `market_context` 传入 `ai_analyzer.analyze_charts(..., market_context=...)`，在 `_build_prompt` 中拼入「当前市场数据」之后、K 线分析说明之前 |

**结论**：AI 仍以 K 线形态与关键位为主；订单簿与成交分布作为补充上下文，不改变现有决策流程。

---

## 8. 已做改进

- **有仓时 current_symbol**：有仓时主循环 5 分钟分支内先设 `trading_simulator.current_symbol = pos.symbol`，再分析，保证 API 展示正确。
- **无仓时 current_price 展示**：无仓时 WebSocket 未订阅，`/api/status` 的 `current_price` 改为：若无仓且有 `current_symbol`，则用 REST `get_current_price(current_symbol)` 拉一次，供前端展示当前分析币种最新价。
- **云端选币失败时的降级**：`do_select_coin()` 若云端返回 `detail` 或无 symbol，在存在 `_current_selected_symbol` 时仍会执行 `do_chart_and_analyze(_current_selected_symbol)`，避免本轮回完全空转。
- **市场微观数据**：分析前并行拉取订单簿与 aggTrades，生成订单簿摘要 + 买卖压力 + 成交分布文本，注入 AI prompt，不改变 K 线分析主流程。
