# 系统重构任务清单

## 项目概述
将复杂的ML预测交易系统重构为简洁高效的动量剥头皮系统。

**目标**: 5U → 500U（30天100倍）
**策略**: 动量突破 + 波动率过滤 + 复利滚仓

---

## 重构进度总览

| 模块 | 状态 | 说明 |
|------|------|------|
| 核心配置 (core/) | ✅ 完成 | config.py, constants.py, logging.py, exceptions.py |
| 剥头皮模块 (scalping/) | ✅ 完成 | 全部核心模块已实现并审查 |
| 交易所客户端 (exchange/) | ✅ 完成 | 新架构已创建并审查 |
| API层 (api/) | ✅ 完成 | scalping端点已创建并审查 |
| 旧模块清理 | ✅ 完成 | git已标记删除，待提交 |
| 测试 | ⏳ 待处理 | 需要添加单元测试 |

---

## 详细任务列表

### 阶段1: 清理旧代码 (优先级: 高)

#### 1.1 删除旧ML模块
- [x] 确认 `app/model/` 目录已删除 ✅ (git已标记删除)
- [x] 确认 `app/services/` 目录已删除 ✅ (git已标记删除)
- [x] 确认 `app/trading/` 目录已删除 ✅ (git已标记删除)
- [x] 清理 `requirements.txt` 中不需要的ML依赖 ✅ (已清理，只保留必要依赖)

#### 1.2 清理旧API端点
- [x] 删除 `app/api/endpoints/performance.py` ✅ (git已标记删除)
- [x] 删除 `app/api/endpoints/positions.py` ✅ (git已标记删除)
- [x] 删除 `app/api/endpoints/signals.py` ✅ (git已标记删除)
- [x] 删除 `app/api/endpoints/trading.py` ✅ (git已标记删除)
- [x] 删除 `app/api/endpoints/training.py` ✅ (git已标记删除)
- [x] 删除 `app/api/endpoints/websocket.py` ✅ (git已标记删除)
- [x] 更新 `app/api/routes.py` 移除旧路由 ✅ (已清理，只保留scalping和system)

#### 1.3 清理旧核心模块
- [x] 删除 `app/core/cache.py` ✅ (git已标记删除)
- [x] 删除 `app/core/database.py` ✅ (git已标记删除)
- [x] 删除 `app/core/database_schema.py` ✅ (git已标记删除)
- [x] 删除 `app/core/executor.py` ✅ (git已标记删除)
- [x] 删除 `app/core/gpu_config.py` ✅ (git已标记删除)

#### 1.4 清理旧交易所模块
- [x] 删除 `app/exchange/clients/okx/` ✅ (git已标记删除)
- [x] 删除 `app/exchange/mock_client.py` ✅ (git已标记删除)
- [x] 删除 `app/exchange/exceptions.py` ✅ (git已标记删除)
- [x] 删除 `app/exchange/exchange_factory.py` ✅ (git已标记删除)
- [x] 删除 `app/exchange/base_exchange_client.py` ✅ (git已标记删除)

---

### 阶段2: 完善交易所模块 (优先级: 高)

#### 2.1 Binance客户端
- [x] 审查 `app/exchange/clients/binance/binance_client.py` ✅ (代码完善)
- [x] 完善 `app/exchange/clients/binance/reconnector.py` ✅ (已集成到binance_client.py)
- [x] 完善 `app/exchange/clients/binance/rest_client.py` ✅ (已集成到binance_client.py)
- [x] 确保WebSocket重连机制稳定 ✅ (ExponentialBackoffReconnector已实现)

#### 2.2 基础类型和工厂
- [x] 审查 `app/exchange/base/types.py` ✅ (统一数据类型定义完善)
- [x] 审查 `app/exchange/base/client.py` ✅ (抽象基类定义完善)
- [x] 审查 `app/exchange/base/websocket.py` ✅ (WebSocket基类)
- [x] 审查 `app/exchange/factory.py` ✅ (工厂模式实现完善)
- [x] 审查 `app/exchange/mappers.py` ✅ (符号映射)

---

### 阶段3: 完善剥头皮模块 (优先级: 中)

#### 3.1 核心引擎
- [x] 审查 `app/scalping/scalping_engine.py` - 主引擎 ✅ (代码完善)
- [x] 审查 `app/scalping/signal_generator.py` - 信号生成 ✅ (代码完善)
- [x] 审查 `app/scalping/orderflow_analyzer.py` - 订单流分析 ✅ (已被momentum_analyzer替代)
- [x] 审查 `app/scalping/momentum_analyzer.py` - 动量分析 ✅ (代码完善)

#### 3.2 仓位和风控
- [x] 审查 `app/scalping/position_manager.py` - 仓位管理 ✅ (支持多仓位，代码完善)
- [x] 审查 `app/scalping/risk_controller.py` - 风控系统 ✅ (分级追踪止盈，代码完善)

#### 3.3 辅助模块
- [x] 审查 `app/scalping/symbol_scanner.py` - 币种扫描 ✅
- [x] 审查 `app/scalping/multi_symbol_monitor.py` - 多币种监控 ✅
- [x] 审查 `app/scalping/backtest.py` - 回测系统 ✅
- [x] 审查 `app/scalping/config.py` - 配置 ✅ (参数已优化)

---

### 阶段4: 完善API层 (优先级: 中)

#### 4.1 端点
- [x] 审查 `app/api/endpoints/scalping.py` ✅ (代码完善)
- [x] 审查 `app/api/endpoints/system.py` ✅

#### 4.2 基础设施
- [x] 审查 `app/api/routes.py` ✅ (已清理)
- [x] 审查 `app/api/models.py` ✅
- [x] 审查 `app/api/dependencies.py` ✅
- [x] 审查 `app/api/middleware.py` ✅

---

### 阶段5: 测试和文档 (优先级: 低)

#### 5.1 单元测试
- [ ] 添加 `tests/test_signal_generator.py`
- [ ] 添加 `tests/test_position_manager.py`
- [ ] 添加 `tests/test_risk_controller.py`
- [ ] 添加 `tests/test_backtest.py`

#### 5.2 集成测试
- [ ] 添加 `tests/test_scalping_engine.py`
- [ ] 添加 `tests/test_api_endpoints.py`

#### 5.3 文档
- [ ] 更新 `README.md`
- [ ] 清理临时文档 (IMPROVEMENTS.md, OPTIMIZATION_LOG.md等)

---

## 当前任务

### 正在进行
> 记录当前正在处理的任务

**任务**: 代码审查完成
**状态**: ✅ 完成
**备注**: 阶段1-4全部完成，代码质量良好

---

### 下一步建议
> 下一个要处理的任务

1. **提交当前更改** - 将已完成的重构提交到git
2. **验证系统启动** - 确保系统能正常运行
3. **添加单元测试** - 阶段5待处理

---

## 完成记录

| 日期 | 任务 | 说明 |
|------|------|------|
| 2026-02-01 | 代码审查 | 审查阶段1-4所有模块，代码质量良好 |
| 2026-01-30 | 参数优化 | 调整止损、止盈、信号阈值等参数 |
| - | 剥头皮模块创建 | 创建scalping/目录下所有核心模块 |
| - | 新exchange架构 | 创建exchange/base/和factory.py |

---

## 注意事项

1. **上下文限制**: 由于上下文长度限制，每次只处理一个小任务
2. **增量提交**: 每完成一个任务后建议提交git
3. **测试验证**: 修改后需要验证系统能正常启动
4. **配置备份**: 修改配置前先备份

---

## 文件结构目标

```
app/
├── api/                    # API层
│   ├── endpoints/
│   │   ├── scalping.py     # 剥头皮交易端点
│   │   └── system.py       # 系统端点
│   ├── routes.py           # 路由注册
│   ├── models.py           # 请求/响应模型
│   ├── dependencies.py     # 依赖注入
│   └── middleware.py       # 中间件
├── scalping/               # 剥头皮交易核心
│   ├── scalping_engine.py  # 交易引擎
│   ├── signal_generator.py # 信号生成器
│   ├── orderflow_analyzer.py # 订单流分析
│   ├── momentum_analyzer.py  # 动量分析
│   ├── position_manager.py # 仓位管理
│   ├── risk_controller.py  # 风控系统
│   ├── symbol_scanner.py   # 币种扫描
│   ├── multi_symbol_monitor.py # 多币种监控
│   ├── backtest.py         # 回测系统
│   └── config.py           # 配置
├── exchange/               # 交易所客户端
│   ├── base/
│   │   ├── types.py        # 类型定义
│   │   ├── client.py       # 基础客户端
│   │   └── websocket.py    # WebSocket基类
│   ├── clients/
│   │   └── binance/
│   │       ├── binance_client.py
│   │       ├── rest_client.py
│   │       └── reconnector.py
│   ├── factory.py          # 工厂类
│   └── mappers.py          # 格式映射
├── core/                   # 核心配置
│   ├── config.py           # 全局配置
│   ├── constants.py        # 常量
│   ├── logging.py          # 日志配置
│   └── exceptions.py       # 异常定义
└── utils/                  # 工具函数
    └── helpers.py
```

---

*最后更新: 2026-02-01*
