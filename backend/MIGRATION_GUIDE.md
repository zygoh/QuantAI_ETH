# OKX交易所集成迁移指南

## 概述

本指南帮助你将QuantAI-ETH交易系统从单一Binance交易所迁移到支持多交易所架构（Binance + OKX）。

## 新增功能

### 1. 多交易所支持
- ✅ 支持Binance交易所（原有功能）
- ✅ 支持OKX交易所（新增）
- ✅ 支持Mock模式（用于测试）
- ✅ 通过配置文件灵活切换

### 2. 统一接口设计
- 所有交易所实现相同的`BaseExchangeClient`接口
- 统一的数据格式（UnifiedKlineData, UnifiedTickerData, UnifiedOrderData）
- 业务代码无需关心具体使用哪个交易所

### 3. 工厂模式管理
- 使用`ExchangeFactory`集中管理客户端创建
- 单例模式确保每个交易所只有一个实例
- 简化客户端获取流程

## 配置更新

### 1. 环境变量配置

在`.env`文件中添加以下配置：

```bash
# 交易所选择（BINANCE, OKX, MOCK）
EXCHANGE_TYPE=BINANCE

# Binance配置（保持不变）
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=True

# OKX配置（新增）
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_PASSPHRASE=your_okx_passphrase
OKX_TESTNET=False
```

### 2. 切换到OKX

修改`EXCHANGE_TYPE`配置：

```bash
EXCHANGE_TYPE=OKX
```

然后重启系统：

```powershell
python main.py
```

## 代码迁移

### 方式1: 使用ExchangeFactory（推荐）

**旧代码**:
```python
from app.exchange.binance_client import binance_client

# 获取K线数据
klines = binance_client.get_klines("ETHUSDT", "5m", limit=100)

# 获取价格
price = binance_client.get_ticker_price("ETHUSDT")

# 下单
result = binance_client.place_order(
    symbol="ETHUSDT",
    side="BUY",
    order_type="MARKET",
    quantity=0.1
)
```

**新代码**:
```python
from app.exchange.exchange_factory import ExchangeFactory

# 获取当前配置的交易所客户端
client = ExchangeFactory.get_current_client()

# 获取K线数据（接口相同）
klines = client.get_klines("ETHUSDT", "5m", limit=100)

# 获取价格（接口相同）
price = client.get_ticker_price("ETHUSDT")

# 下单（接口相同）
result = client.place_order(
    symbol="ETHUSDT",
    side="BUY",
    order_type="MARKET",
    quantity=0.1
)
```

### 方式2: 保持向后兼容（临时方案）

如果暂时不想修改代码，可以继续使用`binance_client`：

```python
from app.exchange.binance_client import binance_client

# 原有代码继续工作（会有deprecation警告）
klines = binance_client.get_klines("ETHUSDT", "5m")
```

**注意**: 这种方式只支持Binance，无法切换到OKX。

## 系统模块更新

以下模块已更新为使用`ExchangeFactory`：

### 1. Trading Engine
```python
# 旧代码
from app.exchange.binance_client import binance_client
self.exchange_client = binance_client

# 新代码
from app.exchange.exchange_factory import ExchangeFactory
self.exchange_client = ExchangeFactory.get_current_client()
```

### 2. Signal Generator
```python
# 旧代码
from app.exchange.binance_client import binance_client
ticker = binance_client.get_ticker_price(symbol)

# 新代码
from app.exchange.exchange_factory import ExchangeFactory
client = ExchangeFactory.get_current_client()
ticker = client.get_ticker_price(symbol)
```

### 3. Data Service
```python
# 旧代码
from app.exchange.binance_client import binance_ws_client
binance_ws_client.subscribe_kline(symbol, interval, callback)

# 新代码
from app.exchange.exchange_factory import ExchangeFactory
client = ExchangeFactory.get_current_client()
# WebSocket订阅需要通过客户端的ws_client属性
```

## 验证迁移

### 1. 检查配置
```powershell
# 启动系统，查看日志
python main.py
```

查找以下日志：
```
✅ 从配置读取交易所类型: BINANCE
✅ Binance客户端创建成功
✅ Binance配置验证通过
```

### 2. 测试连接
系统启动时会自动测试API连接：
```
✓ 服务器时间获取成功: 1234567890000
✓ 账户信息获取成功
```

### 3. 测试数据获取
```python
from app.exchange.exchange_factory import ExchangeFactory

client = ExchangeFactory.get_current_client()

# 测试K线数据
klines = client.get_klines("ETHUSDT", "5m", limit=10)
print(f"获取到 {len(klines)} 条K线数据")

# 测试价格数据
ticker = client.get_ticker_price("ETHUSDT")
print(f"当前价格: {ticker.price}")
```

## 常见问题

### Q1: 切换交易所后数据格式会变吗？
**A**: 不会。所有交易所返回统一的数据格式（UnifiedKlineData等），业务代码无需修改。

### Q2: 可以同时使用多个交易所吗？
**A**: 可以。使用`ExchangeFactory.create_client()`创建不同交易所的客户端：
```python
binance_client = ExchangeFactory.create_client("BINANCE")
okx_client = ExchangeFactory.create_client("OKX")
```

### Q3: OKX的交易对格式不同怎么办？
**A**: 系统自动处理格式转换。你只需使用标准格式（如"ETHUSDT"），系统会自动转换为OKX格式（"ETH-USDT-SWAP"）。

### Q4: 如何在测试环境使用Mock客户端？
**A**: 设置`EXCHANGE_TYPE=MOCK`：
```python
# .env
EXCHANGE_TYPE=MOCK
```

### Q5: 迁移后性能会受影响吗？
**A**: 不会。工厂模式使用单例，不会增加额外开销。

### Q6: 如何回滚到旧版本？
**A**: 
1. 设置`EXCHANGE_TYPE=BINANCE`
2. 重启系统
3. 如需完全回滚，恢复旧代码即可

## 故障排查

### 问题1: 启动时报错"Unsupported exchange type"
**原因**: `EXCHANGE_TYPE`配置错误

**解决**:
```bash
# 检查.env文件
EXCHANGE_TYPE=BINANCE  # 必须是 BINANCE, OKX, 或 MOCK
```

### 问题2: OKX连接失败
**原因**: API密钥配置错误或权限不足

**解决**:
1. 检查API密钥是否正确
2. 确认API密钥已启用合约交易权限
3. 检查Passphrase是否正确

### 问题3: 数据格式错误
**原因**: 使用了交易所特定的格式

**解决**: 使用统一的数据格式类：
```python
# ✅ 正确
kline: UnifiedKlineData = client.get_klines(...)[0]
price = kline.close

# ❌ 错误
kline = client.get_klines(...)[0]
price = kline['close']  # 不要假设是字典
```

## 性能优化建议

### 1. 复用客户端实例
```python
# ✅ 推荐：复用实例
client = ExchangeFactory.get_current_client()
for symbol in symbols:
    klines = client.get_klines(symbol, "5m")

# ❌ 不推荐：重复创建
for symbol in symbols:
    client = ExchangeFactory.get_current_client()  # 每次都创建
    klines = client.get_klines(symbol, "5m")
```

### 2. 使用分页获取大量数据
```python
# 获取超过单次限制的数据
klines = client.get_klines_paginated(
    symbol="ETHUSDT",
    interval="5m",
    limit=2000  # 自动分页
)
```

### 3. 配置合理的限流延迟
```python
klines = client.get_klines_paginated(
    symbol="ETHUSDT",
    interval="5m",
    limit=2000,
    rate_limit_delay=0.2  # 每批次间隔200ms
)
```

## 下一步

1. ✅ 完成配置更新
2. ✅ 验证系统正常运行
3. ✅ 测试交易功能
4. ⏭️ 监控系统运行状态
5. ⏭️ 根据需要优化配置

## 技术支持

如有问题，请查看：
- 设计文档: `.kiro/specs/okx-exchange-integration/design.md`
- 需求文档: `.kiro/specs/okx-exchange-integration/requirements.md`
- 系统日志: `logs/trading_system.log`

---

**版本**: v1.0  
**更新日期**: 2025-01-26  
**适用系统**: QuantAI-ETH v10.1+
