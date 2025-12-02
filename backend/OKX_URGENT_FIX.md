# OKX紧急修复说明

## 修复时间
2025-12-02

## 问题描述

### 1. 交易对格式错误
**错误信息**: `Instrument ID, Instrument ID code, or Spread ID doesn't exist`

**原因**: 
- 配置文件中使用 `SYMBOL: str = "ETHUSDT"` (Binance格式)
- OKX需要标准格式 `ETH/USDT`，然后由SymbolMapper转换为 `ETH-USDT-SWAP`

### 2. WebSocket频繁断开
**错误信息**: `No data received in 30s`

**原因**:
- OKX WebSocket要求每30秒发送一次ping保持连接
- 原代码的`run_forever()`没有配置ping参数

## 修复内容

### 修复1: 更新配置文件 (app/core/config.py)

```python
# 修改前
SYMBOL: str = "ETHUSDT"

# 修改后
SYMBOL: str = "ETH/USDT"  # 使用标准格式，系统会自动转换为交易所格式
```

### 修复2: 添加WebSocket Ping机制 (app/exchange/okx_client.py)

```python
# 修改前
self.ws.run_forever(sslopt=sslopt)

# 修改后
self.ws.run_forever(
    sslopt=sslopt,
    ping_interval=25,  # 每25秒发送一次ping（小于OKX的30秒超时）
    ping_timeout=10    # ping超时时间10秒
)
```

## 技术说明

### Symbol格式转换流程
1. 配置: `ETH/USDT` (标准格式)
2. SymbolMapper转换: `ETH-USDT-SWAP` (OKX永续合约格式)
3. API调用: 使用转换后的格式

### WebSocket保活机制
- OKX要求: 30秒内必须有数据交互
- 解决方案: 每25秒自动发送ping帧
- websocket-client库自动处理pong响应

## 验证步骤

1. 重启应用程序
2. 检查日志中是否还有 "Instrument ID doesn't exist" 错误
3. 观察WebSocket连接是否稳定（不再出现30秒断开）
4. 确认K线数据正常接收

## 预期结果

- ✅ 杠杆设置成功
- ✅ K线数据正常获取
- ✅ WebSocket连接稳定
- ✅ 不再出现"No data received in 30s"错误
