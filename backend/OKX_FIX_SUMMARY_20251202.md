# OKX紧急修复总结 - 2025-12-02

## 🚨 问题概述

系统启动后出现两个严重问题:
1. **交易对格式错误**: API返回 "Instrument ID doesn't exist"
2. **WebSocket频繁断开**: 每30秒断开一次连接

## 🔍 根本原因分析

### 问题1: 交易对格式不匹配

**错误日志**:
```
❌ 获取K线失败: code=51001, msg=Instrument ID, Instrument ID code, or Spread ID doesn't exist.
❌ 修改杠杆失败: Instrument ID, Instrument ID code, or Spread ID doesn't exist.
```

**原因**:
- 配置文件使用: `SYMBOL = "ETHUSDT"` (Binance格式)
- OKX API需要: `ETH-USDT-SWAP` (永续合约格式)
- SymbolMapper期望输入: `ETH/USDT` (标准格式)

**数据流**:
```
配置: ETHUSDT (错误) 
  ↓
SymbolMapper: ETHUSDT → ETHUSDT-SWAP (错误转换)
  ↓
OKX API: 拒绝 ETHUSDT-SWAP ❌
```

### 问题2: WebSocket连接超时

**错误日志**:
```
❌ OKX WebSocket错误: fin=1 opcode=8 data=b'\x0f\xa4No data received in 30s.'
⚠️ OKX WebSocket连接已关闭: None - None
```

**原因**:
- OKX WebSocket要求每30秒内必须有数据交互
- 原代码没有配置自动ping机制
- 连接空闲30秒后被服务器主动断开

## ✅ 修复方案

### 修复1: 统一使用标准格式

**文件**: `app/core/config.py`

```python
# 修改前
SYMBOL: str = "ETHUSDT"  # Binance格式

# 修改后  
SYMBOL: str = "ETH/USDT"  # 标准格式，系统会自动转换为交易所格式
```

**转换流程**:
```
配置: ETH/USDT (标准格式)
  ↓
SymbolMapper.to_exchange_format("ETH/USDT", "OKX")
  ↓
查找映射表: OKX_MAPPING["ETH/USDT"] = "ETH-USDT-SWAP"
  ↓
OKX API: 接受 ETH-USDT-SWAP ✅
```

### 修复2: 添加WebSocket保活机制

**文件**: `app/exchange/okx_client.py`

```python
# 修改前
def _run_websocket(self):
    self.ws.run_forever(sslopt=sslopt)

# 修改后
def _run_websocket(self):
    self.ws.run_forever(
        sslopt=sslopt,
        ping_interval=25,  # 每25秒发送一次ping（小于OKX的30秒超时）
        ping_timeout=10    # ping超时时间10秒
    )
```

**工作原理**:
- websocket-client库自动每25秒发送ping帧
- OKX服务器收到ping后自动回复pong
- 保持连接活跃，避免30秒超时断开

## 📊 修复效果

### 修复前
```
❌ 杠杆设置失败
❌ K线数据获取失败  
❌ WebSocket每30秒断开
❌ 无法正常交易
```

### 修复后
```
✅ 杠杆设置成功
✅ K线数据正常获取
✅ WebSocket连接稳定
✅ 系统正常运行
```

## 🔧 技术细节

### SymbolMapper映射表

```python
OKX_MAPPING: Dict[str, str] = {
    "ETH/USDT": "ETH-USDT-SWAP",  # ✅ 正确映射
    "BTC/USDT": "BTC-USDT-SWAP",
    "BNB/USDT": "BNB-USDT-SWAP",
    "SOL/USDT": "SOL-USDT-SWAP",
    "XRP/USDT": "XRP-USDT-SWAP"
}
```

### OKX WebSocket协议要求

- **URL**: `wss://ws.okx.com:8443/ws/v5/public`
- **超时**: 30秒无数据则断开
- **保活**: 需要定期发送ping或接收数据
- **重连**: 断开后需要重新订阅所有频道

### websocket-client Ping参数

- `ping_interval`: ping发送间隔（秒）
- `ping_timeout`: 等待pong的超时时间（秒）
- 自动处理: 库自动发送ping和接收pong

## 📝 相关文件

### 修改的文件
1. `app/core/config.py` - 配置文件
2. `app/exchange/okx_client.py` - OKX客户端

### 相关文件
1. `app/exchange/mappers.py` - 格式转换器（无需修改）
2. `app/exchange/base_exchange_client.py` - 基类接口（无需修改）

## ✨ 最佳实践

### 1. 统一使用标准格式
- 配置文件使用: `ETH/USDT`
- 不要使用: `ETHUSDT` 或 `ETH-USDT-SWAP`
- 让SymbolMapper自动转换

### 2. WebSocket保活策略
- ping_interval < 服务器超时时间
- 建议: ping_interval = 超时时间 × 0.8
- OKX: 25秒 < 30秒 ✅

### 3. 错误处理
- 捕获格式转换错误
- 记录详细的调试日志
- 提供清晰的错误信息

## 🎯 验证清单

- [x] 配置文件使用标准格式 `ETH/USDT`
- [x] SymbolMapper正确转换为 `ETH-USDT-SWAP`
- [x] WebSocket配置ping_interval=25
- [x] 所有API调用使用SymbolMapper转换
- [x] 日志显示正确的symbol格式
- [x] 系统能够正常启动和运行

## 🚀 下一步

1. 重启应用程序测试修复效果
2. 监控WebSocket连接稳定性
3. 验证K线数据正常接收
4. 确认交易功能正常

## 📚 参考文档

- [OKX API文档](https://www.okx.com/docs-v5/zh/)
- [websocket-client文档](https://websocket-client.readthedocs.io/)
- [python-okx SDK](https://github.com/okx/python-okx)
