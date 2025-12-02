# OKX SDK 迁移指南

## 概述

本文档说明如何将 OKX 交易所集成从手动实现迁移到使用官方 python-okx 0.4.0 SDK。

## 主要变更

### 1. 依赖更新

**之前**：手动实现 HTTP 请求和签名
```python
import requests
import hmac
import hashlib
import base64
```

**现在**：使用官方 SDK
```python
from okx import Account, MarketData, Trade, PublicData
from okx.exceptions import OkxAPIException, OkxRequestException, OkxParamsException
```

**安装**：
```bash
pip install python-okx==0.4.0
```

### 2. 认证和签名

**之前**：手动实现 HMAC-SHA256 签名
```python
def _generate_signature(self, timestamp, method, request_path, body=""):
    message = timestamp + method + request_path + body
    mac = hmac.new(
        bytes(self.secret_key, encoding='utf8'),
        bytes(message, encoding='utf-8'),
        digestmod=hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode()

def _get_headers(self, method, request_path, body=""):
    timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
    signature = self._generate_signature(timestamp, method, request_path, body)
    return {
        'OK-ACCESS-KEY': self.api_key,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': self.passphrase,
        'Content-Type': 'application/json'
    }
```

**现在**：SDK 自动处理
```python
# SDK 初始化时配置认证信息
self.market_api = MarketData(
    api_key=self.api_key,
    api_secret_key=self.secret_key,
    passphrase=self.passphrase,
    flag='0',  # 0=实盘, 1=模拟盘
    proxy=proxy_url
)

# SDK 自动处理签名和请求头，无需手动实现
```

### 3. API 调用

**之前**：手动构建 HTTP 请求
```python
def get_klines(self, symbol, interval, limit=500):
    url = f"{self.base_url}/api/v5/market/candles"
    params = {
        'instId': okx_symbol,
        'bar': okx_interval,
        'limit': str(limit)
    }
    response = requests.get(url, params=params, proxies=self.proxies, timeout=30)
    response.raise_for_status()
    data = response.json()
    # 处理响应...
```

**现在**：使用 SDK 方法
```python
def get_klines(self, symbol, interval, limit=500):
    # 使用 SDK 的市场数据 API
    response = self.market_api.get_candlesticks(
        instId=okx_symbol,
        bar=okx_interval,
        limit=str(limit)
    )
    
    if response['code'] != '0':
        logger.error(f"获取K线失败: {response['msg']}")
        return []
    
    # 转换为统一格式...
```

### 4. 错误处理

**之前**：手动解析错误码
```python
def _handle_response(self, response):
    data = response.json()
    code = data.get('code', '')
    msg = data.get('msg', '')
    
    if code != '0':
        if code in ['50011', '50014']:
            raise ExchangeRateLimitError(f"Rate limit exceeded: {msg}")
        if code in ['50100', '50101', '50102', '50103']:
            raise ExchangeAuthError(f"Authentication failed: {msg}")
        raise ExchangeAPIError(code, msg)
    
    return data
```

**现在**：捕获 SDK 异常并转换
```python
def _handle_sdk_exception(self, e):
    """转换 SDK 异常为统一异常类型"""
    if isinstance(e, OkxAPIException):
        code = e.code
        message = e.message
        
        if code in ['50011', '50014']:
            raise ExchangeRateLimitError(f"Rate limit exceeded: {message}")
        if code in ['50100', '50101', '50102', '50103']:
            raise ExchangeAuthError(f"Authentication failed: {message}")
        raise ExchangeAPIError(code, message)
        
    elif isinstance(e, OkxRequestException):
        raise ExchangeConnectionError(f"Request failed: {str(e)}")
        
    elif isinstance(e, OkxParamsException):
        raise ExchangeInvalidParameterError(f"Invalid parameters: {str(e)}")
    
    else:
        raise ExchangeError(f"Unknown error: {str(e)}")
```

### 5. 代理配置

**之前**：手动配置 requests 代理
```python
self.proxies = None
if settings.USE_PROXY:
    proxy_url = f"{settings.PROXY_TYPE}://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
    self.proxies = {
        "http": proxy_url,
        "https": proxy_url
    }

response = requests.get(url, proxies=self.proxies, timeout=10)
```

**现在**：SDK 原生支持代理
```python
proxy = None
if settings.USE_PROXY:
    proxy_type = settings.PROXY_TYPE.lower()
    if proxy_type == "socks5":
        proxy = f"socks5h://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
    else:
        proxy = f"{proxy_type}://{settings.PROXY_HOST}:{settings.PROXY_PORT}"

# SDK 初始化时传入代理配置
self.market_api = MarketData(
    api_key=self.api_key,
    api_secret_key=self.secret_key,
    passphrase=self.passphrase,
    flag='0',
    proxy=proxy  # SDK 自动处理代理
)
```

## SDK 优势

### 1. 官方维护
- ✅ OKX 官方团队维护
- ✅ API 变更会及时更新
- ✅ 持续的 bug 修复和改进

### 2. 自动认证和签名
- ✅ 无需手动实现 HMAC-SHA256 算法
- ✅ 自动生成正确的请求头
- ✅ 减少认证相关的错误

### 3. 类型安全
- ✅ 提供类型提示
- ✅ IDE 自动补全
- ✅ 减少参数错误

### 4. 标准化异常
- ✅ 定义了标准异常类型
- ✅ 便于统一错误处理
- ✅ 更好的错误信息

### 5. 原生代理支持
- ✅ 支持 HTTP/HTTPS/SOCKS5 代理
- ✅ 配置简单
- ✅ 无需额外依赖

## 迁移步骤

### 步骤 1: 安装 SDK
```bash
pip install python-okx==0.4.0
```

### 步骤 2: 更新 OKXClient 初始化
```python
from okx import Account, MarketData, Trade, PublicData

class OKXClient(BaseExchangeClient):
    def __init__(self, config=None):
        # 初始化 SDK API 客户端
        self.account_api = Account(
            api_key=self.api_key,
            api_secret_key=self.secret_key,
            passphrase=self.passphrase,
            flag='0',
            proxy=proxy
        )
        
        self.market_api = MarketData(...)
        self.trade_api = Trade(...)
        self.public_api = PublicData(...)
```

### 步骤 3: 替换 API 调用
将所有手动 HTTP 请求替换为 SDK 方法调用：
- `requests.get()` → `self.market_api.get_candlesticks()`
- `requests.post()` → `self.trade_api.place_order()`
- 等等

### 步骤 4: 更新错误处理
捕获 SDK 异常并转换为统一异常类型。

### 步骤 5: 测试
- 测试连接
- 测试市场数据获取
- 测试交易功能
- 测试错误处理

## 注意事项

### WebSocket
python-okx SDK 0.4.0 主要提供 REST API 封装，WebSocket 功能仍需使用 websocket-client 库手动实现。

### 统一接口
虽然底层使用 SDK，但仍需保持 BaseExchangeClient 统一接口，确保业务代码无需修改。

### 数据转换
SDK 返回的数据格式需要转换为系统的统一格式（UnifiedKlineData、UnifiedTickerData 等）。

## 配置示例

```bash
# .env
EXCHANGE_TYPE=OKX

# OKX SDK 配置
OKX_API_KEY=your_api_key
OKX_SECRET_KEY=your_secret_key
OKX_PASSPHRASE=your_passphrase
OKX_TESTNET=false  # false=实盘, true=模拟盘

# 代理配置
USE_PROXY=true
PROXY_HOST=127.0.0.1
PROXY_PORT=10808
PROXY_TYPE=socks5
```

## 故障排查

### 问题 1: SDK 导入失败
```
ImportError: No module named 'okx'
```
**解决方案**：安装 SDK
```bash
pip install python-okx==0.4.0
```

### 问题 2: 认证失败
```
OkxAPIException: Authentication failed
```
**解决方案**：
1. 检查 API Key、Secret Key、Passphrase 是否正确
2. 检查 API Key 是否启用了合约交易权限
3. 检查 flag 参数是否正确（0=实盘, 1=模拟盘）

### 问题 3: 代理连接失败
```
OkxRequestException: Request failed
```
**解决方案**：
1. 检查代理服务器是否运行
2. 检查代理地址和端口是否正确
3. 尝试不使用代理（USE_PROXY=false）

## 参考资料

- [python-okx GitHub](https://github.com/okx/python-okx)
- [OKX API 文档](https://www.okx.com/docs-v5/zh/)
- [项目规范文档](.kiro/specs/okx-exchange-integration/)
