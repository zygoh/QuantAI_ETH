# OKX交易所运行指南

## 📋 快速开始

### 步骤1: 配置环境变量

在项目根目录创建或编辑`.env`文件，添加以下配置：

```bash
# 交易所选择（必须设置为OKX）
EXCHANGE_TYPE=OKX

# OKX API配置（必须填写）
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_PASSPHRASE=your_okx_passphrase
OKX_TESTNET=True  # True=测试网, False=生产环境

# 其他配置（可选，根据需要设置）
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/quantai_eth

# 日志级别
LOG_LEVEL=INFO
```

### 步骤2: 获取OKX API密钥

1. **登录OKX官网**
   - 访问: https://www.okx.com
   - 登录你的账户

2. **创建API密钥**
   - 进入"API管理"页面
   - 点击"创建API密钥"
   - 设置API密钥名称（如：QuantAI-ETH）
   - **重要**: 选择权限：
     - ✅ 读取权限（必须）
     - ✅ 交易权限（必须，用于下单）
     - ✅ 提币权限（可选，根据需求）
   - 设置IP白名单（可选，建议设置以提高安全性）
   - 设置Passphrase（记住这个密码，配置时需要）

3. **复制密钥信息**
   - API Key
   - Secret Key
   - Passphrase

### 步骤3: 配置API密钥

#### 方式1: 使用.env文件（推荐）

在项目根目录的`.env`文件中设置：

```bash
OKX_API_KEY=your_actual_api_key
OKX_SECRET_KEY=your_actual_secret_key
OKX_PASSPHRASE=your_actual_passphrase
OKX_TESTNET=True  # 首次使用建议True（测试网）
```

#### 方式2: 直接修改config.py（不推荐，仅用于测试）

编辑`app/core/config.py`：

```python
# 交易所选择配置
EXCHANGE_TYPE: str = "OKX"  # 确保是OKX

# OKX API配置
OKX_API_KEY: str = "your_actual_api_key"
OKX_SECRET_KEY: str = "your_actual_secret_key"
OKX_PASSPHRASE: str = "your_actual_passphrase"
OKX_TESTNET: bool = True  # True=测试网, False=生产环境
```

**⚠️ 注意**: 不要将API密钥提交到Git仓库！

### 步骤4: 安装依赖

确保已安装所有必需的Python包：

```powershell
# 激活虚拟环境（如果使用）
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 步骤5: 启动系统

#### 方式1: 直接运行（开发环境）

```powershell
python main.py
```

#### 方式2: 使用uvicorn（生产环境）

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 步骤6: 验证连接

启动后，查看日志输出，应该看到：

```
✅ 从配置读取交易所类型: OKX
🔧 创建新的OKX客户端实例...
✅ OKX客户端创建成功
✅ OKX配置验证通过
✓ 服务器时间获取成功: 1234567890000
✓ 账户信息获取成功
✅ OKX客户端初始化完成
   - 交易所类型: OKX
```

## 🔍 常见问题排查

### 问题1: 启动失败 - "Unsupported exchange type"

**原因**: `EXCHANGE_TYPE`配置错误

**解决**:
```bash
# 检查.env文件或config.py
EXCHANGE_TYPE=OKX  # 必须是 OKX（大写）
```

### 问题2: API连接失败 - "OKX API连接失败"

**可能原因**:
1. API密钥配置错误
2. API密钥权限不足
3. IP白名单限制
4. 网络连接问题

**解决步骤**:
1. 检查API密钥是否正确：
   ```python
   # 在Python中测试
   from app.exchange.exchange_factory import ExchangeFactory
   client = ExchangeFactory.create_client("OKX")
   result = await client.test_connection()
   print(result)
   ```

2. 检查API密钥权限：
   - 登录OKX → API管理
   - 确认已启用"读取"和"交易"权限

3. 检查IP白名单：
   - 如果设置了IP白名单，确保服务器IP在列表中
   - 或者临时移除IP白名单进行测试

4. 检查网络连接：
   ```powershell
   # 测试OKX API连接
   curl https://www.okx.com/api/v5/public/time
   ```

### 问题3: WebSocket连接失败

**可能原因**:
1. 防火墙阻止WebSocket连接
2. 代理配置问题
3. OKX WebSocket服务暂时不可用

**解决**:
1. 检查防火墙设置
2. 如果使用代理，配置代理：
   ```bash
   USE_PROXY=True
   PROXY_TYPE=socks5  # 或 http
   PROXY_HOST=127.0.0.1
   PROXY_PORT=1080
   ```

3. 查看日志中的WebSocket错误信息

### 问题4: 交易对格式错误

**原因**: OKX使用不同的交易对格式

**解决**: 系统已自动处理格式转换
- 输入: `ETHUSDT`（标准格式）
- 自动转换为: `ETH-USDT-SWAP`（OKX格式）

**注意**: 确保交易对在OKX上存在且可交易

### 问题5: 测试网vs生产环境

**测试网配置**:
```bash
OKX_TESTNET=True
```

**生产环境配置**:
```bash
OKX_TESTNET=False
```

**⚠️ 重要**: 
- 测试网使用虚拟资金，不会产生真实交易
- 生产环境使用真实资金，请谨慎操作
- 首次使用强烈建议先在测试网验证

## 📊 验证系统运行状态

### 1. 检查API连接

```python
from app.exchange.exchange_factory import ExchangeFactory

client = ExchangeFactory.get_current_client()
result = await client.test_connection()
print(f"连接状态: {result}")
```

### 2. 测试数据获取

```python
# 获取K线数据
klines = client.get_klines("ETHUSDT", "5m", limit=10)
print(f"获取到 {len(klines)} 条K线数据")

# 获取价格
ticker = client.get_ticker_price("ETHUSDT")
print(f"当前价格: {ticker.price}")
```

### 3. 检查WebSocket连接

查看日志中是否有：
```
✅ WebSocket连接成功
✅ 订阅K线数据: ETHUSDT-5m
✅ 订阅价格数据: ETHUSDT
```

### 4. 检查系统状态

访问API端点：
```powershell
# 检查系统状态
curl http://localhost:8000/api/system/status

# 检查账户信息
curl http://localhost:8000/api/account/info
```

## 🔄 从Binance切换到OKX

如果你之前使用的是Binance，切换到OKX的步骤：

1. **备份当前配置**
   ```bash
   # 备份.env文件
   copy .env .env.backup
   ```

2. **修改交易所类型**
   ```bash
   # 在.env文件中
   EXCHANGE_TYPE=OKX  # 从BINANCE改为OKX
   ```

3. **配置OKX API密钥**
   ```bash
   OKX_API_KEY=your_okx_api_key
   OKX_SECRET_KEY=your_okx_secret_key
   OKX_PASSPHRASE=your_okx_passphrase
   ```

4. **重启系统**
   ```powershell
   # 停止当前运行的系统（Ctrl+C）
   # 重新启动
   python main.py
   ```

5. **验证切换成功**
   - 查看日志确认使用OKX客户端
   - 测试数据获取功能
   - 验证交易功能（建议先在测试网）

## 🎯 生产环境部署建议

### 1. 安全配置

```bash
# 使用环境变量而不是硬编码
# 不要将API密钥提交到Git
# 使用密钥管理服务（如AWS Secrets Manager）

# 设置IP白名单
# 限制API密钥权限（只启用必要的权限）
```

### 2. 监控和日志

```bash
# 设置日志级别
LOG_LEVEL=INFO  # 或 WARNING（生产环境）

# 配置日志轮转
# 监控系统运行状态
# 设置告警机制
```

### 3. 性能优化

```bash
# 配置合理的限流延迟
# 使用连接池
# 启用缓存机制
```

## 📝 配置检查清单

启动前请确认：

- [ ] `EXCHANGE_TYPE=OKX`已设置
- [ ] OKX API密钥已正确配置
- [ ] OKX Secret Key已正确配置
- [ ] OKX Passphrase已正确配置
- [ ] `OKX_TESTNET`已根据需求设置（测试/生产）
- [ ] API密钥已启用必要权限（读取、交易）
- [ ] IP白名单已配置（如果使用）
- [ ] 数据库连接正常
- [ ] 所有依赖已安装
- [ ] 网络连接正常

## 🆘 获取帮助

如果遇到问题：

1. **查看日志文件**
   ```
   logs/trading_system.log
   ```

2. **检查API文档**
   - OKX API文档: https://www.okx.com/docs-v5/zh/

3. **查看系统文档**
   - 迁移指南: `MIGRATION_GUIDE.md`
   - 集成总结: `OKX_INTEGRATION_SUMMARY.md`

4. **测试连接**
   ```python
   from app.exchange.exchange_factory import ExchangeFactory
   client = ExchangeFactory.create_client("OKX")
   await client.test_connection()
   ```

---

**版本**: v1.0  
**更新日期**: 2025-01-26  
**适用系统**: QuantAI-ETH v10.1+

