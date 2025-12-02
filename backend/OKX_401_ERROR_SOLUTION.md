# OKX 401 Unauthorized 错误解决方案

## 问题描述

从日志可以看到：
```
❌ 获取OKX账户信息失败: 401 Client Error: Unauthorized for url: https://www.okx.com/api/v5/account/balance
```

这是OKX API认证失败的典型错误。

## 快速诊断

运行诊断脚本：
```bash
python test_okx_auth.py
```

这个脚本会：
1. ✅ 检查配置信息
2. ✅ 测试公共接口（无需认证）
3. ✅ 测试签名生成
4. ✅ 测试私有接口（需要认证）
5. 💡 提供具体的错误原因和解决方案

## 常见原因和解决方案

### 1. API Key权限不足 ⭐ 最常见

**症状**: 401 Unauthorized

**解决方案**:
1. 登录 [OKX官网](https://www.okx.com)
2. 进入 **个人中心** → **API**
3. 找到你的API Key
4. 确保已启用以下权限：
   - ✅ **读取** (Read)
   - ✅ **交易** (Trade)
   - ✅ **合约交易** (Futures) - **必需！**

**注意**: 如果没有启用"合约交易"权限，即使API Key正确也会返回401错误。

### 2. IP白名单限制

**症状**: 401 Unauthorized 或 50111错误码

**解决方案**:

**方案A - 添加IP到白名单**:
1. 在API管理页面查看IP限制
2. 获取当前IP地址：
   ```bash
   curl https://api.ipify.org
   ```
3. 将IP添加到白名单

**方案B - 暂时移除IP限制（推荐用于测试）**:
1. 在API管理页面
2. 编辑API Key
3. 移除IP白名单限制
4. 测试是否能正常访问

### 3. API密钥配置错误

**症状**: 401 Unauthorized 或 50100/50104/50105错误码

**检查清单**:
- [ ] API Key长度是否为36个字符
- [ ] Secret Key长度是否为64个字符
- [ ] Passphrase是否正确（区分大小写）
- [ ] 配置文件中是否有多余的空格或换行符

**验证方法**:
```python
from app.core.config import settings

print(f"API Key长度: {len(settings.OKX_API_KEY)}")  # 应该是36
print(f"Secret Key长度: {len(settings.OKX_SECRET_KEY)}")  # 应该是64
print(f"Passphrase: {settings.OKX_PASSPHRASE}")  # 检查是否正确
```

### 4. 时间戳不同步

**症状**: 50102错误码

**解决方案**:
```bash
# Windows同步时间
w32tm /resync

# 检查系统时间
echo %date% %time%
```

### 5. 代理配置问题

**症状**: 连接超时或代理错误

**检查清单**:
- [ ] 代理服务器是否正在运行
- [ ] 代理地址和端口是否正确
- [ ] 代理类型是否正确（SOCKS5/HTTP）

**测试代理**:
```bash
# 测试代理连接
curl -x socks5://127.0.0.1:10808 https://www.okx.com/api/v5/public/time
```

## 推荐的排查顺序

### 第1步：检查API Key权限 ⭐
这是最常见的问题！
1. 登录OKX官网
2. 检查API Key权限
3. 确保启用了"合约交易"权限

### 第2步：检查IP白名单
1. 暂时移除IP限制
2. 测试是否能访问
3. 如果可以，说明是IP问题

### 第3步：验证API密钥
1. 运行诊断脚本 `python test_okx_auth.py`
2. 检查配置信息是否正确
3. 如果需要，重新创建API Key

### 第4步：同步时间
```bash
w32tm /resync
```

### 第5步：测试代理
1. 确认代理服务器运行正常
2. 测试代理连接
3. 如果代理有问题，暂时禁用代理测试

## 创建新的API Key

如果以上方法都不行，建议创建新的API Key：

1. 登录 [OKX官网](https://www.okx.com)
2. 进入 **个人中心** → **API**
3. 点击 **创建API Key**
4. 设置权限：
   - ✅ 读取
   - ✅ 交易
   - ✅ 合约交易 ⭐
5. 设置IP白名单（可选，建议先不设置）
6. 记录以下信息：
   - API Key
   - Secret Key
   - Passphrase
7. 更新`.env`文件：
   ```bash
   OKX_API_KEY=your_new_api_key
   OKX_SECRET_KEY=your_new_secret_key
   OKX_PASSPHRASE=your_new_passphrase
   ```

## 临时解决方案

如果暂时无法解决OKX API问题，可以：

### 方案1：切换到Mock模式
```bash
# .env
EXCHANGE_TYPE=MOCK
```

### 方案2：切换回Binance
```bash
# .env
EXCHANGE_TYPE=BINANCE
```

## 验证修复

修复后，重启系统并查看日志：

```bash
python main.py
```

成功的日志应该显示：
```
✅ OKX客户端创建成功
✓ OKX服务器时间获取成功
✓ OKX账户信息获取成功  ← 这行很重要！
```

## OKX API错误码参考

| 错误码 | 含义 | 解决方案 |
|--------|------|----------|
| 401 | 未授权 | 检查API密钥、权限、IP白名单 |
| 403 | 禁止访问 | 检查API权限，确保启用合约交易 |
| 50100 | API Key不正确 | 检查OKX_API_KEY配置 |
| 50101 | API Key已过期 | 重新创建API Key |
| 50102 | 时间戳错误 | 同步系统时间 |
| 50103 | 请求头不正确 | 检查请求头格式 |
| 50104 | Passphrase不正确 | 检查OKX_PASSPHRASE配置 |
| 50105 | 签名不正确 | 检查OKX_SECRET_KEY配置 |
| 50111 | IP不在白名单 | 添加IP或移除IP限制 |
| 50113 | API Key权限不足 | 启用合约交易权限 |
| 429 | 请求过多 | 降低请求频率 |
| 50011 | 请求频率过高 | 增加请求间隔 |

## 需要帮助？

如果问题仍然存在：

1. 运行诊断脚本获取详细信息：
   ```bash
   python test_okx_auth.py > okx_diagnosis.txt
   ```

2. 检查OKX官方文档：
   - [API认证说明](https://www.okx.com/docs-v5/zh/#overview-rest-authentication)
   - [错误码说明](https://www.okx.com/docs-v5/zh/#error-code)

3. 联系OKX客服

4. 在项目Issues中报告问题（记得隐藏敏感信息）

---

**重要提示**: 
- 请妥善保管API密钥，不要在代码中硬编码或分享给他人
- 建议定期更换API密钥
- 生产环境建议设置IP白名单
- 建议只授予必需的权限
