# Windows系统运行指南

## 🖥️ Windows环境要求

### 系统要求
- Windows 10/11 (64位)
- 内存: 8GB以上
- 存储: 20GB以上可用空间
- 网络: 稳定的互联网连接

### 必需软件
1. **Docker Desktop for Windows**
2. **Git for Windows** (可选)
3. **文本编辑器** (如Notepad++, VS Code)

## 📦 安装Docker Desktop

### 1. 下载Docker Desktop
访问官网下载: https://www.docker.com/products/docker-desktop

### 2. 安装步骤
1. 运行下载的安装程序
2. 勾选 "Use WSL 2 instead of Hyper-V" (推荐)
3. 完成安装后重启电脑
4. 启动Docker Desktop
5. 等待Docker引擎启动完成

### 3. 验证安装
打开命令提示符(CMD)或PowerShell，运行：
```cmd
docker --version
docker-compose --version
```

## 🚀 快速启动

### 方法1: 使用批处理脚本 (推荐)

1. **配置API密钥**
   ```cmd
   # 编辑配置文件
   notepad backend\.env
   ```
   
   在文件中设置您的Binance API密钥：
   ```
   BINANCE_API_KEY=your-api-key
   BINANCE_SECRET_KEY=your-secret-key
   BINANCE_PASSPHRASE=your-passphrase
   ```

2. **启动系统**
   双击运行 `start.bat` 文件，或在命令行中运行：
   ```cmd
   start.bat
   ```

3. **访问应用**
   - 前端界面: http://localhost:3000
   - 后端API: http://localhost:8000
   - InfluxDB: http://localhost:8086

### 方法2: 使用命令行

1. **打开命令提示符**
   - 按 `Win + R`，输入 `cmd`，回车
   - 或按 `Win + X`，选择 "Windows PowerShell"

2. **导航到项目目录**
   ```cmd
   cd C:\path\to\eth-trading-system
   ```

3. **启动服务**
   ```cmd
   docker-compose up -d
   ```

## 🔧 系统管理

### 查看服务状态
```cmd
docker-compose ps
```

### 查看日志
```cmd
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
```

### 重启服务
```cmd
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart backend
```

### 停止服务
```cmd
docker-compose down
```

### 系统状态检查
双击运行 `check-system.bat` 或：
```cmd
check-system.bat
```

## 🛠️ 常见问题解决

### 1. Docker Desktop启动失败
**问题**: Docker Desktop无法启动
**解决方案**:
- 确保已启用Windows的Hyper-V或WSL 2功能
- 以管理员身份运行Docker Desktop
- 重启Windows服务中的Docker相关服务

### 2. 端口占用问题
**问题**: 端口3000或8000被占用
**解决方案**:
```cmd
# 查看端口占用
netstat -ano | findstr :3000
netstat -ano | findstr :8000

# 结束占用进程
taskkill /PID <进程ID> /F
```

### 3. 内存不足
**问题**: Docker容器启动失败，提示内存不足
**解决方案**:
- 在Docker Desktop设置中增加内存分配
- 关闭不必要的应用程序
- 重启电脑释放内存

### 4. 网络连接问题
**问题**: 无法连接到Binance API
**解决方案**:
- 检查网络连接
- 确认防火墙设置
- 验证API密钥是否正确

### 5. 权限问题
**问题**: 文件访问权限不足
**解决方案**:
- 以管理员身份运行命令提示符
- 检查文件夹权限设置
- 确保Docker有足够的权限

## 📁 Windows特定配置

### 环境变量设置
在Windows中，您可以通过以下方式设置环境变量：

1. **临时设置** (当前会话有效):
   ```cmd
   set BINANCE_API_KEY=your-api-key
   set BINANCE_SECRET_KEY=your-secret-key
   ```

2. **永久设置**:
   - 右键"此电脑" → "属性" → "高级系统设置"
   - 点击"环境变量"
   - 在"用户变量"中添加新变量

### 文件路径注意事项
- Windows使用反斜杠 `\` 作为路径分隔符
- 在配置文件中使用正斜杠 `/` 或双反斜杠 `\\`
- 避免路径中包含空格和特殊字符

## 🔄 自动启动设置

### 设置开机自启动
1. 按 `Win + R`，输入 `shell:startup`
2. 将 `start.bat` 的快捷方式复制到启动文件夹
3. 重启电脑验证自动启动

### 创建桌面快捷方式
1. 右键桌面 → "新建" → "快捷方式"
2. 浏览到 `start.bat` 文件
3. 设置快捷方式名称为 "ETH交易系统"

## 📊 性能优化

### Docker Desktop设置优化
1. 打开Docker Desktop设置
2. 在"Resources"中调整：
   - CPU: 分配2-4个核心
   - Memory: 分配4-8GB内存
   - Disk: 确保有足够空间

### Windows系统优化
1. 关闭不必要的启动程序
2. 定期清理磁盘空间
3. 保持系统更新
4. 使用SSD硬盘提升性能

## 🆘 获取帮助

如果遇到问题：

1. **查看日志**:
   ```cmd
   docker-compose logs -f
   ```

2. **检查系统状态**:
   ```cmd
   check-system.bat
   ```

3. **重启服务**:
   ```cmd
   docker-compose restart
   ```

4. **完全重置**:
   ```cmd
   docker-compose down
   docker system prune -f
   start.bat
   ```

## 📞 技术支持

- 查看项目文档: [README.md](README.md)
- 部署指南: [DEPLOYMENT.md](DEPLOYMENT.md)
- 项目总结: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

**🎉 现在您可以在Windows系统上轻松运行ETH合约中频智能交易系统了！**