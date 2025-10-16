# ETH合约中频智能交易系统部署指南

## 系统要求

### 硬件要求
- CPU: 4核心以上
- 内存: 8GB以上
- 存储: 50GB以上可用空间
- GPU: NVIDIA GPU (可选，用于模型训练加速)

### 软件要求
- Docker 20.10+
- Docker Compose 2.0+
- Git
- 操作系统: Linux/macOS/Windows

## 快速部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd eth-trading-system
```

### 2. 配置环境变量
```bash
# 复制环境配置文件
cp .env.example .env
cp .env.prod.example .env.prod

# 编辑配置文件，设置API密钥
nano .env.prod
```

### 3. 开发环境部署
```bash
# 启动开发环境
./start.sh
```

### 4. 生产环境部署
```bash
# 部署到生产环境
./deploy.sh
```

## 详细配置

### API密钥配置
在 `.env.prod` 文件中配置您的Binance API密钥：

```env
BINANCE_API_KEY=your-actual-api-key
BINANCE_SECRET_KEY=your-actual-secret-key
BINANCE_PASSPHRASE=your-actual-passphrase
```

### 数据库配置
系统使用InfluxDB存储时序数据，Redis作为缓存：

```env
INFLUXDB_PASSWORD=your-secure-password
INFLUXDB_TOKEN=your-secure-token
```

### 安全配置
生产环境建议：
- 使用HTTPS/WSS加密连接
- 设置防火墙规则
- 定期更新密钥
- 启用访问日志

## 服务管理

### 查看服务状态
```bash
docker-compose ps
```

### 查看日志
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
```

### 重启服务
```bash
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart backend
```

### 停止服务
```bash
docker-compose down
```

## 监控和维护

### 系统监控
访问监控面板：
- Grafana: http://localhost:3001 (admin/admin123)
- Prometheus: http://localhost:9090

### 数据备份
```bash
# 备份InfluxDB数据
docker exec eth-trading-influxdb influx backup /backup

# 备份Redis数据
docker exec eth-trading-redis redis-cli BGSAVE
```

### 日志管理
日志文件位置：
- 应用日志: `backend/logs/`
- Nginx日志: `/var/log/nginx/`
- 容器日志: `docker logs <container-name>`

## 故障排除

### 常见问题

1. **API连接失败**
   - 检查API密钥是否正确
   - 确认网络连接正常
   - 验证Binance API权限

2. **数据库连接失败**
   - 检查InfluxDB服务状态
   - 验证数据库配置
   - 查看数据库日志

3. **前端无法访问**
   - 检查Nginx配置
   - 验证端口映射
   - 查看前端构建日志

4. **WebSocket连接失败**
   - 检查防火墙设置
   - 验证代理配置
   - 查看WebSocket日志

### 性能优化

1. **数据库优化**
   - 定期清理旧数据
   - 优化查询索引
   - 调整内存配置

2. **缓存优化**
   - 监控Redis内存使用
   - 调整缓存过期时间
   - 优化缓存策略

3. **模型训练优化**
   - 使用GPU加速
   - 调整训练参数
   - 优化特征选择

## 安全建议

1. **API安全**
   - 使用只读API密钥（如果只需要数据）
   - 定期轮换API密钥
   - 限制API访问IP

2. **网络安全**
   - 使用HTTPS/WSS
   - 配置防火墙
   - 启用访问控制

3. **数据安全**
   - 定期备份数据
   - 加密敏感信息
   - 监控异常访问

## 更新升级

### 应用更新
```bash
# 拉取最新代码
git pull origin main

# 重新构建和部署
./deploy.sh
```

### 依赖更新
```bash
# 更新Docker镜像
docker-compose pull

# 重新构建
docker-compose build --no-cache
```

## 联系支持

如遇到问题，请：
1. 查看日志文件
2. 检查系统状态
3. 参考故障排除指南
4. 提交Issue到项目仓库