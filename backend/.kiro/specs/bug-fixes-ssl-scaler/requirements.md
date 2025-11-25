# 需求文档 - SSL连接和梯度缩放器问题修复

## 简介

本规范旨在修复QuantAI-ETH交易系统中的两个关键生产问题：
1. Binance WebSocket SSL连接不稳定导致的频繁断连
2. Informer2模型训练时混合精度梯度缩放器(GradScaler)的scale值异常增长

这些问题影响系统稳定性和模型训练质量，需要以生产级别的专业标准进行修复。

## 术语表

- **WebSocket**: 全双工通信协议，用于实时接收Binance市场数据
- **SSL/TLS**: 安全套接字层协议，用于加密网络通信
- **GradScaler**: PyTorch混合精度训练中的梯度缩放器，用于防止数值下溢
- **AMP**: Automatic Mixed Precision，自动混合精度训练
- **Informer2**: 基于Transformer的时间序列预测模型
- **GMADL**: Gradient-weighted Multi-task Adaptive Deep Learning损失函数
- **Binance Client**: 币安交易所API客户端服务
- **Hyperparameter Optimizer**: 超参数优化器，使用Optuna进行自动调优

## 需求

### 需求1: WebSocket SSL连接稳定性

**用户故事**: 作为量化交易系统运维人员，我希望WebSocket连接能够稳定运行，这样系统就能持续接收实时市场数据而不会因SSL错误频繁断连。

#### 验收标准

1. WHEN Binance Client检测到SSL错误"DECRYPTION_FAILED_OR_BAD_RECORD_MAC"，THE Binance Client SHALL实现指数退避重连策略，初始延迟为1秒，最大延迟为60秒
2. WHILE WebSocket连接处于活动状态，THE Binance Client SHALL每30秒发送一次心跳ping消息以保持连接活跃
3. IF WebSocket连接在5分钟内重连失败次数超过3次，THEN THE Binance Client SHALL记录ERROR级别日志并通知监控系统
4. WHEN WebSocket连接成功重建后，THE Binance Client SHALL清零重连计数器并恢复正常数据订阅
5. THE Binance Client SHALL在SSL握手阶段设置30秒超时，防止无限等待

### 需求2: 梯度缩放器数值稳定性

**用户故事**: 作为量化模型开发工程师，我希望Informer2模型训练时梯度缩放器的scale值保持在合理范围内，这样模型训练就能稳定收敛而不会出现数值溢出。

#### 验收标准

1. WHEN Hyperparameter Optimizer初始化GradScaler时，THE Hyperparameter Optimizer SHALL根据模型参数量动态设置初始缩放因子：小于1M参数使用2^16，1M-10M参数使用2^14，大于10M参数使用2^12
2. WHILE 模型训练过程中，THE Hyperparameter Optimizer SHALL设置growth_factor为1.2（从1.5降低），growth_interval为2000（从1000增加），以减缓scale增长速度
3. IF GradScaler的scale值超过100000，THEN THE Hyperparameter Optimizer SHALL记录WARNING级别日志并在连续3个epoch后自动重置scale为初始值的50%
4. WHEN 检测到连续5个batch出现梯度溢出时，THE Hyperparameter Optimizer SHALL将scale值减半并记录详细诊断信息
5. THE Hyperparameter Optimizer SHALL在每个epoch结束时记录当前scale值，便于监控和调试

### 需求3: 错误监控和诊断

**用户故事**: 作为系统管理员，我希望系统能够详细记录SSL错误和梯度缩放问题的诊断信息，这样我就能快速定位和解决问题根源。

#### 验收标准

1. WHEN Binance Client发生SSL错误时，THE Binance Client SHALL记录完整的错误堆栈、连接状态、重连次数和时间戳
2. WHEN Hyperparameter Optimizer检测到scale异常时，THE Hyperparameter Optimizer SHALL记录当前epoch、batch、scale值、梯度范数和模型输出统计信息
3. THE Binance Client SHALL维护最近10次连接失败的历史记录，包括失败原因和时间间隔
4. THE Hyperparameter Optimizer SHALL在训练开始前验证输入数据范围，确保无NaN/Inf值
5. IF 系统检测到重复性错误模式（如每小时固定时间断连），THEN THE System SHALL生成分析报告并发送告警

### 需求4: 配置灵活性

**用户故事**: 作为系统配置管理员，我希望能够通过配置文件调整重连策略和缩放器参数，这样就能根据不同环境和网络条件优化系统行为。

#### 验收标准

1. THE System SHALL在配置文件中提供WebSocket重连参数：max_retries、initial_delay、max_delay、backoff_factor
2. THE System SHALL在配置文件中提供GradScaler参数：init_scale_strategy、growth_factor、growth_interval、max_scale_threshold
3. WHEN 配置文件更新后，THE System SHALL在下次服务重启时应用新配置，无需修改代码
4. THE System SHALL验证配置参数的合理性，拒绝不合法的配置值（如负数延迟、零增长因子）
5. THE System SHALL为所有配置参数提供合理的默认值，确保在配置缺失时系统仍能正常运行

### 需求5: 向后兼容性

**用户故事**: 作为系统维护人员，我希望修复不会破坏现有功能，这样系统升级就能平滑进行而不影响正在运行的交易策略。

#### 验收标准

1. THE System SHALL保持现有API接口不变，所有公共方法签名保持兼容
2. THE System SHALL保持日志格式兼容，确保现有日志解析工具继续工作
3. THE System SHALL保持模型文件格式兼容，已训练的模型能够正常加载和使用
4. WHEN 引入新的配置参数时，THE System SHALL提供默认值，使得旧配置文件无需修改即可使用
5. THE System SHALL通过单元测试验证所有现有功能未被破坏
