# Requirements Document

## Introduction

本需求文档定义了为QuantAI-ETH交易系统添加OKX交易所支持的功能需求。系统当前仅支持Binance交易所，需要扩展为支持多交易所架构，允许用户在Binance和OKX之间灵活切换，同时保持现有功能的完整性和系统的稳定性。

**技术方案更新**：使用官方 python-okx 0.4.0 SDK 作为底层实现，利用其提供的认证、签名、API调用等功能，同时保持系统的统一接口设计。

## Glossary

- **Exchange**: 加密货币交易所，提供交易、行情数据等服务
- **Trading System**: QuantAI-ETH量化交易系统
- **Exchange Client**: 交易所客户端，封装与交易所API的交互逻辑
- **Exchange Factory**: 交易所工厂类，负责创建和管理不同交易所的客户端实例
- **REST API**: 交易所提供的HTTP接口，用于获取数据和执行交易操作
- **WebSocket**: 交易所提供的实时数据推送接口
- **Trading Engine**: 交易执行引擎，负责执行交易信号
- **Signal Generator**: 信号生成器，负责生成交易信号
- **Position Manager**: 仓位管理器，负责管理持仓信息
- **Configuration**: 系统配置，包括API密钥、交易参数等
- **Proxy**: 代理服务器，用于访问交易所API
- **Unified Interface**: 统一接口，不同交易所客户端实现相同的接口规范
- **python-okx SDK**: OKX官方提供的Python SDK，版本0.4.0，用于处理OKX API的认证、签名和调用

## Requirements

### Requirement 1

**User Story:** 作为系统管理员，我希望能够通过配置文件选择使用Binance或OKX交易所，以便根据不同的交易需求灵活切换交易所。

#### Acceptance Criteria

1. WHEN 系统启动时 THEN Trading System SHALL 从配置文件读取当前选择的交易所类型
2. WHEN 配置文件中指定交易所为"BINANCE" THEN Trading System SHALL 初始化Binance客户端
3. WHEN 配置文件中指定交易所为"OKX" THEN Trading System SHALL 初始化OKX客户端
4. WHEN 配置文件中交易所类型无效 THEN Trading System SHALL 记录错误日志并使用默认交易所（Binance）
5. WHEN 交易所配置发生变化 THEN Trading System SHALL 在下次重启时应用新的交易所配置

### Requirement 2

**User Story:** 作为开发人员，我希望所有交易所客户端实现统一的接口规范，以便系统其他模块无需关心具体使用哪个交易所。

#### Acceptance Criteria

1. WHEN 定义交易所接口时 THEN Exchange Client SHALL 包含所有必需的交易和数据获取方法
2. WHEN Binance客户端实现接口时 THEN Binance Client SHALL 实现所有接口定义的方法
3. WHEN OKX客户端实现接口时 THEN OKX Client SHALL 实现所有接口定义的方法
4. WHEN 调用接口方法时 THEN Exchange Client SHALL 返回统一格式的数据结构
5. WHEN 接口方法执行失败时 THEN Exchange Client SHALL 抛出统一的异常类型

### Requirement 3

**User Story:** 作为系统架构师，我希望使用工厂模式创建交易所客户端实例，以便集中管理客户端的创建逻辑和生命周期。

#### Acceptance Criteria

1. WHEN 请求创建交易所客户端时 THEN Exchange Factory SHALL 根据配置返回对应的客户端实例
2. WHEN 请求的交易所类型为"BINANCE" THEN Exchange Factory SHALL 返回Binance客户端实例
3. WHEN 请求的交易所类型为"OKX" THEN Exchange Factory SHALL 返回OKX客户端实例
4. WHEN 请求的交易所类型不支持时 THEN Exchange Factory SHALL 抛出异常并记录错误日志
5. WHEN 客户端实例已存在时 THEN Exchange Factory SHALL 返回已有实例（单例模式）

### Requirement 4

**User Story:** 作为系统管理员，我希望能够为不同的交易所配置独立的API密钥和参数，以便同时管理多个交易所账户。

#### Acceptance Criteria

1. WHEN 配置Binance交易所时 THEN Configuration SHALL 包含Binance的API Key和Secret Key
2. WHEN 配置OKX交易所时 THEN Configuration SHALL 包含OKX的API Key、Secret Key和Passphrase
3. WHEN 配置交易所代理时 THEN Configuration SHALL 支持为每个交易所独立配置代理参数
4. WHEN 读取交易所配置时 THEN Configuration SHALL 验证必需参数的完整性
5. WHEN 配置参数缺失或无效时 THEN Configuration SHALL 记录警告日志并使用默认值

### Requirement 5

**User Story:** 作为交易员，我希望OKX客户端能够获取实时K线数据，以便系统能够基于最新市场数据生成交易信号。

#### Acceptance Criteria

1. WHEN 请求获取K线数据时 THEN OKX Client SHALL 调用OKX REST API获取历史K线
2. WHEN 订阅K线WebSocket时 THEN OKX Client SHALL 建立WebSocket连接并接收实时K线推送
3. WHEN 接收到K线数据时 THEN OKX Client SHALL 将数据转换为统一格式
4. WHEN K线数据包含必需字段时 THEN OKX Client SHALL 验证数据完整性
5. WHEN K线数据获取失败时 THEN OKX Client SHALL 记录错误并返回空列表

### Requirement 6

**User Story:** 作为交易员，我希望OKX客户端能够获取实时价格数据，以便系统能够监控市场价格变化。

#### Acceptance Criteria

1. WHEN 请求获取实时价格时 THEN OKX Client SHALL 调用OKX REST API获取最新价格
2. WHEN 订阅价格WebSocket时 THEN OKX Client SHALL 建立WebSocket连接并接收实时价格推送
3. WHEN 接收到价格数据时 THEN OKX Client SHALL 将数据转换为统一格式
4. WHEN 价格数据包含symbol和price字段时 THEN OKX Client SHALL 提取并返回这些字段
5. WHEN 价格数据获取失败时 THEN OKX Client SHALL 记录错误并返回None

### Requirement 7

**User Story:** 作为交易员，我希望OKX客户端能够执行市价单和限价单交易，以便系统能够自动执行交易策略。

#### Acceptance Criteria

1. WHEN 下市价单时 THEN OKX Client SHALL 调用OKX下单API并传递正确的参数
2. WHEN 下限价单时 THEN OKX Client SHALL 调用OKX下单API并包含价格参数
3. WHEN 下单成功时 THEN OKX Client SHALL 返回订单ID和订单详情
4. WHEN 下单失败时 THEN OKX Client SHALL 记录错误并返回失败状态
5. WHEN 订单参数无效时 THEN OKX Client SHALL 在发送请求前验证参数并抛出异常

### Requirement 8

**User Story:** 作为交易员，我希望OKX客户端能够查询和管理订单，以便系统能够跟踪订单状态和执行情况。

#### Acceptance Criteria

1. WHEN 查询订单状态时 THEN OKX Client SHALL 调用OKX查询订单API并返回订单详情
2. WHEN 取消订单时 THEN OKX Client SHALL 调用OKX取消订单API并返回取消结果
3. WHEN 查询未成交订单时 THEN OKX Client SHALL 返回所有未完成订单的列表
4. WHEN 订单查询失败时 THEN OKX Client SHALL 记录错误并返回空结果
5. WHEN 取消订单失败时 THEN OKX Client SHALL 记录错误并返回失败状态

### Requirement 9

**User Story:** 作为交易员，我希望OKX客户端能够查询账户信息和持仓情况，以便系统能够管理资金和风险。

#### Acceptance Criteria

1. WHEN 查询账户余额时 THEN OKX Client SHALL 调用OKX账户API并返回可用余额
2. WHEN 查询持仓信息时 THEN OKX Client SHALL 调用OKX持仓API并返回当前持仓列表
3. WHEN 账户信息包含多个币种时 THEN OKX Client SHALL 返回所有币种的余额信息
4. WHEN 持仓信息包含未实现盈亏时 THEN OKX Client SHALL 计算并返回盈亏数据
5. WHEN 账户查询失败时 THEN OKX Client SHALL 记录错误并返回空字典

### Requirement 10

**User Story:** 作为交易员，我希望OKX客户端能够设置和修改杠杆倍数，以便根据风险偏好调整交易杠杆。

#### Acceptance Criteria

1. WHEN 设置杠杆倍数时 THEN OKX Client SHALL 调用OKX杠杆设置API
2. WHEN 杠杆倍数在允许范围内时 THEN OKX Client SHALL 成功设置杠杆并返回确认
3. WHEN 杠杆倍数超出范围时 THEN OKX Client SHALL 记录错误并返回失败状态
4. WHEN 杠杆设置失败时 THEN OKX Client SHALL 记录详细错误信息
5. WHEN 查询当前杠杆倍数时 THEN OKX Client SHALL 从持仓信息中提取杠杆数据

### Requirement 11

**User Story:** 作为系统架构师，我希望OKX WebSocket客户端支持自动重连机制，以便在网络中断时自动恢复连接。

#### Acceptance Criteria

1. WHEN WebSocket连接断开时 THEN OKX WebSocket Client SHALL 自动尝试重新连接
2. WHEN 重连失败时 THEN OKX WebSocket Client SHALL 使用指数退避策略延迟下次重连
3. WHEN 重连成功时 THEN OKX WebSocket Client SHALL 恢复所有之前的订阅
4. WHEN 达到最大重连次数时 THEN OKX WebSocket Client SHALL 停止重连并记录错误
5. WHEN 连接稳定运行24小时时 THEN OKX WebSocket Client SHALL 主动重建连接以保持稳定性

### Requirement 12

**User Story:** 作为系统架构师，我希望OKX WebSocket客户端实现心跳保活机制，以便及时检测连接状态并防止连接超时。

#### Acceptance Criteria

1. WHEN WebSocket连接建立后 THEN OKX WebSocket Client SHALL 定期发送ping消息
2. WHEN 收到pong响应时 THEN OKX WebSocket Client SHALL 更新最后响应时间
3. WHEN pong响应超时时 THEN OKX WebSocket Client SHALL 记录警告并触发重连
4. WHEN 长时间未收到任何消息时 THEN OKX WebSocket Client SHALL 触发健康检查
5. WHEN 健康检查失败时 THEN OKX WebSocket Client SHALL 主动断开并重连

### Requirement 13

**User Story:** 作为开发人员，我希望Trading Engine能够无缝切换交易所，以便在不修改业务逻辑的情况下支持多个交易所。

#### Acceptance Criteria

1. WHEN Trading Engine初始化时 THEN Trading Engine SHALL 通过工厂获取当前配置的交易所客户端
2. WHEN 执行交易操作时 THEN Trading Engine SHALL 调用统一接口方法而非特定交易所方法
3. WHEN 切换交易所配置时 THEN Trading Engine SHALL 在重启后使用新的交易所客户端
4. WHEN 交易所客户端方法调用失败时 THEN Trading Engine SHALL 记录错误并执行降级策略
5. WHEN 虚拟交易模式时 THEN Trading Engine SHALL 使用相同的接口进行模拟交易

### Requirement 14

**User Story:** 作为开发人员，我希望Signal Generator能够从不同交易所获取数据，以便生成准确的交易信号。

#### Acceptance Criteria

1. WHEN Signal Generator初始化时 THEN Signal Generator SHALL 通过工厂获取当前配置的交易所客户端
2. WHEN 获取市场数据时 THEN Signal Generator SHALL 调用统一接口获取K线和价格数据
3. WHEN 交易所数据格式不同时 THEN Signal Generator SHALL 接收统一格式的数据
4. WHEN 数据获取失败时 THEN Signal Generator SHALL 记录错误并跳过当前信号生成周期
5. WHEN 切换交易所时 THEN Signal Generator SHALL 无需修改代码即可正常工作

### Requirement 15

**User Story:** 作为开发人员，我希望Data Service能够从不同交易所获取和存储数据，以便为系统提供统一的数据访问接口。

#### Acceptance Criteria

1. WHEN Data Service初始化时 THEN Data Service SHALL 通过工厂获取当前配置的交易所客户端
2. WHEN 订阅实时数据时 THEN Data Service SHALL 使用统一接口订阅WebSocket数据流
3. WHEN 接收到数据时 THEN Data Service SHALL 将数据存储到数据库中
4. WHEN 查询历史数据时 THEN Data Service SHALL 从数据库返回统一格式的数据
5. WHEN 切换交易所时 THEN Data Service SHALL 继续正常存储和提供数据

### Requirement 16

**User Story:** 作为系统管理员，我希望系统能够记录详细的交易所操作日志，以便追踪问题和审计交易行为。

#### Acceptance Criteria

1. WHEN 调用交易所API时 THEN Exchange Client SHALL 记录请求参数和响应结果
2. WHEN API调用失败时 THEN Exchange Client SHALL 记录详细的错误信息和堆栈跟踪
3. WHEN WebSocket连接状态变化时 THEN Exchange Client SHALL 记录连接事件
4. WHEN 执行交易操作时 THEN Exchange Client SHALL 记录订单详情和执行结果
5. WHEN 日志级别为DEBUG时 THEN Exchange Client SHALL 记录所有API交互细节

### Requirement 17

**User Story:** 作为系统架构师，我希望系统能够优雅地处理交易所API限流，以便避免因请求过多而被封禁。

#### Acceptance Criteria

1. WHEN 检测到API限流错误时 THEN Exchange Client SHALL 自动延迟后续请求
2. WHEN 连续触发限流时 THEN Exchange Client SHALL 增加延迟时间
3. WHEN 限流恢复后 THEN Exchange Client SHALL 逐步恢复正常请求频率
4. WHEN 分页获取大量数据时 THEN Exchange Client SHALL 在请求之间添加延迟
5. WHEN 达到每日请求限制时 THEN Exchange Client SHALL 记录错误并停止非关键请求

### Requirement 18

**User Story:** 作为测试人员，我希望能够在测试环境中使用模拟交易所客户端，以便在不连接真实交易所的情况下测试系统功能。

#### Acceptance Criteria

1. WHEN 配置为测试模式时 THEN Exchange Factory SHALL 返回模拟交易所客户端
2. WHEN 调用模拟客户端方法时 THEN Mock Exchange Client SHALL 返回预定义的测试数据
3. WHEN 执行模拟交易时 THEN Mock Exchange Client SHALL 记录交易操作但不发送真实请求
4. WHEN 测试WebSocket功能时 THEN Mock Exchange Client SHALL 模拟数据推送
5. WHEN 测试错误处理时 THEN Mock Exchange Client SHALL 能够模拟各种错误场景

### Requirement 19

**User Story:** 作为系统管理员，我希望系统能够验证交易所API连接的有效性，以便在启动时及早发现配置问题。

#### Acceptance Criteria

1. WHEN 系统启动时 THEN Exchange Client SHALL 执行连接测试
2. WHEN 测试REST API连接时 THEN Exchange Client SHALL 调用服务器时间接口
3. WHEN 测试API密钥有效性时 THEN Exchange Client SHALL 调用账户信息接口
4. WHEN 连接测试成功时 THEN Exchange Client SHALL 记录成功日志并继续启动
5. WHEN 连接测试失败时 THEN Exchange Client SHALL 记录详细错误并根据配置决定是否继续启动

### Requirement 20

**User Story:** 作为开发人员，我希望交易所客户端代码遵循项目的开发标准，以便保持代码质量和可维护性。

#### Acceptance Criteria

1. WHEN 编写交易所客户端代码时 THEN Exchange Client SHALL 遵循零容忍低级错误标准
2. WHEN 定义方法时 THEN Exchange Client SHALL 包含完整的类型提示和文档字符串
3. WHEN 处理错误时 THEN Exchange Client SHALL 使用全面的try-except块和详细日志
4. WHEN 导入模块时 THEN Exchange Client SHALL 将所有导入放在文件顶部
5. WHEN 使用缩进时 THEN Exchange Client SHALL 使用4个空格的一致缩进

### Requirement 21

**User Story:** 作为开发人员，我希望OKX客户端使用官方python-okx SDK处理底层API调用，以便利用官方维护的认证、签名和API封装功能。

#### Acceptance Criteria

1. WHEN 初始化OKX客户端时 THEN OKX Client SHALL 使用python-okx SDK的认证模块配置API密钥
2. WHEN 调用OKX REST API时 THEN OKX Client SHALL 使用python-okx SDK提供的API方法
3. WHEN 生成API签名时 THEN OKX Client SHALL 使用python-okx SDK的签名功能而非手动实现
4. WHEN 处理API响应时 THEN OKX Client SHALL 将SDK返回的数据转换为统一格式
5. WHEN SDK方法调用失败时 THEN OKX Client SHALL 捕获SDK异常并转换为统一异常类型
