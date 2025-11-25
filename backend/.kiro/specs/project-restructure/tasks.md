# 实施计划 - 项目重构

## 任务列表

- [x] 1. 创建models目录结构


  - 创建`models/`根目录
  - 创建`models/training/`子目录用于训练脚本
  - 创建`models/models/`子目录用于存储模型文件
  - 创建必要的`__init__.py`文件
  - 添加`.gitkeep`到`models/models/`以保留空目录
  - _需求: 2.1, 2.2, 2.3_






- [x] 2. 迁移trading模块文件


- [ ] 2.1 复制signal_generator.py到app/trading/
  - 从`app/services/signal_generator.py`复制到`app/trading/signal_generator.py`
  - 保持文件内容完全一致（此阶段不修改导入）

  - _需求: 3.1, 3.2_

- [ ] 2.2 复制position_manager.py到app/trading/
  - 从`app/services/position_manager.py`复制到`app/trading/position_manager.py`

  - 保持文件内容完全一致
  - _需求: 3.1, 3.2_

- [x] 2.3 复制trading_engine.py到app/trading/

  - 从`app/services/trading_engine.py`复制到`app/trading/trading_engine.py`
  - 保持文件内容完全一致
  - _需求: 3.1, 3.3_





- [ ] 2.4 复制trading_controller.py到app/trading/
  - 从`app/services/trading_controller.py`复制到`app/trading/trading_controller.py`
  - 保持文件内容完全一致

  - _需求: 3.1, 3.4_

- [ ] 2.5 创建app/trading/__init__.py
  - 导出主要类和函数：SignalGenerator, TradingSignal, PositionManager, position_manager, TradingEngine, TradingMode, TradingController

  - 添加模块文档字符串
  - _需求: 3.8, 4.5_

- [ ] 3. 更新trading模块内部导入
- [x] 3.1 更新signal_generator.py的导入语句

  - 将`from app.services.ml_service`保持不变（ML服务未迁移）
  - 将`from app.services.data_service`保持不变（数据服务未迁移）
  - 确保所有导入在文件顶部
  - _需求: 3.5, 4.2, 4.9, 6.2_

- [ ] 3.2 更新position_manager.py的导入语句
  - 将`from app.exchange.binance_client`保持不变


  - 确保所有导入在文件顶部
  - _需求: 3.5, 4.2, 4.9, 6.2_

- [ ] 3.3 更新trading_engine.py的导入语句
  - 更新`from app.services.signal_generator`为`from app.trading.signal_generator`
  - 将`from app.exchange.binance_client`保持不变

  - 确保所有导入在文件顶部



  - _需求: 3.5, 4.2, 4.9, 6.2_

- [x] 3.4 更新trading_controller.py的导入语句

  - 更新`from app.services.trading_engine`为`from app.trading.trading_engine`
  - 更新`from app.services.signal_generator`为`from app.trading.signal_generator`
  - 更新`from app.services.position_manager`为`from app.trading.position_manager`
  - 将`from app.services.ml_service`保持不变
  - 将`from app.services.data_service`保持不变

  - 确保所有导入在文件顶部
  - _需求: 3.5, 4.2, 4.9, 6.2_

- [ ] 4. 更新main.py导入路径
  - 更新`from app.services.trading_engine`为`from app.trading.trading_engine`


  - 更新`from app.services.signal_generator`为`from app.trading.signal_generator`

  - 更新`from app.services.trading_controller`为`from app.trading.trading_controller`
  - 保持其他导入不变
  - 验证导入顺序符合标准（stdlib, third-party, local）
  - _需求: 6.1, 4.2, 4.9_



- [ ] 5. 更新API端点导入路径
- [x] 5.1 更新app/api/endpoints/signals.py

  - 更新`from app.services.signal_generator`为`from app.trading.signal_generator`
  - 确保所有导入在文件顶部
  - _需求: 6.2, 4.2_



- [ ] 5.2 更新app/api/endpoints/trading.py
  - 更新`from app.services.trading_controller`为`from app.trading.trading_controller`
  - 确保所有导入在文件顶部
  - _需求: 6.2, 4.2_

- [ ] 5.3 更新app/api/endpoints/positions.py
  - 检查是否引用了position_manager，如有则更新为`from app.trading.position_manager`
  - 确保所有导入在文件顶部
  - _需求: 6.2, 4.2_

- [ ] 5.4 更新app/api/endpoints/performance.py
  - 检查是否引用了trading相关模块，如有则更新导入路径
  - 确保所有导入在文件顶部
  - _需求: 6.2, 4.2_

- [ ] 6. 更新services模块交叉引用
- [ ] 6.1 检查并更新scheduler.py
  - 检查是否引用了signal_generator，如有则更新为`from app.trading.signal_generator`
  - 确保所有导入在文件顶部
  - _需求: 6.3, 4.2_

- [ ] 6.2 检查并更新health_monitor.py
  - 检查是否引用了signal_generator，如有则更新为`from app.trading.signal_generator`
  - 确保所有导入在文件顶部
  - _需求: 6.3, 4.2_

- [x] 6.3 检查并更新drawdown_monitor.py

  - 检查是否引用了trading相关模块，如有则更新导入路径


  - 确保所有导入在文件顶部
  - _需求: 6.3, 4.2_


- [-] 7. 验证导入路径正确性

  - 使用getDiagnostics工具检查所有Python文件
  - 确认没有ImportError
  - 确认没有循环依赖警告
  - 记录所有诊断结果
  - _需求: 6.4, 6.6, 4.4_


- [ ] 8. 删除services目录中的旧文件
- [ ] 8.1 删除app/services/signal_generator.py
  - 确认新文件已正常工作后删除

  - _需求: 5.1_


- [ ] 8.2 删除app/services/position_manager.py
  - 确认新文件已正常工作后删除
  - _需求: 5.1_

- [ ] 8.3 删除app/services/trading_engine.py
  - 确认新文件已正常工作后删除

  - _需求: 5.1_

- [ ] 8.4 删除app/services/trading_controller.py
  - 确认新文件已正常工作后删除
  - _需求: 5.1_


- [ ] 9. 更新app/services/__init__.py
  - 移除已迁移模块的导出（signal_generator, position_manager, trading_engine, trading_controller）
  - 保留ML和数据相关服务的导出
  - 添加注释说明迁移情况



  - _需求: 5.5_

- [ ] 10. 创建模型训练脚本框架
- [x] 10.1 创建models/training/__init__.py

  - 添加模块文档字符串
  - 导出主要训练类
  - _需求: 2.4, 4.5_


- [ ] 10.2 创建models/training/train_model.py
  - 实现ModelTrainer类框架
  - 添加train_timeframe_model方法
  - 添加train_all_models方法

  - 添加save_model和load_model方法
  - 包含完整的类型提示和文档字符串
  - 添加错误处理（try-except块）
  - _需求: 2.4, 4.3, 4.4, 4.5_


- [ ] 10.3 创建models/__init__.py
  - 添加模块文档字符串
  - 导出训练相关功能

  - _需求: 2.5, 4.5_

- [ ] 11. 创建重构文档
- [ ] 11.1 创建REFACTORING.md
  - 描述新的项目结构

  - 说明各目录的职责
  - 提供目录树可视化
  - 记录迁移注意事项
  - 包含开发者迁移指南

  - _需求: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 11.2 更新models/README.md
  - 说明models目录的用途

  - 描述training和models子目录
  - 提供模型文件命名规范
  - 包含训练脚本使用示例
  - _需求: 7.2, 7.4_


- [ ] 11.3 更新app/trading/README.md
  - 说明trading模块的职责
  - 列出包含的主要组件

  - 描述模块间依赖关系
  - _需求: 7.2, 7.4_

- [ ] 12. 代码质量检查
- [x] 12.1 验证所有文件使用4空格缩进

  - 检查所有迁移的文件
  - 使用autopep8自动修复（如需要）
  - _需求: 4.1_

- [x] 12.2 验证所有导入在文件顶部

  - 检查所有Python文件
  - 确认没有函数内导入
  - _需求: 4.2, 4.9_

- [x] 12.3 验证错误处理完整性

  - 检查所有I/O操作有try-except
  - 确认错误日志包含足够信息
  - _需求: 4.3_

- [x] 12.4 验证类型提示完整性




  - 检查所有函数参数有类型提示
  - 检查所有函数有返回类型注解
  - _需求: 4.4_


- [ ] 12.5 验证文档字符串完整性
  - 检查所有公共方法有文档字符串
  - 确认文档字符串包含Args/Returns/Raises
  - _需求: 4.5_


- [ ] 12.6 验证变量命名一致性
  - 检查函数使用snake_case
  - 检查类使用PascalCase

  - 检查常量使用UPPERCASE
  - _需求: 4.6_

- [ ] 12.7 验证无硬编码值
  - 检查所有配置使用settings或常量
  - 确认没有魔法数字
  - _需求: 4.7_

- [ ] 12.8 验证日志语句规范
  - 检查日志包含适当的emoji前缀
  - 确认日志级别使用正确
  - _需求: 4.8_

- [ ] 12.9 验证无未定义变量
  - 使用getDiagnostics检查
  - 确认所有变量在使用前已定义
  - _需求: 4.9_

- [ ] 12.10 验证无重复定义
  - 检查没有重复的函数或类定义
  - 确认没有重复的变量定义
  - _需求: 4.10_

- [ ] 13. 系统集成测试
- [ ] 13.1 测试系统启动
  - 运行main.py
  - 确认所有服务正常启动
  - 检查日志无错误
  - _需求: 3.7, 6.4_

- [ ] 13.2 测试信号生成功能
  - 触发信号生成
  - 验证信号正常生成
  - 检查信号数据完整性
  - _需求: 3.7_

- [ ] 13.3 测试虚拟交易执行
  - 执行虚拟交易
  - 验证订单记录正确
  - 检查仓位管理正常
  - _需求: 3.7_

- [ ] 13.4 测试API端点
  - 测试所有交易相关API
  - 验证响应数据正确
  - 检查错误处理正常
  - _需求: 3.7_

- [ ] 14. 最终验证
- [ ] 14.1 运行完整的诊断检查
  - 对所有Python文件运行getDiagnostics
  - 确认零错误零警告
  - _需求: 6.4_

- [ ] 14.2 验证文档完整性
  - 检查所有README文件已创建
  - 确认REFACTORING.md内容完整
  - _需求: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 14.3 验证代码质量标准
  - 确认符合general.mdc所有规则
  - 检查无低级错误
  - _需求: 4.1-4.10_

- [ ] 14.4 创建重构总结报告
  - 记录迁移的文件数量
  - 记录更新的导入语句数量
  - 记录发现和修复的问题
  - 提供重构前后对比
  - _需求: 7.5_
