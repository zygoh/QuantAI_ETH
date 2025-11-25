# 设计文档 - 项目重构

## 概述

本文档描述QuantAI-ETH量化交易系统的项目结构重构设计方案。重构目标是优化代码组织，将功能模块化，提升可维护性和可扩展性，同时严格遵循项目的专业开发标准。

### 重构原则

1. **最小化破坏性变更** - 保持现有功能完整性，仅调整文件位置和导入路径
2. **单一职责原则** - 每个模块目录有明确的职责边界
3. **零容忍错误** - 严格遵循general.mdc中的代码质量标准
4. **向后兼容** - 确保API接口和数据库结构不受影响

## 架构设计

### 当前目录结构

```
project/
├── app/
│   ├── api/          # API端点
│   ├── core/         # 核心配置
│   ├── exchange/     # 交易所接口（已存在）
│   │   └── binance_client.py
│   ├── services/     # 服务层（混合了多种职责）
│   │   ├── signal_generator.py      # 信号生成 → 需迁移
│   │   ├── position_manager.py      # 仓位管理 → 需迁移
│   │   ├── trading_engine.py        # 交易引擎 → 需迁移
│   │   ├── trading_controller.py    # 交易控制 → 需迁移
│   │   ├── ml_service.py            # 保留
│   │   ├── ensemble_ml_service.py   # 保留
│   │   ├── data_service.py          # 保留
│   │   └── ...
│   ├── trading/      # 交易模块（空目录）
│   └── utils/        # 工具函数
├── models/           # 不存在，需创建
└── main.py
```

### 目标目录结构

```
project/
├── app/
│   ├── api/          # API端点（不变）
│   ├── core/         # 核心配置（不变）
│   ├── exchange/     # 交易所接口（保持现状）
│   │   └── binance_client.py
│   ├── services/     # 服务层（仅保留数据和ML相关）
│   │   ├── ml_service.py
│   │   ├── ensemble_ml_service.py
│   │   ├── data_service.py
│   │   ├── feature_engineering.py
│   │   ├── risk_service.py
│   │   ├── health_monitor.py
│   │   ├── drawdown_monitor.py
│   │   ├── scheduler.py
│   │   └── ...
│   ├── trading/      # 交易模块（新增内容）
│   │   ├── __init__.py
│   │   ├── signal_generator.py      # 从services迁移
│   │   ├── position_manager.py      # 从services迁移
│   │   ├── trading_engine.py        # 从services迁移
│   │   └── trading_controller.py    # 从services迁移
│   └── utils/        # 工具函数（不变）
├── models/           # 模型训练模块（新建）
│   ├── __init__.py
│   ├── training/     # 训练脚本目录
│   │   ├── __init__.py
│   │   ├── train_model.py           # 模型训练主脚本
│   │   ├── hyperparameter_tuning.py # 超参数优化
│   │   └── model_evaluation.py      # 模型评估
│   └── models/       # 训练好的模型文件存储
│       └── .gitkeep
└── main.py
```

## 组件设计

### 1. Exchange模块（app/exchange/）

**职责**: 交易所API接口封装

**现状**: 已存在binance_client.py，功能完整

**设计决策**: 
- 保持现有结构不变
- 每个交易所一个独立文件
- 未来扩展其他交易所时，添加新文件（如okx_client.py）

**关键类**:
- `BinanceClient`: REST API客户端
- `BinanceWebSocketClient`: WebSocket客户端

### 2. Trading模块（app/trading/）

**职责**: 交易信号生成、仓位管理、订单执行、交易控制

**迁移文件**:
1. `signal_generator.py` - 交易信号生成器
2. `position_manager.py` - 仓位管理器
3. `trading_engine.py` - 交易执行引擎
4. `trading_controller.py` - 交易控制器

**模块依赖关系**:
```
trading_controller
    ├── signal_generator (生成信号)
    ├── trading_engine (执行交易)
    ├── position_manager (管理仓位)
    ├── ml_service (模型预测)
    └── data_service (数据获取)
```

**接口设计**:
```python
# app/trading/__init__.py
from app.trading.signal_generator import SignalGenerator, TradingSignal
from app.trading.position_manager import PositionManager, position_manager
from app.trading.trading_engine import TradingEngine, TradingMode
from app.trading.trading_controller import TradingController

__all__ = [
    'SignalGenerator',
    'TradingSignal',
    'PositionManager',
    'position_manager',
    'TradingEngine',
    'TradingMode',
    'TradingController'
]
```

### 3. Models模块（models/）

**职责**: 机器学习模型训练、评估、存储

**目录结构**:
```
models/
├── __init__.py
├── training/              # 训练脚本
│   ├── __init__.py
│   ├── train_model.py     # 主训练脚本
│   ├── hyperparameter_tuning.py
│   └── model_evaluation.py
└── models/                # 模型文件存储
    ├── .gitkeep
    ├── lgb_3m_v1.pkl      # 示例：LightGBM 3分钟模型
    ├── xgb_5m_v1.pkl
    └── ...
```

**训练脚本设计**:
```python
# models/training/train_model.py
"""
模型训练主脚本

功能:
1. 从数据库加载历史K线数据
2. 特征工程处理
3. 训练多个时间框架的模型
4. 保存模型到models/models/目录
5. 记录训练日志和指标
"""

import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_dir: str = "models/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    async def train_timeframe_model(
        self,
        timeframe: str,
        model_type: str = 'lgb'
    ) -> Dict[str, Any]:
        """训练单个时间框架的模型"""
        pass
    
    async def train_all_models(self) -> Dict[str, Any]:
        """训练所有时间框架的模型"""
        pass
    
    def save_model(self, model: Any, filename: str):
        """保存模型到文件"""
        pass
    
    def load_model(self, filename: str) -> Any:
        """从文件加载模型"""
        pass
```

### 4. Services模块（app/services/）

**职责**: 数据服务、机器学习服务、风险管理、监控

**保留文件**:
- `ml_service.py` - 机器学习服务
- `ensemble_ml_service.py` - 集成学习服务
- `data_service.py` - 数据服务
- `feature_engineering.py` - 特征工程
- `risk_service.py` - 风险管理
- `health_monitor.py` - 健康监控
- `drawdown_monitor.py` - 回撤监控
- `scheduler.py` - 任务调度
- `historical_data.py` - 历史数据管理
- 其他ML相关服务

**移除文件**:
- `signal_generator.py` → `app/trading/`
- `position_manager.py` → `app/trading/`
- `trading_engine.py` → `app/trading/`
- `trading_controller.py` → `app/trading/`

## 数据模型

### 模型文件命名规范

```
{model_type}_{timeframe}_v{version}.{extension}

示例:
- lgb_3m_v1.pkl        # LightGBM 3分钟模型 版本1
- xgb_5m_v2.joblib     # XGBoost 5分钟模型 版本2
- cat_15m_v1.pkl       # CatBoost 15分钟模型 版本1
- meta_ensemble_v3.pkl # 元学习器 版本3
```

### 模型元数据

每个模型文件应配套一个JSON元数据文件：

```json
{
  "model_type": "lgb",
  "timeframe": "3m",
  "version": "v1",
  "training_date": "2025-01-15T10:30:00Z",
  "training_samples": 50000,
  "features": ["rsi", "macd", "volume_ratio", ...],
  "performance": {
    "accuracy": 0.65,
    "precision": 0.68,
    "recall": 0.62,
    "f1_score": 0.65
  },
  "hyperparameters": {
    "learning_rate": 0.05,
    "num_leaves": 31,
    ...
  }
}
```

## 错误处理

### 导入路径更新错误处理

**问题**: 文件迁移后，旧的导入路径会失效

**解决方案**:
1. 使用IDE的全局搜索替换功能
2. 分阶段验证：
   - 第一阶段：更新所有导入语句
   - 第二阶段：运行语法检查
   - 第三阶段：运行单元测试

**错误检测**:
```python
# 使用getDiagnostics工具检查所有Python文件
# 确保没有ImportError
```

### 循环依赖处理

**潜在风险**: trading模块和services模块可能存在循环依赖

**预防措施**:
1. 明确依赖方向：trading → services（单向依赖）
2. 使用依赖注入模式
3. 避免在模块级别导入，使用函数内导入（仅在必要时）

**示例**:
```python
# ✅ 正确：依赖注入
class TradingController:
    def __init__(self, ml_service: MLService, data_service: DataService):
        self.ml_service = ml_service
        self.data_service = data_service

# ❌ 错误：模块级循环导入
from app.services.ml_service import ml_service  # 可能导致循环依赖
```

## 测试策略

### 单元测试

**测试范围**:
1. 导入路径验证 - 确保所有模块可正常导入
2. 功能完整性 - 验证迁移后功能不变
3. 接口兼容性 - 确保API端点正常工作

**测试工具**:
- pytest
- getDiagnostics (Kiro内置)

### 集成测试

**测试场景**:
1. 系统启动测试 - main.py能否正常启动
2. 信号生成测试 - 信号生成器能否正常工作
3. 交易执行测试 - 虚拟交易能否正常执行
4. 模型加载测试 - ML服务能否加载模型

### 回归测试

**验证点**:
1. WebSocket连接正常
2. 数据库读写正常
3. API端点响应正常
4. 日志输出正常

## 迁移步骤

### 阶段1: 创建新目录结构

1. 创建`models/`目录及子目录
2. 创建`models/training/`目录
3. 创建`models/models/`目录
4. 创建必要的`__init__.py`文件

### 阶段2: 迁移trading模块

1. 复制文件到`app/trading/`
2. 更新文件内的导入语句
3. 创建`app/trading/__init__.py`
4. 验证语法正确性

### 阶段3: 更新导入路径

1. 更新`main.py`中的导入
2. 更新`app/api/endpoints/`中的导入
3. 更新`app/services/`中的交叉引用
4. 验证所有导入路径

### 阶段4: 清理旧文件

1. 从`app/services/`删除已迁移的文件
2. 更新`app/services/__init__.py`
3. 验证系统启动

### 阶段5: 创建模型训练脚本

1. 创建`models/training/train_model.py`
2. 实现基本训练逻辑
3. 添加模型保存/加载功能
4. 创建示例训练脚本

### 阶段6: 文档更新

1. 创建`REFACTORING.md`
2. 更新项目README
3. 添加目录结构说明
4. 记录迁移注意事项

## 风险评估

### 高风险项

1. **导入路径错误** - 可能导致系统无法启动
   - 缓解措施：使用全局搜索替换，分阶段验证
   
2. **循环依赖** - 可能导致ImportError
   - 缓解措施：明确依赖方向，使用依赖注入

3. **运行时错误** - 迁移后可能出现未预见的错误
   - 缓解措施：完整的集成测试，逐步部署

### 中风险项

1. **配置文件路径** - 模型文件路径可能需要更新
   - 缓解措施：使用相对路径，配置化路径管理

2. **日志输出** - 模块名称变化可能影响日志
   - 缓解措施：保持logger名称一致

### 低风险项

1. **代码格式** - 可能需要重新格式化
   - 缓解措施：使用autopep8自动格式化

2. **注释更新** - 文件路径注释需要更新
   - 缓解措施：手动检查和更新

## 性能考虑

### 导入性能

**影响**: 文件迁移不会影响导入性能

**原因**: Python的导入机制基于模块路径，与文件物理位置无关

### 运行时性能

**影响**: 零性能影响

**原因**: 仅调整文件位置，不修改业务逻辑

### 内存占用

**影响**: 零内存影响

**原因**: 模块加载机制不变

## 兼容性

### Python版本

**要求**: Python 3.12+（与现有项目一致）

### 依赖库

**无新增依赖**: 重构不引入新的第三方库

### 数据库

**无影响**: 数据库结构和查询不变

### API接口

**向后兼容**: 所有API端点保持不变

## 安全考虑

### 模型文件安全

**措施**:
1. 模型文件不提交到Git（添加到.gitignore）
2. 使用环境变量配置模型路径
3. 验证模型文件完整性（MD5校验）

### 代码安全

**措施**:
1. 遵循general.mdc的安全标准
2. 所有I/O操作添加错误处理
3. 验证文件路径，防止路径遍历攻击

## 可维护性

### 代码组织

**优势**:
1. 职责清晰 - 每个目录有明确的职责
2. 易于扩展 - 新增交易所或模型类型更方便
3. 降低耦合 - 模块间依赖关系更清晰

### 文档维护

**要求**:
1. 每个新目录添加README.md
2. 更新项目整体文档
3. 记录重要设计决策

### 代码审查

**检查点**:
1. 所有导入语句在文件顶部
2. 完整的类型提示
3. 完善的错误处理
4. 清晰的注释和文档字符串

## 总结

本重构设计遵循以下原则：

1. **最小化变更** - 仅调整文件位置，不修改业务逻辑
2. **职责分离** - 明确各模块职责边界
3. **向后兼容** - 保持API和数据结构不变
4. **质量保证** - 严格遵循代码质量标准
5. **可扩展性** - 为未来功能扩展预留空间

重构完成后，项目结构将更加清晰，代码可维护性显著提升，为后续功能开发奠定良好基础。
