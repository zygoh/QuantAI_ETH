# 需求文档 - 项目重构

## 简介

本文档定义了QuantAI-ETH量化交易系统的项目结构重构需求。重构目标是优化代码组织结构，将交易所接口、机器学习模型训练、交易信号处理等功能模块化，提升代码可维护性和可扩展性。重构必须严格遵循项目的专业开发标准（.cursor/rules/general.mdc）。

## 术语表

- **System**: QuantAI-ETH量化交易系统
- **Exchange Module**: 交易所接口模块，负责与加密货币交易所API交互
- **Model Module**: 机器学习模型模块，负责模型训练、存储和加载
- **Trading Module**: 交易模块，负责信号接收、订单执行和模拟交易
- **Model File**: 训练完成的机器学习模型文件（.pkl, .joblib等格式）
- **Production-Grade Code**: 符合专业量化工程师标准的生产级代码
- **Zero-Tolerance Error**: 不可容忍的低级错误（重复定义、未定义变量、逻辑错误等）

## 需求

### 需求 1: 交易所模块重构

**用户故事:** 作为量化开发工程师，我希望将所有交易所相关代码集中在exchange文件夹中，每个交易所一个独立文件，以便于管理和扩展不同交易所的接口。

#### 验收标准

1. THE System SHALL maintain the existing `app/exchange/` directory structure
2. THE System SHALL keep each exchange implementation in a separate file (e.g., `binance_client.py`)
3. THE System SHALL preserve all existing exchange functionality without breaking changes
4. THE System SHALL ensure all exchange files follow the import standards defined in general.mdc (all imports at file top)
5. THE System SHALL validate that no duplicate variable or function definitions exist in exchange modules

### 需求 2: 模型训练模块创建

**用户故事:** 作为机器学习工程师，我希望有一个专门的model文件夹用于存放模型训练代码和训练好的模型文件，以便于模型开发和版本管理。

#### 验收标准

1. THE System SHALL create a new `models/` directory at the project root level
2. THE System SHALL create a `models/training/` subdirectory for model training scripts
3. THE System SHALL create a `models/models/` subdirectory for storing trained model files
4. WHEN model training completes, THE System SHALL save model files to `models/models/` directory
5. THE System SHALL implement proper error handling for model file I/O operations with try-except blocks
6. THE System SHALL include type hints for all model training functions
7. THE System SHALL validate model file paths and existence before loading operations

### 需求 3: 交易信号与订单模块重构

**用户故事:** 作为交易系统开发者，我希望将信号接收、订单执行、模拟交易等功能整合到trading文件夹中，以便于交易逻辑的集中管理和测试。

#### 验收标准

1. THE System SHALL move signal generation logic from `app/services/signal_generator.py` to `app/trading/` directory
2. THE System SHALL move position management logic from `app/services/position_manager.py` to `app/trading/` directory
3. THE System SHALL move trading execution logic from `app/services/trading_engine.py` to `app/trading/` directory
4. THE System SHALL move trading control logic from `app/services/trading_controller.py` to `app/trading/` directory
5. THE System SHALL update all import statements in dependent modules to reflect new file locations
6. THE System SHALL ensure no circular dependencies are introduced during the refactoring
7. THE System SHALL maintain all existing trading functionality without regression
8. THE System SHALL follow single responsibility principle for each trading module file

### 需求 4: 代码质量保证

**用户故事:** 作为专业量化工程师，我希望重构后的代码严格遵循项目开发标准，确保生产级代码质量，避免任何低级错误。

#### 验收标准

1. THE System SHALL ensure all Python files use 4-space indentation consistently
2. THE System SHALL place all module imports at the top of each file (no local imports)
3. THE System SHALL include comprehensive error handling with try-except blocks for all I/O operations
4. THE System SHALL provide complete type hints for all function parameters and return values
5. THE System SHALL include docstrings with Args/Returns/Raises sections for all public methods
6. THE System SHALL use consistent variable naming conventions (snake_case for functions, PascalCase for classes)
7. THE System SHALL validate that no hardcoded values exist without proper constants
8. THE System SHALL ensure all logging statements include appropriate emoji prefixes
9. THE System SHALL validate that no undefined variables are used in any module
10. THE System SHALL ensure no duplicate function or class definitions exist across modules

### 需求 5: 服务模块清理

**用户故事:** 作为系统架构师，我希望在重构后清理app/services目录，移除已迁移的模块，保持清晰的模块职责划分。

#### 验收标准

1. WHEN trading modules are moved to `app/trading/`, THE System SHALL remove the original files from `app/services/`
2. THE System SHALL keep ML-related services (ml_service.py, ensemble_ml_service.py, etc.) in `app/services/`
3. THE System SHALL keep data services (data_service.py, historical_data.py) in `app/services/`
4. THE System SHALL keep monitoring services (health_monitor.py, drawdown_monitor.py) in `app/services/`
5. THE System SHALL update `app/services/__init__.py` to reflect the new module structure
6. THE System SHALL ensure all remaining services maintain their existing interfaces

### 需求 6: 导入路径更新

**用户故事:** 作为开发者，我希望所有模块的导入路径在重构后自动更新，确保系统能够正常运行。

#### 验收标准

1. THE System SHALL update all import statements in `main.py` to reflect new module locations
2. THE System SHALL update all import statements in `app/api/` modules to reflect new module locations
3. THE System SHALL update all cross-module imports within `app/services/` to reflect new structure
4. THE System SHALL validate that all import paths are correct and no ImportError occurs
5. THE System SHALL ensure import ordering follows the standard (stdlib, third-party, local)
6. THE System SHALL verify that no circular import dependencies exist after refactoring

### 需求 7: 文档更新

**用户故事:** 作为团队成员，我希望项目文档能够反映新的目录结构，方便理解和维护系统。

#### 验收标准

1. THE System SHALL create a `REFACTORING.md` document describing the new project structure
2. THE System SHALL document the purpose of each major directory (exchange, models, trading)
3. THE System SHALL provide migration guide for developers working on existing branches
4. THE System SHALL update module responsibility descriptions in the documentation
5. THE System SHALL include directory tree visualization in the documentation
