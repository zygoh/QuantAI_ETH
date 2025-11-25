"""
模型训练模块

职责：
1. 机器学习模型训练
2. 模型评估和优化
3. 模型版本管理
4. 特征工程
5. 超参数优化

注意：
- 训练好的模型文件存储在项目根目录的 models/ 文件夹
- 本模块包含所有ML相关的代码

模块：
- ml_service: 基础ML服务
- ensemble_ml_service: 集成学习服务
- feature_engineering: 特征工程
- hyperparameter_optimizer: 超参数优化
- informer2_model: Informer2深度学习模型
- gmadl_loss: GMADL损失函数
- model_stability_enhancer: 模型稳定性增强
"""

# 注意：这些导入在迁移完成后才可用
# from app.model.ml_service import MLService
# from app.model.ensemble_ml_service import EnsembleMLService, ensemble_ml_service
# from app.model.feature_engineering import FeatureEngineer, feature_engineer

__version__ = "1.0.0"
