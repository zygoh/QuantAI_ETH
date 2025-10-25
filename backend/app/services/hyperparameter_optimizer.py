"""
超参数自动优化器 - 使用Optuna
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    超参数自动优化器
    
    使用Optuna的TPE（Tree-structured Parzen Estimator）算法
    自动搜索最佳超参数组合
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: pd.Series,
        timeframe: str,
        model_type: str = "lightgbm",
        use_gpu: bool = True
    ):
        """
        初始化优化器
        
        Args:
            X: 特征数据（已缩放）
            y: 标签数据
            timeframe: 时间框架（15m/2h/4h）
            model_type: 模型类型（lightgbm/xgboost/catboost）
            use_gpu: 是否使用GPU加速
        """
        self.X = X
        self.y = y
        self.timeframe = timeframe
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0
        
        # HOLD惩罚系数（与训练保持一致）
        self.hold_penalty = 0.65
        
        logger.info(f"🔧 初始化超参数优化器: {timeframe} - {model_type}")
        if len(X.shape) == 3:
            logger.info(f"   样本数: {len(X)}, 序列长度: {X.shape[1]}, 特征数: {X.shape[2]}")
        else:
            logger.info(f"   样本数: {len(X)}, 特征数: {X.shape[1]}")
        logger.info(f"   GPU加速: {'启用' if use_gpu else '关闭'}")
    
    def _get_lightgbm_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        LightGBM搜索空间
        
        根据时间框架差异化配置搜索范围
        """
        # 基础参数
        base_params = {}
        
        if self.timeframe == "15m":
            # 15m: 样本多，可以复杂一些
            base_params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'num_leaves': trial.suggest_int('num_leaves', 63, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True
            }
        
        elif self.timeframe == "2h":
            # 2h: 样本中等，简化模型
            base_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'num_leaves': trial.suggest_int('num_leaves', 11, 31),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 40, 80),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 1.2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.2),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.2),
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True
            }
        
        else:  # 4h
            # 4h: 样本少，极简模型
            base_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'num_leaves': trial.suggest_int('num_leaves', 7, 21),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.8, 1.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.8, 1.5),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True
            }
        
        # 🎮 GPU加速（如果启用）
        if self.use_gpu:
            base_params['device'] = 'gpu'
            base_params['gpu_platform_id'] = 0
            base_params['gpu_device_id'] = 0
        
        return base_params
    
    def _get_xgboost_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """XGBoost搜索空间"""
        if self.timeframe == "15m":
            base_params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'verbosity': 0
            }
        else:
            # 2h/4h简化
            base_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 1.2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.2),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
                'random_state': 42,
                'verbosity': 0
            }
        
        # 🎮 GPU加速（如果启用）
        if self.use_gpu:
            base_params['tree_method'] = 'gpu_hist'
            base_params['gpu_id'] = 0
        else:
            base_params['tree_method'] = 'hist'
        
        return base_params
    
    def _get_catboost_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """CatBoost搜索空间"""
        if self.timeframe == "15m":
            base_params = {
                'iterations': trial.suggest_int('iterations', 200, 500),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0),
                'border_count': trial.suggest_int('border_count', 32, 128),
                'random_state': 42,
                'verbose': False
            }
        else:
            # 2h/4h简化
            base_params = {
                'iterations': trial.suggest_int('iterations', 100, 250),
                'depth': trial.suggest_int('depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 64),
                'random_state': 42,
                'verbose': False
            }
        
        # 🎮 GPU加速（如果启用）
        if self.use_gpu:
            base_params['task_type'] = 'GPU'
            base_params['devices'] = '0'
        
        return base_params
    
    def _get_informer2_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Informer-2搜索空间（基于Transformer理论的最佳实践 + 精确复杂度匹配）"""
        
        # 🔑 序列长度配置（与ensemble_ml_service.py保持一致）
        seq_len_config = {
            '15m': 96,   # 96 × 15分钟 = 24小时
            '2h': 48,    # 48 × 2小时 = 4天
            '4h': 24     # 24 × 4小时 = 4天
        }
        
        seq_len = seq_len_config.get(self.timeframe, 96)
        
        # 🎯 基于Transformer理论的最佳实践
        # 1. d_model与序列长度的关系：d_model ≈ sqrt(seq_len) * 8-16
        # 2. n_heads与d_model的关系：n_heads = d_model / 64 (标准比例)
        # 3. n_layers与序列长度的关系：n_layers ≈ log2(seq_len) + 1
        
        if self.timeframe == "15m":
            # 15m: 长序列(96)，精确复杂度匹配
            # d_model = sqrt(96) * 12 ≈ 118 → 128
            # n_heads = 128 / 64 = 2 → 4,8,16 (渐进式搜索)
            # n_layers = log2(96) + 1 ≈ 7 → 2,3,4 (渐进式搜索)
            base_params = {
                'd_model': trial.suggest_categorical('d_model', [128, 256]),      # 精确匹配
                'n_heads': trial.suggest_categorical('n_heads', [4, 8]),          # 精确匹配
                'n_layers': trial.suggest_int('n_layers', 2, 3),  # 精确匹配
                'epochs': trial.suggest_int('epochs', 20, 40),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                'lr': trial.suggest_float('lr', 0.0005, 0.005, log=True),
                'dropout': trial.suggest_float('dropout', 0.05, 0.2),
                'alpha': trial.suggest_float('alpha', 0.5, 2.0),  # GMADL参数
                'beta': trial.suggest_float('beta', 0.3, 0.7)    # GMADL参数
            }
        elif self.timeframe == "2h":
            # 2h: 中等序列(48)，精确复杂度匹配
            # d_model = sqrt(48) * 12 ≈ 83 → 64,128
            # n_heads = 64/128 / 64 = 1/2 → 2,4,8 (渐进式搜索)
            # n_layers = log2(48) + 1 ≈ 6 → 1,2,3 (渐进式搜索)
            base_params = {
                'd_model': trial.suggest_categorical('d_model', [64, 128]),       # 精确匹配
                'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),       # 精确匹配
                'n_layers': trial.suggest_int('n_layers', 1, 3),  # 精确匹配
                'epochs': trial.suggest_int('epochs', 15, 30),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
                'lr': trial.suggest_float('lr', 0.001, 0.005, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'alpha': trial.suggest_float('alpha', 0.8, 1.5),
                'beta': trial.suggest_float('beta', 0.4, 0.6)
            }
        else:  # 4h
            # 4h: 短序列(24)，精确复杂度匹配
            # d_model = sqrt(24) * 12 ≈ 59 → 64
            # n_heads = 64 / 64 = 1 → 2,4 (渐进式搜索)
            # n_layers = log2(24) + 1 ≈ 5 → 1,2 (渐进式搜索)
            base_params = {
                'd_model': trial.suggest_categorical('d_model', [64]),            # 精确匹配
                'n_heads': trial.suggest_categorical('n_heads', [2, 4]),          # 精确匹配
                'n_layers': trial.suggest_int('n_layers', 1, 2),  # 精确匹配
                'epochs': trial.suggest_int('epochs', 10, 25),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
                'lr': trial.suggest_float('lr', 0.002, 0.01, log=True),
                'dropout': trial.suggest_float('dropout', 0.15, 0.35),
                'alpha': trial.suggest_float('alpha', 1.0, 2.0),
                'beta': trial.suggest_float('beta', 0.5, 0.7)
            }
        
        # 添加序列长度信息到参数中（用于日志记录）
        base_params['seq_len'] = seq_len
        
        return base_params
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        优化目标函数
        
        使用5折时间序列交叉验证评估超参数组合
        
        Args:
            trial: Optuna试验对象
        
        Returns:
            负的CV平均准确率（Optuna默认最小化）
        """
        # 获取搜索空间
        if self.model_type == "lightgbm":
            params = self._get_lightgbm_search_space(trial)
        elif self.model_type == "xgboost":
            params = self._get_xgboost_search_space(trial)
        elif self.model_type == "catboost":
            params = self._get_catboost_search_space(trial)
        elif self.model_type == "informer2":
            params = self._get_informer2_search_space(trial)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 时间序列5折交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        # 🔑 修复：对于3D序列输入，需要基于样本数量而不是特征进行分割
        n_samples = len(self.X) if isinstance(self.X, np.ndarray) else self.X.shape[0]
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.arange(n_samples))):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            # 🔑 修复：兼容 numpy 数组和 pandas Series
            if isinstance(self.y, np.ndarray):
                y_train, y_val = self.y[train_idx], self.y[val_idx]
            else:
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # 计算样本权重（类别平衡 × 时间衰减 × HOLD惩罚）
            class_weights = compute_sample_weight('balanced', y_train)
            # ✅ 添加时间衰减权重（与基础模型训练保持一致）
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            hold_penalty_weights = np.where(y_train == 1, self.hold_penalty, 1.0)
            sample_weights = class_weights * time_decay * hold_penalty_weights
            
            # 训练模型
            try:
                if self.model_type == "lightgbm":
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                
                elif self.model_type == "xgboost":
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                
                elif self.model_type == "catboost":
                    model = cb.CatBoostClassifier(**params)
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                
                elif self.model_type == "informer2":
                    # Informer-2需要特殊处理（深度学习模型 + 序列输入）
                    from app.services.informer2_model import Informer2ForClassification
                    from app.services.gmadl_loss import GMADLossWithHOLDPenalty
                    import torch
                    import torch.nn as nn
                    from torch.utils.data import DataLoader, TensorDataset
                    
                    # 🔑 检查输入维度（2D或3D）
                    if len(X_train.shape) == 2:
                        # 2D输入：需要构造序列（这不应该发生，但作为降级处理）
                        logger.warning(f"⚠️ Informer-2收到2D输入，将跳过此fold")
                        cv_scores.append(0.0)
                        continue
                    
                    # 3D序列输入：(n_samples, seq_len, n_features)
                    n_features = X_train.shape[2]
                    
                    # 转换为PyTorch张量
                    device = torch.device('cuda:0' if self.use_gpu and torch.cuda.is_available() else 'cpu')
                    X_train_tensor = torch.FloatTensor(X_train).to(device)
                    # ✅ 兼容pandas Series和numpy ndarray
                    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
                    y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
                    y_train_tensor = torch.LongTensor(y_train_np).to(device)
                    X_val_tensor = torch.FloatTensor(X_val).to(device)
                    y_val_tensor = torch.LongTensor(y_val_np).to(device)
                    
                    # 创建数据加载器
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
                    
                    # 创建模型（支持序列输入）
                    model = Informer2ForClassification(
                        n_features=n_features,  # 特征数量（从序列的最后一维获取）
                        n_classes=3,  # 类别数
                        d_model=params['d_model'],
                        n_heads=params['n_heads'],
                        n_layers=params['n_layers'],
                        dropout=params['dropout'],
                        use_distilling=True  # 启用蒸馏层（完整Informer架构）
                    ).to(device)
                    
                    # 定义损失函数和优化器
                    criterion = GMADLossWithHOLDPenalty(
                        hold_penalty=self.hold_penalty,
                        alpha=params['alpha'],
                        beta=params['beta']
                    )
                    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
                    
                    # 训练模型
                    model.train()
                    for epoch in range(params['epochs']):
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                    
                    # 评估模式
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        y_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
                
                # 预测并评估
                if self.model_type != "informer2":
                    y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                cv_scores.append(acc)
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} Fold {fold_idx+1} 失败: {e}")
                # 失败的trial返回很差的分数
                cv_scores.append(0.0)
        
        # 计算平均CV准确率
        mean_cv_acc = np.mean(cv_scores)
        
        # 每10次试验报告一次进度
        if trial.number % 10 == 0:
            logger.info(f"   Trial {trial.number}: CV准确率={mean_cv_acc:.4f}")
        
        # 返回负值（Optuna最小化目标）
        return -mean_cv_acc
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: int = 1800,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            n_trials: 试验次数（默认100次）
            timeout: 超时时间（秒，默认30分钟）
            show_progress: 是否显示进度条
        
        Returns:
            最佳参数字典
        """
        logger.info(f"🚀 开始超参数优化: {self.timeframe} - {self.model_type}")
        logger.info(f"   试验次数: {n_trials}, 超时: {timeout}秒 (~{timeout//60}分钟)")
        
        # 创建study（静默模式，避免过多日志）
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(
            direction='minimize',  # 最小化负准确率
            sampler=TPESampler(seed=42)
        )
        
        # 执行优化
        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=show_progress,
                n_jobs=1  # 单线程（避免并发问题）
            )
        except KeyboardInterrupt:
            logger.warning("⚠️ 优化被用户中断")
        
        # 保存最佳参数
        self.best_params = study.best_params
        self.best_score = -study.best_value  # 转回正准确率
        
        logger.info(f"✅ {self.timeframe} 超参数优化完成!")
        logger.info(f"   最佳CV准确率: {self.best_score:.4f} ({self.best_score*100:.2f}%)")
        logger.info(f"   最佳参数: {self.best_params}")
        logger.info(f"   总试验次数: {len(study.trials)}")
        
        # 显示参数重要性（Top 5）
        try:
            importances = optuna.importance.get_param_importances(study)
            logger.info(f"   参数重要性（Top 5）:")
            for i, (param, importance) in enumerate(list(importances.items())[:5]):
                logger.info(f"      {i+1}. {param}: {importance:.4f}")
        except:
            pass
        
        return self.best_params
    
    def get_optimized_params(self) -> Dict[str, Any]:
        """
        获取优化后的参数
        
        Returns:
            最佳参数字典，如果未优化则返回None
        """
        return self.best_params

