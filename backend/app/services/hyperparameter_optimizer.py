"""
超参数自动优化器 - 使用Optuna
"""

import logging
import gc
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from app.services.informer2_model import Informer2ForClassification
from app.services.gmadl_loss import create_trade_loss
from app.core.config import settings

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
        timeframe: 时间框架（3m/5m/15m）
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
    
    def clear_gpu_memory(self):
        """
        统一GPU内存清理方法
        
        功能：
        - 清空PyTorch缓存
        - 同步GPU操作
        - 强制垃圾回收
        - 记录清理状态
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # 记录GPU内存状态
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_used = torch.cuda.memory_allocated(0)
            gpu_free = gpu_memory - gpu_used
            logger.debug(f"🧹 GPU内存已清理 (使用: {gpu_used/1024**3:.1f}GB, 可用: {gpu_free/1024**3:.1f}GB)")
        else:
            logger.debug("🧹 CPU模式，无需清理GPU内存")
    
    def monitor_gpu_memory(self):
        """
        监控GPU内存使用情况
        
        Returns:
            Dict: GPU内存状态信息
        """
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_used = torch.cuda.memory_allocated(0)
            gpu_free = gpu_memory - gpu_used
            gpu_reserved = torch.cuda.memory_reserved(0)
            
            return {
                'total': gpu_memory,
                'used': gpu_used,
                'free': gpu_free,
                'reserved': gpu_reserved,
                'usage_percent': (gpu_used / gpu_memory) * 100
            }
        else:
            return {'error': 'GPU不可用'}
    
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
        else:
            # 3m/5m 简化搜索（与15m区分）
            base_params = {
                'n_estimators': trial.suggest_int('n_estimators', 120, 320),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'num_leaves': trial.suggest_int('num_leaves', 31, 63),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 30, 70),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.3, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 1.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.2),
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
            # 3m/5m 简化
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
            base_params['tree_method'] = 'hist'  # 新版本使用 hist
            base_params['device'] = 'cuda'  # 使用 device 参数指定 GPU
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
            # 3m/5m 简化
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
        # 🎯 优化：减少序列长度以降低内存占用（减少80-90%）
        seq_len_config = {
            '3m': 96,   # 96 × 3分钟 = 4.8小时（足够短期模式识别）
            '5m': 96,   # 96 × 5分钟 = 8小时（主时间框架）
            '15m': 64   # 64 × 15分钟 = 16小时（趋势确认）
        }
        
        seq_len = seq_len_config.get(self.timeframe, 96)
        
        # 🎯 基于Transformer理论的最佳实践
        # 1. d_model与序列长度的关系：d_model ≈ sqrt(seq_len) * 8-16
        # 2. n_heads与d_model的关系：n_heads = d_model / 64 (标准比例)
        # 3. n_layers与序列长度的关系：n_layers ≈ log2(seq_len) + 1
        
        if self.timeframe == "15m":
            # 15m: 短序列(64)，精确复杂度匹配
            # d_model = sqrt(64) * 12 ≈ 96 → 128
            # n_heads = 128 / 64 = 2 → 4,8 (渐进式搜索)
            # n_layers = log2(64) + 1 ≈ 7 → 2,3 (渐进式搜索)
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
        else:
            # 3m/5m：中序列(96)
            base_params = {
                'd_model': trial.suggest_categorical('d_model', [64, 128]),
                'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
                'n_layers': trial.suggest_int('n_layers', 1, 2),
                'epochs': trial.suggest_int('epochs', 10, 30),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'lr': trial.suggest_float('lr', 0.0008, 0.006, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'alpha': trial.suggest_float('alpha', 0.8, 1.8),
                'beta': trial.suggest_float('beta', 0.4, 0.6)
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
        fold_fail_count = 0
        
        # 🔑 修复：对于3D序列输入，需要基于样本数量而不是特征进行分割
        n_samples = len(self.X) if isinstance(self.X, np.ndarray) else self.X.shape[0]
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.arange(n_samples))):
            # 🔥 关键修复：确保索引转换为numpy数组（兼容内存映射）
            train_idx = np.asarray(train_idx)
            val_idx = np.asarray(val_idx)
            
            # 🔥 关键修复：对于内存映射数组，需要复制数据到内存
            if hasattr(self.X, 'filename') and self.X.filename:
                # 内存映射数组：复制到内存
                X_train = np.array(self.X[train_idx], dtype=np.float32)
                X_val = np.array(self.X[val_idx], dtype=np.float32)
            else:
                # 普通数组：直接切片
                X_train, X_val = self.X[train_idx], self.X[val_idx]
            
            # 🔑 修复：统一转换为numpy数组（兼容pandas Series和numpy数组）
            if isinstance(self.y, pd.Series):
                y_train = self.y.iloc[train_idx].values
                y_val = self.y.iloc[val_idx].values
            elif isinstance(self.y, np.ndarray):
                if hasattr(self.y, 'filename') and self.y.filename:
                    # 内存映射数组：复制到内存
                    y_train = np.array(self.y[train_idx], dtype=np.int64)
                    y_val = np.array(self.y[val_idx], dtype=np.int64)
                else:
                    y_train, y_val = self.y[train_idx], self.y[val_idx]
            else:
                # 其他类型：尝试转换
                y_train = np.asarray(self.y)[train_idx]
                y_val = np.asarray(self.y)[val_idx]
            
            # 计算样本权重（有效样本数 × 时间衰减 × HOLD惩罚）
            try:
                from app.services.ml_service import MLService
                temp_svc = MLService()
                class_weights = temp_svc._compute_effective_sample_weights(y_train, self.timeframe)
            except Exception:
                class_weights = compute_sample_weight('balanced', y_train)
            # ✅ 添加时间衰减权重（与基础模型训练保持一致）
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            # HOLD惩罚自适应
            hold_ratio_tmp = float((y_train == 1).sum()) / max(len(y_train), 1)
            if self.timeframe == '3m':
                hold_weight_tmp = float(max(0.35, min(0.70, 0.80 - 0.6 * hold_ratio_tmp)))
            else:
                hold_weight_tmp = float(max(0.50, min(0.75, 0.85 - 0.5 * hold_ratio_tmp)))
            hold_penalty_weights = np.where(y_train == 1, hold_weight_tmp, 1.0)
            sample_weights = class_weights * time_decay * hold_penalty_weights
            
            # 训练模型
            try:
                # 🎮 统一GPU内存管理：训练前清理
                self.clear_gpu_memory()
                
                if self.model_type == "lightgbm":
                    try:
                        model = lgb.LGBMClassifier(**params)
                        model.fit(X_train, y_train, sample_weight=sample_weights)
                        
                        # 🎮 统一GPU内存管理：训练后清理
                        self.clear_gpu_memory()
                            
                    except Exception as e:
                        logger.error(f"❌ LightGBM训练失败: {e}")
                        # 降级到CPU
                        params['device'] = 'cpu'
                        model = lgb.LGBMClassifier(**params)
                        model.fit(X_train, y_train, sample_weight=sample_weights)
                        self.clear_gpu_memory()
                
                elif self.model_type == "xgboost":
                    try:
                        model = xgb.XGBClassifier(**params)
                        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                        
                        # 🎮 统一GPU内存管理：训练后清理
                        self.clear_gpu_memory()
                            
                    except Exception as e:
                        logger.error(f"❌ XGBoost训练失败: {e}")
                        # 降级到CPU
                        params['tree_method'] = 'hist'
                        params['device'] = 'cpu'
                        model = xgb.XGBClassifier(**params)
                        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                        self.clear_gpu_memory()
                
                elif self.model_type == "catboost":
                    # 检查GPU内存可用性
                    if torch.cuda.is_available():
                        gpu_status = self.monitor_gpu_memory()
                        if gpu_status.get('free', 0) > 500 * 1024**2:  # 500MB
                            params['task_type'] = 'GPU'
                            params['devices'] = '0'
                            logger.debug(f"🚀 CatBoost使用GPU训练 (可用内存: {gpu_status['free']/1024**3:.1f}GB)")
                        else:
                            params['task_type'] = 'CPU'
                            logger.warning(f"⚠️ GPU内存不足({gpu_status['free']/1024**3:.1f}GB)，切换到CPU")
                    else:
                        params['task_type'] = 'CPU'
                        logger.debug("🔄 GPU不可用，CatBoost使用CPU训练")
                    
                    try:
                        model = cb.CatBoostClassifier(**params)
                        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                        
                        # 🎮 统一GPU内存管理：训练后清理
                        self.clear_gpu_memory()
                            
                    except Exception as e:
                        logger.error(f"❌ CatBoost GPU训练失败: {e}")
                        # 降级到CPU
                        params['task_type'] = 'CPU'
                        model = cb.CatBoostClassifier(**params)
                        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                        self.clear_gpu_memory()
                
                elif self.model_type == "informer2":
                    # Informer-2需要特殊处理（深度学习模型 + 序列输入）
                    # 🎮 统一GPU内存管理：训练前清理
                    self.clear_gpu_memory()
                    
                    # 🔑 检查输入维度（2D或3D）
                    if len(X_train.shape) == 2:
                        # 2D输入：需要构造序列（这不应该发生，但作为降级处理）
                        logger.warning(f"⚠️ Informer-2收到2D输入，将跳过此fold")
                        cv_scores.append(0.0)
                        fold_fail_count += 1
                        continue
                    
                    # 3D序列输入：(n_samples, seq_len, n_features)
                    n_features = X_train.shape[2]
                    
                    # 转换为PyTorch张量（内存优化）
                    device = torch.device('cuda:0' if self.use_gpu and torch.cuda.is_available() else 'cpu')
                    
                    # 🔥 内存优化：确保输入数据为float32（减少内存占用）
                    if X_train.dtype != np.float32:
                        logger.debug(f"   转换X_train为float32（原类型: {X_train.dtype}）")
                        X_train = X_train.astype(np.float32)
                    if X_val.dtype != np.float32:
                        logger.debug(f"   转换X_val为float32（原类型: {X_val.dtype}）")
                        X_val = X_val.astype(np.float32)
                    
                    # 🔥 关键修复：统一转换为numpy数组（确保是连续内存）
                    if not isinstance(y_train, np.ndarray):
                        y_train_np = np.asarray(y_train, dtype=np.int64)
                    else:
                        y_train_np = y_train.astype(np.int64) if y_train.dtype != np.int64 else y_train
                    
                    if not isinstance(y_val, np.ndarray):
                        y_val_np = np.asarray(y_val, dtype=np.int64)
                    else:
                        y_val_np = y_val.astype(np.int64) if y_val.dtype != np.int64 else y_val
                    
                    # 🔥 关键修复：创建张量用于内存监控（但不用于训练）
                    # DataLoader会自动处理数据转换
                    train_memory_mb = (X_train.nbytes + y_train_np.nbytes) / (1024 ** 2)
                    logger.debug(f"   训练集内存: {train_memory_mb:.1f} MB")
                    
                    # 🚀 梯度累积配置（解决GPU OOM问题）
                    effective_batch_size = params['batch_size']
                    actual_batch_size = max(8, params['batch_size'] // 8)
                    accumulation_steps = effective_batch_size // actual_batch_size
                    
                    # 🔥 关键修复：确保数据是连续的numpy数组（避免内存映射问题）
                    if not X_train.flags['C_CONTIGUOUS']:
                        logger.debug(f"   转换X_train为连续数组")
                        X_train = np.ascontiguousarray(X_train)
                    if not y_train_np.flags['C_CONTIGUOUS']:
                        logger.debug(f"   转换y_train为连续数组")
                        y_train_np = np.ascontiguousarray(y_train_np)
                    
                    # 创建数据加载器（使用更小的物理批次）
                    class NumpyTimeSeriesDataset(Dataset):
                        def __init__(self, X_np, y_np):
                            # 确保数据是连续的numpy数组
                            self.X_np = np.ascontiguousarray(X_np) if not X_np.flags['C_CONTIGUOUS'] else X_np
                            self.y_np = np.ascontiguousarray(y_np) if not y_np.flags['C_CONTIGUOUS'] else y_np
                        def __len__(self):
                            return len(self.y_np)
                        def __getitem__(self, idx):
                            return (
                                torch.from_numpy(self.X_np[idx].copy()).to(dtype=torch.float32),
                                torch.tensor(self.y_np[idx], dtype=torch.long)
                            )

                    train_dataset = NumpyTimeSeriesDataset(X_train, y_train_np)
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=actual_batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True if device.type == 'cuda' else False
                    )
                    
                    # 创建模型（支持序列输入 + 梯度检查点）
                    model = Informer2ForClassification(
                        n_features=n_features,  # 特征数量（从序列的最后一维获取）
                        n_classes=3,  # 类别数
                        d_model=params['d_model'],
                        n_heads=params['n_heads'],
                        n_layers=params['n_layers'],
                        dropout=params['dropout'],
                        use_distilling=True,  # 启用蒸馏层（完整Informer架构）
                        use_gradient_checkpointing=True  # 🔥 启用梯度检查点（节省50-70%内存）
                    ).to(device)
                    
                    # 定义损失函数（与训练流程保持一致）
                    hold_ratio_opt = float((y_train_np == 1).sum()) / max(len(y_train_np), 1)
                    if self.timeframe == '3m':
                        hold_penalty_nn = float(max(0.35, min(0.70, 0.80 - 0.6 * hold_ratio_opt)))
                    else:
                        hold_penalty_nn = float(max(0.50, min(0.75, 0.85 - 0.5 * hold_ratio_opt)))

                    criterion = create_trade_loss(
                        use_gmadl=settings.USE_GMADL_LOSS,
                        hold_penalty=hold_penalty_nn,
                        alpha=params.get('alpha', settings.GMADL_ALPHA),
                        beta=params.get('beta', settings.GMADL_BETA)
                    )

                    if settings.USE_GMADL_LOSS:
                        logger.debug(
                            f"   损失函数: GMADL + HOLD惩罚 (alpha={params.get('alpha', settings.GMADL_ALPHA):.2f}, beta={params.get('beta', settings.GMADL_BETA):.2f})"
                        )
                    else:
                        logger.debug("   损失函数: 交叉熵 + HOLD惩罚 (稳定模式)")
                    
                    # 🔥 尝试使用8-bit Adam优化器（节省75%优化器内存）
                    optimizer_created = False
                    if self.use_gpu and device.type == 'cuda':
                        try:
                            import bitsandbytes as bnb
                            optimizer = bnb.optim.Adam8bit(
                                model.parameters(),
                                lr=params['lr'],
                                betas=(0.9, 0.999)
                            )
                            optimizer_created = True
                        except (ImportError, Exception):
                            pass
                    
                    if not optimizer_created:
                        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
                    
                    # 🚀 混合精度训练（使用新的torch.amp API + 激进优化）
                    use_amp = device.type == 'cuda' and torch.cuda.is_available()
                    if settings.USE_GMADL_LOSS and use_amp:
                        logger.debug("   ⚠️ GMADL开启 → Optuna试验禁用AMP改用FP32训练")
                        use_amp = False
                    if use_amp:
                        # 🔥 激进混合精度优化
                        scaler = torch.amp.GradScaler('cuda', init_scale=2.**16)
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    else:
                        scaler = None
                    
                    # 训练模型（带梯度累积和混合精度）
                    model.train()
                    for epoch in range(params['epochs']):
                        optimizer.zero_grad()
                        
                        for i, (batch_X, batch_y) in enumerate(train_loader):
                            # 🎯 混合精度前向传播
                            # 将批次移动到目标设备
                            batch_X = batch_X.to(device, non_blocking=True)
                            batch_y = batch_y.to(device, non_blocking=True)
                            if use_amp:
                                with torch.amp.autocast('cuda'):
                                    outputs = model(batch_X)
                                    # 统一dtype与loss输入：logits用float32，targets用long
                                    loss = criterion(outputs.float(), batch_y.long()) / accumulation_steps
                            else:
                                outputs = model(batch_X)
                                loss = criterion(outputs.float(), batch_y.long()) / accumulation_steps
                            
                            # 🎯 混合精度反向传播
                            if use_amp:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                            
                            # 🎯 梯度累积
                            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                                if use_amp:
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    optimizer.step()
                                
                                optimizer.zero_grad()
                                
                                # 定期清理GPU缓存
                                if (i + 1) % (accumulation_steps * 10) == 0 and device.type == 'cuda':
                                    torch.cuda.empty_cache()
                        
                        # 每个epoch结束后清理GPU缓存
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # 🔥 关键修复：确保验证数据也是连续数组
                    if not X_val.flags['C_CONTIGUOUS']:
                        logger.debug(f"   转换X_val为连续数组")
                        X_val = np.ascontiguousarray(X_val)
                    if not y_val_np.flags['C_CONTIGUOUS']:
                        logger.debug(f"   转换y_val为连续数组")
                        y_val_np = np.ascontiguousarray(y_val_np)
                    
                    # 评估模式
                    model.eval()
                    with torch.no_grad():
                        # 🔥 关键修复：使用copy()避免内存映射问题
                        X_val_tensor = torch.from_numpy(X_val.copy()).to(device, dtype=torch.float32)
                        val_outputs = model(X_val_tensor)
                        y_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
                    
                    # 🎮 统一GPU内存管理：训练后清理
                    self.clear_gpu_memory()
                    
                    # 删除模型和张量释放内存
                    del model, X_val_tensor
                    self.clear_gpu_memory()
                
                # 预测并评估
                if self.model_type != "informer2":
                    y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                cv_scores.append(acc)
                
            except Exception as e:
                fold_fail_count += 1
                logger.debug(f"Trial {trial.number} Fold {fold_idx+1} 失败: {e}")
                # 失败的trial返回很差的分数
                cv_scores.append(0.0)
        
        # 计算平均CV准确率
        mean_cv_acc = np.mean(cv_scores)
        if fold_fail_count > 0:
            logger.info(f"   Trial {trial.number} 汇总：失败fold={fold_fail_count}/5，CV={mean_cv_acc:.4f}")
        
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

