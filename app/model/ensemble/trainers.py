"""
集成模型训练器模块
"""
# StdLib
import logging
import time
from typing import Dict, Any, Optional

# Third-Party
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

# Local App
from app.core.constants import HOLD_WEIGHT_MULTIPLIER
from app.model.ensemble.informer_wrapper import InformerWrapper
from app.model.ensemble.utils import clear_gpu_memory

logger = logging.getLogger(__name__)

# 深度学习模型（PyTorch）- 可选依赖
try:
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset
    from app.model.gmadl_loss import create_trade_loss
    from app.model.informer2_model import Informer2ForClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    timeframe: str,
    lgb_params: dict,
    lgb_params_by_timeframe: dict,
    use_gpu: bool,
    compute_effective_sample_weights: callable,
    custom_params: Optional[Dict[str, Any]] = None
) -> lgb.LGBMClassifier:
    """训练LightGBM模型"""
    try:
        clear_gpu_memory()
        
        class_weights = compute_effective_sample_weights(y_train, timeframe)
        time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
        
        # 样本加权：有效样本数 × 时间衰减
        # 注意：HOLD 权重惩罚只在损失函数中应用（避免双重加权导致模型永远预测 HOLD）
        sample_weights = class_weights * time_decay
        
        hold_count = int((y_train == 1).sum())
        hold_ratio = hold_count / max(len(y_train), 1)
        logger.info(f"✅ 样本加权已启用：有效样本数 × 时间衰减（HOLD权重×{HOLD_WEIGHT_MULTIPLIER}仅在损失函数中应用）")
        logger.info(f"📊 HOLD样本: {hold_count}个 ({hold_ratio:.2%})")
        
        if custom_params:
            params = custom_params
            logger.info(f"使用Optuna优化参数")
        else:
            timeframe_params = lgb_params_by_timeframe.get(timeframe, {})
            params = {**lgb_params, **timeframe_params}
        
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            logger.info(f"LightGBM GPU加速已启用")
        
        logger.info(f"{timeframe} LightGBM参数: num_leaves={params.get('num_leaves')}, "
                   f"reg_alpha={params.get('reg_alpha', 0)}, reg_lambda={params.get('reg_lambda', 0)}")
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        clear_gpu_memory()
        
        return model
        
    except Exception as e:
        logger.error(f"LightGBM训练失败: {e}")
        clear_gpu_memory()
        raise


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    timeframe: str,
    use_gpu: bool,
    compute_effective_sample_weights: callable,
    custom_params: Optional[Dict[str, Any]] = None
) -> xgb.XGBClassifier:
    """训练XGBoost模型"""
    try:
        clear_gpu_memory()
        
        class_weights = compute_effective_sample_weights(y_train, timeframe)
        time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
        
        # 样本加权：有效样本数 × 时间衰减
        # 注意：HOLD 权重惩罚只在损失函数中应用（避免双重加权导致模型永远预测 HOLD）
        sample_weights = class_weights * time_decay
        
        hold_count = int((y_train == 1).sum())
        hold_ratio = hold_count / max(len(y_train), 1)
        logger.info(f"✅ 样本加权已启用：有效样本数 × 时间衰减（HOLD权重×{HOLD_WEIGHT_MULTIPLIER}仅在损失函数中应用）")
        logger.info(f"📊 HOLD样本: {hold_count}个 ({hold_ratio:.2%})")
        
        if custom_params:
            params = custom_params.copy()
            logger.info(f"使用Optuna优化参数")
        else:
            if timeframe == '15m':
                params = {
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 300,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.3
                }
            elif timeframe == '5m':
                params = {
                    'max_depth': 5,
                    'learning_rate': 0.06,
                    'n_estimators': 220,
                    'reg_alpha': 0.5,
                    'reg_lambda': 0.5
                }
            else:  # 3m
                params = {
                    'max_depth': 5,
                    'learning_rate': 0.07,
                    'n_estimators': 180,
                    'reg_alpha': 0.6,
                    'reg_lambda': 0.6
                }
        
        params.update({
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        })
        
        if use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            logger.info(f"XGBoost GPU加速已启用")
        else:
            params['tree_method'] = 'hist'
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
        
        clear_gpu_memory()
        
        return model
        
    except Exception as e:
        logger.error(f"XGBoost训练失败: {e}")
        clear_gpu_memory()
        raise


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    timeframe: str,
    use_gpu: bool,
    compute_effective_sample_weights: callable,
    custom_params: Optional[Dict[str, Any]] = None
) -> CatBoostClassifier:
    """训练CatBoost模型"""
    try:
        clear_gpu_memory()
        
        class_weights = compute_effective_sample_weights(y_train, timeframe)
        time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
        
        # 样本加权：有效样本数 × 时间衰减
        # 注意：HOLD 权重惩罚只在损失函数中应用（避免双重加权导致模型永远预测 HOLD）
        sample_weights = class_weights * time_decay
        
        hold_count = int((y_train == 1).sum())
        hold_ratio = hold_count / max(len(y_train), 1)
        logger.info(f"✅ 样本加权已启用：有效样本数 × 时间衰减（HOLD权重×{HOLD_WEIGHT_MULTIPLIER}仅在损失函数中应用）")
        logger.info(f"📊 HOLD样本: {hold_count}个 ({hold_ratio:.2%})")
        
        if custom_params:
            params = custom_params.copy()
            logger.info(f"使用Optuna优化参数")
        else:
            if timeframe == '15m':
                params = {
                    'iterations': 300,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3.0
                }
            elif timeframe == '5m':
                params = {
                    'iterations': 220,
                    'learning_rate': 0.06,
                    'depth': 6,
                    'l2_leaf_reg': 4.0
                }
            else:  # 3m
                params = {
                    'iterations': 180,
                    'learning_rate': 0.07,
                    'depth': 6,
                    'l2_leaf_reg': 5.0
                }
        
        params.update({
            'loss_function': 'MultiClass',
            'random_seed': 42,
            'verbose': False,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'allow_writing_files': False
        })
        
        if use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
            logger.info(f"CatBoost GPU加速已启用")
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
        
        clear_gpu_memory()
        
        return model
        
    except Exception as e:
        logger.error(f"CatBoost训练失败: {e}")
        clear_gpu_memory()
        raise


def train_informer2(
    X_seq_train: np.ndarray,
    y_seq_train: np.ndarray,
    timeframe: str,
    d_model: int,
    n_heads: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    use_gpu: bool,
    gpu_device: str,
    use_gradient_checkpointing: bool,
    use_8bit_adam: bool,
    use_aggressive_amp: bool,
    custom_params: Optional[Dict[str, Any]] = None
) -> Optional[InformerWrapper]:
    """训练Informer-2深度学习模型"""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch未安装，跳过Informer-2训练")
        return None
    
    try:
        start_time = time.time()
        clear_gpu_memory()
        
        device = torch.device(gpu_device if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Informer-2训练设备: {device}")
        
        n_features = X_seq_train.shape[2]
        n_classes = 3
        
        if custom_params:
            model_params = custom_params.copy()
            logger.info(f"使用Optuna优化参数")
        else:
            model_params = {
                'd_model': d_model,
                'n_heads': n_heads,
                'n_layers': n_layers,
                'd_ff': d_model * 4,
                'dropout': 0.1,
                'n_classes': n_classes,
                'seq_len': X_seq_train.shape[1]
            }
        
        model = Informer2ForClassification(
            n_features=n_features,
            **model_params
        ).to(device)
        
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        trade_loss_fn = create_trade_loss()
        
        if use_8bit_adam and TORCH_AVAILABLE:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr)
                logger.info("使用8-bit Adam优化器")
            except ImportError:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                logger.warning("bitsandbytes未安装，使用标准Adam")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        scaler = None
        if use_aggressive_amp and device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            logger.info("使用混合精度训练 (FP16)")
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_seq_train),
            torch.LongTensor(y_seq_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(batch_X)
                        loss = trade_loss_fn(logits, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(batch_X)
                    loss = trade_loss_fn(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Informer-2 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Informer-2训练完成，耗时: {training_time:.1f}秒")
        
        clear_gpu_memory()
        
        return InformerWrapper(model, device)
        
    except Exception as e:
        logger.error(f"Informer-2训练失败: {e}")
        clear_gpu_memory()
        return None

