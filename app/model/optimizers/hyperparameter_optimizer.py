"""
超参数自动优化器 - 使用Optuna
"""

import logging
import gc
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
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
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local App
from app.core.constants import (
    GMADL_ALPHA,
    GMADL_BETA,
    GRAD_SCALER_AUTO_RESET,
    GRAD_SCALER_GROWTH_FACTOR,
    GRAD_SCALER_GROWTH_INTERVAL,
    GRAD_SCALER_MAX_CONSECUTIVE_OVERFLOW,
    GRAD_SCALER_MAX_SCALE,
    GRAD_SCALER_RESET_THRESHOLD_EPOCHS,
    HOLD_WEIGHT_MULTIPLIER,
    OPTIMIZER_LR_REDUCE_FACTOR,
    OPTIMIZER_LR_REDUCE_THRESHOLD,
    USE_GMADL_LOSS
)
from app.model.informer2_model import Informer2ForClassification
from app.model.gmadl_loss import create_trade_loss
from app.model.base.ml_service import MLService

# 可选依赖：bitsandbytes（8-bit优化器）
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ScaleRecord:
    """Scale监控记录"""
    epoch: int
    batch: int
    scale_value: float
    has_overflow: bool
    consecutive_overflow_count: int
    timestamp: datetime


class DynamicGradScalerConfig:
    """
    动态GradScaler配置器
    
    根据模型规模和训练阶段动态调整GradScaler参数
    """
    
    def __init__(self, model_param_count: int):
        """
        初始化配置器
        
        Args:
            model_param_count: 模型参数量
        """
        self.model_param_count = model_param_count
        self.init_scale = self.calculate_init_scale(model_param_count)
        self.growth_factor = GRAD_SCALER_GROWTH_FACTOR
        self.backoff_factor = 0.5
        self.growth_interval = GRAD_SCALER_GROWTH_INTERVAL
        self.max_scale = GRAD_SCALER_MAX_SCALE
        
        logger.info(f"🔧 GradScaler配置器初始化:")
        logger.info(f"   模型参数量: {model_param_count/1e6:.2f}M")
        logger.info(f"   初始scale: {self.init_scale}")
        logger.info(f"   增长因子: {self.growth_factor}")
        logger.info(f"   增长间隔: {self.growth_interval}")
        logger.info(f"   最大scale阈值: {self.max_scale}")
    
    @staticmethod
    def calculate_init_scale(param_count: int) -> float:
        """
        根据模型参数量计算初始scale
        
        Args:
            param_count: 模型参数量
        
        Returns:
            初始scale值
        """
        # 🔧 优化：降低初始scale值，减少Scale值过大问题
        if param_count > 10_000_000:  # >10M参数：大模型
            init_scale = 2.**11  # 从4096降低到2048
            logger.debug(f"   检测到大模型({param_count/1e6:.1f}M参数)，使用init_scale=2^11={init_scale}")
        elif param_count > 1_000_000:  # 1M-10M参数：中等模型
            init_scale = 2.**13  # 从16384降低到8192
            logger.debug(f"   检测到中等模型({param_count/1e6:.1f}M参数)，使用init_scale=2^13={init_scale}")
        else:  # <1M参数：小模型
            init_scale = 2.**15  # 从65536降低到32768（这是导致Scale值过大的主要原因）
            logger.debug(f"   检测到小模型({param_count/1e6:.1f}M参数)，使用init_scale=2^15={init_scale}")
        
        return init_scale
    
    def create_scaler(self, device: str) -> torch.amp.GradScaler:
        """
        创建配置好的GradScaler
        
        Args:
            device: 设备类型（'cuda'或'cpu'）
        
        Returns:
            配置好的GradScaler实例
        """
        scaler = torch.amp.GradScaler(
            device,
            init_scale=self.init_scale,
            growth_factor=self.growth_factor,
            backoff_factor=self.backoff_factor,
            growth_interval=self.growth_interval,
            enabled=True
        )
        
        logger.info(f"✅ GradScaler已创建 (初始scale={self.init_scale})")
        return scaler


class GradScalerMonitor:
    """
    GradScaler监控器
    
    监控scale值变化，检测异常并触发自动重置
    """
    
    def __init__(self, scaler: torch.amp.GradScaler, init_scale: float):
        """
        初始化监控器
        
        Args:
            scaler: GradScaler实例
            init_scale: 初始scale值
        """
        self.scaler = scaler
        self.init_scale = init_scale
        self.scale_history: List[float] = []
        self.max_scale_threshold = GRAD_SCALER_MAX_SCALE
        self.consecutive_overflow_count = 0
        self.max_consecutive_overflow = GRAD_SCALER_MAX_CONSECUTIVE_OVERFLOW
        self.epoch_scale_records: Dict[int, float] = {}
        self.scale_records: List[ScaleRecord] = []
        self.abnormal_epoch_count = 0
        self.reset_threshold_epochs = GRAD_SCALER_RESET_THRESHOLD_EPOCHS
        
        logger.info(f"📊 GradScaler监控器初始化:")
        logger.info(f"   最大scale阈值: {self.max_scale_threshold}")
        logger.info(f"   最大连续溢出: {self.max_consecutive_overflow}")
        logger.info(f"   重置阈值epoch数: {self.reset_threshold_epochs}")
    
    def record_scale(self, epoch: int, batch: int) -> bool:
        """
        记录当前scale值
        
        Args:
            epoch: 当前epoch
            batch: 当前batch
        
        Returns:
            是否超过阈值
        """
        current_scale = self.scaler.get_scale()
        self.scale_history.append(current_scale)
        
        if epoch not in self.epoch_scale_records:
            self.epoch_scale_records[epoch] = current_scale
        
        # 检查是否超过阈值
        if current_scale > self.max_scale_threshold:
            logger.warning(f"⚠️ Epoch {epoch} Batch {batch}: Scale值过大 ({current_scale:.2f} > {self.max_scale_threshold})")
            return True
        
        return False
    
    def check_overflow(self, has_overflow: bool, epoch: int, batch: int) -> bool:
        """
        检查梯度溢出
        
        Args:
            has_overflow: 是否发生溢出
            epoch: 当前epoch
            batch: 当前batch
        
        Returns:
            是否需要重置scale
        """
        if has_overflow:
            self.consecutive_overflow_count += 1
            
            # 记录溢出
            record = ScaleRecord(
                epoch=epoch,
                batch=batch,
                scale_value=self.scaler.get_scale(),
                has_overflow=True,
                consecutive_overflow_count=self.consecutive_overflow_count,
                timestamp=datetime.now()
            )
            self.scale_records.append(record)
            
            if self.consecutive_overflow_count >= self.max_consecutive_overflow:
                logger.error(f"❌ Epoch {epoch} Batch {batch}: 连续{self.consecutive_overflow_count}次梯度溢出")
                return True
        else:
            self.consecutive_overflow_count = 0
        
        return False
    
    def check_epoch_abnormal(self, epoch: int) -> bool:
        """
        检查epoch级别的异常
        
        Args:
            epoch: 当前epoch
        
        Returns:
            是否需要重置scale
        """
        if epoch in self.epoch_scale_records:
            scale = self.epoch_scale_records[epoch]
            if scale > self.max_scale_threshold:
                self.abnormal_epoch_count += 1
                logger.warning(f"⚠️ Epoch {epoch}: 异常epoch计数 {self.abnormal_epoch_count}/{self.reset_threshold_epochs}")
                
                if self.abnormal_epoch_count >= self.reset_threshold_epochs:
                    logger.error(f"❌ 连续{self.abnormal_epoch_count}个epoch的scale异常")
                    return True
            else:
                self.abnormal_epoch_count = 0
        
        return False
    
    def reset_scale(self, reset_to_percent: float = 0.5):
        """
        重置scale值
        
        Args:
            reset_to_percent: 重置到初始值的百分比
        """
        current_scale = self.scaler.get_scale()
        new_scale = self.init_scale * reset_to_percent
        
        # 直接修改scaler的内部scale值
        self.scaler._scale = torch.tensor(new_scale).to(self.scaler._scale.device)
        
        # 重置计数器
        self.consecutive_overflow_count = 0
        self.abnormal_epoch_count = 0
        
        logger.warning(f"🔄 Scale已重置: {current_scale:.2f} -> {new_scale:.2f} (初始值的{reset_to_percent*100:.0f}%)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取监控统计信息
        
        Returns:
            统计信息字典
        """
        if not self.scale_history:
            return {
                'current_scale': self.scaler.get_scale(),
                'min_scale': 0,
                'max_scale': 0,
                'avg_scale': 0,
                'consecutive_overflow': self.consecutive_overflow_count,
                'abnormal_epoch_count': self.abnormal_epoch_count,
                'epoch_records': self.epoch_scale_records
            }
        
        return {
            'current_scale': self.scaler.get_scale(),
            'min_scale': min(self.scale_history),
            'max_scale': max(self.scale_history),
            'avg_scale': sum(self.scale_history) / len(self.scale_history),
            'consecutive_overflow': self.consecutive_overflow_count,
            'abnormal_epoch_count': self.abnormal_epoch_count,
            'epoch_records': self.epoch_scale_records,
            'total_records': len(self.scale_records),
            'overflow_count': sum(1 for r in self.scale_records if r.has_overflow)
        }


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
        
        # 🔑 添加固定参数（多分类任务）
        base_params['objective'] = 'multiclass'
        base_params['num_class'] = 3
        base_params['metric'] = 'multi_logloss'
        base_params['boosting_type'] = 'gbdt'
        
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
        
        # 🔑 添加固定参数（多分类任务）
        base_params['objective'] = 'multi:softprob'
        base_params['num_class'] = 3
        base_params['eval_metric'] = 'mlogloss'
        
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
                'verbose': False,
                'allow_writing_files': False  # 禁用输出文件
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
                'verbose': False,
                'allow_writing_files': False  # 禁用输出文件
            }
        
        # 🎮 GPU加速（如果启用）
        if self.use_gpu:
            base_params['task_type'] = 'GPU'
            base_params['devices'] = '0'
        
        # 🔑 添加固定参数（多分类任务）
        base_params['loss_function'] = 'MultiClass'
        
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
            # 🔥 修复：降低学习率上限，避免梯度爆炸导致nan/inf
            base_params = {
                'd_model': trial.suggest_categorical('d_model', [128, 256]),      # 精确匹配
                'n_heads': trial.suggest_categorical('n_heads', [4, 8]),          # 精确匹配
                'n_layers': trial.suggest_int('n_layers', 2, 3),  # 精确匹配
                'epochs': trial.suggest_int('epochs', 20, 40),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                'lr': trial.suggest_float('lr', 0.0001, 0.002, log=True),  # 降低上限: 0.005→0.002
                'dropout': trial.suggest_float('dropout', 0.05, 0.2),
                'alpha': trial.suggest_float('alpha', 0.5, 2.0),  # GMADL参数
                'beta': trial.suggest_float('beta', 0.3, 0.7)    # GMADL参数
            }
        else:
            # 3m/5m：中序列(96)
            # 🔥 修复：降低学习率上限，避免梯度爆炸导致nan/inf
            base_params = {
                'd_model': trial.suggest_categorical('d_model', [64, 128]),
                'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
                'n_layers': trial.suggest_int('n_layers', 1, 2),
                'epochs': trial.suggest_int('epochs', 10, 30),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'lr': trial.suggest_float('lr', 0.0001, 0.001, log=True),  # 🔧 优化：从0.002降低到0.001，减少梯度爆炸
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
        # ✅ 详细日志：记录Trial开始信息
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"🚀 开始Trial {trial.number} - 模型类型: {self.model_type}, 时间框架: {self.timeframe}")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 获取搜索空间
        try:
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
            
            # ✅ 详细日志：记录所有超参数
            logger.info(f"📋 Trial {trial.number} 超参数配置:")
            for key, value in params.items():
                if isinstance(value, float):
                    logger.info(f"   {key}: {value:.6f}")
                else:
                    logger.info(f"   {key}: {value}")
        except Exception as e:
            logger.error(f"❌ Trial {trial.number}: 获取搜索空间失败: {e}")
            logger.error(f"   错误类型: {type(e).__name__}")
            logger.error(f"   堆栈跟踪:\n{traceback.format_exc()}")
            raise
        
        # ✅ 详细日志：记录输入数据基本信息
        n_samples = len(self.X) if isinstance(self.X, np.ndarray) else self.X.shape[0]
        logger.info(f"📊 Trial {trial.number} 输入数据统计:")
        logger.info(f"   总样本数: {n_samples}")
        logger.info(f"   数据形状: {self.X.shape}")
        logger.info(f"   数据类型: {type(self.X).__name__}, dtype: {self.X.dtype}")
        if isinstance(self.X, np.ndarray):
            logger.info(f"   数据范围: [{self.X.min():.4f}, {self.X.max():.4f}]")
            logger.info(f"   数据均值: {self.X.mean():.4f}, 标准差: {self.X.std():.4f}")
            nan_count = np.isnan(self.X).sum()
            inf_count = np.isinf(self.X).sum()
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"   ⚠️ 输入数据包含异常值: NaN={nan_count}, INF={inf_count}")
        
        # 时间序列5折交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        fold_fail_count = 0
        
        # 🔑 修复：对于3D序列输入，需要基于样本数量而不是特征进行分割
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.arange(n_samples))):
            # ✅ 详细日志：记录Fold开始信息
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"📦 Trial {trial.number} Fold {fold_idx+1}/5 开始处理...")
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.debug(f"   训练索引范围: [{train_idx.min()}, {train_idx.max()}], 数量: {len(train_idx)}")
            logger.debug(f"   验证索引范围: [{val_idx.min()}, {val_idx.max()}], 数量: {len(val_idx)}")
            
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
            
            # ✅ 详细诊断：检查数据分割后的情况（Informer2专用）
            if self.model_type == "informer2":
                logger.info(f"📊 Trial {trial.number} Fold {fold_idx+1}/5 数据分割统计:")
                logger.info(f"   训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")
                logger.info(f"   训练标签形状: {y_train.shape}, 验证标签形状: {y_val.shape}")
                
                # 详细统计信息
                logger.debug(f"   训练集统计:")
                logger.debug(f"      数据类型: {X_train.dtype}, 内存占用: {X_train.nbytes / 1024**2:.2f} MB")
                logger.debug(f"      数据范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
                logger.debug(f"      数据均值: {X_train.mean():.4f}, 标准差: {X_train.std():.4f}")
                logger.debug(f"   标签统计:")
                unique_labels, counts = np.unique(y_train, return_counts=True)
                label_dist = dict(zip(unique_labels, counts))
                logger.debug(f"      标签分布: {label_dist}")
                logger.debug(f"      标签范围: [{y_train.min()}, {y_train.max()}]")
                
                # 检查分割后数据是否为空
                if len(X_train) == 0 or len(y_train) == 0:
                    logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 数据分割后训练集为空！")
                    logger.error(f"   训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
                    logger.error(f"   训练索引: {train_idx[:10]}..." if len(train_idx) > 10 else f"   训练索引: {train_idx}")
                    raise ValueError(f"Fold {fold_idx+1}训练集为空")
                
                if len(X_val) == 0 or len(y_val) == 0:
                    logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 数据分割后验证集为空！")
                    logger.error(f"   验证集形状: {X_val.shape}, 标签形状: {y_val.shape}")
                    logger.error(f"   验证索引: {val_idx[:10]}..." if len(val_idx) > 10 else f"   验证索引: {val_idx}")
                    raise ValueError(f"Fold {fold_idx+1}验证集为空")
                
                # 检查数据维度
                if len(X_train.shape) != 3:
                    logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 训练数据维度错误！")
                    logger.error(f"   期望3D (n_samples, seq_len, n_features), 实际: {X_train.shape}")
                    logger.error(f"   原始数据形状: {self.X.shape}")
                    logger.error(f"   训练索引数量: {len(train_idx)}")
                    raise ValueError(f"训练数据维度错误: {X_train.shape}，期望3D")
                
                # 检查序列长度和特征数
                seq_len = X_train.shape[1]
                n_features = X_train.shape[2]
                logger.info(f"   序列长度: {seq_len}, 特征数: {n_features}")
                if seq_len != params.get('seq_len', 96):
                    logger.warning(f"   ⚠️ 序列长度不匹配: 数据={seq_len}, 参数={params.get('seq_len', 96)}")
            
            # 计算样本权重（有效样本数 × 时间衰减 × HOLD惩罚）
            try:
                temp_svc = MLService()
                class_weights = temp_svc._compute_effective_sample_weights(y_train, self.timeframe)
            except Exception:
                class_weights = compute_sample_weight('balanced', y_train)
            # ✅ 添加时间衰减权重（与基础模型训练保持一致）
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            # HOLD增强权重
            hold_penalty_weights = np.where(y_train == 1, HOLD_WEIGHT_MULTIPLIER, 1.0)
            sample_weights = class_weights * time_decay * hold_penalty_weights
            
            # 训练模型
            try:
                # 🎮 统一GPU内存管理：训练前清理
                self.clear_gpu_memory()

                if self.model_type == "lightgbm":
                    try:
                        model = lgb.LGBMClassifier(**params)
                        # 添加验证集和早停机制
                        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=0)]
                        model.fit(
                            X_train, y_train,
                            sample_weight=sample_weights,
                            eval_set=[(X_val, y_val)],
                            callbacks=callbacks
                        )
                        
                        # 🎮 统一GPU内存管理：训练后清理
                        self.clear_gpu_memory()
                            
                    except Exception as e:
                        logger.error(f"❌ LightGBM训练失败: {e}")
                        # 降级到CPU
                        params['device'] = 'cpu'
                        model = lgb.LGBMClassifier(**params)
                        # 添加验证集和早停机制
                        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=0)]
                        model.fit(
                            X_train, y_train,
                            sample_weight=sample_weights,
                            eval_set=[(X_val, y_val)],
                            callbacks=callbacks
                        )
                        self.clear_gpu_memory()
                
                elif self.model_type == "xgboost":
                    try:
                        # 🔑 XGBoost 2.0+ API变更：callbacks不能传入fit()，只能在构造函数中配置
                        # XGBoost 1.6-1.9: callbacks可传入fit()
                        # XGBoost <1.6: 使用early_stopping_rounds参数
                        xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
                        
                        if xgb_version >= (2, 0):
                            # 🔑 XGBoost 2.0+ API: 早停通过构造函数参数控制
                            params['early_stopping_rounds'] = 100
                            params['eval_metric'] = 'mlogloss'
                            model = xgb.XGBClassifier(**params)
                            model.fit(
                                X_train, y_train,
                                sample_weight=sample_weights,
                                eval_set=[(X_val, y_val)],
                                verbose=False
                            )
                        elif xgb_version >= (1, 6):
                            # XGBoost 1.6-1.9 API: callbacks可传入fit()
                            early_stop = xgb.callback.EarlyStopping(
                                rounds=100, 
                                save_best=True,
                                maximize=False
                            )
                            model = xgb.XGBClassifier(**params)
                            model.fit(
                                X_train, y_train,
                                sample_weight=sample_weights,
                                eval_set=[(X_val, y_val)],
                                callbacks=[early_stop],
                                verbose=False
                            )
                        else:
                            # XGBoost <1.6 API: 使用early_stopping_rounds参数
                            logger.warning(f"⚠️ XGBoost版本{xgb.__version__} < 1.6.0，使用旧版API")
                            model = xgb.XGBClassifier(**params)
                            model.fit(
                                X_train, y_train,
                                sample_weight=sample_weights,
                                eval_set=[(X_val, y_val)],
                                early_stopping_rounds=100,
                                verbose=False
                            )
                        
                        # 🎮 统一GPU内存管理：训练后清理
                        self.clear_gpu_memory()
                            
                    except Exception as e:
                        logger.error(f"❌ XGBoost训练失败: {e}")
                        logger.error(f"   XGBoost版本: {xgb.__version__}")
                        logger.error(traceback.format_exc())
                        
                        # 降级到CPU重试
                        params['tree_method'] = 'hist'
                        params['device'] = 'cpu'
                        
                        try:
                            xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
                            if xgb_version >= (2, 0):
                                # XGBoost 2.0+ API
                                params['early_stopping_rounds'] = 100
                                params['eval_metric'] = 'mlogloss'
                                model = xgb.XGBClassifier(**params)
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    verbose=False
                                )
                            elif xgb_version >= (1, 6):
                                # XGBoost 1.6-1.9 API
                                early_stop = xgb.callback.EarlyStopping(
                                    rounds=100,
                                    save_best=True,
                                    maximize=False
                                )
                                model = xgb.XGBClassifier(**params)
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    callbacks=[early_stop],
                                    verbose=False
                                )
                            else:
                                # XGBoost <1.6 API
                                model = xgb.XGBClassifier(**params)
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    early_stopping_rounds=100,
                                    verbose=False
                                )
                            logger.info(f"✅ XGBoost降级到CPU后训练成功")
                        except Exception as e2:
                            logger.error(f"❌ XGBoost CPU降级也失败: {e2}")
                            logger.error(traceback.format_exc())
                            raise
                        
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
                        # 添加验证集和早停机制
                        model.fit(
                            X_train, y_train, 
                            sample_weight=sample_weights,
                            eval_set=(X_val, y_val),
                            early_stopping_rounds=100,  # 早停轮数
                            verbose=False
                        )
                        
                        # 🎮 统一GPU内存管理：训练后清理
                        self.clear_gpu_memory()
                            
                    except Exception as e:
                        logger.error(f"❌ CatBoost GPU训练失败: {e}")
                        # 降级到CPU
                        params['task_type'] = 'CPU'
                        model = cb.CatBoostClassifier(**params)
                        # 添加验证集和早停机制
                        model.fit(
                            X_train, y_train, 
                            sample_weight=sample_weights,
                            eval_set=(X_val, y_val),
                            early_stopping_rounds=100,  # 早停轮数
                            verbose=False
                        )
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
                    
                    # 🎯 使用滚动窗口标准化（避免数据泄露）
                    # 核心原理：对于时间点 t，只使用 [t-window_size+1, t] 的历史数据计算均值和标准差
                    # 不使用未来数据（t+1, t+2, ...），确保时序数据无泄露
                    logger.info(f"🔧 Trial {trial.number} Fold {fold_idx+1}/5 使用滚动窗口标准化...")
                    logger.info(f"   标准化前统计: 范围=[{X_train.min():.4f}, {X_train.max():.4f}], 均值={X_train.mean():.4f}, 标准差={X_train.std():.4f}")
                    
                    # 从 constants.py 读取滚动窗口大小
                    from app.core.constants import ROLLING_SCALER_WINDOW_SIZE
                    window_size = ROLLING_SCALER_WINDOW_SIZE
                    
                    # 方法：对每个样本的序列，使用滚动窗口计算均值和标准差
                    # 输入：(n_samples, seq_len, n_features)
                    # 输出：(n_samples, seq_len, n_features)
                    original_shape_train = X_train.shape
                    original_shape_val = X_val.shape
                    n_samples_train, seq_len, n_features = original_shape_train
                    n_samples_val = original_shape_val[0]
                    
                    # 初始化标准化后的数组
                    X_train_scaled = np.zeros_like(X_train, dtype=np.float32)
                    X_val_scaled = np.zeros_like(X_val, dtype=np.float32)
                    
                    # 对训练集进行滚动窗口标准化
                    for sample_idx in range(n_samples_train):
                        sample_seq = X_train[sample_idx]  # (seq_len, n_features)
                        
                        # 对序列中的每个时间步，使用滚动窗口标准化
                        for t in range(seq_len):
                            if t < window_size:
                                # 前 window_size 个时间步：使用扩展窗口 [0, t+1]
                                window_data = sample_seq[:t+1]  # (t+1, n_features)
                            else:
                                # 后续时间步：使用固定大小的滚动窗口 [t-window_size+1, t+1]
                                window_data = sample_seq[t-window_size+1:t+1]  # (window_size, n_features)
                            
                            # 计算窗口内的均值和标准差（按特征维度）
                            window_mean = window_data.mean(axis=0)  # (n_features,)
                            window_std = window_data.std(axis=0)  # (n_features,)
                            # 防止除零
                            window_std = np.where(window_std < 1e-10, 1e-10, window_std)
                            
                            # 标准化当前时间步
                            X_train_scaled[sample_idx, t] = (sample_seq[t] - window_mean) / window_std
                    
                    # 对验证集进行滚动窗口标准化（使用相同的逻辑）
                    for sample_idx in range(n_samples_val):
                        sample_seq = X_val[sample_idx]  # (seq_len, n_features)
                        
                        for t in range(seq_len):
                            if t < window_size:
                                window_data = sample_seq[:t+1]
                            else:
                                window_data = sample_seq[t-window_size+1:t+1]
                            
                            window_mean = window_data.mean(axis=0)
                            window_std = window_data.std(axis=0)
                            window_std = np.where(window_std < 1e-10, 1e-10, window_std)
                            
                            X_val_scaled[sample_idx, t] = (sample_seq[t] - window_mean) / window_std
                    
                    # 更新数据
                    X_train = X_train_scaled
                    X_val = X_val_scaled
                    
                    logger.info(f"   ✅ 滚动窗口标准化完成 (window_size={window_size})")
                    logger.info(f"   标准化后统计: 范围=[{X_train.min():.4f}, {X_train.max():.4f}], 均值={X_train.mean():.4f}, 标准差={X_train.std():.4f}")
                    
                    # 转换为PyTorch张量（内存优化）
                    device = torch.device('cuda:0' if self.use_gpu and torch.cuda.is_available() else 'cpu')
                    
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

                    # 🔍 修复E: 训练前数据质量检查（Optuna试验模式）
                    logger.info(f"🔍 Trial {trial.number} Fold {fold_idx+1}/5 执行训练前数据质量检查...")
                    
                    # 检查特征数据
                    nan_count = np.isnan(X_train).sum()
                    inf_count = np.isinf(X_train).sum()
                    if nan_count > 0:
                        logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 训练数据包含{nan_count}个NaN值！")
                        logger.error(f"   数据形状: {X_train.shape}, NaN比例: {100*nan_count/X_train.size:.2f}%")
                        logger.error(f"   NaN位置统计（前10个）:")
                        nan_positions = np.where(np.isnan(X_train))
                        for i in range(min(10, len(nan_positions[0]))):
                            logger.error(f"      位置: ({nan_positions[0][i]}, {nan_positions[1][i]}, {nan_positions[2][i]})")
                        raise ValueError(f"训练数据包含NaN值：{nan_count}个（{100*nan_count/X_train.size:.2f}%）")
                    
                    if inf_count > 0:
                        logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 训练数据包含{inf_count}个INF值！")
                        logger.error(f"   数据形状: {X_train.shape}, INF比例: {100*inf_count/X_train.size:.2f}%")
                        logger.error(f"   INF位置统计（前10个）:")
                        inf_positions = np.where(np.isinf(X_train))
                        for i in range(min(10, len(inf_positions[0]))):
                            logger.error(f"      位置: ({inf_positions[0][i]}, {inf_positions[1][i]}, {inf_positions[2][i]})")
                        raise ValueError(f"训练数据包含INF值：{inf_count}个（{100*inf_count/X_train.size:.2f}%）")
                    
                    # 检查标签数据
                    label_nan_count = np.isnan(y_train_np).sum() if np.isnan(y_train_np).any() else 0
                    label_inf_count = np.isinf(y_train_np).sum() if np.isinf(y_train_np).any() else 0
                    if label_nan_count > 0 or label_inf_count > 0:
                        logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 训练标签包含NaN/INF值！")
                        logger.error(f"   标签形状: {y_train_np.shape}, NaN: {label_nan_count}, INF: {label_inf_count}")
                        if label_nan_count > 0:
                            nan_indices = np.where(np.isnan(y_train_np))[0]
                            logger.error(f"   NaN标签位置（前10个）: {nan_indices[:10]}")
                        if label_inf_count > 0:
                            inf_indices = np.where(np.isinf(y_train_np))[0]
                            logger.error(f"   INF标签位置（前10个）: {inf_indices[:10]}")
                        raise ValueError(f"训练标签包含NaN/INF值（NaN: {label_nan_count}, INF: {label_inf_count}）")
                    
                    # 检查标签范围
                    unique_labels = np.unique(y_train_np)
                    if not all(label in [0, 1, 2] for label in unique_labels):
                        logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 训练标签包含非法值！")
                        logger.error(f"   期望标签: [0, 1, 2], 实际标签: {unique_labels.tolist()}")
                        logger.error(f"   标签统计: {np.bincount(y_train_np.astype(int))}")
                        illegal_indices = np.where(~np.isin(y_train_np, [0, 1, 2]))[0]
                        logger.error(f"   非法标签位置（前10个）: {illegal_indices[:10]}")
                        logger.error(f"   非法标签值（前10个）: {y_train_np[illegal_indices[:10]]}")
                        raise ValueError(f"训练标签包含非法值：{unique_labels.tolist()}，期望[0,1,2]")
                    
                    # 统计数据范围
                    logger.info(f"   ✅ Fold {fold_idx+1} 数据质量检查通过")
                    logger.info(f"      特征范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
                    logger.info(f"      特征均值: {X_train.mean():.4f}, 标准差: {X_train.std():.4f}")
                    logger.info(f"      标签分布: {np.bincount(y_train_np.astype(int)).tolist()}")
                    logger.info(f"      样本数: {len(y_train_np)}")

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
                    
                    # 🔍 修复E: 模型权重初始化检查
                    logger.info(f"🔍 Trial {trial.number} Fold {fold_idx+1}/5 检查模型权重初始化...")
                    has_nan_weights = False
                    has_inf_weights = False
                    weight_stats = {}
                    
                    for name, param in model.named_parameters():
                        param_nan = torch.isnan(param).sum().item()
                        param_inf = torch.isinf(param).sum().item()
                        param_total = param.numel()
                        
                        if param_nan > 0:
                            logger.error(f"❌ 模型参数 {name} 包含{param_nan}个NaN值（共{param_total}个参数）！")
                            logger.error(f"   参数形状: {param.shape}, 参数范围: [{param.min().item():.4f}, {param.max().item():.4f}]")
                            has_nan_weights = True
                        if param_inf > 0:
                            logger.error(f"❌ 模型参数 {name} 包含{param_inf}个INF值（共{param_total}个参数）！")
                            logger.error(f"   参数形状: {param.shape}, 参数范围: [{param.min().item():.4f}, {param.max().item():.4f}]")
                            has_inf_weights = True
                        
                        # 记录参数统计
                        weight_stats[name] = {
                            'shape': list(param.shape),
                            'min': param.min().item(),
                            'max': param.max().item(),
                            'mean': param.mean().item(),
                            'std': param.std().item(),
                            'nan': param_nan,
                            'inf': param_inf
                        }
                    
                    if has_nan_weights or has_inf_weights:
                        logger.error("❌ 模型权重初始化异常，训练终止！")
                        logger.error("   详细参数统计:")
                        for name, stats in weight_stats.items():
                            if stats['nan'] > 0 or stats['inf'] > 0:
                                logger.error(f"      {name}: {stats}")
                        raise ValueError("模型权重初始化包含NaN/INF值")
                    
                    logger.info("   ✅ 模型权重初始化正常")
                    logger.debug(f"   模型参数统计（前5个）:")
                    for i, (name, stats) in enumerate(list(weight_stats.items())[:5]):
                        logger.debug(f"      {name}: shape={stats['shape']}, range=[{stats['min']:.4f}, {stats['max']:.4f}], mean={stats['mean']:.4f}")
                    
                    # 定义损失函数（与训练流程保持一致）
                    hold_ratio_opt = float((y_train_np == 1).sum()) / max(len(y_train_np), 1)
                    if self.timeframe == '3m':
                        hold_penalty_nn = float(max(0.35, min(0.70, 0.80 - 0.6 * hold_ratio_opt)))
                    else:
                        hold_penalty_nn = float(max(0.50, min(0.75, 0.85 - 0.5 * hold_ratio_opt)))

                    criterion = create_trade_loss(
                        use_gmadl=USE_GMADL_LOSS,
                        hold_penalty=hold_penalty_nn,
                        alpha=params.get('alpha', GMADL_ALPHA),
                        beta=params.get('beta', GMADL_BETA)
                    )

                    if USE_GMADL_LOSS:
                        logger.debug(
                            f"   损失函数: GMADL + HOLD惩罚 (alpha={params.get('alpha', GMADL_ALPHA):.2f}, beta={params.get('beta', GMADL_BETA):.2f})"
                        )
                    else:
                        logger.debug("   损失函数: 交叉熵 + HOLD惩罚 (稳定模式)")
                    
                    # 🔥 尝试使用8-bit Adam优化器（节省75%优化器内存）
                    optimizer_created = False
                    if self.use_gpu and device.type == 'cuda':
                        try:
                            if not BNB_AVAILABLE:
                                raise ImportError("bitsandbytes未安装")
                            optimizer = bnb.optim.Adam8bit(
                                model.parameters(),
                                lr=params['lr'],
                                betas=(0.9, 0.999)
                            )
                            optimizer_created = True
                        except (ImportError, Exception):
                            pass
                    
                    if not optimizer_created:
                        optimizer = torch.optim.Adam(
                            model.parameters(),
                            lr=params['lr'],
                            weight_decay=1e-5,
                            betas=(0.9, 0.999)
                        )
                    
                    # ✅ 修复C: 添加Warmup + ReduceLROnPlateau组合调度器
                    # Warmup配置
                    warmup_epochs = 5  # 前5个epoch warmup
                    target_lr = params['lr']
                    
                    # 主调度器：ReduceLROnPlateau（用于warmup后的学习率调整）
                    # ✅ 修复：移除已废弃的verbose参数（PyTorch新版本已废弃）
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=OPTIMIZER_LR_REDUCE_FACTOR,
                        patience=5,
                        min_lr=1e-6,
                        threshold=OPTIMIZER_LR_REDUCE_THRESHOLD,
                        threshold_mode='rel',
                        cooldown=2
                    )
                    
                    logger.info(f"   ✅ 学习率调度: Warmup({warmup_epochs}轮) + ReduceLROnPlateau")
                    logger.info(f"      目标LR: {target_lr:.6f}, Warmup后自动调整")
                    
                    # ✅ 关键修复：初始化Warmup学习率（第一个epoch应该从target_lr/warmup_epochs开始）
                    # 注意：优化器创建时已经设置为target_lr，需要在训练前调整为Warmup初始值
                    initial_warmup_lr = target_lr / warmup_epochs  # 第一个epoch的初始学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = initial_warmup_lr
                    logger.info(f"   ✅ Warmup初始学习率已设置: {initial_warmup_lr:.6f} (目标LR的1/{warmup_epochs})")
                    logger.debug(f"   优化器参数组数量: {len(optimizer.param_groups)}")
                    for i, pg in enumerate(optimizer.param_groups):
                        logger.debug(f"      参数组{i}: lr={pg['lr']:.6f}, weight_decay={pg.get('weight_decay', 0)}")
                    
                    # 🚀 动态混合精度配置（使用新的配置器和监控器）
                    use_amp = device.type == 'cuda' and torch.cuda.is_available()
                    if USE_GMADL_LOSS and use_amp:
                        logger.debug("   ⚠️ GMADL开启 → Optuna试验禁用AMP改用FP32训练")
                        use_amp = False
                    
                    scaler = None
                    scaler_monitor = None
                    
                    if use_amp:
                        # 🔥 使用动态GradScaler配置器
                        num_params = sum(p.numel() for p in model.parameters())
                        scaler_config = DynamicGradScalerConfig(num_params)
                        scaler = scaler_config.create_scaler('cuda')
                        
                        # 🔥 创建GradScaler监控器
                        scaler_monitor = GradScalerMonitor(scaler, scaler_config.init_scale)
                        
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        logger.debug(f"   混合精度训练: 启用（动态缩放策略 + 监控器）")
                    else:
                        logger.debug("   混合精度训练: 禁用（CPU环境或GMADL模式）")
                    
                    # 训练模型（带梯度累积和混合精度）
                    model.train()
                    nan_inf_count = 0  # 统计nan/inf出现次数
                    max_nan_inf_tolerance = 30  # ✅ 修复F: 符合文档要求
                    consecutive_nan_inf = 0
                    max_consecutive_nan_inf = 8  # ✅ 修复F: 从5改为8（符合文档）
                    
                    logger.info(f"🚂 Trial {trial.number} Fold {fold_idx+1}/5 开始训练...")
                    logger.info(f"   训练配置:")
                    logger.info(f"      总Epoch数: {params['epochs']}")
                    logger.info(f"      批次大小: {actual_batch_size} (有效批次: {effective_batch_size}, 累积步数: {accumulation_steps})")
                    logger.info(f"      设备: {device}")
                    logger.info(f"      混合精度: {'启用' if use_amp else '禁用'}")
                    logger.info(f"      早期终止阈值: 连续{max_consecutive_nan_inf}次 或 累计{max_nan_inf_tolerance}次")
                    logger.info(f"      总批次数: {len(train_loader)}")
                    
                    for epoch in range(params['epochs']):
                        optimizer.zero_grad()
                        
                        # ✅ 修复C: 初始化epoch统计
                        epoch_loss = 0.0
                        correct = 0
                        total = 0
                        processed_batches = 0
                        epoch_nan_inf_count = 0
                        
                        # ✅ 详细日志：记录epoch开始
                        if epoch == 0:
                            logger.info(f"   📍 Epoch {epoch+1}/{params['epochs']} 开始（第一个epoch，将记录详细诊断信息）...")
                        elif epoch % 5 == 0:
                            logger.info(f"   📍 Epoch {epoch+1}/{params['epochs']} 开始...")
                        
                        for i, (batch_X, batch_y) in enumerate(train_loader):
                            # 🎯 混合精度前向传播
                            # 将批次移动到目标设备
                            batch_X = batch_X.to(device, non_blocking=True)
                            batch_y = batch_y.to(device, non_blocking=True)
                            
                            # ✅ 详细诊断：第一个batch的完整信息（无论是否有错误）
                            if i == 0 and epoch == 0:  # 只在第一个batch的第一个epoch检查
                                logger.info(f"   🔍 第一个Batch详细诊断:")
                                logger.info(f"      Batch形状: {batch_X.shape}")
                                logger.info(f"      输入范围: [{batch_X.min().item():.4f}, {batch_X.max().item():.4f}]")
                                logger.info(f"      输入均值: {batch_X.mean().item():.4f}, 标准差: {batch_X.std().item():.4f}")
                                batch_nan = torch.isnan(batch_X).sum().item()
                                batch_inf = torch.isinf(batch_X).sum().item()
                                if batch_nan > 0 or batch_inf > 0:
                                    logger.error(f"      ❌ 第一个batch输入数据异常！")
                                    logger.error(f"         NaN: {batch_nan}, INF: {batch_inf}")
                                    logger.error(f"         NaN位置（前5个）:")
                                    nan_pos = torch.where(torch.isnan(batch_X))
                                    for j in range(min(5, len(nan_pos[0]))):
                                        logger.error(f"           位置: ({nan_pos[0][j]}, {nan_pos[1][j]}, {nan_pos[2][j]})")
                                    raise ValueError(f"Batch输入数据包含NaN/INF（NaN: {batch_nan}, INF: {batch_inf}）")
                                else:
                                    logger.info(f"      ✅ 输入数据正常（无NaN/INF）")
                                
                                logger.info(f"      标签形状: {batch_y.shape}")
                                logger.info(f"      标签分布: {torch.bincount(batch_y.long()).tolist()}")
                                logger.info(f"      标签范围: [{batch_y.min().item()}, {batch_y.max().item()}]")
                            
                            if use_amp:
                                with torch.amp.autocast('cuda'):
                                    outputs = model(batch_X)
                                    # 统一dtype与loss输入：logits用float32，targets用long
                                    loss = criterion(outputs.float(), batch_y.long()) / accumulation_steps
                            else:
                                outputs = model(batch_X)
                                loss = criterion(outputs.float(), batch_y.long()) / accumulation_steps
                            
                            # ✅ 详细诊断：第一个batch的模型输出和损失（无论是否有错误）
                            if i == 0 and epoch == 0:
                                logger.info(f"      🤖 模型输出统计:")
                                logger.info(f"         输出形状: {outputs.shape}")
                                logger.info(f"         输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                                logger.info(f"         输出均值: {outputs.mean().item():.4f}, 标准差: {outputs.std().item():.4f}")
                                output_nan = torch.isnan(outputs).sum().item()
                                output_inf = torch.isinf(outputs).sum().item()
                                if output_nan > 0 or output_inf > 0:
                                    logger.error(f"         ❌ 模型输出包含NaN/INF: NaN={output_nan}, INF={output_inf}")
                                else:
                                    logger.info(f"         ✅ 模型输出正常（无NaN/INF）")
                                
                                logger.info(f"      📉 损失统计:")
                                logger.info(f"         损失值: {loss.item():.6f}")
                                if torch.isnan(loss) or torch.isinf(loss):
                                    logger.error(f"         ❌ 损失值异常！")
                                else:
                                    logger.info(f"         ✅ 损失值正常")
                            
                            # 🔍 检测数值不稳定（增强诊断）
                            if torch.isnan(loss) or torch.isinf(loss):
                                nan_inf_count += 1
                                consecutive_nan_inf += 1
                                epoch_nan_inf_count += 1
                                
                                # ✅ 关键修复：检测到NaN/INF时，立即跳过反向传播，避免污染梯度
                                optimizer.zero_grad()
                                
                                # ✅ 详细诊断：记录第一个NaN/INF的详细信息
                                if nan_inf_count == 1:
                                    logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                    logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 第一个NaN/INF损失检测！")
                                    logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                    logger.error(f"   📍 位置信息:")
                                    logger.error(f"      Epoch: {epoch+1}/{params['epochs']}, Batch: {i+1}/{len(train_loader)}")
                                    logger.error(f"      累计处理批次数: {processed_batches}")
                                    
                                    logger.error(f"   📊 输入数据统计:")
                                    logger.error(f"      输入形状: {batch_X.shape}")
                                    logger.error(f"      输入范围: min={batch_X.min().item():.4f}, max={batch_X.max().item():.4f}")
                                    logger.error(f"      输入均值: {batch_X.mean().item():.4f}, 标准差: {batch_X.std().item():.4f}")
                                    batch_nan = torch.isnan(batch_X).sum().item()
                                    batch_inf = torch.isinf(batch_X).sum().item()
                                    if batch_nan > 0 or batch_inf > 0:
                                        logger.error(f"      ⚠️ 输入包含异常值: NaN={batch_nan}, INF={batch_inf}")
                                    
                                    logger.error(f"   🏷️ 标签统计:")
                                    logger.error(f"      标签形状: {batch_y.shape}")
                                    logger.error(f"      标签分布: {torch.bincount(batch_y.long()).tolist()}")
                                    logger.error(f"      标签范围: [{batch_y.min().item()}, {batch_y.max().item()}]")
                                    
                                    logger.error(f"   🤖 模型输出(logits)统计:")
                                    logger.error(f"      输出形状: {outputs.shape}")
                                    logger.error(f"      输出范围: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
                                    logger.error(f"      输出均值: {outputs.mean().item():.4f}, 标准差: {outputs.std().item():.4f}")
                                    
                                    # 检查模型输出
                                    output_nan = torch.isnan(outputs).sum().item()
                                    output_inf = torch.isinf(outputs).sum().item()
                                    if output_nan > 0 or output_inf > 0:
                                        logger.error(f"      ❌ 模型输出包含NaN/INF！")
                                        logger.error(f"         NaN数量: {output_nan}, INF数量: {output_inf}")
                                        logger.error(f"         NaN位置（前5个）:")
                                        nan_pos = torch.where(torch.isnan(outputs))
                                        for j in range(min(5, len(nan_pos[0]))):
                                            logger.error(f"           样本{nan_pos[0][j]}, 类别{nan_pos[1][j]}")
                                        if output_inf > 0:
                                            inf_pos = torch.where(torch.isinf(outputs))
                                            logger.error(f"         INF位置（前5个）:")
                                            for j in range(min(5, len(inf_pos[0]))):
                                                logger.error(f"           样本{inf_pos[0][j]}, 类别{inf_pos[1][j]}")
                                    else:
                                        logger.error(f"      ✅ 模型输出正常（无NaN/INF）")
                                    
                                    logger.error(f"   📉 损失函数统计:")
                                    logger.error(f"      损失值: {loss.item()}")
                                    logger.error(f"      损失类型: {type(loss).__name__}")
                                    logger.error(f"      损失设备: {loss.device}")
                                    
                                    logger.error(f"   ⚙️ 训练配置:")
                                    logger.error(f"      当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
                                    logger.error(f"      混合精度: {'启用' if use_amp else '禁用'}")
                                    if use_amp and scaler is not None:
                                        logger.error(f"      缩放器状态: scale={scaler.get_scale():.4f}")
                                    logger.error(f"      梯度累积步数: {accumulation_steps}")
                                    
                                    logger.error(f"   🔍 模型参数检查（前5个）:")
                                    for j, (name, param) in enumerate(list(model.named_parameters())[:5]):
                                        param_nan = torch.isnan(param).sum().item()
                                        param_inf = torch.isinf(param).sum().item()
                                        if param.numel() > 0:
                                            param_min = param.min().item()
                                            param_max = param.max().item()
                                        else:
                                            param_min = float('nan')
                                            param_max = float('nan')
                                        
                                        if param_nan > 0 or param_inf > 0:
                                            logger.error(f"      {name}: NaN={param_nan}, INF={param_inf}, range=[{param_min:.4f}, {param_max:.4f}], shape={list(param.shape)}")
                                        else:
                                            logger.error(f"      {name}: 正常, range=[{param_min:.4f}, {param_max:.4f}], shape={list(param.shape)}")
                                    
                                    # ✅ 关键诊断：检查梯度状态
                                    try:
                                        total_grad_norm = 0.0
                                        grad_count = 0
                                        has_nan_grad = False
                                        for name, param in model.named_parameters():
                                            if param.grad is not None:
                                                grad_nan = torch.isnan(param.grad).sum().item()
                                                grad_inf = torch.isinf(param.grad).sum().item()
                                                if grad_nan > 0 or grad_inf > 0:
                                                    logger.error(f"      ⚠️ {name} 梯度包含NaN/INF: NaN={grad_nan}, INF={grad_inf}")
                                                    has_nan_grad = True
                                                else:
                                                    grad_norm = param.grad.norm().item()
                                                    total_grad_norm += grad_norm ** 2
                                                    grad_count += 1
                                        if grad_count > 0:
                                            total_grad_norm = total_grad_norm ** 0.5
                                            logger.error(f"   📊 总梯度范数: {total_grad_norm:.4f} (来自{grad_count}个参数)")
                                        if has_nan_grad:
                                            logger.error(f"   ⚠️ 检测到梯度中包含NaN/INF，这可能是导致模型输出NaN的根本原因！")
                                    except Exception as grad_error:
                                        logger.error(f"   ⚠️ 梯度检查失败: {grad_error}")
                                    
                                    logger.error(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                
                                # Optuna试验中出现nan/inf直接prune
                                if consecutive_nan_inf >= max_consecutive_nan_inf or nan_inf_count >= max_nan_inf_tolerance:
                                    logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1}: 触发prune条件！")
                                    logger.error(f"   连续NaN/INF: {consecutive_nan_inf}/{max_consecutive_nan_inf}, 累计: {nan_inf_count}/{max_nan_inf_tolerance}")
                                    raise optuna.TrialPruned()
                                
                                # ✅ 已在上方执行了optimizer.zero_grad()，这里直接跳过
                                continue
                            
                            consecutive_nan_inf = 0
                            
                            # ✅ 修复C: 累积loss和accuracy统计
                            epoch_loss += loss.item() * accumulation_steps  # 反归一化（因为loss除以了accumulation_steps）
                            processed_batches += 1
                            
                            # 计算准确率
                            with torch.no_grad():
                                pred = torch.argmax(outputs, dim=1)
                                correct += (pred == batch_y).sum().item()
                                total += batch_y.size(0)
                            
                            # 🎯 混合精度反向传播
                            if use_amp:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                            
                            # 🎯 梯度累积
                            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                                if use_amp:
                                    try:
                                        scaler.unscale_(optimizer)
                                    except RuntimeError as e:
                                        if "unscale_() has already been called" in str(e):
                                            # 如果已经调用过unscale_()，说明之前可能已经处理过，直接跳过
                                            logger.warning(f"⚠️ Trial {trial.number} Fold {fold_idx+1} Epoch {epoch+1} Batch {i+1}: unscale_()已调用，跳过此次更新")
                                            optimizer.zero_grad()
                                            continue
                                        else:
                                            raise
                                
                                # ✅ 关键修复：更严格的梯度裁剪（防止梯度爆炸）
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 
                                    max_norm=0.5,  # ✅ 从1.0降低到0.5，更严格
                                    norm_type=2.0
                                )
                                
                                # ✅ 关键修复：检查梯度裁剪后的梯度范数
                                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 10.0:
                                    logger.warning(f"⚠️ Trial {trial.number} Fold {fold_idx+1} Epoch {epoch+1} Batch {i+1}: 梯度异常 (grad_norm={grad_norm:.4f})，跳过此batch")
                                    # ✅ 修复：如果使用了混合精度，需要清理scaler状态
                                    if use_amp:
                                        try:
                                            scaler.update()  # 更新scaler状态，避免后续unscale_()报错
                                        except:
                                            pass
                                    optimizer.zero_grad()
                                    continue
                                
                                # ✅ 关键修复：定期检查模型参数是否变得不稳定（每100个batch检查一次）
                                if (i + 1) % (accumulation_steps * 100) == 0:
                                    has_unstable_params = False
                                    for name, param in model.named_parameters():
                                        if param.numel() > 0:
                                            param_max = param.abs().max().item()
                                            if param_max > 1e6:  # 参数绝对值超过100万
                                                logger.warning(f"⚠️ Trial {trial.number} Fold {fold_idx+1} Epoch {epoch+1} Batch {i+1}: 参数{name}异常大 (max_abs={param_max:.2e})")
                                                has_unstable_params = True
                                            if torch.isnan(param).any() or torch.isinf(param).any():
                                                logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1} Epoch {epoch+1} Batch {i+1}: 参数{name}包含NaN/INF！")
                                                has_unstable_params = True
                                    
                                    if has_unstable_params:
                                        logger.error(f"❌ 检测到模型参数不稳定，训练终止！")
                                        raise ValueError("模型参数变得不稳定，包含NaN/INF或异常大的值")
                                
                                if use_amp:
                                    # 🔥 混合精度训练梯度更新流程（使用监控器）
                                    scaler.step(optimizer)
                                    scaler.update()
                                    
                                    # 🔥 使用监控器检查scale和溢出
                                    if scaler_monitor:
                                        # 记录当前scale
                                        scale_exceeded = scaler_monitor.record_scale(epoch, i)
                                        
                                        # 检查是否有溢出（通过检查scale是否减小来判断）
                                        has_overflow = scaler.get_scale() < scaler_monitor.scale_history[-2] if len(scaler_monitor.scale_history) > 1 else False
                                        
                                        # 检查溢出并判断是否需要重置
                                        need_reset = scaler_monitor.check_overflow(has_overflow, epoch, i)
                                        
                                        # 如果scale超过阈值或连续溢出，触发重置
                                        if need_reset or scale_exceeded:
                                            if GRAD_SCALER_AUTO_RESET:
                                                scaler_monitor.reset_scale()
                                                logger.warning(f"🔄 Trial {trial.number} Fold {fold_idx+1} Epoch {epoch+1} Batch {i+1}: Scale已自动重置")
                                            else:
                                                logger.warning(f"⚠️ Trial {trial.number} Fold {fold_idx+1} Epoch {epoch+1} Batch {i+1}: 检测到异常但自动重置已禁用")
                                        
                                        # 每100个batch记录一次scale统计
                                        if (i + 1) % (accumulation_steps * 100) == 0:
                                            stats = scaler_monitor.get_statistics()
                                            logger.debug(f"📊 Scale统计: 当前={stats['current_scale']:.2f}, 平均={stats['avg_scale']:.2f}, 最大={stats['max_scale']:.2f}")
                                else:
                                    optimizer.step()
                                
                                optimizer.zero_grad()
                                
                                # 定期清理GPU缓存
                                if (i + 1) % (accumulation_steps * 10) == 0 and device.type == 'cuda':
                                    torch.cuda.empty_cache()
                        
                        # ✅ 修复F: Epoch级别检查
                        total_batches = len(train_loader)
                        
                        if processed_batches == 0:
                            logger.error(f"❌ Epoch {epoch+1}: 没有任何batch成功处理（全部{total_batches}个batch均为nan/inf），训练终止！")
                            raise ValueError(f"Epoch {epoch+1}所有batch均出现nan/inf，训练无法继续")
                        
                        if epoch_nan_inf_count > total_batches * 0.5:
                            logger.error(f"❌ Epoch {epoch+1}: {epoch_nan_inf_count}/{total_batches} batch出现nan/inf "
                                        f"({100*epoch_nan_inf_count/total_batches:.1f}%，超过50%阈值），训练终止！")
                            raise ValueError(f"Epoch {epoch+1}超过50%的batch出现nan/inf，训练质量无法保证")
                        
                        if epoch_nan_inf_count > total_batches * 0.3:
                            logger.warning(f"⚠️ Epoch {epoch+1}: {epoch_nan_inf_count}/{total_batches} batch出现nan/inf "
                                          f"({100*epoch_nan_inf_count/total_batches:.1f}%），数值稳定性问题！")
                        
                        # ✅ 修复C: 计算平均loss和accuracy
                        avg_loss = epoch_loss / max(processed_batches, 1)
                        accuracy = 100.0 * correct / max(total, 1)
                        
                        # ✅ 修复C: 学习率调度（简化的Warmup + ReduceLROnPlateau）
                        if epoch < warmup_epochs:
                            # Warmup阶段：线性增长学习率
                            # epoch 0: target_lr * 1/5, epoch 1: target_lr * 2/5, ...
                            warmup_lr = target_lr * (epoch + 1) / warmup_epochs
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = warmup_lr
                            phase = "Warmup"
                            current_lr = warmup_lr
                        else:
                            # 主调度阶段：根据loss自动调整
                            scheduler.step(avg_loss)
                            phase = "Main"
                            current_lr = optimizer.param_groups[0]['lr']
                        
                        # 打印epoch信息（带学习率）
                        logger.debug(
                            f"   Epoch [{epoch+1}/{params['epochs']}] "
                            f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, "
                            f"LR: {current_lr:.6f} ({phase})"
                        )
                        
                        # 🔥 Epoch结束后检查scale异常
                        if use_amp and scaler_monitor:
                            need_reset = scaler_monitor.check_epoch_abnormal(epoch)
                            if need_reset and GRAD_SCALER_AUTO_RESET:
                                scaler_monitor.reset_scale()
                                logger.warning(f"🔄 Trial {trial.number} Fold {fold_idx+1} Epoch {epoch+1}: 连续epoch异常，Scale已重置")
                            
                            # 记录epoch的scale统计
                            if epoch % 5 == 0 or epoch == params['epochs'] - 1:
                                stats = scaler_monitor.get_statistics()
                                logger.info(f"📊 Epoch {epoch+1} Scale统计: 当前={stats['current_scale']:.2f}, 溢出次数={stats['overflow_count']}")
                        
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
                    
                    # 🔑 修复：确保 y_pred 格式正确（1D 标签数组）
                    # 如果 y_pred 是 2D（概率矩阵），转换为类别标签
                    if y_pred.ndim > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    # 确保 y_pred 是 1D 数组
                    y_pred = y_pred.ravel()
                
                # 🔑 修复：确保 y_val 和 y_pred 格式一致（1D 标签数组）
                # 确保 y_val 是 1D 标签数组（不是 one-hot 编码）
                if y_val.ndim > 1:
                    y_val = np.argmax(y_val, axis=1)
                y_val = y_val.ravel()
                
                # 确保 y_pred 是 1D 数组（Informer2 可能已经是，但确保一致性）
                if y_pred.ndim > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                y_pred = y_pred.ravel()
                
                # 确保数据类型一致
                y_val = y_val.astype(int)
                y_pred = y_pred.astype(int)
                    
                acc = accuracy_score(y_val, y_pred)
                cv_scores.append(acc)
                
            except Exception as e:
                fold_fail_count += 1
                # ✅ 提升日志级别并添加详细错误信息
                logger.error(f"❌ Trial {trial.number} Fold {fold_idx+1} 失败: {e}")
                logger.error(f"   错误详情: {traceback.format_exc()}")
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

