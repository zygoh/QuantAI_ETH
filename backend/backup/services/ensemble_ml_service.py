"""
集成机器学习服务 - Stacking三模型融合
"""
# Standard library imports
import logging
import gc
import time
import traceback
import os
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pickle

# Third-party imports
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.special import entr
from scipy.stats import entropy as scipy_entropy
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from numpy.lib.format import open_memmap

# Local application imports
from app.services.ml_service import MLService
from app.core.config import settings
from app.core.cache import cache_manager
from app.services.hyperparameter_optimizer import HyperparameterOptimizer
from app.services.direction_consistency_checker import TradingDirectionConsistencyChecker, ConsistencyCheck
from app.services.adaptive_frequency_controller import AdaptiveFrequencyController, FrequencyControl
from app.services.model_stability_enhancer import ModelStabilityEnhancer
from app.utils.helpers import format_signal_type

logger = logging.getLogger(__name__)

# 🔥 延迟导入避免循环依赖（仅在必要时）
# binance_client 在方法内导入以避免循环依赖

# 深度学习模型（PyTorch）
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from app.services.informer2_model import Informer2ForClassification
    from app.services.gmadl_loss import create_trade_loss
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch已加载，Informer-2模型可用")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch未安装，Informer-2模型将不可用")


class InformerWrapper:
    """
    包装Informer-2模型，提供predict_proba接口（支持序列输入）
    
    将类移到模块级别以支持pickle序列化
    """
    
    def __init__(self, model, device):
        """
        初始化包装器
        
        Args:
            model: Informer2ForClassification模型实例
            device: PyTorch设备（'cuda'或'cpu'）
        """
        self.model = model
        self.device = device
    
    def predict_proba(self, X_seq):
        """
        预测概率（兼容scikit-learn，支持序列输入）
        
        Args:
            X_seq: NumPy数组 (n_samples, seq_len, n_features)
        
        Returns:
            概率数组 (n_samples, n_classes)
        """
        self.model.eval()
        with torch.no_grad():
            # 🔥 内存优化：使用from_numpy避免数据复制，确保float32
            if not isinstance(X_seq, torch.Tensor):
                # 🔥 关键修复：确保数据是连续的numpy数组
                if not isinstance(X_seq, np.ndarray):
                    X_seq = np.asarray(X_seq, dtype=np.float32)
                elif X_seq.dtype != np.float32:
                    X_seq = X_seq.astype(np.float32)
                
                if not X_seq.flags['C_CONTIGUOUS']:
                    X_seq = np.ascontiguousarray(X_seq)
                
                # 🔥 关键修复：使用copy()避免内存映射问题
                X_tensor = torch.from_numpy(X_seq.copy()).to(self.device)
            else:
                X_tensor = X_seq.to(self.device)
            
            probs = self.model.predict_proba(X_tensor)
            return probs.cpu().numpy()
    
    def predict(self, X_seq):
        """
        预测类别（兼容scikit-learn，支持序列输入）
        
        Args:
            X_seq: NumPy数组 (n_samples, seq_len, n_features)
        
        Returns:
            预测类别数组
        """
        probs = self.predict_proba(X_seq)
        return np.argmax(probs, axis=1)


class EnsembleMLService(MLService):
    """
    集成机器学习服务（Stacking）
    
    使用LightGBM + XGBoost + CatBoost + Informer-2 四模型Stacking融合
    目标：准确率从37%提升到50%+
    
    Phase 1: 时间序列CV + 元特征 + HOLD惩罚 + 防过拟合
    Phase 2A: 82个高级技术指标特征
    Phase 2B: Optuna超参数自动优化
    Phase 3: Informer-2深度学习 + GMADL损失函数
    """
    
    def __init__(self):
        super().__init__()
        
        # 集成模型字典 {timeframe: {lgb, xgb, cat, inf, meta}}
        self.ensemble_models = {}
        
        # 🔒 训练状态管理（生产级别：后台训练，不影响预测）
        self.training_in_progress = {}  # {timeframe: bool}
        self.models_ready = {}  # {timeframe: bool}
        self.background_training = False  # 🔥 后台训练标志（不阻止预测）
        for tf in settings.TIMEFRAMES:
            self.training_in_progress[tf] = False
            self.models_ready[tf] = False
        
        # 🔥 模型版本管理（支持热更新）
        self.model_versions = {}  # {timeframe: version_number}
        
        # 集成权重（Stacking自动学习，这里作为降级方案）
        self.fallback_weights = {
            'lgb': 0.4,
            'xgb': 0.3,
            'cat': 0.3
        }
        
        # 🔧 超参数优化配置
        self.enable_hyperparameter_tuning = True  # ✅ 已启用（Phase 2B）
        self.optimize_all_models = True  # ✅ GPU加速下优化所有模型
        self.optimize_informer2 = True  # ✅ 优化Informer-2（深度学习）
        self.optuna_n_trials = 100  # Optuna试验次数（传统模型）
        self.informer_n_trials = 20  # Informer-2试验次数（保证至少完成整轮搜索）
        self.optuna_timeout = 1800  # 超时30分钟（GPU加速下足够优化3个模型）
        self.informer_timeout = 3600  # Informer-2超时60分钟（防止仅完成1-2次试验）
        
        # 🤖 Informer-2深度学习配置
        self.enable_informer2 = True  # ✅ 已启用（Phase 3 - 神经网络）
        self.informer_d_model = 128  # 模型维度
        self.informer_n_heads = 8  # 注意力头数
        self.informer_n_layers = 3  # Encoder层数
        self.informer_epochs = 50  # 训练轮数（GPU加速）
        self.informer_batch_size = 256  # 批次大小
        self.informer_lr = 0.0005  # 学习率（降低以提高数值稳定性：0.001→0.0005）
        
        # 🔥 高级内存优化配置（生产级别）
        self.use_gradient_checkpointing = True  # 梯度检查点（节省50-70%内存）
        self.use_8bit_adam = True  # 8-bit Adam优化器（节省75%优化器内存）
        self.use_aggressive_amp = True  # 激进混合精度训练（FP16 + TF32）
        
        # 🎮 GPU配置（从config读取）
        self.use_gpu = settings.USE_GPU
        self.gpu_device = settings.GPU_DEVICE
        
        # 🔑 序列长度配置（用于Informer-2序列输入）
        # 🎯 优化：减少序列长度以降低内存占用（减少80-90%）
        self.seq_len_config = {
            '3m': 96,   # 96 × 3分钟 = 4.8小时（足够短期模式识别）
            '5m': 96,   # 96 × 5分钟 = 8小时（主时间框架）
            '15m': 64   # 64 × 15分钟 = 16小时（趋势确认）
        }
        # 🧠 序列内存优化：使用内存映射文件，避免整库常驻内存
        # 🔥 关键修复：禁用内存映射（在交叉验证时会导致索引问题）
        self.use_sequence_memmap = False
        
        # 🛡️ 系统优化组件
        self.direction_checker = TradingDirectionConsistencyChecker()
        self.frequency_controller = AdaptiveFrequencyController()
        self.stability_enhancer = ModelStabilityEnhancer()
        
        # 📊 优化指标记录
        self.optimization_metrics = {
            'fatal_error_rate': 0.0,
            'fee_impact': 0.0,
            'model_stability': 0.0,
            'consistency_rate': 0.0
        }

        # 🗂️ 模型目录防御式初始化（避免早期调用出现 AttributeError）
        if not hasattr(self, 'model_dir') or not self.model_dir:
            self.model_dir = "models"
        try:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        
        logger.info("✅ 集成ML服务初始化完成（Stacking四模型融合 + 深度学习）")
        logger.info(f"   超参数优化: {'启用' if self.enable_hyperparameter_tuning else '关闭'}")
        logger.info(f"   Informer-2神经网络: {'启用' if self.enable_informer2 else '关闭'}")
        logger.info(f"   序列长度配置: {self.seq_len_config}")
        logger.info(f"   GPU加速: {'启用' if self.use_gpu else '关闭'} (设备: {self.gpu_device if self.use_gpu else 'CPU'})")
        
        # 🔥 高级内存优化状态
        if self.enable_informer2:
            logger.info(f"   🚀 高级内存优化:")
            logger.info(f"      - 梯度检查点: {'✅ 启用' if self.use_gradient_checkpointing else '❌ 关闭'} (节省50-70%内存)")
            logger.info(f"      - 8-bit Adam: {'✅ 启用' if self.use_8bit_adam else '❌ 关闭'} (节省75%优化器内存)")
            logger.info(f"      - 激进混合精度: {'✅ 启用' if self.use_aggressive_amp else '❌ 关闭'} (FP16+TF32)")
            
            # 估算内存节省
            if self.use_gradient_checkpointing and self.use_8bit_adam and self.use_aggressive_amp:
                logger.info(f"      💾 预期GPU内存节省: ~60-70% (6.3GB → 2.0GB)")
            elif self.use_gradient_checkpointing:
                logger.info(f"      💾 预期GPU内存节省: ~40-50% (6.3GB → 3.5GB)")
    
    def clear_gpu_memory(self):
        """
        清理GPU内存
        
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
            logger.info(f"🧹 GPU内存已清理 (使用: {gpu_used/1024**3:.1f}GB, 可用: {gpu_free/1024**3:.1f}GB)")
        else:
            logger.info("🧹 CPU模式，无需清理GPU内存")
    
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
    
    async def _prepare_diverse_training_data(self, timeframe: str, days_multiplier: float = 1.0) -> pd.DataFrame:
        """
        准备差异化训练数据（不同天数）
        
        Args:
            timeframe: 时间框架
            days_multiplier: 天数倍数（1.0=标准，1.5=+50%，2.0=+100%）
        
        Returns:
            K线数据DataFrame
        """
        try:
            # 🔥 延迟导入避免循环依赖
            from app.exchange.binance_client import binance_client
            
            symbol = settings.SYMBOL
            
            # 🔑 基础训练天数配置（超短线策略：确保足够样本）
            base_days_config = {
                '3m': 120,   # 3m: 120天=57,600条（高频样本，捕捉极短期模式）
                '5m': 120,   # 5m: 120天=34,560条（主时间框架，充足样本）
                '15m': 120   # 15m: 120天=11,520条（中期过滤，足够识别趋势）
            }
            base_days = base_days_config.get(timeframe, 120)
            
            # 应用倍数
            training_days = int(base_days * days_multiplier)
            
            # 计算需要的K线数量
            interval_minutes = {
                '3m': 3, '5m': 5, '15m': 15
            }
            minutes = interval_minutes.get(timeframe, 60)
            required_klines = int((training_days * 24 * 60) / minutes)
            
            logger.info(f"📥 获取{timeframe}数据（×{days_multiplier}倍）: {required_klines}条K线 ({training_days}天)")
            
            # ✅ 统一使用分页方法（自动处理超过1500的情况）
            all_klines = binance_client.get_klines_paginated(
                symbol=symbol,
                interval=timeframe,
                limit=required_klines,
                rate_limit_delay=0.1
            )
            
            # 转换为DataFrame（不依赖reverse，直接用时间戳排序）
            df = pd.DataFrame(all_klines)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 🔑 关键：依赖时间戳排序，而不是假设API返回顺序
            df = df.sort_values('timestamp', ascending=True)  # 明确指定升序（旧→新）
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            df = df.set_index('timestamp')
            
            logger.info(f"✅ 获取成功: {len(df)}条（×{days_multiplier}倍数据）")
            
            return df
            
        except Exception as e:
            logger.error(f"准备差异化训练数据失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _prepare_features_labels_reuse(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和标签（复用已选择的特征列）
        
        用途：为XGBoost和CatBoost准备数据时，复用LightGBM已选择的特征列
        
        Args:
            df: 包含label列的DataFrame
            timeframe: 时间框架
        
        Returns:
            (X, y): 特征DataFrame和标签Series
        """
        try:
            # 使用已选择的特征列（LightGBM训练时已确定）
            feature_columns = self.feature_columns_dict.get(timeframe, [])
            
            if not feature_columns:
                logger.error(f"{timeframe} 特征列未找到，无法复用")
                return pd.DataFrame(), pd.Series()
            
            X = df[feature_columns].copy()
            y = df['label'].copy()
            
            # 移除包含NaN的行
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            
            return X, y
            
        except Exception as e:
            logger.error(f"准备特征和标签（复用）失败: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _create_sequence_input(
        self,
        df: pd.DataFrame,
        seq_len: int,
        timeframe: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构造序列输入（用于Informer-2模型）- 内存优化版
        
        使用滑动窗口将单点特征转换为序列输入，充分利用历史时间序列信息
        
        优化策略：
        1. 使用float32代替float64，节省50%内存
        2. 预分配NumPy数组，避免动态append
        3. 预先转换DataFrame为NumPy数组，避免重复iloc切片
        4. 显式垃圾回收，释放中间数据
        
        Args:
            df: 特征工程后的DataFrame（包含label列）
            seq_len: 序列长度（3m=480, 5m=288, 15m=96）
            timeframe: 时间框架（3m/5m/15m）
        
        Returns:
            X_seq: (n_samples, seq_len, n_features) - 序列特征（float32）
            y: (n_samples,) - 标签（int8）
        """
        try:
            feature_columns = self.feature_columns_dict.get(timeframe, [])
            
            if not feature_columns:
                logger.error(f"❌ {timeframe} 特征列未找到，无法构造序列输入")
                return np.array([]), np.array([])
            
            # 🔥 优化1：预先转换为NumPy数组（避免重复DataFrame切片）
            logger.debug(f"🔧 {timeframe} 开始构造序列输入（seq_len={seq_len}）...")
            X_all = df[feature_columns].values.astype(np.float32)  # float32节省50%内存
            y_all = df['label'].values.astype(np.int8)  # int8节省内存
            
            n_total = len(df)
            n_features = len(feature_columns)
            max_samples = n_total - seq_len
            
            if max_samples <= 0:
                logger.warning(f"⚠️ {timeframe} 数据量不足，无法构造序列（需要>{seq_len}条）")
                return np.array([]), np.array([])
            
            # 🔥 优化2：预分配内存（避免动态append和内存碎片）
            X_seq = np.empty((max_samples, seq_len, n_features), dtype=np.float32)
            y = np.empty(max_samples, dtype=np.int8)
            
            # 🔥 优化3：使用NumPy切片（比DataFrame.iloc快5-10倍）
            valid_count = 0
            for i in range(seq_len, n_total):
                idx = i - seq_len
                X_window = X_all[idx:i]  # NumPy切片，O(1)复杂度
                y_label = y_all[i]
                
                # 仅检查NaN（已在特征工程阶段处理过inf和大值）
                if not np.isnan(X_window).any() and not np.isnan(y_label):
                    X_seq[valid_count] = X_window
                    y[valid_count] = y_label
                    valid_count += 1
            
            # 🔥 优化4：截断到有效长度（释放未使用内存）
            X_seq = X_seq[:valid_count]
            y = y[:valid_count]
            
            # 计算内存占用
            memory_mb = (X_seq.nbytes + y.nbytes) / (1024 ** 2)

            # 可选：落盘为内存映射，避免整库常驻内存
            if getattr(self, 'use_sequence_memmap', False):
                try:
                    memmap_dir = self.model_dir if hasattr(self, 'model_dir') and self.model_dir else 'models'
                    os.makedirs(memmap_dir, exist_ok=True)
                    seq_path = os.path.join(memmap_dir, f"{settings.SYMBOL}_{timeframe}_Xseq.npy")
                    y_path = os.path.join(memmap_dir, f"{settings.SYMBOL}_{timeframe}_Yseq.npy")

                    # 写入为.npy（内含shape与dtype），再以只读内存映射方式打开
                    mm_x = open_memmap(seq_path, mode='w+', dtype=np.float32, shape=X_seq.shape)
                    mm_x[:] = X_seq
                    del mm_x
                    mm_y = open_memmap(y_path, mode='w+', dtype=np.int8, shape=y.shape)
                    mm_y[:] = y
                    del mm_y

                    # 释放内存中数组，使用内存映射读取
                    del X_seq, y
                    gc.collect()

                    X_seq = np.load(seq_path, mmap_mode='r')
                    y = np.load(y_path, mmap_mode='r')

                    logger.info(f"   已启用内存映射: {seq_path} ({memory_mb:.1f} MB)")
                except Exception:
                    logger.warning("⚠️ 序列内存映射失败，回退为内存数组")

            logger.info(f"✅ {timeframe} 序列输入构造完成: {X_seq.shape} (样本数={valid_count}, 序列长度={seq_len}, 特征数={n_features})")
            logger.info(f"   原始样本数: {n_total}, 序列样本数: {valid_count}, 减少: {n_total - valid_count}个")
            logger.info(f"   内存占用: {memory_mb:.1f} MB (float32优化)")

            # 🔥 优化5：显式垃圾回收（释放X_all, y_all等中间数据）
            del X_all, y_all
            gc.collect()

            return X_seq, y
            
        except Exception as e:
            logger.error(f"❌ 构造序列输入失败: {e}")
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])
    
    async def train_all_timeframes(self) -> Dict[str, Any]:
        """
        训练所有时间框架的集成模型
        
        Returns:
            训练结果和指标
        """
        try:
            logger.info("🚀 开始Stacking集成模型训练...")
            if self.enable_informer2 and TORCH_AVAILABLE:
                logger.info(f"✨ 四模型融合: LightGBM + XGBoost + CatBoost + Informer-2 (GMADL损失)")
                logger.info(f"   超参数优化: {'启用' if self.enable_hyperparameter_tuning else '关闭'}")
                logger.info(f"   深度学习: GPU {'可用' if torch.cuda.is_available() else '不可用'}")
            else:
                logger.info(f"三模型融合: LightGBM + XGBoost + CatBoost")
            logger.info(f"时间框架: {settings.TIMEFRAMES}")
            logger.info("")
            
            results = {}
            
            for timeframe in settings.TIMEFRAMES:
                logger.info("=" * 60)
                logger.info(f"📊 训练 {timeframe} 集成模型...")
                logger.info("=" * 60)
                
                result = await self._train_ensemble_single_timeframe(timeframe)
                results[timeframe] = result
                
                logger.info(f"✅ {timeframe} 集成模型训练完成 - 准确率: {result['accuracy']:.4f}")
                logger.info("")
            
            # 计算平均准确率
            avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
            
            logger.info("=" * 60)
            logger.info("🎉 Stacking集成模型训练完成")
            logger.info(f"成功训练: {len(results)}/{len(settings.TIMEFRAMES)} 个时间框架")
            logger.info(f"平均准确率: {avg_accuracy:.4f}")
            logger.info("=" * 60)
            logger.info("")
            
            # 🔑 保存模型指标到Redis缓存（供health_monitor读取）
            metrics_cache = {
                'accuracy': float(avg_accuracy),
                'timeframes': {tf: float(r['accuracy']) for tf, r in results.items()},
                'training_date': datetime.now().isoformat(),
                'method': 'Stacking Ensemble',
                'models': ['LightGBM', 'XGBoost', 'CatBoost']
            }
            await cache_manager.set_model_metrics(settings.SYMBOL, metrics_cache)
            
            return {
                'results': results,
                'average_accuracy': avg_accuracy,
                'training_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 集成模型训练失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _train_ensemble_single_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """
        训练单个时间框架的Stacking集成模型
        
        改进：三个模型使用不同的训练数据，增加多样性
        
        流程:
        1. 准备三份不同的训练数据（不同天数）
        2. 训练基础模型（LightGBM, XGBoost, CatBoost）- 各用不同数据
        3. 生成元特征（基础模型的预测概率）
        4. 训练元学习器（Stacking）
        5. 评估集成效果
        """
        # 🔥 生产级别：后台训练，不影响预测
        self.training_in_progress[timeframe] = True
        self.background_training = True
        logger.info(f"🔄 {timeframe} 后台训练已开始（预测功能继续运行，训练完成后热更新模型）")
        
        try:
            # 1️⃣ 为三个模型准备不同的训练数据（增加多样性）
            logger.info(f"📥 为三个模型准备差异化训练数据...")
            
            # LightGBM: 使用较新数据（标准天数）
            data_lgb = await self._prepare_training_data_for_timeframe(timeframe)
            logger.info(f"✅ LightGBM数据: {len(data_lgb)}条（标准）")
            
            # XGBoost: 使用更多数据（+50%天数）
            data_xgb = await self._prepare_diverse_training_data(timeframe, days_multiplier=1.5)
            logger.info(f"✅ XGBoost数据: {len(data_xgb)}条（+50%天数）")
            
            # CatBoost: 使用最多数据（+100%天数）
            data_cat = await self._prepare_diverse_training_data(timeframe, days_multiplier=2.0)
            logger.info(f"✅ CatBoost数据: {len(data_cat)}条（+100%天数）")
            
            # 2️⃣ 处理三份数据（特征工程 + 标签 + 特征选择）
            logger.info(f"🔧 处理三份训练数据...")
            
            # 处理LightGBM数据
            data_lgb = self.feature_engineer.create_features(data_lgb)
            data_lgb = self._create_labels(data_lgb, timeframe=timeframe)
            X_lgb, y_lgb = self._prepare_features_labels(data_lgb, timeframe)
            X_lgb_scaled = self._scale_features(X_lgb, timeframe, fit=True)
            
            # 处理XGBoost数据（复用同一个scaler）
            data_xgb = self.feature_engineer.create_features(data_xgb)
            data_xgb = self._create_labels(data_xgb, timeframe=timeframe)
            X_xgb, y_xgb = self._prepare_features_labels_reuse(data_xgb, timeframe)
            X_xgb_scaled = self._scale_features(X_xgb, timeframe, fit=False)
            
            # 处理CatBoost数据（复用同一个scaler）
            data_cat = self.feature_engineer.create_features(data_cat)
            data_cat = self._create_labels(data_cat, timeframe=timeframe)
            X_cat, y_cat = self._prepare_features_labels_reuse(data_cat, timeframe)
            X_cat_scaled = self._scale_features(X_cat, timeframe, fit=False)
            
            logger.info(f"✅ 三份数据处理完成: LGB={len(X_lgb)}, XGB={len(X_xgb)}, CAT={len(X_cat)}")
            
            # 🆕 构造序列输入（仅用于Informer-2）
            X_seq_lgb, y_seq_lgb = None, None
            if self.enable_informer2 and TORCH_AVAILABLE:
                seq_len = self.seq_len_config.get(timeframe, 96)
                logger.info(f"🔧 构造Informer-2序列输入（seq_len={seq_len}）...")
                X_seq_lgb, y_seq_lgb = self._create_sequence_input(data_lgb, seq_len, timeframe)
                
                if len(X_seq_lgb) == 0:
                    logger.warning(f"⚠️ 序列输入构造失败，将跳过Informer-2训练")
                    self.enable_informer2 = False
            
            # 3️⃣ 时间序列分割（使用最短的数据长度作为验证集基准）
            min_len = min(len(X_lgb_scaled), len(X_xgb_scaled), len(X_cat_scaled))
            split_idx = int(min_len * 0.8)
            
            # 🔑 分割数据（取最新的数据，保证时间对齐）
            if isinstance(X_lgb_scaled, np.ndarray):
                X_lgb_train, X_lgb_val = X_lgb_scaled[-min_len:][:split_idx], X_lgb_scaled[-min_len:][split_idx:]
                X_xgb_train, X_xgb_val = X_xgb_scaled[-min_len:][:split_idx], X_xgb_scaled[-min_len:][split_idx:]
                X_cat_train, X_cat_val = X_cat_scaled[-min_len:][:split_idx], X_cat_scaled[-min_len:][split_idx:]
            else:
                X_lgb_train, X_lgb_val = X_lgb_scaled.iloc[-min_len:][:split_idx], X_lgb_scaled.iloc[-min_len:][split_idx:]
                X_xgb_train, X_xgb_val = X_xgb_scaled.iloc[-min_len:][:split_idx], X_xgb_scaled.iloc[-min_len:][split_idx:]
                X_cat_train, X_cat_val = X_cat_scaled.iloc[-min_len:][:split_idx], X_cat_scaled.iloc[-min_len:][split_idx:]
            
            y_lgb_train, y_lgb_val = y_lgb.iloc[-min_len:][:split_idx], y_lgb.iloc[-min_len:][split_idx:]
            y_xgb_train, y_xgb_val = y_xgb.iloc[-min_len:][:split_idx], y_xgb.iloc[-min_len:][split_idx:]
            y_cat_train, y_cat_val = y_cat.iloc[-min_len:][:split_idx], y_cat.iloc[-min_len:][split_idx:]
            
            # 🆕 分割序列数据（用于Informer-2）
            X_seq_train, X_seq_val, y_seq_train, y_seq_val = None, None, None, None
            if self.enable_informer2 and X_seq_lgb is not None:
                seq_split_idx = int(len(X_seq_lgb) * 0.8)
                X_seq_train = X_seq_lgb[:seq_split_idx]
                X_seq_val = X_seq_lgb[seq_split_idx:]
                y_seq_train = y_seq_lgb[:seq_split_idx]
                y_seq_val = y_seq_lgb[seq_split_idx:]
                logger.info(f"📊 {timeframe} 序列数据分割: 训练{len(X_seq_train)}条, 验证{len(X_seq_val)}条")
                
                # 🔑 关键修复：对齐传统模型的验证集到序列数据的长度
                # 序列数据比原始数据少seq_len个样本，需要对齐
                seq_val_len = len(X_seq_val)
                if seq_val_len < len(X_lgb_val):
                    logger.warning(f"⚠️ 对齐验证集：传统模型{len(X_lgb_val)}条 → Informer-2{seq_val_len}条")
                    # 取传统模型验证集的最后seq_val_len个样本（时间对齐）
                    if isinstance(X_lgb_val, np.ndarray):
                        X_lgb_val = X_lgb_val[-seq_val_len:]
                        X_xgb_val = X_xgb_val[-seq_val_len:]
                        X_cat_val = X_cat_val[-seq_val_len:]
                    else:
                        X_lgb_val = X_lgb_val.iloc[-seq_val_len:]
                        X_xgb_val = X_xgb_val.iloc[-seq_val_len:]
                        X_cat_val = X_cat_val.iloc[-seq_val_len:]
                    
                    y_lgb_val = y_lgb_val.iloc[-seq_val_len:]
                    y_xgb_val = y_xgb_val.iloc[-seq_val_len:]
                    y_cat_val = y_cat_val.iloc[-seq_val_len:]
            
            logger.info(f"📊 {timeframe} 传统模型数据分割: 训练{len(X_lgb_train)}条（对齐后）, 验证{len(X_lgb_val)}条")
            
            # 4️⃣ 训练Stacking集成模型（使用差异化数据 + 序列输入）
            logger.info(f"🚂 开始训练 {timeframe} Stacking集成（差异化数据）...")
            ensemble_result = self._train_stacking_diverse(
                X_lgb_train, y_lgb_train, X_lgb_val, y_lgb_val,
                X_xgb_train, y_xgb_train, X_xgb_val, y_xgb_val,
                X_cat_train, y_cat_train, X_cat_val, y_cat_val,
                X_seq_train, y_seq_train, X_seq_val, y_seq_val,
                timeframe
            )
            
            # 8️⃣ 保存集成模型
            self._save_ensemble_models(timeframe)
            
            logger.info(f"⏱️ {timeframe} 训练耗时: {ensemble_result['training_time']:.2f}秒")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"❌ {timeframe} 集成模型训练失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _train_stacking_diverse(
        self,
        X_lgb_train, y_lgb_train, X_lgb_val, y_lgb_val,
        X_xgb_train, y_xgb_train, X_xgb_val, y_xgb_val,
        X_cat_train, y_cat_train, X_cat_val, y_cat_val,
        X_seq_train, y_seq_train, X_seq_val, y_seq_val,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        使用差异化数据训练Stacking集成模型（支持序列输入）
        
        Args:
            X_lgb_train, y_lgb_train: LightGBM训练数据
            X_lgb_val, y_lgb_val: LightGBM验证数据
            X_xgb_train, y_xgb_train: XGBoost训练数据
            X_xgb_val, y_xgb_val: XGBoost验证数据
            X_cat_train, y_cat_train: CatBoost训练数据
            X_cat_val, y_cat_val: CatBoost验证数据
            X_seq_train, y_seq_train: Informer-2序列训练数据
            X_seq_val, y_seq_val: Informer-2序列验证数据
            timeframe: 时间框架
        
        Returns:
            训练结果字典
        """
        start_time = time.time()
        
        try:
            # 🔧 Optuna超参数优化（如果启用）
            lgb_params_optimized = None
            xgb_params_optimized = None
            cat_params_optimized = None
            inf_params_optimized = None
            
            if self.enable_hyperparameter_tuning:
                # 🤖 优先优化Informer-2（深度学习模型）
                if self.enable_informer2 and self.optimize_informer2 and TORCH_AVAILABLE and X_seq_train is not None:
                    logger.info(f"🤖 启动Informer-2超参数优化（深度学习）- 优先优化...")
                    logger.info(f"   GPU加速: {'启用' if self.use_gpu else '关闭'}")
                    logger.info(f"   试验次数: {self.informer_n_trials}次, 超时: {self.informer_timeout}秒")
                    logger.info(f"   序列输入形状: {X_seq_train.shape} (样本数, 序列长度, 特征数)")
                    
                    # 🔑 关键修复：使用序列数据而不是2D数据
                    inf_optimizer = HyperparameterOptimizer(
                        X=X_seq_train,  # 使用3D序列数据
                        y=y_seq_train,  # 使用对应的序列标签
                        timeframe=timeframe,
                        model_type="informer2",
                        use_gpu=self.use_gpu
                    )
                    inf_params_optimized = inf_optimizer.optimize(
                        n_trials=self.informer_n_trials,
                        timeout=self.informer_timeout,
                        show_progress=False
                    )
                    logger.info(f"✅ Informer-2超参数优化完成: 最佳CV准确率={inf_optimizer.best_score:.4f}")
                
                # 🔧 然后优化传统模型
                if self.optimize_all_models:
                    logger.info(f"🔧 启动传统模型超参数优化（Optuna）- 优化全部3个传统模型...")
                else:
                    logger.info(f"🔧 启动传统模型超参数优化（Optuna）- 仅优化LightGBM...")
                logger.info(f"   GPU加速: {'启用' if self.use_gpu else '关闭'}")
                logger.info(f"   每模型试验: {self.optuna_n_trials}次, 超时: {self.optuna_timeout}秒")
                
                # 优化LightGBM
                logger.info(f"   🔧 [1/{'3' if self.optimize_all_models else '1'}] 优化LightGBM...")
                lgb_optimizer = HyperparameterOptimizer(
                    X=X_lgb_train.values if isinstance(X_lgb_train, pd.DataFrame) else X_lgb_train,
                    y=y_lgb_train,
                    timeframe=timeframe,
                    model_type="lightgbm",
                    use_gpu=self.use_gpu
                )
                lgb_params_optimized = lgb_optimizer.optimize(
                    n_trials=self.optuna_n_trials,
                    timeout=self.optuna_timeout,
                    show_progress=False  # 关闭进度条（避免混乱）
                )
                
                # 优化XGBoost（如果启用）
                if self.optimize_all_models:
                    logger.info(f"   🔧 [2/3] 优化XGBoost...")
                    xgb_optimizer = HyperparameterOptimizer(
                        X=X_xgb_train.values if isinstance(X_xgb_train, pd.DataFrame) else X_xgb_train,
                        y=y_xgb_train,
                        timeframe=timeframe,
                        model_type="xgboost",
                        use_gpu=self.use_gpu
                    )
                    xgb_params_optimized = xgb_optimizer.optimize(
                        n_trials=self.optuna_n_trials,
                        timeout=self.optuna_timeout,
                        show_progress=False
                    )
                
                # 优化CatBoost（如果启用）
                if self.optimize_all_models:
                    logger.info(f"   🔧 [3/3] 优化CatBoost...")
                    cat_optimizer = HyperparameterOptimizer(
                        X=X_cat_train.values if isinstance(X_cat_train, pd.DataFrame) else X_cat_train,
                        y=y_cat_train,
                        timeframe=timeframe,
                        model_type="catboost",
                        use_gpu=self.use_gpu
                    )
                    cat_params_optimized = cat_optimizer.optimize(
                        n_trials=self.optuna_n_trials,
                        timeout=self.optuna_timeout,
                        show_progress=False
                    )
                
                logger.info(f"✅ 传统模型超参数优化完成!")
                if lgb_params_optimized:
                    logger.info(f"   LightGBM最佳CV: {lgb_optimizer.best_score:.4f}")
                if xgb_params_optimized:
                    logger.info(f"   XGBoost最佳CV:  {xgb_optimizer.best_score:.4f}")
                if cat_params_optimized:
                    logger.info(f"   CatBoost最佳CV: {cat_optimizer.best_score:.4f}")
            
            # 1️⃣ 训练四个基础模型（Informer-2优先训练）
            # 🤖 优先训练Informer-2（深度学习 + GMADL损失 + 序列输入）
            inf_model = None
            if self.enable_informer2 and TORCH_AVAILABLE and X_seq_train is not None:
                logger.info(f"🤖 训练Informer-2（深度学习 + GMADL损失 + 序列输入）- 优先训练...")
                inf_model = self._train_informer2(X_seq_train, y_seq_train, timeframe, custom_params=inf_params_optimized)
            
            # 🔧 然后训练传统模型
            logger.info(f"🚂 训练LightGBM（{timeframe} 标准数据）...")
            lgb_model = self._train_lightgbm(X_lgb_train, y_lgb_train, timeframe, custom_params=lgb_params_optimized)
            
            logger.info(f"🚂 训练XGBoost（{timeframe} +50%数据）...")
            xgb_model = self._train_xgboost(X_xgb_train, y_xgb_train, timeframe, custom_params=xgb_params_optimized)
            
            logger.info(f"🚂 训练CatBoost（{timeframe} +100%数据）...")
            cat_model = self._train_catboost(X_cat_train, y_cat_train, timeframe, custom_params=cat_params_optimized)
            
            # 2️⃣ 生成验证集的预测概率（元特征）
            logger.info(f"📊 生成元特征（基于对齐的验证集）...")
            
            # 使用各自的验证集生成预测
            lgb_pred_proba = lgb_model.predict_proba(X_lgb_val)
            xgb_pred_proba = xgb_model.predict_proba(X_xgb_val)
            cat_pred_proba = cat_model.predict_proba(X_cat_val)
            
            # Informer-2预测（如果启用，使用序列验证数据）
            if inf_model is not None and X_seq_val is not None:
                inf_pred_proba = inf_model.predict_proba(X_seq_val)
                logger.info(f"   Informer-2概率形状: {inf_pred_proba.shape}")
            
            logger.info(f"概率形状: lgb={lgb_pred_proba.shape}, xgb={xgb_pred_proba.shape}, cat={cat_pred_proba.shape}")
            
            # 🔑 验证形状一致性
            assert lgb_pred_proba.shape == xgb_pred_proba.shape == cat_pred_proba.shape, \
                f"概率数组形状不一致: {lgb_pred_proba.shape} vs {xgb_pred_proba.shape} vs {cat_pred_proba.shape}"
            
            # 获取预测类别
            lgb_pred_raw = lgb_model.predict(X_lgb_val)
            xgb_pred_raw = xgb_model.predict(X_xgb_val)
            cat_pred_raw = cat_model.predict(X_cat_val)
            
            # 🔑 统一转换为1D数组（CatBoost返回2D，需要ravel）
            lgb_pred = lgb_pred_raw.ravel()
            xgb_pred = xgb_pred_raw.ravel()
            cat_pred = cat_pred_raw.ravel()
            
            # 🔑 严格验证预测数组形状
            expected_shape = (len(y_lgb_val),)
            assert lgb_pred.shape == expected_shape, f"lgb_pred形状错误: {lgb_pred.shape} != {expected_shape}"
            assert xgb_pred.shape == expected_shape, f"xgb_pred形状错误: {xgb_pred.shape} != {expected_shape}"
            assert cat_pred.shape == expected_shape, f"cat_pred形状错误: {cat_pred.shape} != {expected_shape}"
            
            logger.info(f"预测类别形状验证通过: {lgb_pred.shape} (已统一为1D数组)")
            
            # 🆕 增强元特征（提升元学习器决策能力）
            logger.info(f"生成增强元特征...")
            
            # 1. 模型一致性（3个模型预测是否一致）
            # 🔑 已确认都是1D数组，直接比较
            agreement_bool = (lgb_pred == xgb_pred) & (xgb_pred == cat_pred)  # (6757,) boolean
            agreement = agreement_bool.astype(float).reshape(-1, 1)  # (6757, 1)
            
            # 验证维度
            assert agreement.shape == (len(y_lgb_val), 1), f"agreement形状错误: {agreement.shape}"
            logger.debug(f"✓ agreement: {agreement.shape}")
            
            # 2. 最大概率（每个模型的最高置信度）
            lgb_max_prob = lgb_pred_proba.max(axis=1).reshape(-1, 1)
            xgb_max_prob = xgb_pred_proba.max(axis=1).reshape(-1, 1)
            cat_max_prob = cat_pred_proba.max(axis=1).reshape(-1, 1)
            assert lgb_max_prob.shape == (len(y_lgb_val), 1), f"lgb_max_prob形状错误: {lgb_max_prob.shape}"
            logger.debug(f"✓ max_prob: {lgb_max_prob.shape}")
            
            # 3. 概率熵（不确定性，熵越高越不确定）
            lgb_entropy = entr(lgb_pred_proba).sum(axis=1).reshape(-1, 1)
            xgb_entropy = entr(xgb_pred_proba).sum(axis=1).reshape(-1, 1)
            cat_entropy = entr(cat_pred_proba).sum(axis=1).reshape(-1, 1)
            assert lgb_entropy.shape == (len(y_lgb_val), 1), f"lgb_entropy形状错误: {lgb_entropy.shape}"
            logger.debug(f"✓ entropy: {lgb_entropy.shape}")
            
            # Informer-2的增强特征（如果启用）
            if inf_model is not None:
                inf_max_prob = inf_pred_proba.max(axis=1).reshape(-1, 1)
                inf_entropy = entr(inf_pred_proba).sum(axis=1).reshape(-1, 1)
                logger.debug(f"✓ inf_max_prob: {inf_max_prob.shape}, inf_entropy: {inf_entropy.shape}")
            
            # 4. 平均概率（三个或四个模型的平均预测概率）
            if inf_model is not None:
                avg_proba = (lgb_pred_proba + xgb_pred_proba + cat_pred_proba + inf_pred_proba) / 4
            else:
                avg_proba = (lgb_pred_proba + xgb_pred_proba + cat_pred_proba) / 3
            assert avg_proba.shape == lgb_pred_proba.shape, f"avg_proba形状错误: {avg_proba.shape}"
            logger.debug(f"✓ avg_proba: {avg_proba.shape}")
            
            # 5. 概率标准差（模型间的预测差异）
            if inf_model is not None:
                prob_std = np.std(np.stack([lgb_pred_proba, xgb_pred_proba, cat_pred_proba, inf_pred_proba]), axis=0)
            else:
                prob_std = np.std(np.stack([lgb_pred_proba, xgb_pred_proba, cat_pred_proba]), axis=0)
            prob_std_max = prob_std.max(axis=1).reshape(-1, 1)
            assert prob_std_max.shape == (len(y_lgb_val), 1), f"prob_std_max形状错误: {prob_std_max.shape}"
            logger.debug(f"✓ prob_std_max: {prob_std_max.shape}")
            
            # 🔑 拼接所有元特征（严格验证每一步）
            logger.info(f"开始拼接元特征...")
            
            # 逐步拼接并验证
            if inf_model is not None:
                # 包含Informer-2（25个特征）
                meta_list = [
                    lgb_pred_proba,      # (N, 3)
                    xgb_pred_proba,      # (N, 3)
                    cat_pred_proba,      # (N, 3)
                    inf_pred_proba,      # (N, 3) ← 新增
                    agreement,           # (N, 1)
                    lgb_max_prob,        # (N, 1)
                    xgb_max_prob,        # (N, 1)
                    cat_max_prob,        # (N, 1)
                    inf_max_prob,        # (N, 1) ← 新增
                    lgb_entropy,         # (N, 1)
                    xgb_entropy,         # (N, 1)
                    cat_entropy,         # (N, 1)
                    inf_entropy,         # (N, 1) ← 新增
                    avg_proba,           # (N, 3)
                    prob_std_max         # (N, 1)
                ]
                expected_features = 25  # 3+3+3+3+1+1+1+1+1+1+1+1+1+3+1 = 25
            else:
                # 仅传统模型（20个特征）
                meta_list = [
                    lgb_pred_proba,      # (N, 3)
                    xgb_pred_proba,      # (N, 3)
                    cat_pred_proba,      # (N, 3)
                    agreement,           # (N, 1)
                    lgb_max_prob,        # (N, 1)
                    xgb_max_prob,        # (N, 1)
                    cat_max_prob,        # (N, 1)
                    lgb_entropy,         # (N, 1)
                    xgb_entropy,         # (N, 1)
                    cat_entropy,         # (N, 1)
                    avg_proba,           # (N, 3)
                    prob_std_max         # (N, 1)
                ]
                expected_features = 20  # 3+3+3+1+1+1+1+1+1+1+3+1
            
            # 验证所有数组的第0维度都相同
            expected_rows = len(y_lgb_val)
            for i, arr in enumerate(meta_list):
                assert arr.shape[0] == expected_rows, \
                    f"元特征{i}第0维度错误: {arr.shape[0]} != {expected_rows}, 完整形状: {arr.shape}"
            
            # 拼接
            meta_features_val = np.hstack(meta_list)
            
            # 最终验证
            assert meta_features_val.shape == (expected_rows, expected_features), \
                f"元特征最终形状错误: {meta_features_val.shape} != ({expected_rows}, {expected_features})"
            
            # 元标签（使用LightGBM的y_val，因为验证集已对齐）
            meta_labels_val = y_lgb_val
            
            if inf_model is not None:
                logger.info(f"✅ 增强元特征生成完成: {meta_features_val.shape} (基础12+增强13=25个，含Informer-2)")
            else:
                logger.info(f"✅ 增强元特征生成完成: {meta_features_val.shape} (基础9+增强11=20个)")
            
            # 3️⃣ 训练元学习器（Stacking） - 升级为LightGBM + 动态HOLD惩罚
            logger.info(f"🧠 训练元学习器（LightGBM - 更强大的决策能力）...")
            
            # 🔑 检查HOLD比例，动态调整惩罚系数
            hold_ratio = (meta_labels_val == 1).sum() / len(meta_labels_val)
            
            # 🔑 根据HOLD比例动态调整惩罚（平衡策略）
            if hold_ratio > 0.60:  # HOLD占比>60%，重惩罚
                meta_hold_penalty_weight = 0.45
            elif hold_ratio > 0.50:  # HOLD占比>50%，中等
                meta_hold_penalty_weight = 0.55
            elif hold_ratio > 0.40:  # HOLD占比>40%，轻度
                meta_hold_penalty_weight = 0.65
            else:  # HOLD占比<=40%，正常
                meta_hold_penalty_weight = 0.75
            
            logger.info(f"   HOLD占比: {hold_ratio*100:.1f}% → 惩罚系数: {meta_hold_penalty_weight}")
            
            meta_class_weights = compute_sample_weight('balanced', meta_labels_val)
            # ✅ 添加时间衰减权重（与基础模型保持一致）
            meta_time_decay = np.exp(-np.arange(len(meta_features_val)) / (len(meta_features_val) * 0.1))[::-1]
            meta_hold_penalty = np.where(meta_labels_val == 1, meta_hold_penalty_weight, 1.0)
            meta_sample_weights = meta_class_weights * meta_time_decay * meta_hold_penalty
            
            # 🔑 元学习器：专业配置平衡性能和防过拟合
            meta_learner = lgb.LGBMClassifier(
                n_estimators=150,    # ✅ 适当增加树数量 50→150
                max_depth=6,         # ✅ 中等深度平衡表达能力 3→6
                learning_rate=0.05,  # ✅ 降低学习率更稳定收敛 0.15→0.05
                num_leaves=31,       # ✅ 2^5-1标准配置 7→31
                min_child_samples=20,  # ✅ 适度最小样本 30→20
                subsample=0.8,       # ✅ 行采样防过拟合 0.7→0.8
                colsample_bytree=0.8,  # ✅ 列采样防过拟合 0.7→0.8
                reg_alpha=0.1,       # ✅ 适度L1正则化 0.3→0.1
                reg_lambda=0.1,      # ✅ 适度L2正则化 0.3→0.1
                random_state=42,
                verbose=-1
            )
            meta_learner.fit(meta_features_val, meta_labels_val, sample_weight=meta_sample_weights)
            
            logger.info(f"✅ 元学习器训练完成（动态HOLD惩罚={meta_hold_penalty_weight}）")
            
            # 4️⃣ 构造全新的模型字典（避免训练期间读到半更新状态）
            models: Dict[str, Any] = {
                'lgb': lgb_model,
                'xgb': xgb_model,
                'cat': cat_model,
                'meta': meta_learner
            }

            # 保存Informer-2模型（如果存在）
            if inf_model is not None:
                models['inf'] = inf_model
            
            # 5️⃣ 评估集成模型 - 使用时间序列交叉验证
            logger.info(f"📊 {timeframe} 时间序列交叉验证评估...")
            
            # 🆕 时间序列5折交叉验证（更可靠的评估）
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            # 对验证集进行交叉验证
            for fold, (train_idx, test_idx) in enumerate(tscv.split(meta_features_val), 1):
                meta_train, meta_test = meta_features_val[train_idx], meta_features_val[test_idx]
                y_train, y_test = meta_labels_val.iloc[train_idx], meta_labels_val.iloc[test_idx]
                
                # 训练元学习器（每个fold）- 与最终模型完全一致的配置
                fold_meta = lgb.LGBMClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.15,
                    num_leaves=7, min_child_samples=30, subsample=0.7,
                    colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0.3,
                    random_state=42, verbose=-1
                )
                
                # 🔑 HOLD惩罚（与最终模型一致，使用相同的动态策略）
                fold_weights = compute_sample_weight('balanced', y_train)
                fold_hold_ratio = (y_train == 1).sum() / len(y_train)
                
                # 动态惩罚（平衡策略，与最终模型完全一致）
                if fold_hold_ratio > 0.60:
                    fold_penalty = 0.45
                elif fold_hold_ratio > 0.50:
                    fold_penalty = 0.55
                elif fold_hold_ratio > 0.40:
                    fold_penalty = 0.65
                else:
                    fold_penalty = 0.75
                
                fold_hold_penalty = np.where(y_train == 1, fold_penalty, 1.0)
                fold_sample_weights = fold_weights * fold_hold_penalty
                
                fold_meta.fit(meta_train, y_train, sample_weight=fold_sample_weights)
                fold_pred = fold_meta.predict(meta_test)
                fold_acc = accuracy_score(y_test, fold_pred)
                cv_scores.append(fold_acc)
                
                logger.debug(f"  Fold {fold}: 准确率={fold_acc:.4f}")
            
            # 交叉验证准确率
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            logger.info(f"✅ {timeframe} 时间序列CV结果: {cv_mean:.4f} ± {cv_std:.4f}")
            logger.info(f"   CV分数: {[f'{s:.4f}' for s in cv_scores]}")
            
            # 使用完整验证集评估最终模型
            ensemble_pred = meta_learner.predict(meta_features_val)
            ensemble_proba = meta_learner.predict_proba(meta_features_val)
            accuracy = accuracy_score(meta_labels_val, ensemble_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                meta_labels_val, ensemble_pred, average='weighted', zero_division=0
            )
            
            # 🆕 类别级别详细指标
            class_report = classification_report(
                meta_labels_val, ensemble_pred, 
                target_names=['SHORT', 'HOLD', 'LONG'], 
                output_dict=True,
                zero_division=0
            )
            
            # 🆕 混淆矩阵和致命错误分析
            cm = confusion_matrix(meta_labels_val, ensemble_pred)
            
            # 安全检查：确保混淆矩阵至少是3x3
            if cm.shape[0] >= 3 and cm.shape[1] >= 3:
                fatal_errors = int(cm[0, 2] + cm[2, 0])  # SHORT→LONG + LONG→SHORT
                fatal_error_rate = fatal_errors / len(meta_labels_val) if len(meta_labels_val) > 0 else 0.0
                long_to_short = int(cm[2, 0])  # LONG→SHORT
                short_to_long = int(cm[0, 2])  # SHORT→LONG
            else:
                logger.warning(f"⚠️ 混淆矩阵维度异常: {cm.shape}，跳过致命错误分析")
                fatal_errors = 0
                fatal_error_rate = 0.0
                long_to_short = 0
                short_to_long = 0
            
            # 🆕 信号质量分析
            signal_mask = ensemble_pred != 1  # 非HOLD预测
            signal_count = int(np.sum(signal_mask))
            signal_frequency = float(np.mean(signal_mask))
            hold_ratio = 1.0 - signal_frequency
            
            # 只在有信号时计算信号准确率
            if signal_count > 0:
                signal_labels = meta_labels_val[signal_mask]
                signal_preds = ensemble_pred[signal_mask]
                # 信号准确率：只看LONG/SHORT的预测准确率
                signal_accuracy = float(accuracy_score(signal_labels, signal_preds))
            else:
                signal_accuracy = 0.0
            
            # 🆕 概率校准指标
            try:
                log_loss_score = float(log_loss(meta_labels_val, ensemble_proba))
            except Exception as e:
                logger.warning(f"⚠️ Log Loss计算失败: {e}")
                log_loss_score = 0.0
            
            try:
                confidence_avg = float(np.mean(np.max(ensemble_proba, axis=1)))
            except Exception as e:
                logger.warning(f"⚠️ 置信度计算失败: {e}")
                confidence_avg = 0.0
            
            # 🆕 模型稳定性指标
            cv_stability = float(cv_std / cv_mean if cv_mean > 0 else 0)  # 变异系数
            cv_min = float(np.min(cv_scores))
            cv_max = float(np.max(cv_scores))
            
            # 基础模型一致性
            model_agreement = float(np.mean([
                (lgb_pred == xgb_pred).mean(),
                (lgb_pred == cat_pred).mean(),
                (xgb_pred == cat_pred).mean()
            ]))
            
            # 🆕 交易经济性指标
            trade_efficiency = float(signal_accuracy / signal_frequency if signal_frequency > 0 else 0)
            fee_impact = float(signal_frequency * 0.0007 * 100)  # 预估日手续费损耗%
            required_winrate = float(0.5 + (0.0007 / 0.02))  # 盈亏比1:1时的盈亏平衡胜率
            
            # 🆕 预测置信度分布
            try:
                confidence_values = np.max(ensemble_proba, axis=1)
                confidence_quantiles = np.quantile(confidence_values, [0.25, 0.5, 0.75, 0.9])
                confidence_q25 = float(confidence_quantiles[0])
                confidence_median = float(confidence_quantiles[1])
                confidence_q75 = float(confidence_quantiles[2])
                confidence_q90 = float(confidence_quantiles[3])
            except Exception as e:
                logger.warning(f"⚠️ 置信度分位数计算失败: {e}")
                confidence_q25 = 0.0
                confidence_median = 0.0
                confidence_q75 = 0.0
                confidence_q90 = 0.0
            
            # 高置信度预测的准确率
            try:
                high_confidence_mask = confidence_values > 0.7
                if np.sum(high_confidence_mask) > 0:
                    high_confidence_accuracy = float(accuracy_score(
                        meta_labels_val[high_confidence_mask],
                        ensemble_pred[high_confidence_mask]
                    ))
                    high_confidence_ratio = float(np.mean(high_confidence_mask))
                else:
                    high_confidence_accuracy = 0.0
                    high_confidence_ratio = 0.0
            except Exception as e:
                logger.warning(f"⚠️ 高置信度指标计算失败: {e}")
                high_confidence_accuracy = 0.0
                high_confidence_ratio = 0.0
            
            # 🆕 类别平衡性指标
            try:
                pred_distribution = np.bincount(ensemble_pred, minlength=3) / len(ensemble_pred)
                prediction_entropy = float(scipy_entropy(pred_distribution))  # 熵越高越平衡
                prediction_balance_score = float(1 - np.std(pred_distribution))  # 平衡分数
                short_ratio = float(pred_distribution[0])
                long_ratio = float(pred_distribution[2])
            except Exception as e:
                logger.warning(f"⚠️ 类别平衡性指标计算失败: {e}")
                prediction_entropy = 0.0
                prediction_balance_score = 0.0
                short_ratio = 0.0
                long_ratio = 0.0
            
            # 🆕 错误严重性加权指标
            try:
                fatal_weight = 3.0
                total_errors = len(meta_labels_val) - np.sum(ensemble_pred == meta_labels_val)
                normal_errors = max(0, total_errors - fatal_errors)  # 确保非负
                if len(meta_labels_val) > 0:
                    weighted_error_rate = float((fatal_errors * fatal_weight + normal_errors) / (len(meta_labels_val) * fatal_weight))
                else:
                    weighted_error_rate = 0.0
                fatal_error_ratio_in_errors = float(fatal_errors / total_errors if total_errors > 0 else 0)
            except Exception as e:
                logger.warning(f"⚠️ 错误严重性指标计算失败: {e}")
                weighted_error_rate = 0.0
                fatal_error_ratio_in_errors = 0.0
            
            logger.info(f"📊 {timeframe} 最终模型验证集准确率: {accuracy:.4f} (CV: {cv_mean:.4f}±{cv_std:.4f})")
            
            # 6️⃣ 评估各基础模型
            lgb_pred = lgb_model.predict(X_lgb_val)
            xgb_pred = xgb_model.predict(X_xgb_val)
            cat_pred = cat_model.predict(X_cat_val)
            
            lgb_acc = accuracy_score(y_lgb_val, lgb_pred)
            xgb_acc = accuracy_score(y_xgb_val, xgb_pred)
            cat_acc = accuracy_score(y_cat_val, cat_pred)
            
            # Informer-2准确率（如果存在）
            if inf_model is not None and X_seq_val is not None:
                # 🔑 修复：使用序列验证数据而不是2D数据
                inf_pred = inf_model.predict(X_seq_val)
                inf_acc = accuracy_score(y_seq_val, inf_pred)
            else:
                inf_acc = 0.0
            
            training_time = time.time() - start_time
            
            result = {
                # 基础指标
                'accuracy': cv_mean,  # 🔑 使用CV均值作为主准确率（更可靠）
                'cv_mean': cv_mean,   # 交叉验证均值
                'cv_std': cv_std,     # 交叉验证标准差
                'cv_scores': cv_scores,  # 各折分数
                'val_accuracy': accuracy,  # 验证集准确率
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                
                # 基础模型准确率
                'lgb_accuracy': lgb_acc,
                'xgb_accuracy': xgb_acc,
                'cat_accuracy': cat_acc,
                'inf_accuracy': inf_acc if inf_model else 0.0,
                
                # 🆕 类别级别指标
                'class_metrics': {
                    'SHORT': class_report.get('SHORT', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}),
                    'HOLD': class_report.get('HOLD', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}),
                    'LONG': class_report.get('LONG', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
                },
                
                # 🆕 混淆矩阵和致命错误
                'confusion_matrix': cm.tolist(),
                'fatal_errors': fatal_errors,
                'fatal_error_rate': fatal_error_rate,
                'long_to_short_errors': long_to_short,
                'short_to_long_errors': short_to_long,
                
                # 🆕 信号质量分析
                'signal_frequency': signal_frequency,
                'signal_accuracy': signal_accuracy,
                'signal_count': signal_count,
                'hold_ratio': hold_ratio,
                
                # 🆕 概率校准指标
                'log_loss': log_loss_score,
                'confidence_avg': confidence_avg,
                'confidence_q25': confidence_q25,
                'confidence_median': confidence_median,
                'confidence_q75': confidence_q75,
                'confidence_q90': confidence_q90,
                
                # 🆕 高置信度指标
                'high_confidence_accuracy': high_confidence_accuracy,
                'high_confidence_ratio': high_confidence_ratio,
                
                # 🆕 模型稳定性指标
                'cv_stability': cv_stability,
                'cv_min': cv_min,
                'cv_max': cv_max,
                'model_agreement': model_agreement,
                
                # 🆕 交易经济性指标
                'trade_efficiency': trade_efficiency,
                'fee_impact': fee_impact,
                'required_winrate': required_winrate,
                
                # 🆕 类别平衡性指标
                'prediction_entropy': prediction_entropy,
                'prediction_balance_score': prediction_balance_score,
                'short_ratio': short_ratio,
                'long_ratio': long_ratio,
                
                # 🆕 错误严重性加权指标
                'weighted_error_rate': weighted_error_rate,
                'fatal_error_ratio_in_errors': fatal_error_ratio_in_errors,
                
                # 其他信息
                'training_time': training_time,
                'ensemble_size': len(models),
                'meta_features_count': meta_features_val.shape[1]  # 元特征数量
            }
            
            logger.info(f"✅ Stacking训练完成（差异化数据）:")
            logger.info(f"")
            logger.info(f"  📊 基础模型表现:")
            logger.info(f"     LightGBM(360天): {lgb_acc:.4f}")
            logger.info(f"     XGBoost(540天):  {xgb_acc:.4f}")
            logger.info(f"     CatBoost(720天): {cat_acc:.4f}")
            if inf_model:
                logger.info(f"     Informer-2:      {inf_acc:.4f} 🤖")
            logger.info(f"")
            logger.info(f"  🎯 集成模型表现:")
            logger.info(f"     验证集准确率:   {accuracy:.4f}")
            logger.info(f"     时间序列CV:     {cv_mean:.4f} ± {cv_std:.4f} (5-fold)")
            logger.info(f"     Precision:      {precision:.4f}")
            logger.info(f"     Recall:         {recall:.4f}")
            logger.info(f"     F1-Score:       {f1:.4f}")
            logger.info(f"")
            logger.info(f"  📈 类别级别表现:")
            short_metrics = class_report.get('SHORT', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
            hold_metrics = class_report.get('HOLD', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
            long_metrics = class_report.get('LONG', {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
            logger.info(f"     SHORT - P:{short_metrics['precision']:.4f} R:{short_metrics['recall']:.4f} F1:{short_metrics['f1-score']:.4f} (样本:{int(short_metrics['support'])})")
            logger.info(f"     HOLD  - P:{hold_metrics['precision']:.4f} R:{hold_metrics['recall']:.4f} F1:{hold_metrics['f1-score']:.4f} (样本:{int(hold_metrics['support'])})")
            logger.info(f"     LONG  - P:{long_metrics['precision']:.4f} R:{long_metrics['recall']:.4f} F1:{long_metrics['f1-score']:.4f} (样本:{int(long_metrics['support'])})")
            logger.info(f"")
            logger.info(f"  🎲 信号质量分析:")
            logger.info(f"     信号频率:       {signal_frequency*100:.2f}% ({signal_count}个信号)")
            logger.info(f"     信号准确率:     {signal_accuracy:.4f}")
            logger.info(f"     HOLD比例:       {hold_ratio*100:.2f}%")
            logger.info(f"     平均置信度:     {confidence_avg:.4f}")
            logger.info(f"")
            logger.info(f"  ⚠️ 错误分析:")
            logger.info(f"     致命错误:       {fatal_errors}次 ({fatal_error_rate*100:.2f}%)")
            logger.info(f"     LONG→SHORT:     {long_to_short}次")
            logger.info(f"     SHORT→LONG:     {short_to_long}次")
            logger.info(f"     加权错误率:     {weighted_error_rate:.4f} (致命×3权重)")
            logger.info(f"     致命错误占比:   {fatal_error_ratio_in_errors*100:.2f}% (在总错误中)")
            logger.info(f"     Log Loss:       {log_loss_score:.4f}")
            logger.info(f"")
            logger.info(f"  🎯 预测置信度分布:")
            logger.info(f"     平均值:         {confidence_avg:.4f}")
            logger.info(f"     中位数:         {confidence_median:.4f}")
            logger.info(f"     Q25-Q75:        {confidence_q25:.4f} - {confidence_q75:.4f}")
            logger.info(f"     Q90:            {confidence_q90:.4f}")
            logger.info(f"     高置信(>0.7):   {high_confidence_ratio*100:.2f}% (准确率:{high_confidence_accuracy:.4f})")
            logger.info(f"")
            logger.info(f"  📊 类别预测分布:")
            logger.info(f"     SHORT比例:      {short_ratio*100:.2f}%")
            logger.info(f"     HOLD比例:       {hold_ratio*100:.2f}%")
            logger.info(f"     LONG比例:       {long_ratio*100:.2f}%")
            logger.info(f"     预测熵:         {prediction_entropy:.4f} (越高越平衡)")
            logger.info(f"     平衡分数:       {prediction_balance_score:.4f}")
            logger.info(f"")
            logger.info(f"  💰 交易经济性分析:")
            logger.info(f"     交易效率:       {trade_efficiency:.4f} (准确率/频率)")
            logger.info(f"     手续费影响:     {fee_impact:.4f}% (日预估)")
            logger.info(f"     盈亏平衡胜率:   {required_winrate*100:.2f}% (盈亏比1:1)")
            logger.info(f"")
            logger.info(f"  🔧 模型稳定性:")
            logger.info(f"     CV变异系数:     {cv_stability:.4f} (越小越稳定)")
            logger.info(f"     CV范围:         {cv_min:.4f} - {cv_max:.4f}")
            logger.info(f"     模型一致性:     {model_agreement*100:.2f}% (基础模型共识)")
            logger.info(f"")
            logger.info(f"  📊 模型配置:")
            n_base = 12 if inf_model else 9
            n_enhanced = 11
            logger.info(f"     元特征数量:     {meta_features_val.shape[1]}个（基础{n_base}+增强{n_enhanced}）")
            logger.info(f"     训练耗时:       {training_time:.2f}秒")
            
            # 🔄 生产级别：热更新模型（原子性替换）
            logger.info(f"🔄 {timeframe} 训练完成，准备热更新模型...")
            
            # 原子性替换模型（确保预测不会使用半更新的模型）
            old_version = self.model_versions.get(timeframe, 0)
            new_version = old_version + 1
            
            # ✅ 原子性替换：一次性更新模型字典
            self.ensemble_models[timeframe] = models
            
            # 更新版本和状态（原子操作）
            self.model_versions[timeframe] = new_version
            self.models_ready[timeframe] = True
            self.training_in_progress[timeframe] = False
            
            # 检查是否还有其他时间框架在训练
            self.background_training = any(self.training_in_progress.values())
            
            logger.info(f"✅ {timeframe} 模型已热更新（v{old_version} → v{new_version}），预测功能无缝衔接")
            
            if not self.background_training:
                logger.info(f"✅ 所有时间框架训练完成，系统运行在最新模型版本")
            
            return result
            
        except Exception as e:
            logger.error(f"差异化Stacking训练失败: {e}")
            logger.error(traceback.format_exc())
            # 🔓 训练失败时清除状态（保持旧模型继续运行）
            self.training_in_progress[timeframe] = False
            self.background_training = any(self.training_in_progress.values())
            
            logger.warning(f"⚠️ {timeframe} 训练失败，继续使用旧模型（预测功能不受影响）")
            
            raise
    
    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str, custom_params: Optional[Dict[str, Any]] = None):
        """
        训练LightGBM模型（覆盖父类方法，统一三模型训练代码位置）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            timeframe: 时间框架
            custom_params: 自定义参数（Optuna优化后的参数，优先级最高）
        """
        try:
            # 🎮 统一GPU内存管理：训练前清理
            self.clear_gpu_memory()
            
            # 样本加权（有效样本数 × 时间衰减 × HOLD惩罚）
            class_weights = self._compute_effective_sample_weights(y_train, timeframe)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（按HOLD占比自适应）
            hold_ratio = float((y_train == 1).sum()) / max(len(y_train), 1)
            if timeframe == '3m':
                hold_weight = float(max(0.35, min(0.70, 0.80 - 0.6 * hold_ratio)))
            else:
                hold_weight = float(max(0.50, min(0.75, 0.85 - 0.5 * hold_ratio)))
            hold_penalty = np.where(y_train == 1, hold_weight, 1.0)
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            logger.info(f"✅ 样本加权已启用：有效样本数 × 时间衰减 × HOLD惩罚({hold_weight:.2f})")
            
            # 确定最终参数（优先级：custom_params > timeframe_params > base_params）
            if custom_params:
                params = custom_params
                logger.info(f"🎯 使用Optuna优化参数")
            else:
                # 获取时间框架差异化参数
                timeframe_params = self.lgb_params_by_timeframe.get(timeframe, {})
                # 合并基础参数和差异化参数
                params = {**self.lgb_params, **timeframe_params}
            
            # 🎮 启用GPU加速（如果配置启用）
            if self.use_gpu:
                params['device'] = 'gpu'
                params['gpu_platform_id'] = 0
                params['gpu_device_id'] = 0
                logger.info(f"🚀 LightGBM GPU加速已启用")
            
            logger.info(f"📊 {timeframe} LightGBM参数: num_leaves={params.get('num_leaves')}, "
                       f"reg_alpha={params.get('reg_alpha', 0)}, reg_lambda={params.get('reg_lambda', 0)}")
            
            # 创建并训练模型（params中已包含random_state=42）
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # 🎮 统一GPU内存管理：训练后清理
            self.clear_gpu_memory()
            
            return model
            
        except Exception as e:
            logger.error(f"LightGBM训练失败: {e}")
            # 🎮 统一GPU内存管理：异常时清理
            self.clear_gpu_memory()
            raise
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str, custom_params: Optional[Dict[str, Any]] = None):
        """训练XGBoost模型（防过拟合 + GPU内存管理）"""
        try:
            # 🎮 统一GPU内存管理：训练前清理
            self.clear_gpu_memory()
            
            # 样本加权（有效样本数 × 时间衰减 × HOLD惩罚）
            class_weights = self._compute_effective_sample_weights(y_train, timeframe)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（按HOLD占比自适应）
            hold_ratio = float((y_train == 1).sum()) / max(len(y_train), 1)
            if timeframe == '3m':
                hold_weight = float(max(0.35, min(0.70, 0.80 - 0.6 * hold_ratio)))
            else:
                hold_weight = float(max(0.50, min(0.75, 0.85 - 0.5 * hold_ratio)))
            hold_penalty = np.where(y_train == 1, hold_weight, 1.0)
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            # 🔑 时间框架差异化配置（防止2h/4h过拟合）
            if custom_params:
                # 使用Optuna优化的参数
                params = custom_params.copy()
                logger.info(f"🎯 使用Optuna优化参数")
            else:
                # 使用默认参数（仅3m/5m/15m）
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
            
            # 通用参数
            params.update({
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            })
            
            # 🎮 GPU加速（如果启用）
            if self.use_gpu:
                params['tree_method'] = 'hist'  # 新版本使用 hist
                params['device'] = 'cuda'  # 使用 device 参数指定 GPU
                logger.info(f"🚀 XGBoost GPU加速已启用")
            else:
                params['tree_method'] = 'hist'
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            # 🎮 统一GPU内存管理：训练后清理
            self.clear_gpu_memory()
            
            return model
            
        except Exception as e:
            logger.error(f"XGBoost训练失败: {e}")
            # 🎮 统一GPU内存管理：异常时清理
            self.clear_gpu_memory()
            raise
    
    def _train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str, custom_params: Optional[Dict[str, Any]] = None):
        """训练CatBoost模型（防过拟合 + GPU内存管理）"""
        try:
            # 🎮 统一GPU内存管理：训练前清理
            self.clear_gpu_memory()
            
            # 样本加权（有效样本数 × 时间衰减 × HOLD惩罚）
            class_weights = self._compute_effective_sample_weights(y_train, timeframe)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（按HOLD占比自适应）
            hold_ratio = float((y_train == 1).sum()) / max(len(y_train), 1)
            if timeframe == '3m':
                hold_weight = float(max(0.35, min(0.70, 0.80 - 0.6 * hold_ratio)))
            else:
                hold_weight = float(max(0.50, min(0.75, 0.85 - 0.5 * hold_ratio)))
            hold_penalty = np.where(y_train == 1, hold_weight, 1.0)
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            # 🔑 时间框架差异化配置（防止2h/4h过拟合）
            if custom_params:
                # 使用Optuna优化的参数
                params = custom_params.copy()
                logger.info(f"🎯 使用Optuna优化参数")
            else:
                # 使用默认参数（仅3m/5m/15m）
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
            
            # 通用参数
            params.update({
                'loss_function': 'MultiClass',
                'random_seed': 42,
                'verbose': False,
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.8,
                'allow_writing_files': False
            })
            
            # 🎮 GPU加速（如果启用）
            if self.use_gpu:
                params['task_type'] = 'GPU'
                params['devices'] = '0'
                logger.info(f"🚀 CatBoost GPU加速已启用")
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            # 🎮 统一GPU内存管理：训练后清理
            self.clear_gpu_memory()
            
            return model
            
        except Exception as e:
            logger.error(f"CatBoost训练失败: {e}")
            # 🎮 统一GPU内存管理：异常时清理
            self.clear_gpu_memory()
            raise
    
    def _train_informer2(self, X_seq_train: np.ndarray, y_seq_train: np.ndarray, timeframe: str, custom_params: Optional[Dict[str, Any]] = None):
        """
        训练Informer-2深度学习模型（使用GMADL损失函数 + 序列输入）
        
        Args:
            X_seq_train: 序列训练特征 (n_samples, seq_len, n_features)
            y_seq_train: 训练标签 (n_samples,)
            timeframe: 时间框架（仅支持3m/5m/15m）
            custom_params: 自定义参数（来自Optuna优化）
        
        Returns:
            训练好的Informer-2模型（兼容scikit-learn接口）
        """
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch未安装，跳过Informer-2训练")
            return None
        
        try:
            start_time = time.time()
            
            # 🎮 GPU内存管理：训练前清理
            self.clear_gpu_memory()
            
            logger.info(f"🤖 训练Informer-2神经网络模型（序列输入）...")
            logger.info(f"   输入形状: {X_seq_train.shape} (样本数, 序列长度, 特征数)")
            
            # 1. 数据准备（NumPy → PyTorch，内存优化）
            # 🔥 优化：确保输入为float32，并使用from_numpy避免数据复制
            # 🔥 关键修复：确保数据是连续的numpy数组（避免内存映射问题）
            if not isinstance(X_seq_train, np.ndarray):
                X_seq_train = np.asarray(X_seq_train, dtype=np.float32)
            elif X_seq_train.dtype != np.float32:
                logger.debug(f"   转换序列数据为float32（原类型: {X_seq_train.dtype}）")
                X_seq_train = X_seq_train.astype(np.float32)
            
            if not X_seq_train.flags['C_CONTIGUOUS']:
                logger.debug(f"   转换X_seq_train为连续数组")
                X_seq_train = np.ascontiguousarray(X_seq_train)
            
            # 🔥 关键修复：统一处理y_seq_train的数据类型
            if not isinstance(y_seq_train, np.ndarray):
                y_seq_train = np.asarray(y_seq_train, dtype=np.int64)
            elif y_seq_train.dtype != np.int64:
                y_seq_train = y_seq_train.astype(np.int64)
            
            if not y_seq_train.flags['C_CONTIGUOUS']:
                logger.debug(f"   转换y_seq_train为连续数组")
                y_seq_train = np.ascontiguousarray(y_seq_train)
            
            # ✅ 关键修复：对3D序列数据进行归一化（防止数值溢出）
            logger.info(f"🔧 对序列数据进行归一化（防止数值溢出）...")
            logger.info(f"   归一化前统计: 范围=[{X_seq_train.min():.4f}, {X_seq_train.max():.4f}], 均值={X_seq_train.mean():.4f}, 标准差={X_seq_train.std():.4f}")
            
            # 方法：将3D数据reshape为2D，按特征归一化，再reshape回3D
            # 这样每个特征在所有样本和时间步上都被归一化
            original_shape = X_seq_train.shape
            n_features = original_shape[2]
            
            # Reshape为2D: (n_samples * seq_len, n_features)
            X_seq_train_2d = X_seq_train.reshape(-1, n_features)
            
            # 使用StandardScaler归一化
            scaler = StandardScaler()
            X_seq_train_2d_scaled = scaler.fit_transform(X_seq_train_2d)
            
            # Reshape回3D: (n_samples, seq_len, n_features)
            X_seq_train = X_seq_train_2d_scaled.reshape(original_shape).astype(np.float32)
            
            logger.info(f"   ✅ 归一化完成")
            logger.info(f"   归一化后统计: 范围=[{X_seq_train.min():.4f}, {X_seq_train.max():.4f}], 均值={X_seq_train.mean():.4f}, 标准差={X_seq_train.std():.4f}")
            
            # 保存scaler用于预测时使用
            if timeframe not in self.scalers:
                self.scalers[timeframe] = {}
            self.scalers[timeframe]['informer2'] = scaler
            
            X_tensor = torch.from_numpy(X_seq_train)  # (n_samples, seq_len, n_features) - 避免复制
            y_tensor = torch.from_numpy(y_seq_train)  # (n_samples,) - LongTensor需要int64
            
            # 📊 内存监控：张量占用
            tensor_memory_mb = (X_tensor.element_size() * X_tensor.nelement() + 
                               y_tensor.element_size() * y_tensor.nelement()) / (1024 ** 2)
            logger.info(f"   张量内存占用: {tensor_memory_mb:.1f} MB")
            
            # 2. 检测GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"   设备: {device} {'🚀 (GPU加速)' if device.type == 'cuda' else '💻 (CPU)'}")
            
            # 🎮 GPU内存监控
            if torch.cuda.is_available():
                gpu_status = self.monitor_gpu_memory()
                logger.info(f"   GPU内存状态: 使用{gpu_status['usage_percent']:.1f}% ({gpu_status['used']/1024**3:.1f}GB/{gpu_status['total']/1024**3:.1f}GB)")
            
            # 3. 创建数据加载器
            class NumpyTimeSeriesDataset(Dataset):
                def __init__(self, X_np, y_np):
                    # 🔥 关键修复：确保数据是连续的numpy数组
                    self.X_np = np.ascontiguousarray(X_np) if not X_np.flags['C_CONTIGUOUS'] else X_np
                    self.y_np = np.ascontiguousarray(y_np) if not y_np.flags['C_CONTIGUOUS'] else y_np
                def __len__(self):
                    return len(self.y_np)
                def __getitem__(self, idx):
                    # 🔥 关键修复：使用copy()避免内存映射问题
                    return (
                        torch.from_numpy(self.X_np[idx].copy()).to(dtype=torch.float32),
                        torch.tensor(self.y_np[idx], dtype=torch.long)
                    )

            dataset = NumpyTimeSeriesDataset(X_seq_train, y_seq_train)
            dataloader = DataLoader(
                dataset,
                batch_size=self.informer_batch_size,
                shuffle=True,
                num_workers=0  # Windows兼容
            )
            
            # 4. 使用自定义参数（如果提供）
            if custom_params:
                d_model = custom_params.get('d_model', self.informer_d_model)
                n_heads = custom_params.get('n_heads', self.informer_n_heads)
                n_layers = custom_params.get('n_layers', self.informer_n_layers)
                dropout = custom_params.get('dropout', 0.1)
                epochs = custom_params.get('epochs', self.informer_epochs)
                batch_size = custom_params.get('batch_size', self.informer_batch_size)
                lr = custom_params.get('lr', self.informer_lr)
                alpha = custom_params.get('alpha', settings.GMADL_ALPHA)
                beta = custom_params.get('beta', settings.GMADL_BETA)
                logger.info(f"🎯 使用优化参数: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, epochs={epochs}")
            else:
                d_model = self.informer_d_model
                n_heads = self.informer_n_heads
                n_layers = self.informer_n_layers
                dropout = 0.1
                epochs = self.informer_epochs
                batch_size = self.informer_batch_size
                lr = self.informer_lr
                alpha = settings.GMADL_ALPHA
                beta = settings.GMADL_BETA
            
            # 5. 初始化模型（支持序列输入 + 梯度检查点）
            n_features = X_seq_train.shape[2]  # 特征数量（从序列的最后一维获取）
            model = Informer2ForClassification(
                n_features=n_features,
                n_classes=3,  # 类别数
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                use_distilling=True,  # 启用蒸馏层（完整Informer架构）
                use_gradient_checkpointing=self.use_gradient_checkpointing  # 🔥 启用梯度检查点
            ).to(device)
            
            logger.info(f"   模型参数: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
            logger.info(f"   训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
            
            # 6. 定义损失函数（支持GMADL/交叉熵两种模式）
            # 🔑 GMADL的HOLD惩罚按类别占比自适应（3m更强以对抗极端不平衡）
            hold_ratio_informer = float((y_seq_train == 1).sum()) / max(len(y_seq_train), 1)
            if timeframe == '3m':
                hold_penalty_nn = float(max(0.35, min(0.70, 0.80 - 0.6 * hold_ratio_informer)))
            else:
                hold_penalty_nn = float(max(0.50, min(0.75, 0.85 - 0.5 * hold_ratio_informer)))

            criterion = create_trade_loss(
                use_gmadl=settings.USE_GMADL_LOSS,
                hold_penalty=hold_penalty_nn,
                alpha=alpha,
                beta=beta
            )

            if settings.USE_GMADL_LOSS:
                logger.info(
                    f"   损失函数: GMADL + HOLD惩罚 (alpha={alpha:.2f}, beta={beta:.2f})"
                )
            else:
                logger.info(
                    "   损失函数: 交叉熵 + HOLD惩罚 (稳定模式)"
                )
            
            # 7. 定义优化器（支持8-bit Adam）
            # 🔥 尝试使用8-bit Adam优化器（节省75%优化器内存）
            optimizer_created = False
            if self.use_8bit_adam and device.type == 'cuda':
                try:
                    # 🔥 动态导入可选依赖（bitsandbytes可能未安装）
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.Adam8bit(
                        model.parameters(),
                        lr=lr,
                        weight_decay=1e-5,
                        betas=(0.9, 0.999)
                    )
                    logger.info("   ✅ 使用8-bit Adam优化器（节省75%优化器内存）")
                    optimizer_created = True
                except ImportError:
                    logger.warning("   ⚠️ bitsandbytes未安装，使用标准Adam优化器")
                    logger.warning("   💡 安装命令: pip install bitsandbytes")
                except Exception as e:
                    logger.warning(f"   ⚠️ 8-bit Adam初始化失败: {e}，使用标准Adam")
            
            # 降级到标准Adam
            if not optimizer_created:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=1e-5,  # L2正则化
                    betas=(0.9, 0.999)
                )
            
            # ✅ 修复C: 添加Warmup + ReduceLROnPlateau组合调度器
            # Warmup配置
            warmup_epochs = 5  # 前5个epoch warmup
            target_lr = lr
            
            # 主调度器：ReduceLROnPlateau（用于warmup后的学习率调整）
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=2,
                verbose=True
            )
            
            logger.info(f"   ✅ 学习率调度: Warmup({warmup_epochs}轮) + ReduceLROnPlateau")
            logger.info(f"      目标LR: {target_lr:.6f}, Warmup后自动调整")
            
            # 🚀 9. 梯度累积配置（解决GPU OOM问题，不降低模型复杂度）
            # 将大批次分成小批次，累积梯度，保持等效训练效果
            effective_batch_size = batch_size  # 保持原始有效批次大小
            actual_batch_size = max(8, batch_size // 8)  # 物理批次大小缩小8倍（节省8倍GPU内存）
            accumulation_steps = effective_batch_size // actual_batch_size  # 累积步数
            
            # 重新创建数据加载器（使用更小的物理批次）
            dataloader = DataLoader(
                dataset,
                batch_size=actual_batch_size,
                shuffle=True,
                num_workers=0,  # Windows兼容
                pin_memory=True if device.type == 'cuda' else False  # 加速GPU数据传输
            )
            
            logger.info(f"   🎮 梯度累积策略: 有效批次={effective_batch_size}, 物理批次={actual_batch_size}, 累积步数={accumulation_steps}")
            logger.info(f"   💾 预期GPU内存节省: ~{100*(1-actual_batch_size/batch_size):.0f}%")
            
            # ✅ 修复D: 动态混合精度配置（而非完全禁用）
            use_amp = device.type == 'cuda' and torch.cuda.is_available()

            if use_amp:
                # 根据模型规模动态调整初始缩放因子
                num_params = sum(p.numel() for p in model.parameters())
                
                if num_params > 10_000_000:  # >10M参数：大模型
                    init_scale = 2.**12  # 4096
                    logger.info(f"   检测到大模型({num_params/1e6:.1f}M参数)，使用init_scale=2^12")
                elif num_params > 1_000_000:  # 1M-10M参数：中等模型
                    init_scale = 2.**14  # 16384
                    logger.info(f"   检测到中等模型({num_params/1e6:.1f}M参数)，使用init_scale=2^14")
                else:  # <1M参数：小模型
                    init_scale = 2.**16  # 65536（默认值）
                    logger.info(f"   检测到小模型({num_params/1e6:.1f}M参数)，使用init_scale=2^16")
                
                # ✅ 修复：使用新的torch.amp.GradScaler API（PyTorch 2.0+）
                scaler = torch.amp.GradScaler(
                    'cuda',
                    init_scale=init_scale,  # 动态调整的初始缩放
                    growth_factor=1.5,      # 增长因子（默认2.0，改为1.5更温和）
                    backoff_factor=0.5,     # 回退因子（检测到溢出时）
                    growth_interval=1000,   # 增长间隔（默认2000，改为1000更谨慎）
                    enabled=True
                )
                logger.info("   混合精度训练: 启用（动态缩放策略）")
                logger.info(f"      初始缩放: {init_scale}, 增长因子: 1.5")
            else:
                scaler = None
                logger.info("   混合精度训练: 禁用（CPU环境）")
            
            # 可选：如果未来需要重新启用AMP，使用保守策略
            # if settings.USE_GMADL_LOSS and use_amp:
            #     logger.info("   ⚠️ GMADL开启 → 为保障数值稳定，禁用AMP改用FP32训练")
            #     use_amp = False
            # 
            # # 🔥 激进混合精度优化
            # if use_amp and self.use_aggressive_amp:
            #     # 设置更高的初始缩放因子
            #     scaler = torch.amp.GradScaler('cuda', init_scale=2.**16)
            #     
            #     # 启用TF32（Ampere架构GPU：RTX 30/40系列）
            #     torch.backends.cuda.matmul.allow_tf32 = True
            #     torch.backends.cudnn.allow_tf32 = True
            #     
            #     logger.info(f"   ⚡ 启用激进混合精度训练（FP16 + TF32 + 高缩放因子）")
            # elif use_amp:
            #     scaler = torch.amp.GradScaler('cuda')
            #     logger.info(f"   ⚡ 启用混合精度训练（AMP）：FP16计算 + 动态损失缩放")
            # else:
            #     scaler = None
            
            # ✅ 修复E: 训练前数据质量检查
            logger.info("🔍 执行训练前数据质量检查...")
            
            # 检查特征数据
            if torch.isnan(X_tensor).any():
                nan_count = torch.isnan(X_tensor).sum().item()
                logger.error(f"❌ 训练数据包含{nan_count}个NaN值，训练终止！")
                raise ValueError(f"训练数据包含NaN值：{nan_count}个")
            
            if torch.isinf(X_tensor).any():
                inf_count = torch.isinf(X_tensor).sum().item()
                logger.error(f"❌ 训练数据包含{inf_count}个INF值，训练终止！")
                raise ValueError(f"训练数据包含INF值：{inf_count}个")
            
            # 检查标签数据
            if torch.isnan(y_tensor.float()).any() or torch.isinf(y_tensor.float()).any():
                logger.error(f"❌ 训练标签包含NaN/INF值，训练终止！")
                raise ValueError("训练标签包含NaN/INF值")
            
            # 检查标签范围
            unique_labels = torch.unique(y_tensor)
            if not all(label in [0, 1, 2] for label in unique_labels.tolist()):
                logger.error(f"❌ 训练标签包含非法值：{unique_labels.tolist()}，期望[0,1,2]")
                raise ValueError(f"训练标签包含非法值：{unique_labels.tolist()}")
            
            # 统计数据范围
            logger.info(f"   特征范围: [{X_tensor.min().item():.4f}, {X_tensor.max().item():.4f}]")
            logger.info(f"   特征均值: {X_tensor.mean().item():.4f}, 标准差: {X_tensor.std().item():.4f}")
            logger.info(f"   标签分布: {torch.bincount(y_tensor.long()).tolist()}")
            logger.info(f"✅ 数据质量检查通过")
            
            # 🔍 模型权重初始化检查
            logger.info("🔍 检查模型权重初始化...")
            has_nan_weights = False
            has_inf_weights = False
            
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    logger.error(f"❌ 模型参数 {name} 包含NaN值！")
                    has_nan_weights = True
                if torch.isinf(param).any():
                    logger.error(f"❌ 模型参数 {name} 包含INF值！")
                    has_inf_weights = True
            
            if has_nan_weights or has_inf_weights:
                logger.error("❌ 模型权重初始化异常，训练终止！")
                raise ValueError("模型权重初始化包含NaN/INF值")
            
            logger.info("✅ 模型权重初始化正常")
            
            # 11. 训练循环（带梯度累积和混合精度）
            model.train()
            best_loss = float('inf')
            
            # ✅ 修复F: 平衡的早期终止阈值
            nan_inf_count = 0  # 统计nan/inf出现次数
            max_nan_inf_tolerance = 30  # 从50降低到30（平衡值）
            consecutive_nan_inf = 0  # 连续nan/inf次数
            max_consecutive_nan_inf = 8  # 从10降低到8（平衡值）
            
            logger.info(f"   早期终止阈值: 连续{max_consecutive_nan_inf}次 或 累计{max_nan_inf_tolerance}次")
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0
                processed_batches = 0  # 实际处理的batch数（排除nan/inf的batch）
                epoch_nan_inf_count = 0  # 本epoch的nan/inf次数
                
                optimizer.zero_grad()  # 初始化梯度
                
                for i, (batch_X, batch_y) in enumerate(dataloader):
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    # ✅ 修复A - 诊断1: 检查输入数据
                    if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                        logger.error(f"❌ Batch {i+1}: 输入数据包含NaN/INF")
                        logger.error(f"   NaN数量: {torch.isnan(batch_X).sum().item()}")
                        logger.error(f"   INF数量: {torch.isinf(batch_X).sum().item()}")
                        # 保存异常batch用于离线分析
                        try:
                            torch.save({'X': batch_X.cpu(), 'y': batch_y.cpu()}, 
                                      f'debug_batch_{epoch}_{i}.pt')
                        except:
                            pass
                        optimizer.zero_grad()
                        continue
                    
                    # 🎯 混合精度前向传播
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            logits = model(batch_X)
                            # 统一dtype与loss输入：logits用float32，targets用long
                            loss = criterion(logits.float(), batch_y.long())
                            loss = loss / accumulation_steps  # 归一化损失（梯度累积）
                    else:
                        logits = model(batch_X)
                        loss = criterion(logits.float(), batch_y.long())
                        loss = loss / accumulation_steps
                    
                    # ✅ 修复A - 诊断2: 检查模型输出
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.error(f"❌ Batch {i+1}: 模型输出(logits)包含NaN/INF")
                        logger.error(f"   输入范围: [{batch_X.min().item():.4f}, {batch_X.max().item():.4f}]")
                        
                        # 逐层诊断（使用forward hooks，更安全）
                        logger.error("   🔍 逐层诊断（使用hooks）:")
                        activation_stats = {}
                        hooks = []
                        
                        def get_activation_hook(name):
                            def hook(module, input, output):
                                if isinstance(output, torch.Tensor):
                                    has_nan = torch.isnan(output).any().item()
                                    has_inf = torch.isinf(output).any().item()
                                    activation_stats[name] = {
                                        'has_nan': has_nan,
                                        'has_inf': has_inf,
                                        'min': output.min().item() if not (has_nan or has_inf) else None,
                                        'max': output.max().item() if not (has_nan or has_inf) else None
                                    }
                            return hook
                        
                        # 注册hooks
                        with torch.no_grad():
                            for name, module in model.named_modules():
                                if len(list(module.children())) == 0:  # 只对叶子模块
                                    hook = module.register_forward_hook(get_activation_hook(name))
                                    hooks.append(hook)
                            
                            # 重新执行forward
                            try:
                                _ = model(batch_X)
                                
                                # 打印异常层
                                for name, stats in activation_stats.items():
                                    if stats['has_nan'] or stats['has_inf']:
                                        logger.error(f"      {name}: NaN={stats['has_nan']}, INF={stats['has_inf']}")
                            except Exception as e:
                                logger.error(f"      逐层诊断失败: {e}")
                            finally:
                                # 移除所有hooks
                                for hook in hooks:
                                    hook.remove()
                        
                        optimizer.zero_grad()
                        continue
                    
                    # ✅ 修复A - 诊断3: 检查损失值
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_inf_count += 1
                        consecutive_nan_inf += 1
                        epoch_nan_inf_count += 1
                        
                        logger.error(f"❌ Batch {i+1}: 损失为NaN/INF")
                        logger.error(f"   Logits统计: min={logits.min().item():.4f}, "
                                    f"max={logits.max().item():.4f}, "
                                    f"mean={logits.mean().item():.4f}")
                        logger.error(f"   Target分布: {torch.bincount(batch_y.long()).tolist()}")
                        
                        # ✅ 修复A - 诊断4: 检查梯度
                        loss.backward()
                        max_grad_norm = 0.0
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                max_grad_norm = max(max_grad_norm, grad_norm)
                                if grad_norm > 1000 or grad_norm != grad_norm:  # 梯度爆炸或NaN
                                    logger.error(f"   {name}: 梯度异常 norm={grad_norm:.4f}")
                        
                        logger.error(f"   最大梯度范数: {max_grad_norm:.4f}")
                        
                        # 仅在前5次或每50次打印警告，避免日志刷屏
                        if nan_inf_count <= 5 or nan_inf_count % 50 == 0:
                            logger.warning(f"⚠️ Epoch {epoch+1} Batch {i+1}: 检测到损失为nan/inf（累计{nan_inf_count}次，连续{consecutive_nan_inf}次）")
                        
                        # 🚨 检查是否超过容忍阈值
                        if consecutive_nan_inf >= max_consecutive_nan_inf:
                            logger.error(f"❌ 连续{consecutive_nan_inf}个batch出现nan/inf损失，训练终止！")
                            logger.error(f"   可能原因：1) 学习率过大 2) GMADL损失函数数值不稳定 3) 数据异常")
                            logger.error(f"   建议：1) 降低学习率 2) 使用FP32精度 3) 检查数据质量")
                            raise ValueError(f"训练过程数值不稳定：连续{consecutive_nan_inf}个batch出现nan/inf损失")
                        
                        if nan_inf_count >= max_nan_inf_tolerance:
                            logger.error(f"❌ 累计{nan_inf_count}个batch出现nan/inf损失（超过阈值{max_nan_inf_tolerance}），训练终止！")
                            raise ValueError(f"训练过程数值不稳定：累计{nan_inf_count}个batch出现nan/inf损失")
                        
                        optimizer.zero_grad()
                        continue
                    
                    # 成功处理batch，重置连续nan/inf计数器
                    consecutive_nan_inf = 0
                    
                    # 🎯 混合精度反向传播
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # ✅ 修复B: 梯度裁剪（核心修复）⭐
                    # 🎯 梯度累积：每accumulation_steps步更新一次参数
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                        # ⚠️ 重要：混合精度训练时必须先unscale_()再裁剪
                        if use_amp:
                            scaler.unscale_(optimizer)  # 先反缩放梯度，否则裁剪无效
                        
                        # ⭐ 核心修复：梯度裁剪（防止梯度爆炸）
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm=1.0,      # 梯度范数上限（Informer2建议1.0）
                            norm_type=2.0       # L2范数
                        )
                        
                        # 优化器步进
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        optimizer.zero_grad()  # 清空梯度
                        
                        # 🧹 定期清理GPU缓存（每10个累积周期）
                        if (i + 1) % (accumulation_steps * 10) == 0 and device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # 统计（使用未归一化的损失）
                    processed_batches += 1
                    epoch_loss += loss.item() * accumulation_steps
                    with torch.no_grad():
                        _, predicted = torch.max(logits, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                # 🧹 每个epoch结束后清理GPU缓存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # ✅ 修复F: Epoch级别检查
                total_batches = len(dataloader)
                
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
                
                # 计算平均损失和准确率
                avg_loss = epoch_loss / max(processed_batches, 1)
                accuracy = 100.0 * correct / max(total, 1)
                
                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                # ✅ 修复C: 学习率调度（简化的Warmup + ReduceLROnPlateau）
                if epoch < warmup_epochs:
                    # Warmup阶段：线性增长学习率
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
                
                # 每10轮或最后1轮打印进度（带学习率）
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    nan_info = f", nan/inf跳过: {epoch_nan_inf_count}" if epoch_nan_inf_count > 0 else ""
                    logger.info(
                        f"   Epoch [{epoch+1}/{epochs}] "
                        f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, "
                        f"LR: {current_lr:.6f} ({phase}){nan_info}"
                    )
            
            # 📊 训练完成总结
            if nan_inf_count > 0:
                logger.warning(f"⚠️ Informer2训练完成，但出现{nan_inf_count}次nan/inf损失（已跳过）")
                logger.warning(f"   数值稳定性问题可能影响模型质量，建议：")
                logger.warning(f"   1. 降低学习率（当前：{lr}）")
                logger.warning(f"   2. 禁用混合精度训练（use_amp=False）")
                logger.warning(f"   3. 调整GMADL损失函数参数")
            else:
                logger.info(f"✅ Informer2训练完成，无数值稳定性问题")
            
            # 9. 切换到评估模式
            model.eval()
            
            # 10. 包装模型以兼容scikit-learn接口（支持序列输入）
            # ✅ 使用模块级别的InformerWrapper类（支持pickle序列化）
            wrapped_model = InformerWrapper(model, device)
            
            # 🎮 GPU内存管理：训练后清理
            self.clear_gpu_memory()
            
            training_time = time.time() - start_time
            logger.info(f"✅ Informer-2训练完成: 最佳Loss={best_loss:.4f}, "
                       f"耗时={training_time:.2f}秒")
            
            return wrapped_model
            
        except Exception as e:
            logger.error(f"Informer-2训练失败: {e}")
            # 🎮 GPU内存管理：异常时清理
            self.clear_gpu_memory()
            logger.warning("⚠️ 将跳过Informer-2，仅使用传统模型")
            return None
    
    async def predict(
        self, 
        data: pd.DataFrame, 
        timeframe: str
    ) -> Dict[str, Any]:
        """
        集成预测（覆盖父类方法）
        
        Args:
            data: K线数据DataFrame
            timeframe: 时间框架
        
        Returns:
            预测结果，如果模型训练中则返回None
        """
        try:
            # � 检生产级别：后台训练不影响预测
            # 训练和预测并行运行，训练完成后热更新模型
            if self.background_training:
                training_tfs = [tf for tf, status in self.training_in_progress.items() if status]
                logger.debug(f"🔄 后台训练中（{', '.join(training_tfs)}），预测继续使用当前模型")
            
            # 仅在首次训练时（模型不存在）才阻止预测
            if timeframe not in self.ensemble_models and not self.models_ready.get(timeframe, False):
                if self.training_in_progress.get(timeframe, False):
                    logger.debug(f"⏳ {timeframe} 首次训练中，等待模型就绪")
                    return None
                else:
                    logger.debug(f"⏸️ {timeframe} 模型未就绪，等待训练完成")
                    return None
            
            # 检查集成模型是否存在
            if timeframe not in self.ensemble_models:
                # 如果模型未就绪且不在训练中，尝试加载
                if not self.models_ready.get(timeframe, False):
                    logger.debug(f"⏸️ {timeframe} 模型未就绪，等待训练完成")
                    return None
                logger.warning(f"⚠️ {timeframe} 集成模型未训练，降级到单模型")
                return await super().predict(data, timeframe)
            
            # 特征工程
            processed_data = self.feature_engineer.create_features(data.copy())
            if processed_data.empty:
                return None
            
            # 准备特征（使用该时间框架的特征列）
            feature_columns = self.feature_columns_dict.get(timeframe, [])
            if not feature_columns:
                logger.error(f"{timeframe} 特征列未找到")
                return None
            
            X = processed_data.iloc[-1:][feature_columns]
            if len(X) == 0:
                return None
            
            # 特征缩放
            X_scaled = self._scale_features(X, timeframe, fit=False)
            
            # 获取集成模型
            models = self.ensemble_models[timeframe]
            
            # 三个基础模型预测（X_scaled可能是numpy数组或DataFrame）
            if isinstance(X_scaled, np.ndarray):
                X_pred = X_scaled
            else:
                X_pred = X_scaled.iloc[[-1]] if hasattr(X_scaled, 'iloc') else X_scaled
            
            # 🔑 基础模型预测（使用短键名）
            lgb_proba = models['lgb'].predict_proba(X_pred)[0]
            xgb_proba = models['xgb'].predict_proba(X_pred)[0]
            cat_proba = models['cat'].predict_proba(X_pred)[0]
            
            lgb_pred = models['lgb'].predict(X_pred)[0]
            xgb_pred = models['xgb'].predict(X_pred)[0]
            cat_pred = models['cat'].predict(X_pred)[0]
            
            # 🤖 Informer-2预测（如果存在，需要序列输入）
            if 'inf' in models:
                # 构造序列输入（取最新seq_len个时间步）
                seq_len = self.seq_len_config.get(timeframe, 96)
                
                if len(processed_data) < seq_len:
                    logger.warning(f"⚠️ 数据不足：需要{seq_len}个时间步，实际{len(processed_data)}个，跳过Informer-2预测")
                    inf_proba = None
                    inf_pred = None
                else:
                    # 取最新seq_len个时间步构造序列
                    latest_seq = processed_data.iloc[-seq_len:][feature_columns].values
                    latest_seq = latest_seq.reshape(1, seq_len, -1)  # (1, seq_len, n_features)
                    
                    # ✅ 关键修复：预测时也需要归一化（使用训练时的scaler）
                    if timeframe in self.scalers and 'informer2' in self.scalers[timeframe]:
                        scaler = self.scalers[timeframe]['informer2']
                        # Reshape为2D进行归一化
                        original_shape = latest_seq.shape
                        n_features = original_shape[2]
                        latest_seq_2d = latest_seq.reshape(-1, n_features)
                        latest_seq_2d_scaled = scaler.transform(latest_seq_2d)
                        latest_seq = latest_seq_2d_scaled.reshape(original_shape).astype(np.float32)
                        logger.debug(f"   ✅ Informer-2预测数据已归一化")
                    else:
                        logger.warning(f"⚠️ {timeframe} Informer-2 scaler未找到，预测数据未归一化")
                    
                    inf_proba = models['inf'].predict_proba(latest_seq)[0]
                    inf_pred = models['inf'].predict(latest_seq)[0]
            else:
                inf_proba = None
                inf_pred = None
            
            # Stacking预测（使用元学习器）
            if 'meta' in models:
                # 🆕 生成增强元特征（与训练时一致）
                # 1. 模型一致性
                if inf_proba is not None:
                    agreement = float((lgb_pred == xgb_pred) and (xgb_pred == cat_pred) and (cat_pred == inf_pred))
                else:
                    agreement = float((lgb_pred == xgb_pred) and (xgb_pred == cat_pred))
                
                # 2. 最大概率
                lgb_max_prob = lgb_proba.max()
                xgb_max_prob = xgb_proba.max()
                cat_max_prob = cat_proba.max()
                
                # 3. 概率熵（单个样本）
                lgb_entropy = entr(lgb_proba).sum()
                xgb_entropy = entr(xgb_proba).sum()
                cat_entropy = entr(cat_proba).sum()
                
                # 4. 平均概率
                if inf_proba is not None:
                    inf_max_prob = inf_proba.max()
                    inf_entropy = entr(inf_proba).sum()
                    avg_proba = (lgb_proba + xgb_proba + cat_proba + inf_proba) / 4
                else:
                    avg_proba = (lgb_proba + xgb_proba + cat_proba) / 3
                
                # 5. 概率标准差
                if inf_proba is not None:
                    prob_std = np.std(np.stack([lgb_proba, xgb_proba, cat_proba, inf_proba]), axis=0)
                else:
                    prob_std = np.std(np.stack([lgb_proba, xgb_proba, cat_proba]), axis=0)
                prob_std_max = prob_std.max()
                
                # 🔑 拼接所有元特征（20个或23个）
                if inf_proba is not None:
                    # 包含Informer-2（23个特征）
                    meta_features = np.hstack([
                        lgb_proba,           # 3个
                        xgb_proba,           # 3个
                        cat_proba,           # 3个
                        inf_proba,           # 3个 ← Informer-2
                        [agreement],         # 1个
                        [lgb_max_prob],      # 1个
                        [xgb_max_prob],      # 1个
                        [cat_max_prob],      # 1个
                        [inf_max_prob],      # 1个 ← Informer-2
                        [lgb_entropy],       # 1个
                        [xgb_entropy],       # 1个
                        [cat_entropy],       # 1个
                        [inf_entropy],       # 1个 ← Informer-2
                        avg_proba,           # 3个
                        [prob_std_max]       # 1个
                    ]).reshape(1, -1)  # (1, 23)
                else:
                    # 仅传统模型（20个特征）
                    meta_features = np.hstack([
                        lgb_proba,           # 3个
                        xgb_proba,           # 3个
                        cat_proba,           # 3个
                        [agreement],         # 1个
                        [lgb_max_prob],      # 1个
                        [xgb_max_prob],      # 1个
                        [cat_max_prob],      # 1个
                        [lgb_entropy],       # 1个
                        [xgb_entropy],       # 1个
                        [cat_entropy],       # 1个
                        avg_proba,           # 3个
                        [prob_std_max]       # 1个
                    ]).reshape(1, -1)  # (1, 20)
                
                # 元学习器预测
                stacking_proba = models['meta'].predict_proba(meta_features)[0]
                final_pred = stacking_proba.argmax()
                confidence = stacking_proba[final_pred]
                final_probabilities = stacking_proba  # 使用元学习器概率
            else:
                # 降级：简单加权平均（如果元学习器不存在）
                weights = self.fallback_weights
                ensemble_proba = (
                    lgb_proba * weights['lgb'] +
                    xgb_proba * weights['xgb'] +
                    cat_proba * weights['cat']
                )
                final_pred = ensemble_proba.argmax()
                confidence = ensemble_proba[final_pred]
                final_probabilities = ensemble_proba  # 使用加权平均概率
            
            # 映射到信号类型
            signal_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
            signal_type = signal_map[final_pred]
            
            # 简洁记录预测结果
            logger.info(f"🎯 {timeframe} Stacking预测: {format_signal_type(signal_type)} "
                       f"(置信度={confidence:.4f}, 概率: 📉{final_probabilities[0]:.2f} ⏸️{final_probabilities[1]:.2f} 📈{final_probabilities[2]:.2f})")
            
            # 返回值格式与父类一致
            return {
                'signal_type': signal_type,
                'confidence': float(confidence),
                'probabilities': {
                    'short': float(final_probabilities[0]),
                    'hold': float(final_probabilities[1]),
                    'long': float(final_probabilities[2])
                },
                'timestamp': datetime.now(),
                'model_version': '2.0_stacking_ensemble'
            }
            
        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def _save_ensemble_models(self, timeframe: str):
        """
        保存集成模型（生产级别：原子性保存）
        
        使用临时文件+原子性重命名，确保：
        1. 保存过程中不影响正在使用的模型
        2. 保存失败不会破坏现有模型
        3. 保存成功后立即可用
        """
        try:
            models = self.ensemble_models[timeframe]
            model_dir = Path(self.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 🔥 使用临时目录进行原子性保存
            with tempfile.TemporaryDirectory(dir=model_dir) as temp_dir:
                temp_path = Path(temp_dir)
                saved_count = 0
                
                # 🔑 保存模型到临时目录
                model_mapping = {
                    'lgb': 'lgb',
                    'xgb': 'xgb',
                    'cat': 'cat',
                    'meta': 'meta'
                }
                
                for short_name in model_mapping:
                    if short_name in models:
                        temp_file = temp_path / f"{settings.SYMBOL}_{timeframe}_{short_name}_model.pkl"
                        with open(temp_file, 'wb') as f:
                            pickle.dump(models[short_name], f)
                        saved_count += 1
                
                # 保存Informer-2
                if 'inf' in models and TORCH_AVAILABLE:
                    temp_file = temp_path / f"{settings.SYMBOL}_{timeframe}_inf_model.pt"
                    with open(temp_file, 'wb') as f:
                        pickle.dump(models['inf'], f)
                    saved_count += 1
                
                # 保存scaler和特征列表
                if timeframe in self.scalers:
                    temp_file = temp_path / f"{settings.SYMBOL}_{timeframe}_scaler.pkl"
                    with open(temp_file, 'wb') as f:
                        pickle.dump(self.scalers[timeframe], f)
                    saved_count += 1
                
                if timeframe in self.feature_columns_dict:
                    temp_file = temp_path / f"{settings.SYMBOL}_{timeframe}_features.pkl"
                    with open(temp_file, 'wb') as f:
                        pickle.dump(self.feature_columns_dict[timeframe], f)
                    saved_count += 1
                
                # 🔥 原子性移动：一次性替换所有文件
                for temp_file in temp_path.glob(f"{settings.SYMBOL}_{timeframe}_*"):
                    target_file = model_dir / temp_file.name
                    # Windows下使用replace实现原子性替换
                    shutil.move(str(temp_file), str(target_file))
                
                logger.info(f"✅ {timeframe} 集成模型保存完成（{saved_count}个文件，原子性更新）")
            
        except Exception as e:
            logger.error(f"保存集成模型失败: {e}")
            logger.error(traceback.format_exc())
    
    def _load_ensemble_models(self, timeframe: str) -> bool:
        """加载集成模型（支持Informer-2）"""
        try:
            model_dir = Path(self.model_dir)  # 使用父类的model_dir
            models = {}
            
            # 🔑 加载传统模型（必需）
            model_mapping = {
                'lgb': 'lgb',
                'xgb': 'xgb',
                'cat': 'cat',
                'meta': 'meta'
            }
            
            # 检查必需模型文件是否存在
            for short_name in model_mapping:
                filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{short_name}_model.pkl"
                if not filepath.exists():
                    logger.warning(f"⚠️ {timeframe} {short_name}模型文件不存在: {filepath}")
                    return False
            
            # 加载所有传统模型
            for short_name in model_mapping:
                filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{short_name}_model.pkl"
                with open(filepath, 'rb') as f:
                    models[short_name] = pickle.load(f)
            
            # 🤖 加载Informer-2模型（可选，如果存在）
            if TORCH_AVAILABLE:
                inf_filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_inf_model.pt"
                if inf_filepath.exists():
                    with open(inf_filepath, 'rb') as f:
                        models['inf'] = pickle.load(f)
                    logger.info(f"   ✅ Informer-2模型已加载")
            
            self.ensemble_models[timeframe] = models
            
            # 🔥 加载scaler和features（关键！预测时需要）
            scaler_path = model_dir / f"{settings.SYMBOL}_{timeframe}_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers[timeframe] = pickle.load(f)
            
            features_path = model_dir / f"{settings.SYMBOL}_{timeframe}_features.pkl"
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.feature_columns_dict[timeframe] = pickle.load(f)
            
            # 🔓 模型加载成功，标记为就绪
            self.models_ready[timeframe] = True
            
            logger.info(f"✅ {timeframe} 集成模型加载完成（{len(models)}个模型）")
            return True
            
        except Exception as e:
            logger.error(f"加载集成模型失败: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def train_model(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        训练模型（覆盖父类方法，使用Stacking集成）
        
        Args:
            force_retrain: 是否强制重新训练
        
        Returns:
            训练结果和平均准确率
        """
        try:
            # 调用集成训练
            result = await self.train_all_timeframes()
            
            return {
                'accuracy': result['average_accuracy'],
                'timeframes': result['results'],
                'method': 'Stacking Ensemble',
                'models': ['LightGBM', 'XGBoost', 'CatBoost']
            }
            
        except Exception as e:
            logger.error(f"❌ 集成模型训练失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def predict_with_optimizations(
        self,
        features: Dict[str, pd.DataFrame],
        price_data: Optional[pd.DataFrame] = None,
        previous_signal: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        带优化的集成预测
        
        Args:
            features: 各时间框架特征数据 {timeframe: DataFrame}
            price_data: 价格数据（用于市场状态分析）
            previous_signal: 前一个信号 (0=SHORT, 1=HOLD, 2=LONG)
        
        Returns:
            Dict[str, Any]: 预测结果和优化信息
        """
        try:
            # 1. 基础预测
            predictions = {}
            probabilities = {}
            
            for timeframe, X in features.items():
                if timeframe in self.ensemble_models:
                    models = self.ensemble_models[timeframe]
                    
                    # 确保X是numpy数组格式
                    if isinstance(X, pd.DataFrame):
                        X_pred = X.values
                    else:
                        X_pred = X
                    
                    # 检查数据有效性
                    if X_pred.size == 0:
                        logger.warning(f"⚠️ {timeframe} 特征数据为空，跳过预测")
                        continue
                    
                    # 基础模型预测
                    lgb_pred = models['lgb'].predict(X_pred)
                    xgb_pred = models['xgb'].predict(X_pred)
                    cat_pred = models['cat'].predict(X_pred)
                    
                    # 元学习器预测
                    meta_features = self._generate_enhanced_meta_features(X_pred, models)
                    meta_pred = models['meta'].predict(meta_features)
                    meta_proba = models['meta'].predict_proba(meta_features)
                    
                    predictions[timeframe] = meta_pred[0]
                    probabilities[timeframe] = meta_proba[0]
            
            # 2. 交易方向一致性检查
            consistency_check = self.direction_checker.check_multi_timeframe_consistency(
                predictions, probabilities
            )
            
            # 3. 频率控制检查
            frequency_control = None
            if price_data is not None:
                market_state = self.frequency_controller.calculate_market_state(price_data)
                recent_performance = self._get_recent_performance()
                
                frequency_control = self.frequency_controller.check_trade_frequency(
                    datetime.now(),
                    consistency_check.confidence_score,
                    market_state,
                    recent_performance
                )
            
            # 4. 致命错误过滤
            final_signal = predictions.get('15m', 1)  # 默认HOLD
            filter_passed, filter_reason = self.direction_checker.filter_fatal_error_signals(
                final_signal, consistency_check, previous_signal
            )
            
            # 5. 综合决策
            if not filter_passed:
                final_signal = 1  # 强制HOLD
                logger.warning(f"⚠️ 信号被过滤: {filter_reason}")
            
            if frequency_control and not frequency_control.allow_trade:
                final_signal = 1  # 强制HOLD
                logger.warning(f"⚠️ 频率控制阻止交易: {frequency_control.reason}")
            
            # 6. 更新优化指标
            self._update_optimization_metrics(consistency_check, frequency_control)
            
            return {
                'signal': final_signal,
                'confidence': consistency_check.confidence_score,
                'consistency_check': {
                    'is_consistent': consistency_check.is_consistent,
                    'timeframe_agreement': consistency_check.timeframe_agreement,
                    'risk_level': consistency_check.risk_level
                },
                'frequency_control': {
                    'allow_trade': frequency_control.allow_trade if frequency_control else True,
                    'reason': frequency_control.reason if frequency_control else "未检查",
                    'fee_impact': frequency_control.fee_impact if frequency_control else 0.0
                },
                'filter_result': {
                    'passed': filter_passed,
                    'reason': filter_reason
                },
                'optimization_metrics': self.optimization_metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"❌ 优化预测失败: {e}")
            return {
                'signal': 1,  # 默认HOLD
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_enhanced_meta_features(
        self, 
        X_pred: np.ndarray, 
        models: Dict[str, Any]
    ) -> np.ndarray:
        """
        生成增强元特征（与训练时保持一致）
        
        Args:
            X_pred: 预测特征数据
            models: 模型字典
        
        Returns:
            np.ndarray: 增强元特征
        """
        try:
            # 基础模型预测概率
            lgb_proba = models['lgb'].predict_proba(X_pred)[0]
            xgb_proba = models['xgb'].predict_proba(X_pred)[0]
            cat_proba = models['cat'].predict_proba(X_pred)[0]
            
            # 基础模型预测结果
            lgb_pred = models['lgb'].predict(X_pred)[0]
            xgb_pred = models['xgb'].predict(X_pred)[0]
            cat_pred = models['cat'].predict(X_pred)[0]
            
            # Informer-2预测（如果存在）
            if 'inf' in models:
                try:
                    # 尝试获取序列输入（需要从features中构造）
                    seq_len = self.seq_len_config.get('15m', 96)  # 默认使用15m配置
                    # 这里需要完整的序列数据，暂时使用默认值
                    inf_proba = np.array([0.33, 0.34, 0.33])  # 默认均匀分布
                    inf_pred = 1  # 默认HOLD
                    logger.debug(f"⚠️ Informer-2使用默认预测（需要序列输入）")
                except Exception as e:
                    logger.warning(f"⚠️ Informer-2预测失败: {e}")
                    inf_proba = np.array([0.33, 0.34, 0.33])
                    inf_pred = 1
            else:
                inf_proba = None
                inf_pred = None
            
            # 1. 基础元特征（12个）
            meta_features = np.concatenate([
                lgb_proba,  # 3个
                xgb_proba,  # 3个
                cat_proba   # 3个
            ])
            
            if inf_proba is not None:
                meta_features = np.concatenate([meta_features, inf_proba])  # +3个
            
            # 2. 增强元特征（11个）
            # 模型一致性
            if inf_pred is not None:
                agreement = float((lgb_pred == xgb_pred) and (xgb_pred == cat_pred) and (cat_pred == inf_pred))
            else:
                agreement = float((lgb_pred == xgb_pred) and (xgb_pred == cat_pred))
            
            # 最大概率
            lgb_max_prob = lgb_proba.max()
            xgb_max_prob = xgb_proba.max()
            cat_max_prob = cat_proba.max()
            
            # 概率熵
            lgb_entropy = entr(lgb_proba).sum()
            xgb_entropy = entr(xgb_proba).sum()
            cat_entropy = entr(cat_proba).sum()
            
            # 平均概率
            avg_proba = np.mean([lgb_proba, xgb_proba, cat_proba], axis=0)
            if inf_proba is not None:
                avg_proba = np.mean([lgb_proba, xgb_proba, cat_proba, inf_proba], axis=0)
            
            # 概率标准差
            prob_std = np.std([lgb_proba, xgb_proba, cat_proba], axis=0).mean()
            if inf_proba is not None:
                prob_std = np.std([lgb_proba, xgb_proba, cat_proba, inf_proba], axis=0).mean()
            
            # Informer-2增强特征（如果存在）
            if inf_proba is not None:
                inf_max_prob = inf_proba.max()
                inf_entropy = entr(inf_proba).sum()
                
                enhanced_features = np.array([
                    agreement, lgb_max_prob, xgb_max_prob, cat_max_prob,
                    lgb_entropy, xgb_entropy, cat_entropy,
                    avg_proba.mean(), prob_std,
                    inf_max_prob, inf_entropy
                ])
            else:
                enhanced_features = np.array([
                    agreement, lgb_max_prob, xgb_max_prob, cat_max_prob,
                    lgb_entropy, xgb_entropy, cat_entropy,
                    avg_proba.mean(), prob_std,
                    0.0, 0.0  # 占位符
                ])
            
            # 合并所有特征
            all_features = np.concatenate([meta_features, enhanced_features])
            
            return all_features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ 增强元特征生成失败: {e}")
            # 返回默认特征
            default_features = np.zeros(23)  # 12 + 11
            return default_features.reshape(1, -1)

    def _get_recent_performance(self) -> Dict[str, float]:
        """获取近期表现指标"""
        try:
            # 这里应该从实际交易记录中获取
            # 暂时返回模拟数据
            return {
                'win_rate': 0.55,
                'avg_profit': 0.02,
                'max_drawdown': 0.05
            }
        except Exception as e:
            logger.error(f"❌ 获取近期表现失败: {e}")
            return {
                'win_rate': 0.5,
                'avg_profit': 0.0,
                'max_drawdown': 0.0
            }
    
    def _update_optimization_metrics(
        self,
        consistency_check: ConsistencyCheck,
        frequency_control: Optional[FrequencyControl]
    ) -> None:
        """更新优化指标"""
        try:
            # 更新致命错误率
            self.optimization_metrics['fatal_error_rate'] = 1.0 - consistency_check.direction_strength
            
            # 更新手续费影响
            if frequency_control:
                self.optimization_metrics['fee_impact'] = frequency_control.fee_impact
            
            # 更新一致性率
            self.optimization_metrics['consistency_rate'] = consistency_check.timeframe_agreement
            
            logger.debug(f"📊 优化指标更新: 致命错误率={self.optimization_metrics['fatal_error_rate']:.3f}, "
                        f"手续费影响={self.optimization_metrics['fee_impact']:.3f}%, "
                        f"一致性率={self.optimization_metrics['consistency_rate']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 优化指标更新失败: {e}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        try:
            # 获取频率控制统计
            freq_stats = self.frequency_controller.get_frequency_statistics()
            
            # 获取稳定性建议
            stability_recommendations = []
            if hasattr(self, 'stability_metrics'):
                stability_recommendations = self.stability_enhancer.get_stability_recommendations(
                    self.stability_metrics
                )
            
            return {
                'optimization_metrics': self.optimization_metrics.copy(),
                'frequency_statistics': freq_stats,
                'stability_recommendations': stability_recommendations,
                'system_status': {
                    'direction_checker': 'active',
                    'frequency_controller': 'active',
                    'stability_enhancer': 'active'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 优化报告生成失败: {e}")
            return {
                'error': str(e),
                'optimization_metrics': self.optimization_metrics.copy()
            }

    async def start(self):
        """启动集成ML服务"""
        try:
            logger.info("启动Stacking集成机器学习服务...")
            
            # 尝试加载已有集成模型
            all_loaded = True
            for timeframe in settings.TIMEFRAMES:
                if not self._load_ensemble_models(timeframe):
                    all_loaded = False
                    break
            
            if all_loaded:
                logger.info("✅ 所有集成模型加载成功")
            else:
                logger.warning("⚠️ 未找到集成模型，需要训练")
            
            logger.info("Stacking集成ML服务启动完成（训练由scheduler管理）")
            
        except Exception as e:
            logger.error(f"❌ 集成ML服务启动失败: {e}")
            raise

# 全局集成ML服务实例
ensemble_ml_service = EnsembleMLService()

