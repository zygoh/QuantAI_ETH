"""
集成机器学习服务 - Stacking三模型融合
"""
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import pickle
from pathlib import Path

from app.services.ml_service import MLService
from app.core.config import settings
from app.core.cache import cache_manager
from app.services.hyperparameter_optimizer import HyperparameterOptimizer

logger = logging.getLogger(__name__)

# 深度学习模型（PyTorch）
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from app.services.informer2_model import Informer2ForClassification
    from app.services.gmadl_loss import GMADLossWithHOLDPenalty
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch已加载，Informer-2模型可用")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch未安装，Informer-2模型将不可用")

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
        self.informer_n_trials = 50  # Informer-2试验次数（减少以控制时间）
        self.optuna_timeout = 1800  # 超时30分钟（GPU加速下足够优化3个模型）
        self.informer_timeout = 1200  # Informer-2超时20分钟
        
        # 🤖 Informer-2深度学习配置
        self.enable_informer2 = True  # ✅ 已启用（Phase 3 - 神经网络）
        self.informer_d_model = 128  # 模型维度
        self.informer_n_heads = 8  # 注意力头数
        self.informer_n_layers = 3  # Encoder层数
        self.informer_epochs = 50  # 训练轮数（GPU加速）
        self.informer_batch_size = 256  # 批次大小
        self.informer_lr = 0.001  # 学习率
        
        # 🎮 GPU配置（从config读取）
        self.use_gpu = settings.USE_GPU
        self.gpu_device = settings.GPU_DEVICE
        
        logger.info("✅ 集成ML服务初始化完成（Stacking四模型融合 + 深度学习）")
        logger.info(f"   超参数优化: {'启用' if self.enable_hyperparameter_tuning else '关闭'}")
        logger.info(f"   Informer-2神经网络: {'启用' if self.enable_informer2 else '关闭'}")
        logger.info(f"   GPU加速: {'启用' if self.use_gpu else '关闭'} (设备: {self.gpu_device if self.use_gpu else 'CPU'})")
    
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
            from app.services.binance_client import binance_client
            
            symbol = settings.SYMBOL
            
            # 🔑 基础训练天数配置（2h/4h增加数据量防过拟合）
            base_days_config = {
                '15m': 360,  # 保持360天
                '2h': 540,   # 270→540（翻倍）
                '4h': 720    # 360→720（翻倍）
            }
            base_days = base_days_config.get(timeframe, 360)
            
            # 应用倍数
            training_days = int(base_days * days_multiplier)
            
            # 计算需要的K线数量
            interval_minutes = {
                '15m': 15, '2h': 120, '4h': 240
            }
            minutes = interval_minutes.get(timeframe, 60)
            required_klines = int((training_days * 24 * 60) / minutes)
            
            logger.info(f"📥 获取{timeframe}数据（×{days_multiplier}倍）: {required_klines}条K线 ({training_days}天)")
            
            # 分批获取
            all_klines = []
            batch_size = 1500
            batches_needed = (required_klines + batch_size - 1) // batch_size
            
            end_time = None
            for batch in range(batches_needed):
                remaining = required_klines - len(all_klines)
                batch_limit = min(batch_size, remaining)
                
                klines = binance_client.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=batch_limit,
                    end_time=end_time
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                if len(klines) < batch_limit:
                    break
                
                end_time = klines[0]['timestamp'] - 1
            
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
            import traceback
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
            
            logger.info(f"📊 {timeframe} 数据分割: 训练{len(X_lgb_train)}条（对齐后）, 验证{len(X_lgb_val)}条")
            
            # 4️⃣ 训练Stacking集成模型（使用差异化数据）
            logger.info(f"🚂 开始训练 {timeframe} Stacking集成（差异化数据）...")
            ensemble_result = self._train_stacking_diverse(
                X_lgb_train, y_lgb_train, X_lgb_val, y_lgb_val,
                X_xgb_train, y_xgb_train, X_xgb_val, y_xgb_val,
                X_cat_train, y_cat_train, X_cat_val, y_cat_val,
                timeframe
            )
            
            # 8️⃣ 保存集成模型
            self._save_ensemble_models(timeframe)
            
            logger.info(f"⏱️ {timeframe} 训练耗时: {ensemble_result['training_time']:.2f}秒")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"❌ {timeframe} 集成模型训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _train_stacking_diverse(
        self,
        X_lgb_train, y_lgb_train, X_lgb_val, y_lgb_val,
        X_xgb_train, y_xgb_train, X_xgb_val, y_xgb_val,
        X_cat_train, y_cat_train, X_cat_val, y_cat_val,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        使用差异化数据训练Stacking集成模型
        
        Args:
            X_lgb_train, y_lgb_train: LightGBM训练数据
            X_lgb_val, y_lgb_val: LightGBM验证数据
            X_xgb_train, y_xgb_train: XGBoost训练数据
            X_xgb_val, y_xgb_val: XGBoost验证数据
            X_cat_train, y_cat_train: CatBoost训练数据
            X_cat_val, y_cat_val: CatBoost验证数据
            timeframe: 时间框架
        
        Returns:
            训练结果字典
        """
        import time
        start_time = time.time()
        
        try:
            # 🔧 Optuna超参数优化（如果启用）
            lgb_params_optimized = None
            xgb_params_optimized = None
            cat_params_optimized = None
            inf_params_optimized = None
            
            if self.enable_hyperparameter_tuning:
                if self.optimize_all_models:
                    logger.info(f"🔧 启动超参数自动优化（Optuna）- 优化全部3个传统模型...")
                else:
                    logger.info(f"🔧 启动超参数自动优化（Optuna）- 仅优化LightGBM...")
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
            
            # 🤖 Informer-2超参数优化（如果启用）
            if self.enable_informer2 and self.optimize_informer2 and TORCH_AVAILABLE:
                logger.info(f"🤖 启动Informer-2超参数优化（深度学习）...")
                logger.info(f"   GPU加速: {'启用' if self.use_gpu else '关闭'}")
                logger.info(f"   试验次数: {self.informer_n_trials}次, 超时: {self.informer_timeout}秒")
                
                inf_optimizer = HyperparameterOptimizer(
                    X=X_lgb_train.values if isinstance(X_lgb_train, pd.DataFrame) else X_lgb_train,
                    y=y_lgb_train,
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
            
            # 1️⃣ 训练四个基础模型（各用自己的数据）
            logger.info(f"🚂 训练LightGBM（360天数据）...")
            lgb_model = self._train_lightgbm(X_lgb_train, y_lgb_train, timeframe, custom_params=lgb_params_optimized)
            
            logger.info(f"🚂 训练XGBoost（540天数据）...")
            xgb_model = self._train_xgboost(X_xgb_train, y_xgb_train, timeframe, custom_params=xgb_params_optimized)
            
            logger.info(f"🚂 训练CatBoost（720天数据）...")
            cat_model = self._train_catboost(X_cat_train, y_cat_train, timeframe, custom_params=cat_params_optimized)
            
            # 🤖 训练Informer-2（深度学习 + GMADL损失）
            inf_model = None
            if self.enable_informer2 and TORCH_AVAILABLE:
                logger.info(f"🤖 训练Informer-2（深度学习 + GMADL损失）...")
                inf_model = self._train_informer2(X_lgb_train, y_lgb_train, timeframe, custom_params=inf_params_optimized)
            
            # 2️⃣ 生成验证集的预测概率（元特征）
            logger.info(f"📊 生成元特征（基于对齐的验证集）...")
            
            # 使用各自的验证集生成预测
            lgb_pred_proba = lgb_model.predict_proba(X_lgb_val)
            xgb_pred_proba = xgb_model.predict_proba(X_xgb_val)
            cat_pred_proba = cat_model.predict_proba(X_cat_val)
            
            # Informer-2预测（如果启用）
            if inf_model is not None:
                inf_pred_proba = inf_model.predict_proba(X_lgb_val)
            
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
            from scipy.special import entr
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
                # 包含Informer-2（23个特征）
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
                expected_features = 23  # 3+3+3+3+1+1+1+1+1+1+1+1+1+3+1
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
                logger.info(f"✅ 增强元特征生成完成: {meta_features_val.shape} (基础12+增强11=23个，含Informer-2)")
            else:
                logger.info(f"✅ 增强元特征生成完成: {meta_features_val.shape} (基础9+增强11=20个)")
            
            # 3️⃣ 训练元学习器（Stacking） - 升级为LightGBM + 动态HOLD惩罚
            logger.info(f"🧠 训练元学习器（LightGBM - 更强大的决策能力）...")
            
            # 🔑 检查HOLD比例，动态调整惩罚系数
            from sklearn.utils.class_weight import compute_sample_weight
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
            meta_hold_penalty = np.where(meta_labels_val == 1, meta_hold_penalty_weight, 1.0)
            meta_sample_weights = meta_class_weights * meta_hold_penalty
            
            import lightgbm as lgb
            # 🔑 元学习器：极简配置防止过拟合
            meta_learner = lgb.LGBMClassifier(
                n_estimators=50,     # 减少树数量 100→50
                max_depth=3,         # 更浅的树 4→3
                learning_rate=0.15,  # 提高学习率 0.1→0.15（少量树）
                num_leaves=7,        # 大幅减少叶子 15→7
                min_child_samples=30,  # 增加最小样本 20→30
                subsample=0.7,       # 降低采样 0.8→0.7
                colsample_bytree=0.7,  # 降低特征采样 0.8→0.7
                reg_alpha=0.3,       # 加强L1正则 0.1→0.3
                reg_lambda=0.3,      # 加强L2正则 0.1→0.3
                random_state=42,
                verbose=-1
            )
            meta_learner.fit(meta_features_val, meta_labels_val, sample_weight=meta_sample_weights)
            
            logger.info(f"✅ 元学习器训练完成（动态HOLD惩罚={meta_hold_penalty_weight}）")
            
            # 4️⃣ 保存模型到字典
            if timeframe not in self.ensemble_models:
                self.ensemble_models[timeframe] = {}
            
            self.ensemble_models[timeframe]['lightgbm'] = lgb_model
            self.ensemble_models[timeframe]['xgboost'] = xgb_model
            self.ensemble_models[timeframe]['catboost'] = cat_model
            self.ensemble_models[timeframe]['meta_learner'] = meta_learner
            
            # 5️⃣ 评估集成模型 - 使用时间序列交叉验证
            logger.info(f"📊 {timeframe} 时间序列交叉验证评估...")
            
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
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
            accuracy = accuracy_score(meta_labels_val, ensemble_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                meta_labels_val, ensemble_pred, average='weighted', zero_division=0
            )
            
            logger.info(f"📊 {timeframe} 最终模型验证集准确率: {accuracy:.4f} (CV: {cv_mean:.4f}±{cv_std:.4f})")
            
            # 6️⃣ 评估各基础模型
            lgb_pred = lgb_model.predict(X_lgb_val)
            xgb_pred = xgb_model.predict(X_xgb_val)
            cat_pred = cat_model.predict(X_cat_val)
            
            lgb_acc = accuracy_score(y_lgb_val, lgb_pred)
            xgb_acc = accuracy_score(y_xgb_val, xgb_pred)
            cat_acc = accuracy_score(y_cat_val, cat_pred)
            
            # Informer-2准确率（如果存在）
            if inf_model is not None:
                inf_pred = inf_model.predict(X_lgb_val)
                inf_acc = accuracy_score(y_lgb_val, inf_pred)
            else:
                inf_acc = 0.0
            
            training_time = time.time() - start_time
            
            result = {
                'accuracy': cv_mean,  # 🔑 使用CV均值作为主准确率（更可靠）
                'cv_mean': cv_mean,   # 交叉验证均值
                'cv_std': cv_std,     # 交叉验证标准差
                'cv_scores': cv_scores,  # 各折分数
                'val_accuracy': accuracy,  # 验证集准确率
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'lgb_accuracy': lgb_acc,
                'xgb_accuracy': xgb_acc,
                'cat_accuracy': cat_acc,
                'inf_accuracy': inf_acc if inf_model else 0.0,
                'training_time': training_time,
                'ensemble_size': len(self.ensemble_models[timeframe]),
                'meta_features_count': meta_features_val.shape[1]  # 元特征数量
            }
            
            logger.info(f"✅ Stacking训练完成（差异化数据）:")
            logger.info(f"  LightGBM(360天): {lgb_acc:.4f}")
            logger.info(f"  XGBoost(540天):  {xgb_acc:.4f}")
            logger.info(f"  CatBoost(720天): {cat_acc:.4f}")
            if inf_model:
                logger.info(f"  Informer-2:      {inf_acc:.4f} 🤖")
            logger.info(f"  Stacking验证集:  {accuracy:.4f}")
            logger.info(f"  🎯 时间序列CV:  {cv_mean:.4f} ± {cv_std:.4f} (5-fold)")
            n_base = 12 if inf_model else 9
            n_enhanced = 11 if not inf_model else 11
            logger.info(f"  📊 元特征: {meta_features_val.shape[1]}个（基础{n_base}+增强{n_enhanced}）")
            logger.info(f"  训练耗时: {training_time:.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"差异化Stacking训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            import lightgbm as lgb
            from sklearn.utils.class_weight import compute_sample_weight
            
            # 样本加权（类别平衡 × 时间衰减 × HOLD惩罚）
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（适度惩罚策略）
            hold_penalty = np.where(y_train == 1, 0.65, 1.0)  # HOLD权重0.65（适度惩罚 0.5→0.65）
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            logger.info(f"✅ 样本加权已启用：类别平衡 × 时间衰减 × HOLD惩罚(0.65)")
            
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
            
            return model
            
        except Exception as e:
            logger.error(f"LightGBM训练失败: {e}")
            raise
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str, custom_params: Optional[Dict[str, Any]] = None):
        """训练XGBoost模型（防过拟合）"""
        try:
            import xgboost as xgb
            
            # 样本加权（与LightGBM一致 + HOLD惩罚）
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（适度惩罚，与LightGBM一致）
            hold_penalty = np.where(y_train == 1, 0.65, 1.0)
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            # 🔑 时间框架差异化配置（防止2h/4h过拟合）
            if custom_params:
                # 使用Optuna优化的参数
                params = custom_params.copy()
                logger.info(f"🎯 使用Optuna优化参数")
            else:
                # 使用默认参数
                if timeframe == '15m':
                    params = {
                        'max_depth': 6,
                        'learning_rate': 0.05,
                        'n_estimators': 300,
                        'reg_alpha': 0.3,
                        'reg_lambda': 0.3
                    }
                elif timeframe == '2h':
                    params = {
                        'max_depth': 4,  # 6→4（简化）
                        'learning_rate': 0.08,  # 0.05→0.08（少量树）
                        'n_estimators': 150,  # 300→150（减半）
                        'reg_alpha': 0.8,  # 加强正则化
                        'reg_lambda': 0.8
                    }
                else:  # 4h
                    params = {
                        'max_depth': 3,  # 6→3（极简）
                        'learning_rate': 0.1,  # 0.05→0.1
                        'n_estimators': 100,  # 300→100（大幅减少）
                        'reg_alpha': 1.0,  # 极强正则化
                        'reg_lambda': 1.0
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
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
                logger.info(f"🚀 XGBoost GPU加速已启用")
            else:
                params['tree_method'] = 'hist'
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            return model
            
        except Exception as e:
            logger.error(f"XGBoost训练失败: {e}")
            raise
    
    def _train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str, custom_params: Optional[Dict[str, Any]] = None):
        """训练CatBoost模型（防过拟合）"""
        try:
            from catboost import CatBoostClassifier
            
            # 样本加权（与LightGBM一致 + HOLD惩罚）
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（适度惩罚，与LightGBM一致）
            hold_penalty = np.where(y_train == 1, 0.65, 1.0)
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            # 🔑 时间框架差异化配置（防止2h/4h过拟合）
            if custom_params:
                # 使用Optuna优化的参数
                params = custom_params.copy()
                logger.info(f"🎯 使用Optuna优化参数")
            else:
                # 使用默认参数
                if timeframe == '15m':
                    params = {
                        'iterations': 300,
                        'learning_rate': 0.05,
                        'depth': 6,
                        'l2_leaf_reg': 3.0
                    }
                elif timeframe == '2h':
                    params = {
                        'iterations': 150,  # 300→150（减半）
                        'learning_rate': 0.08,
                        'depth': 4,  # 6→4（简化）
                        'l2_leaf_reg': 5.0  # 3.0→5.0（加强正则）
                    }
                else:  # 4h
                    params = {
                        'iterations': 100,  # 300→100（大幅减少）
                        'learning_rate': 0.1,
                        'depth': 3,  # 6→3（极简）
                        'l2_leaf_reg': 8.0  # 3.0→8.0（极强正则）
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
            
            return model
            
        except Exception as e:
            logger.error(f"CatBoost训练失败: {e}")
            raise
    
    def _train_informer2(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str, custom_params: Optional[Dict[str, Any]] = None):
        """
        训练Informer-2深度学习模型（使用GMADL损失函数）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            timeframe: 时间框架
            custom_params: 自定义参数（来自Optuna优化）
        
        Returns:
            训练好的Informer-2模型（兼容scikit-learn接口）
        """
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch未安装，跳过Informer-2训练")
            return None
        
        try:
            import time
            start_time = time.time()
            
            logger.info(f"🤖 训练Informer-2神经网络模型...")
            
            # 1. 数据准备（Pandas → PyTorch）
            X_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            y_np = y_train.values if isinstance(y_train, pd.Series) else y_train
            
            X_tensor = torch.FloatTensor(X_np)
            y_tensor = torch.LongTensor(y_np)
            
            # 2. 检测GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"   设备: {device} {'🚀 (GPU加速)' if device.type == 'cuda' else '💻 (CPU)'}")
            
            # 3. 创建数据加载器
            dataset = TensorDataset(X_tensor, y_tensor)
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
                alpha = custom_params.get('alpha', 1.0)
                beta = custom_params.get('beta', 0.5)
                logger.info(f"🎯 使用优化参数: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, epochs={epochs}")
            else:
                d_model = self.informer_d_model
                n_heads = self.informer_n_heads
                n_layers = self.informer_n_layers
                dropout = 0.1
                epochs = self.informer_epochs
                batch_size = self.informer_batch_size
                lr = self.informer_lr
                alpha = 1.0
                beta = 0.5
            
            # 5. 初始化模型（修复参数名）
            model = Informer2ForClassification(
                n_features=X_np.shape[1],  # 特征数量
                n_classes=3,  # 类别数
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                use_distilling=True
            ).to(device)
            
            # 6. 定义GMADL损失函数（关键创新！）
            criterion = GMADLossWithHOLDPenalty(
                hold_penalty=0.65,  # 与其他模型保持一致
                alpha=alpha,  # 鲁棒性参数
                beta=beta    # 凸性参数（论文推荐）
            )
            
            # 7. 定义优化器
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=1e-5  # L2正则化
            )
            
            # 8. 学习率调度器（余弦退火）
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
            
            # 9. 训练循环
            model.train()
            best_loss = float('inf')
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    # 前向传播
                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 统计
                    epoch_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                # 更新学习率
                scheduler.step()
                
                # 计算准确率
                epoch_loss /= len(dataloader)
                epoch_acc = 100.0 * correct / total
                
                # 每10轮或最后1轮打印进度
                if (epoch + 1) % 10 == 0 or epoch == self.informer_epochs - 1:
                    logger.info(f"   Epoch [{epoch+1}/{self.informer_epochs}] "
                               f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
                
                # 保存最佳模型
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
            
            # 9. 切换到评估模式
            model.eval()
            
            # 10. 包装模型以兼容scikit-learn接口
            class InformerWrapper:
                """包装Informer-2模型，提供predict_proba接口"""
                
                def __init__(self, model, device):
                    self.model = model
                    self.device = device
                
                def predict_proba(self, X):
                    """
                    预测概率（兼容scikit-learn）
                    
                    Args:
                        X: NumPy数组或Pandas DataFrame
                    
                    Returns:
                        概率数组 (n_samples, n_classes)
                    """
                    self.model.eval()
                    with torch.no_grad():
                        if isinstance(X, pd.DataFrame):
                            X = X.values
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        probs = self.model.predict_proba(X_tensor)
                        return probs.cpu().numpy()
                
                def predict(self, X):
                    """
                    预测类别（兼容scikit-learn）
                    
                    Args:
                        X: NumPy数组或Pandas DataFrame
                    
                    Returns:
                        预测类别数组
                    """
                    probs = self.predict_proba(X)
                    return np.argmax(probs, axis=1)
            
            wrapped_model = InformerWrapper(model, device)
            
            training_time = time.time() - start_time
            logger.info(f"✅ Informer-2训练完成: 最佳Loss={best_loss:.4f}, "
                       f"耗时={training_time:.2f}秒")
            
            return wrapped_model
            
        except Exception as e:
            logger.error(f"Informer-2训练失败: {e}")
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
            预测结果
        """
        try:
            # 检查集成模型是否存在
            if timeframe not in self.ensemble_models:
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
            
            # 🤖 Informer-2预测（如果存在）
            if 'inf' in models:
                inf_proba = models['inf'].predict_proba(X_pred)[0]
                inf_pred = models['inf'].predict(X_pred)[0]
            else:
                inf_proba = None
                inf_pred = None
            
            # Stacking预测（使用元学习器）
            if 'meta' in models:
                # 🆕 生成增强元特征（与训练时一致）
                from scipy.special import entr
                
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
            from app.utils.helpers import format_signal_type
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
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _save_ensemble_models(self, timeframe: str):
        """保存集成模型"""
        try:
            models = self.ensemble_models[timeframe]
            model_dir = Path(self.model_dir)  # 使用父类的model_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 🔑 保存模型（支持Informer-2）
            model_mapping = {
                'lgb': 'lgb',
                'xgb': 'xgb',
                'cat': 'cat',
                'meta': 'meta'
            }
            
            saved_count = 0
            for short_name in model_mapping:
                if short_name in models:
                    filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{short_name}_model.pkl"
                    with open(filepath, 'wb') as f:
                        pickle.dump(models[short_name], f)
                    saved_count += 1
            
            # 保存Informer-2（PyTorch模型，使用torch.save）
            if 'inf' in models and TORCH_AVAILABLE:
                inf_filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_inf_model.pt"
                # 保存整个wrapper对象（包含模型和device）
                with open(inf_filepath, 'wb') as f:
                    pickle.dump(models['inf'], f)
                saved_count += 1
                logger.info(f"   ✅ Informer-2模型已保存: {inf_filepath.name}")
            
            # 🔥 保存scaler和features（关键！预测时需要）
            if timeframe in self.scalers:
                scaler_path = model_dir / f"{settings.SYMBOL}_{timeframe}_scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[timeframe], f)
                saved_count += 1
            
            if timeframe in self.feature_columns_dict:
                features_path = model_dir / f"{settings.SYMBOL}_{timeframe}_features.pkl"
                with open(features_path, 'wb') as f:
                    pickle.dump(self.feature_columns_dict[timeframe], f)
                saved_count += 1
            
            if saved_count > 0:
                logger.info(f"✅ {timeframe} 集成模型保存完成（{saved_count}个文件）")
            else:
                logger.warning(f"⚠️ {timeframe} 没有模型被保存（键名: {list(models.keys())}）")
            
        except Exception as e:
            logger.error(f"保存集成模型失败: {e}")
            import traceback
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
            
            logger.info(f"✅ {timeframe} 集成模型加载完成（{len(models)}个模型）")
            return True
            
        except Exception as e:
            logger.error(f"加载集成模型失败: {e}")
            import traceback
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
            import traceback
            logger.error(traceback.format_exc())
            raise
    
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

