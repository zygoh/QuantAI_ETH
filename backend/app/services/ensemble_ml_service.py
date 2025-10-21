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

logger = logging.getLogger(__name__)

class EnsembleMLService(MLService):
    """
    集成机器学习服务（Stacking）
    
    使用LightGBM + XGBoost + CatBoost三模型Stacking融合
    目标：准确率从42.81%提升到50%+
    """
    
    def __init__(self):
        super().__init__()
        
        # 集成模型字典 {timeframe: {lgb, xgb, cat, meta}}
        self.ensemble_models = {}
        
        # 集成权重（Stacking自动学习，这里作为降级方案）
        self.fallback_weights = {
            'lgb': 0.4,
            'xgb': 0.3,
            'cat': 0.3
        }
        
        logger.info("✅ 集成ML服务初始化完成（Stacking三模型融合）")
    
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
            
            # 基础训练天数配置
            base_days_config = {
                '15m': 360,
                '2h': 270,
                '4h': 360
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
            # 1️⃣ 训练三个基础模型（各用自己的数据）
            logger.info(f"🚂 训练LightGBM（360天数据）...")
            lgb_model = self._train_lightgbm(X_lgb_train, y_lgb_train, timeframe)
            
            logger.info(f"🚂 训练XGBoost（540天数据）...")
            xgb_model = self._train_xgboost(X_xgb_train, y_xgb_train, timeframe)
            
            logger.info(f"🚂 训练CatBoost（720天数据）...")
            cat_model = self._train_catboost(X_cat_train, y_cat_train, timeframe)
            
            # 2️⃣ 生成验证集的预测概率（元特征）
            logger.info(f"📊 生成元特征（基于对齐的验证集）...")
            
            # 使用各自的验证集生成预测
            lgb_pred_proba = lgb_model.predict_proba(X_lgb_val)
            xgb_pred_proba = xgb_model.predict_proba(X_xgb_val)
            cat_pred_proba = cat_model.predict_proba(X_cat_val)
            
            # 拼接元特征（三个模型的预测概率）
            meta_features_val = np.hstack([
                lgb_pred_proba,
                xgb_pred_proba,
                cat_pred_proba
            ])
            
            # 元标签（使用LightGBM的y_val，因为验证集已对齐）
            meta_labels_val = y_lgb_val
            
            logger.info(f"✅ 元特征生成完成: shape={meta_features_val.shape}")
            
            # 3️⃣ 训练元学习器（Stacking） - 升级为LightGBM + HOLD惩罚
            logger.info(f"🧠 训练元学习器（LightGBM - 更强大的决策能力）...")
            
            # 🔑 元学习器也需要HOLD惩罚（关键修复！）
            from sklearn.utils.class_weight import compute_sample_weight
            meta_class_weights = compute_sample_weight('balanced', meta_labels_val)
            meta_hold_penalty = np.where(meta_labels_val == 1, 0.6, 1.0)  # 元学习器HOLD惩罚更重（0.6，更平衡）
            meta_sample_weights = meta_class_weights * meta_hold_penalty
            
            import lightgbm as lgb
            meta_learner = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,  # 浅层树，避免过拟合
                learning_rate=0.1,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            )
            meta_learner.fit(meta_features_val, meta_labels_val, sample_weight=meta_sample_weights)
            
            logger.info(f"✅ 元学习器训练完成（已应用HOLD惩罚0.6，更平衡）")
            
            # 4️⃣ 保存模型到字典
            if timeframe not in self.ensemble_models:
                self.ensemble_models[timeframe] = {}
            
            self.ensemble_models[timeframe]['lightgbm'] = lgb_model
            self.ensemble_models[timeframe]['xgboost'] = xgb_model
            self.ensemble_models[timeframe]['catboost'] = cat_model
            self.ensemble_models[timeframe]['meta_learner'] = meta_learner
            
            # 5️⃣ 评估集成模型
            ensemble_pred = meta_learner.predict(meta_features_val)
            
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(meta_labels_val, ensemble_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                meta_labels_val, ensemble_pred, average='weighted', zero_division=0
            )
            
            # 6️⃣ 评估各基础模型
            lgb_pred = lgb_model.predict(X_lgb_val)
            xgb_pred = xgb_model.predict(X_xgb_val)
            cat_pred = cat_model.predict(X_cat_val)
            
            lgb_acc = accuracy_score(y_lgb_val, lgb_pred)
            xgb_acc = accuracy_score(y_xgb_val, xgb_pred)
            cat_acc = accuracy_score(y_cat_val, cat_pred)
            
            training_time = time.time() - start_time
            
            result = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'lgb_accuracy': lgb_acc,
                'xgb_accuracy': xgb_acc,
                'cat_accuracy': cat_acc,
                'training_time': training_time,
                'ensemble_size': len(self.ensemble_models[timeframe]),
                'meta_features_shape': meta_features_val.shape
            }
            
            logger.info(f"✅ Stacking训练完成（差异化数据）:")
            logger.info(f"  LightGBM(360天): {lgb_acc:.4f}")
            logger.info(f"  XGBoost(540天):  {xgb_acc:.4f}")
            logger.info(f"  CatBoost(720天): {cat_acc:.4f}")
            logger.info(f"  Stacking集成:    {accuracy:.4f}")
            logger.info(f"  训练耗时: {training_time:.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"差异化Stacking训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _train_stacking_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        训练Stacking集成模型
        
        Stacking方法:
        1. 训练基础模型（LightGBM, XGBoost, CatBoost）
        2. 使用基础模型生成元特征（预测概率）
        3. 训练元学习器（LogisticRegression）学习如何组合
        
        优势:
        - 比简单加权更智能
        - 自动学习每个模型的强项
        - 更好的泛化能力
        """
        import time
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        start_time = time.time()
        
        logger.info(f"🎯 Stage 1: 训练3个基础模型...")
        
        # 1. 训练LightGBM（基础模型1）
        logger.info(f"  📊 训练LightGBM...")
        lgb_model = self._train_lightgbm(X_train, y_train, timeframe)
        
        # 2. 训练XGBoost（基础模型2）
        logger.info(f"  📊 训练XGBoost...")
        xgb_model = self._train_xgboost(X_train, y_train, timeframe)
        
        # 3. 训练CatBoost（基础模型3）
        logger.info(f"  📊 训练CatBoost...")
        cat_model = self._train_catboost(X_train, y_train, timeframe)
        
        logger.info(f"✅ 3个基础模型训练完成")
        
        # 4. 生成元特征（训练集）
        logger.info(f"🎯 Stage 2: 生成元特征...")
        lgb_pred_train = lgb_model.predict_proba(X_train)
        xgb_pred_train = xgb_model.predict_proba(X_train)
        cat_pred_train = cat_model.predict_proba(X_train)
        
        # 合并元特征（9维：每个模型3个类别概率）
        meta_features_train = np.hstack([
            lgb_pred_train,
            xgb_pred_train,
            cat_pred_train
        ])
        
        # 5. 训练元学习器（Stacking） - 升级为LightGBM + HOLD惩罚
        logger.info(f"🎯 Stage 3: 训练元学习器（LightGBM - 更强大的决策能力）...")
        
        # 🔑 元学习器也需要HOLD惩罚（关键修复！）
        from sklearn.utils.class_weight import compute_sample_weight
        meta_class_weights = compute_sample_weight('balanced', y_train)
        meta_hold_penalty = np.where(y_train == 1, 0.6, 1.0)  # 元学习器HOLD惩罚更重（0.6，更平衡）
        meta_sample_weights = meta_class_weights * meta_hold_penalty
        
        import lightgbm as lgb
        meta_learner = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,  # 浅层树，避免过拟合
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        meta_learner.fit(meta_features_train, y_train, sample_weight=meta_sample_weights)
        
        logger.info(f"✅ 元学习器训练完成（已应用HOLD惩罚0.6，更平衡）")
        
        # 6. 验证集评估
        logger.info(f"🎯 Stage 4: 验证集评估...")
        
        # 生成验证集元特征
        lgb_pred_val = lgb_model.predict_proba(X_val)
        xgb_pred_val = xgb_model.predict_proba(X_val)
        cat_pred_val = cat_model.predict_proba(X_val)
        
        meta_features_val = np.hstack([
            lgb_pred_val,
            xgb_pred_val,
            cat_pred_val
        ])
        
        # Stacking预测
        stacking_pred = meta_learner.predict(meta_features_val)
        stacking_proba = meta_learner.predict_proba(meta_features_val)
        
        # 计算指标
        accuracy = accuracy_score(y_val, stacking_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, stacking_pred, average='weighted', zero_division=0
        )
        
        # 计算AUC（多分类使用OvR）
        try:
            auc = roc_auc_score(y_val, stacking_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.5
        
        # 各基础模型单独准确率
        lgb_acc = accuracy_score(y_val, lgb_model.predict(X_val))
        xgb_acc = accuracy_score(y_val, xgb_model.predict(X_val))
        cat_acc = accuracy_score(y_val, cat_model.predict(X_val))
        
        training_time = time.time() - start_time
        
        # 保存集成模型
        self.ensemble_models[timeframe] = {
            'lgb': lgb_model,
            'xgb': xgb_model,
            'cat': cat_model,
            'meta': meta_learner
        }
        
        # 日志输出
        logger.info(f"📊 {timeframe} Stacking集成评估:")
        logger.info(f"  基础模型准确率:")
        logger.info(f"    LightGBM: {lgb_acc:.4f}")
        logger.info(f"    XGBoost:  {xgb_acc:.4f}")
        logger.info(f"    CatBoost: {cat_acc:.4f}")
        logger.info(f"  Stacking准确率: {accuracy:.4f}")
        logger.info(f"  提升: +{(accuracy - max(lgb_acc, xgb_acc, cat_acc))*100:.2f}%")
        logger.info(f"  精确率: {precision:.4f}")
        logger.info(f"  召回率: {recall:.4f}")
        logger.info(f"  F1分数: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'base_models': {
                'lgb': lgb_acc,
                'xgb': xgb_acc,
                'cat': cat_acc
            },
            'training_time': training_time
        }
    
    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str):
        """
        训练LightGBM模型（覆盖父类方法，统一三模型训练代码位置）
        
        覆盖原因：保证代码结构统一，三个模型训练都在ensemble_ml_service.py
        """
        try:
            import lightgbm as lgb
            from sklearn.utils.class_weight import compute_sample_weight
            
            # 样本加权（类别平衡 × 时间衰减 × HOLD惩罚）
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（惩罚过度保守）
            hold_penalty = np.where(y_train == 1, 0.7, 1.0)  # HOLD权重0.7，其他1.0
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            logger.info(f"✅ 样本加权已启用：类别平衡 × 时间衰减 × HOLD惩罚(0.7)")
            
            # 获取时间框架差异化参数
            timeframe_params = self.lgb_params_by_timeframe.get(timeframe, {})
            
            # 合并基础参数和差异化参数
            params = {**self.lgb_params, **timeframe_params}
            
            logger.info(f"📊 {timeframe} LightGBM参数: num_leaves={params.get('num_leaves')}, "
                       f"reg_alpha={params.get('reg_alpha', 0)}, reg_lambda={params.get('reg_lambda', 0)}")
            
            # 创建并训练模型（params中已包含random_state=42）
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            return model
            
        except Exception as e:
            logger.error(f"LightGBM训练失败: {e}")
            raise
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str):
        """训练XGBoost模型"""
        try:
            import xgboost as xgb
            
            # 样本加权（与LightGBM一致 + HOLD惩罚）
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（惩罚过度保守）
            hold_penalty = np.where(y_train == 1, 0.7, 1.0)
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'tree_method': 'hist',
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            return model
            
        except Exception as e:
            logger.error(f"XGBoost训练失败: {e}")
            raise
    
    def _train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str):
        """训练CatBoost模型"""
        try:
            from catboost import CatBoostClassifier
            
            # 样本加权（与LightGBM一致 + HOLD惩罚）
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            
            # 🔑 HOLD类别降权（惩罚过度保守）
            hold_penalty = np.where(y_train == 1, 0.7, 1.0)
            
            sample_weights = class_weights * time_decay * hold_penalty
            
            params = {
                'iterations': 300,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'MultiClass',
                'random_seed': 42,
                'verbose': False,
                'l2_leaf_reg': 3.0,
                'bootstrap_type': 'Bernoulli',  # 改用Bernoulli（支持subsample）
                'subsample': 0.8,
                'allow_writing_files': False  # 🔑 禁止生成catboost_info目录
            }
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            return model
            
        except Exception as e:
            logger.error(f"CatBoost训练失败: {e}")
            raise
    
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
            
            # 🔑 基础模型预测（使用完整键名）
            lgb_proba = models['lightgbm'].predict_proba(X_pred)[0]
            xgb_proba = models['xgboost'].predict_proba(X_pred)[0]
            cat_proba = models['catboost'].predict_proba(X_pred)[0]
            
            # Stacking预测（使用元学习器）
            if 'meta_learner' in models:
                # 生成元特征
                meta_features = np.hstack([lgb_proba, xgb_proba, cat_proba]).reshape(1, -1)
                
                # 元学习器预测
                stacking_proba = models['meta_learner'].predict_proba(meta_features)[0]
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
            
            # 🔑 保存4个模型（修复键名映射）
            model_mapping = {
                'lightgbm': 'lgb',
                'xgboost': 'xgb',
                'catboost': 'cat',
                'meta_learner': 'meta'
            }
            
            saved_count = 0
            for full_name, short_name in model_mapping.items():
                if full_name in models:
                    filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{short_name}_model.pkl"
                    with open(filepath, 'wb') as f:
                        pickle.dump(models[full_name], f)
                    saved_count += 1
            
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
        """加载集成模型"""
        try:
            model_dir = Path(self.model_dir)  # 使用父类的model_dir
            models = {}
            
            # 🔑 加载4个模型（修复键名映射）
            model_mapping = {
                'lightgbm': 'lgb',
                'xgboost': 'xgb',
                'catboost': 'cat',
                'meta_learner': 'meta'
            }
            
            # 检查所有模型文件是否存在
            for full_name, short_name in model_mapping.items():
                filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{short_name}_model.pkl"
                
                if not filepath.exists():
                    logger.warning(f"⚠️ {timeframe} {short_name}模型文件不存在: {filepath}")
                    return False
            
            # 加载所有模型
            for full_name, short_name in model_mapping.items():
                filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{short_name}_model.pkl"
                with open(filepath, 'rb') as f:
                    models[full_name] = pickle.load(f)  # 🔑 使用完整键名
            
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

