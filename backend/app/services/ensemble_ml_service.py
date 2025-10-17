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
        
        流程:
        1. 准备数据
        2. 训练基础模型（LightGBM, XGBoost, CatBoost）
        3. 生成元特征（基础模型的预测概率）
        4. 训练元学习器（Stacking）
        5. 评估集成效果
        """
        try:
            # 1️⃣ 准备训练数据
            logger.info(f"📥 获取 {timeframe} 训练数据...")
            data = await self._prepare_training_data_for_timeframe(timeframe)
            logger.info(f"✅ {timeframe} 数据获取成功: {len(data)}条")
            
            # 2️⃣ 特征工程
            data = self.feature_engineer.create_features(data)
            
            # 3️⃣ 创建标签
            data = self._create_labels(data, timeframe=timeframe)
            
            # 4️⃣ 准备特征和标签
            X, y = self._prepare_features_labels(data, timeframe)
            
            # 5️⃣ 特征缩放
            X_scaled = self._scale_features(X, timeframe, fit=True)
            
            # 6️⃣ 时间序列分割（训练集80%，验证集20%）
            split_idx = int(len(X_scaled) * 0.8)
            
            # 🔑 X_scaled是numpy数组，y是Series，分别处理
            if isinstance(X_scaled, np.ndarray):
                X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            else:
                X_train, X_val = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
            
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logger.info(f"📊 {timeframe} 数据分割: 训练{len(X_train)}条, 验证{len(X_val)}条")
            
            # 7️⃣ 训练Stacking集成模型
            logger.info(f"🚂 开始训练 {timeframe} Stacking集成...")
            ensemble_result = self._train_stacking_ensemble(
                X_train, y_train, X_val, y_val, timeframe
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
        
        # 5. 训练元学习器（Stacking）
        logger.info(f"🎯 Stage 3: 训练元学习器（Stacking）...")
        meta_learner = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        meta_learner.fit(meta_features_train, y_train)
        
        logger.info(f"✅ 元学习器训练完成")
        
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
            
            # 样本加权（类别平衡 × 时间衰减）
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            sample_weights = class_weights * time_decay
            
            # 获取时间框架差异化参数
            timeframe_params = self.lgb_params_by_timeframe.get(timeframe, {})
            
            # 合并基础参数和差异化参数
            params = {**self.base_lgb_params, **timeframe_params}
            
            logger.info(f"📊 {timeframe} LightGBM参数: num_leaves={params.get('num_leaves')}, "
                       f"reg_alpha={params.get('reg_alpha', 0)}, reg_lambda={params.get('reg_lambda', 0)}")
            
            # 创建并训练模型
            model = lgb.LGBMClassifier(**params, random_state=42)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            return model
            
        except Exception as e:
            logger.error(f"LightGBM训练失败: {e}")
            raise
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, timeframe: str):
        """训练XGBoost模型"""
        try:
            import xgboost as xgb
            
            # 样本加权（与LightGBM一致）
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            sample_weights = class_weights * time_decay
            
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
            
            # 样本加权（与LightGBM一致）
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced', y_train)
            time_decay = np.exp(-np.arange(len(X_train)) / (len(X_train) * 0.1))[::-1]
            sample_weights = class_weights * time_decay
            
            params = {
                'iterations': 300,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'MultiClass',
                'random_seed': 42,
                'verbose': False,
                'l2_leaf_reg': 3.0,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.0,
                'subsample': 0.8
            }
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            return model
            
        except Exception as e:
            logger.error(f"CatBoost训练失败: {e}")
            raise
    
    async def predict(
        self, 
        symbol: str, 
        timeframe: str, 
        use_stacking: bool = True
    ) -> Dict[str, Any]:
        """
        集成预测（覆盖父类方法）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            use_stacking: 是否使用Stacking（True=元学习器，False=简单加权）
        
        Returns:
            预测结果
        """
        try:
            # 检查集成模型是否存在
            if timeframe not in self.ensemble_models:
                logger.warning(f"⚠️ {timeframe} 集成模型未训练，降级到单模型")
                return await super().predict(symbol, timeframe)
            
            # 准备预测数据
            data = await self._prepare_prediction_data(symbol, timeframe)
            if data.empty:
                return None
            
            # 准备特征
            X = self._prepare_features_for_prediction(data, timeframe)
            if len(X) == 0:
                return None
            
            # 获取集成模型
            models = self.ensemble_models[timeframe]
            
            # 获取最后一行（最新数据）
            X_latest = X.iloc[[-1]]
            
            # 三个基础模型预测
            lgb_proba = models['lgb'].predict_proba(X_latest)[0]
            xgb_proba = models['xgb'].predict_proba(X_latest)[0]
            cat_proba = models['cat'].predict_proba(X_latest)[0]
            
            # Stacking预测
            if use_stacking and 'meta' in models:
                # 生成元特征
                meta_features = np.hstack([lgb_proba, xgb_proba, cat_proba]).reshape(1, -1)
                
                # 元学习器预测
                stacking_proba = models['meta'].predict_proba(meta_features)[0]
                final_pred = stacking_proba.argmax()
                confidence = stacking_proba[final_pred]
                
                method = "Stacking"
            else:
                # 降级：简单加权平均
                weights = self.fallback_weights
                ensemble_proba = (
                    lgb_proba * weights['lgb'] +
                    xgb_proba * weights['xgb'] +
                    cat_proba * weights['cat']
                )
                final_pred = ensemble_proba.argmax()
                confidence = ensemble_proba[final_pred]
                
                method = "Weighted"
            
            # 映射到信号类型
            signal_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
            signal_type = signal_map[final_pred]
            
            return {
                'signal_type': signal_type,
                'confidence': float(confidence),
                'probabilities': {
                    'SHORT': float(lgb_proba[0] + xgb_proba[0] + cat_proba[0]) / 3,
                    'HOLD': float(lgb_proba[1] + xgb_proba[1] + cat_proba[1]) / 3,
                    'LONG': float(lgb_proba[2] + xgb_proba[2] + cat_proba[2]) / 3
                },
                'base_predictions': {
                    'lgb': {'type': signal_map[lgb_proba.argmax()], 'confidence': float(lgb_proba.max())},
                    'xgb': {'type': signal_map[xgb_proba.argmax()], 'confidence': float(xgb_proba.max())},
                    'cat': {'type': signal_map[cat_proba.argmax()], 'confidence': float(cat_proba.max())}
                },
                'method': method,
                'timestamp': datetime.now(),
                'model_version': '2.0_ensemble'
            }
            
        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _save_ensemble_models(self, timeframe: str):
        """保存集成模型"""
        try:
            models = self.ensemble_models[timeframe]
            model_dir = Path(self.model_dir)  # 使用父类的model_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存4个模型
            for model_name in ['lgb', 'xgb', 'cat', 'meta']:
                if model_name in models:
                    filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{model_name}_model.pkl"
                    with open(filepath, 'wb') as f:
                        pickle.dump(models[model_name], f)
            
            logger.info(f"✅ {timeframe} 集成模型保存完成（4个模型）")
            
        except Exception as e:
            logger.error(f"保存集成模型失败: {e}")
    
    def _load_ensemble_models(self, timeframe: str) -> bool:
        """加载集成模型"""
        try:
            model_dir = Path(self.model_dir)  # 使用父类的model_dir
            models = {}
            
            # 加载4个模型
            for model_name in ['lgb', 'xgb', 'cat', 'meta']:
                filepath = model_dir / f"{settings.SYMBOL}_{timeframe}_{model_name}_model.pkl"
                
                if not filepath.exists():
                    logger.warning(f"⚠️ {timeframe} {model_name}模型文件不存在")
                    return False
                
                with open(filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)
            
            self.ensemble_models[timeframe] = models
            logger.info(f"✅ {timeframe} 集成模型加载完成（4个模型）")
            return True
            
        except Exception as e:
            logger.error(f"加载集成模型失败: {e}")
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

