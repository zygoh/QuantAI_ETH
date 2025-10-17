"""
两模型集成服务（临时方案：LightGBM + XGBoost）
CatBoost安装失败时的降级方案
"""
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

from app.services.ensemble_ml_service import EnsembleMLService

logger = logging.getLogger(__name__)

class TwoModelEnsemble(EnsembleMLService):
    """两模型集成（LightGBM + XGBoost）"""
    
    def _train_stacking_ensemble(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        训练两模型Stacking（跳过CatBoost）
        
        降级方案：只用LightGBM + XGBoost
        """
        import time
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        start_time = time.time()
        
        logger.info(f"🎯 Stage 1: 训练2个基础模型（CatBoost跳过）...")
        
        # 1. 训练LightGBM
        logger.info(f"  📊 训练LightGBM...")
        lgb_model = self._train_lightgbm(X_train, y_train, timeframe)
        
        # 2. 训练XGBoost
        logger.info(f"  📊 训练XGBoost...")
        xgb_model = self._train_xgboost(X_train, y_train, timeframe)
        
        logger.info(f"✅ 2个基础模型训练完成")
        
        # 3. 生成元特征（6维：2个模型 × 3个类别）
        logger.info(f"🎯 Stage 2: 生成元特征...")
        lgb_pred_train = lgb_model.predict_proba(X_train)
        xgb_pred_train = xgb_model.predict_proba(X_train)
        
        meta_features_train = np.hstack([
            lgb_pred_train,
            xgb_pred_train
        ])
        
        # 4. 训练元学习器
        logger.info(f"🎯 Stage 3: 训练元学习器（两模型Stacking）...")
        meta_learner = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        meta_learner.fit(meta_features_train, y_train)
        
        logger.info(f"✅ 元学习器训练完成")
        
        # 5. 验证集评估
        logger.info(f"🎯 Stage 4: 验证集评估...")
        
        lgb_pred_val = lgb_model.predict_proba(X_val)
        xgb_pred_val = xgb_model.predict_proba(X_val)
        
        meta_features_val = np.hstack([
            lgb_pred_val,
            xgb_pred_val
        ])
        
        stacking_pred = meta_learner.predict(meta_features_val)
        stacking_proba = meta_learner.predict_proba(meta_features_val)
        
        # 计算指标
        accuracy = accuracy_score(y_val, stacking_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, stacking_pred, average='weighted', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_val, stacking_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.5
        
        # 各基础模型准确率
        lgb_acc = accuracy_score(y_val, lgb_model.predict(X_val))
        xgb_acc = accuracy_score(y_val, xgb_model.predict(X_val))
        
        training_time = time.time() - start_time
        
        # 保存两模型（不包含catboost）
        self.ensemble_models[timeframe] = {
            'lgb': lgb_model,
            'xgb': xgb_model,
            'meta': meta_learner
        }
        
        # 日志输出
        logger.info(f"📊 {timeframe} 两模型Stacking评估:")
        logger.info(f"  基础模型准确率:")
        logger.info(f"    LightGBM: {lgb_acc:.4f}")
        logger.info(f"    XGBoost:  {xgb_acc:.4f}")
        logger.info(f"  Stacking准确率: {accuracy:.4f}")
        logger.info(f"  提升: +{(accuracy - max(lgb_acc, xgb_acc))*100:.2f}%")
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
                'xgb': xgb_acc
            },
            'training_time': training_time
        }
    
    async def predict(
        self, 
        symbol: str, 
        timeframe: str, 
        use_stacking: bool = True
    ) -> Dict[str, Any]:
        """两模型集成预测"""
        try:
            if timeframe not in self.ensemble_models:
                logger.warning(f"⚠️ {timeframe} 集成模型未训练，降级到单模型")
                return await super(EnsembleMLService, self).predict(symbol, timeframe)
            
            # 准备数据
            data = await self._prepare_prediction_data(symbol, timeframe)
            if data.empty:
                return None
            
            X = self._prepare_features_for_prediction(data, timeframe)
            if len(X) == 0:
                return None
            
            models = self.ensemble_models[timeframe]
            X_latest = X.iloc[[-1]]
            
            # 两模型预测
            lgb_proba = models['lgb'].predict_proba(X_latest)[0]
            xgb_proba = models['xgb'].predict_proba(X_latest)[0]
            
            # Stacking预测（6维元特征）
            if use_stacking and 'meta' in models:
                meta_features = np.hstack([lgb_proba, xgb_proba]).reshape(1, -1)
                stacking_proba = models['meta'].predict_proba(meta_features)[0]
                final_pred = stacking_proba.argmax()
                confidence = stacking_proba[final_pred]
                method = "Stacking(2-Model)"
            else:
                # 简单加权
                ensemble_proba = lgb_proba * 0.6 + xgb_proba * 0.4
                final_pred = ensemble_proba.argmax()
                confidence = ensemble_proba[final_pred]
                method = "Weighted(2-Model)"
            
            signal_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
            signal_type = signal_map[final_pred]
            
            return {
                'signal_type': signal_type,
                'confidence': float(confidence),
                'probabilities': {
                    'SHORT': float((lgb_proba[0] + xgb_proba[0]) / 2),
                    'HOLD': float((lgb_proba[1] + xgb_proba[1]) / 2),
                    'LONG': float((lgb_proba[2] + xgb_proba[2]) / 2)
                },
                'base_predictions': {
                    'lgb': {'type': signal_map[lgb_proba.argmax()], 'confidence': float(lgb_proba.max())},
                    'xgb': {'type': signal_map[xgb_proba.argmax()], 'confidence': float(xgb_proba.max())}
                },
                'method': method,
                'model_version': '2.0_two_model_ensemble'
            }
            
        except Exception as e:
            logger.error(f"两模型集成预测失败: {e}")
            return None

# 全局实例
two_model_ensemble = TwoModelEnsemble()

