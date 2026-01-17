"""
模型稳定性增强模块
通过bagging和模型多样性提升系统稳定性

核心功能：
1. Bagging集成策略
2. 模型多样性增强
3. 稳定性指标监控
4. 动态权重调整

作者: QuantAI Team
版本: v3.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.utils import resample

# Local App
from app.core.constants import (
    STABILITY_BOOTSTRAP_RATIO,
    STABILITY_DIVERSITY_THRESHOLD,
    STABILITY_FEATURE_SAMPLING_RATIO,
    STABILITY_N_BAGGING_MODELS,
    STABILITY_STABILITY_THRESHOLD
)

logger = logging.getLogger(__name__)


@dataclass
class ModelStabilityMetrics:
    """模型稳定性指标"""
    cv_stability: float
    model_diversity: float
    prediction_consistency: float
    bagging_effectiveness: float
    stability_score: float


class ModelStabilityEnhancer:
    """
    模型稳定性增强器
    
    核心功能：
    1. Bagging集成策略
    2. 模型多样性增强
    3. 稳定性指标监控
    4. 动态权重调整
    """
    
    def __init__(self):
        # Bagging参数
        self.n_bagging_models = STABILITY_N_BAGGING_MODELS
        self.bootstrap_ratio = STABILITY_BOOTSTRAP_RATIO
        self.feature_sampling_ratio = STABILITY_FEATURE_SAMPLING_RATIO
        
        # 多样性参数
        self.diversity_threshold = STABILITY_DIVERSITY_THRESHOLD
        self.stability_threshold = STABILITY_STABILITY_THRESHOLD
        
        # 历史记录
        self.stability_history: List[ModelStabilityMetrics] = []
        
        logger.info("✅ 模型稳定性增强器初始化完成")
    
    def create_bagging_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        base_params: Dict[str, Any],
        n_models: int = None
    ) -> List[Any]:
        """
        创建Bagging模型集合
        
        Args:
            X: 特征数据
            y: 标签数据
            model_type: 模型类型
            base_params: 基础参数
            n_models: 模型数量
        
        Returns:
            List[Any]: Bagging模型列表
        """
        try:
            if n_models is None:
                n_models = self.n_bagging_models
            
            bagging_models = []
            
            for i in range(n_models):
                # 1. 自助采样
                n_samples = int(len(X) * self.bootstrap_ratio)
                bootstrap_indices = resample(
                    range(len(X)), 
                    n_samples=n_samples, 
                    random_state=42 + i
                )
                
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]
                
                # 2. 特征采样
                if len(X.shape) == 2:  # 2D数据
                    n_features = X.shape[1]
                    n_selected_features = int(n_features * self.feature_sampling_ratio)
                    feature_indices = resample(
                        range(n_features),
                        n_samples=n_selected_features,
                        random_state=42 + i
                    )
                    X_bootstrap = X_bootstrap[:, feature_indices]
                elif len(X.shape) == 3:  # 3D数据（序列）
                    n_features = X.shape[2]
                    n_selected_features = int(n_features * self.feature_sampling_ratio)
                    feature_indices = resample(
                        range(n_features),
                        n_samples=n_selected_features,
                        random_state=42 + i
                    )
                    X_bootstrap = X_bootstrap[:, :, feature_indices]
                
                # 3. 训练模型
                model = self._create_single_model(model_type, base_params, i)
                
                if model_type == "lightgbm":
                    model.fit(X_bootstrap, y_bootstrap)
                elif model_type == "xgboost":
                    model.fit(X_bootstrap, y_bootstrap)
                elif model_type == "catboost":
                    model.fit(X_bootstrap, y_bootstrap, verbose=False)
                
                bagging_models.append({
                    'model': model,
                    'feature_indices': feature_indices if len(X.shape) == 2 else None,
                    'bootstrap_indices': bootstrap_indices
                })
                
                logger.debug(f"✅ Bagging模型 {i+1}/{n_models} 训练完成")
            
            logger.info(f"🎯 创建了 {len(bagging_models)} 个Bagging模型")
            return bagging_models
            
        except Exception as e:
            logger.error(f"❌ Bagging模型创建失败: {e}")
            return []
    
    def _create_single_model(self, model_type: str, base_params: Dict[str, Any], seed: int) -> Any:
        """创建单个模型"""
        try:
            params = base_params.copy()
            
            if model_type == "lightgbm":
                params['random_state'] = seed
                params['verbose'] = -1
                return lgb.LGBMClassifier(**params)
            elif model_type == "xgboost":
                params['random_state'] = seed
                params['verbosity'] = 0
                return xgb.XGBClassifier(**params)
            elif model_type == "catboost":
                params['random_seed'] = seed
                params['verbose'] = False
                return cb.CatBoostClassifier(**params)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
        except Exception as e:
            logger.error(f"❌ 单模型创建失败: {e}")
            return None
    
    def predict_with_bagging(
        self,
        bagging_models: List[Dict],
        X: np.ndarray,
        return_proba: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Bagging模型进行预测
        
        Args:
            bagging_models: Bagging模型列表
            X: 特征数据
            return_proba: 是否返回概率
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (预测结果, 预测概率)
        """
        try:
            if not bagging_models:
                raise ValueError("Bagging模型列表为空")
            
            predictions = []
            probabilities = []
            
            for model_info in bagging_models:
                model = model_info['model']
                feature_indices = model_info.get('feature_indices')
                
                # 特征选择
                if feature_indices is not None:
                    if len(X.shape) == 2:
                        X_selected = X[:, feature_indices]
                    else:  # 3D数据
                        X_selected = X[:, :, feature_indices]
                else:
                    X_selected = X
                
                # 预测
                if return_proba:
                    proba = model.predict_proba(X_selected)
                    probabilities.append(proba)
                
                pred = model.predict(X_selected)
                predictions.append(pred)
            
            # 集成预测
            predictions = np.array(predictions)
            ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
            
            if return_proba and probabilities:
                probabilities = np.array(probabilities)
                ensemble_proba = np.mean(probabilities, axis=0)
            else:
                ensemble_proba = None
            
            logger.debug(f"🎯 Bagging预测完成: {len(bagging_models)}个模型集成")
            
            return ensemble_pred, ensemble_proba
            
        except Exception as e:
            logger.error(f"❌ Bagging预测失败: {e}")
            return np.array([]), np.array([])
    
    def calculate_model_diversity(
        self,
        predictions_list: List[np.ndarray]
    ) -> float:
        """
        计算模型多样性
        
        Args:
            predictions_list: 预测结果列表
        
        Returns:
            float: 多样性分数
        """
        try:
            if len(predictions_list) < 2:
                return 0.0
            
            n_models = len(predictions_list)
            n_samples = len(predictions_list[0])
            
            # 计算模型间的不一致度
            disagreements = 0
            total_comparisons = 0
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    pred_i = predictions_list[i]
                    pred_j = predictions_list[j]
                    
                    # 计算不一致的样本数
                    disagreement = np.sum(pred_i != pred_j)
                    disagreements += disagreement
                    total_comparisons += n_samples
            
            # 多样性分数（不一致度比例）
            diversity = disagreements / total_comparisons if total_comparisons > 0 else 0.0
            
            logger.debug(f"🔍 模型多样性: {diversity:.3f}")
            return diversity
            
        except Exception as e:
            logger.error(f"❌ 模型多样性计算失败: {e}")
            return 0.0
    
    def calculate_prediction_consistency(
        self,
        predictions_list: List[np.ndarray],
        true_labels: np.ndarray
    ) -> float:
        """
        计算预测一致性
        
        Args:
            predictions_list: 预测结果列表
            true_labels: 真实标签
        
        Returns:
            float: 一致性分数
        """
        try:
            if not predictions_list or len(true_labels) == 0:
                return 0.0
            
            # 计算每个模型的准确率
            accuracies = []
            for pred in predictions_list:
                if len(pred) == len(true_labels):
                    acc = accuracy_score(true_labels, pred)
                    accuracies.append(acc)
            
            if not accuracies:
                return 0.0
            
            # 一致性分数（准确率的标准差，越小越一致）
            consistency = 1.0 - np.std(accuracies)
            
            logger.debug(f"🔍 预测一致性: {consistency:.3f}")
            return max(0.0, consistency)
            
        except Exception as e:
            logger.error(f"❌ 预测一致性计算失败: {e}")
            return 0.0
    
    def calculate_bagging_effectiveness(
        self,
        individual_accuracies: List[float],
        ensemble_accuracy: float
    ) -> float:
        """
        计算Bagging有效性
        
        Args:
            individual_accuracies: 单个模型准确率
            ensemble_accuracy: 集成模型准确率
        
        Returns:
            float: Bagging有效性分数
        """
        try:
            if not individual_accuracies:
                return 0.0
            
            # 单个模型平均准确率
            avg_individual_acc = np.mean(individual_accuracies)
            
            # Bagging有效性（集成提升）
            effectiveness = ensemble_accuracy - avg_individual_acc
            
            logger.debug(f"🔍 Bagging有效性: {effectiveness:.3f}")
            return max(0.0, effectiveness)
            
        except Exception as e:
            logger.error(f"❌ Bagging有效性计算失败: {e}")
            return 0.0
    
    def calculate_stability_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bagging_models: List[Dict],
        cv_folds: int = 5
    ) -> ModelStabilityMetrics:
        """
        计算模型稳定性指标
        
        Args:
            X: 特征数据
            y: 标签数据
            bagging_models: Bagging模型列表
            cv_folds: 交叉验证折数
        
        Returns:
            ModelStabilityMetrics: 稳定性指标
        """
        try:
            # 1. 交叉验证稳定性
            cv_stability = self._calculate_cv_stability(X, y, bagging_models, cv_folds)
            
            # 2. 模型多样性
            predictions_list = []
            for model_info in bagging_models:
                model = model_info['model']
                feature_indices = model_info.get('feature_indices')
                
                if feature_indices is not None:
                    if len(X.shape) == 2:
                        X_selected = X[:, feature_indices]
                    else:
                        X_selected = X[:, :, feature_indices]
                else:
                    X_selected = X
                
                pred = model.predict(X_selected)
                predictions_list.append(pred)
            
            model_diversity = self.calculate_model_diversity(predictions_list)
            
            # 3. 预测一致性
            prediction_consistency = self.calculate_prediction_consistency(predictions_list, y)
            
            # 4. Bagging有效性
            individual_accuracies = []
            for pred in predictions_list:
                if len(pred) == len(y):
                    acc = accuracy_score(y, pred)
                    individual_accuracies.append(acc)
            
            ensemble_pred, _ = self.predict_with_bagging(bagging_models, X, return_proba=False)
            ensemble_accuracy = accuracy_score(y, ensemble_pred) if len(ensemble_pred) == len(y) else 0.0
            
            bagging_effectiveness = self.calculate_bagging_effectiveness(individual_accuracies, ensemble_accuracy)
            
            # 5. 综合稳定性分数
            stability_score = (
                cv_stability * 0.3 +
                model_diversity * 0.2 +
                prediction_consistency * 0.3 +
                bagging_effectiveness * 0.2
            )
            
            metrics = ModelStabilityMetrics(
                cv_stability=cv_stability,
                model_diversity=model_diversity,
                prediction_consistency=prediction_consistency,
                bagging_effectiveness=bagging_effectiveness,
                stability_score=stability_score
            )
            
            logger.info(f"📊 稳定性指标: CV稳定性={cv_stability:.3f}, "
                       f"模型多样性={model_diversity:.3f}, "
                       f"预测一致性={prediction_consistency:.3f}, "
                       f"Bagging有效性={bagging_effectiveness:.3f}, "
                       f"综合稳定性={stability_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ 稳定性指标计算失败: {e}")
            return ModelStabilityMetrics(
                cv_stability=0.0,
                model_diversity=0.0,
                prediction_consistency=0.0,
                bagging_effectiveness=0.0,
                stability_score=0.0
            )
    
    def _calculate_cv_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bagging_models: List[Dict],
        cv_folds: int
    ) -> float:
        """计算交叉验证稳定性"""
        try:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                # 使用Bagging模型预测
                ensemble_pred, _ = self.predict_with_bagging(bagging_models, X_val, return_proba=False)
                
                if len(ensemble_pred) == len(y_val):
                    score = accuracy_score(y_val, ensemble_pred)
                    cv_scores.append(score)
            
            if not cv_scores:
                return 0.0
            
            # 稳定性 = 1 - 变异系数
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            stability = 1.0 - (cv_std / cv_mean) if cv_mean > 0 else 0.0
            
            return max(0.0, stability)
            
        except Exception as e:
            logger.error(f"❌ CV稳定性计算失败: {e}")
            return 0.0
    
    def get_stability_recommendations(
        self,
        metrics: ModelStabilityMetrics
    ) -> List[str]:
        """
        获取稳定性改进建议
        
        Args:
            metrics: 稳定性指标
        
        Returns:
            List[str]: 改进建议
        """
        recommendations = []
        
        if metrics.cv_stability < self.stability_threshold:
            recommendations.append("CV稳定性较低，建议增加正则化参数")
        
        if metrics.model_diversity < self.diversity_threshold:
            recommendations.append("模型多样性不足，建议增加特征采样比例")
        
        if metrics.prediction_consistency < 0.7:
            recommendations.append("预测一致性较低，建议调整模型参数")
        
        if metrics.bagging_effectiveness < 0.05:
            recommendations.append("Bagging效果不明显，建议增加模型数量")
        
        if metrics.stability_score < 0.6:
            recommendations.append("综合稳定性较低，建议全面优化模型配置")
        
        return recommendations
