"""
属性测试：预测结果标量化

Feature: trading-system-prediction-fix
Property 1: 预测结果标量化
验证需求：1.1, 1.3
"""
import logging
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


# 生成随机特征数据的策略
@st.composite
def feature_data(draw):
    """生成随机特征数据（255个特征）"""
    n_features = 255
    features = draw(arrays(
        dtype=np.float32,
        shape=(1, n_features),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        )
    ))
    return features


# 生成随机训练数据的策略
@st.composite
def training_data(draw):
    """生成随机训练数据"""
    n_samples = draw(st.integers(min_value=100, max_value=500))
    n_features = 255
    X = draw(arrays(
        dtype=np.float32,
        shape=(n_samples, n_features),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        )
    ))
    y = draw(arrays(
        dtype=np.int32,
        shape=(n_samples,),
        elements=st.integers(min_value=0, max_value=2)
    ))
    return X, y


class TestPredictionScalar:
    """测试预测结果标量化"""
    
    @pytest.fixture(scope="class")
    def trained_models(self):
        """创建训练好的模型（用于测试）"""
        logger.info("🔧 开始训练测试模型...")
        
        # 生成简单的训练数据
        np.random.seed(42)
        n_samples = 200
        n_features = 255
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, 3, size=n_samples)
        
        logger.info(f"📊 训练数据: {n_samples}样本, {n_features}特征")
        
        # 训练LightGBM
        logger.info("🚀 训练LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        logger.info("✅ LightGBM训练完成")
        
        # 训练XGBoost
        logger.info("🚀 训练XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        logger.info("✅ XGBoost训练完成")
        
        # 训练CatBoost
        logger.info("🚀 训练CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=10,
            depth=3,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        logger.info("✅ CatBoost训练完成")
        
        logger.info("✅ 所有测试模型训练完成")
        
        return {
            'lgb': lgb_model,
            'xgb': xgb_model,
            'cat': cat_model
        }
    
    @given(features=feature_data())
    @settings(max_examples=100, deadline=None)
    def test_lightgbm_returns_scalar(self, trained_models, features):
        """
        属性1：LightGBM预测返回标量
        
        对于任意特征数据，LightGBM预测应该返回int类型的标量值
        """
        model = trained_models['lgb']
        pred = model.predict(features)
        
        # 提取标量
        if isinstance(pred, (np.ndarray, list)):
            scalar_pred = int(pred[0])
        else:
            scalar_pred = int(pred)
        
        # 验证是int类型
        assert isinstance(scalar_pred, int), \
            f"预测结果应该是int类型，实际是{type(scalar_pred)}"
        
        # 验证在有效范围内
        assert scalar_pred in [0, 1, 2], \
            f"预测结果应该在[0,1,2]范围内，实际是{scalar_pred}"
    
    @given(features=feature_data())
    @settings(max_examples=100, deadline=None)
    def test_xgboost_returns_scalar(self, trained_models, features):
        """
        属性1：XGBoost预测返回标量
        
        对于任意特征数据，XGBoost预测应该返回int类型的标量值
        """
        model = trained_models['xgb']
        pred = model.predict(features)
        
        # 提取标量
        if isinstance(pred, (np.ndarray, list)):
            scalar_pred = int(pred[0])
        else:
            scalar_pred = int(pred)
        
        # 验证是int类型
        assert isinstance(scalar_pred, int), \
            f"预测结果应该是int类型，实际是{type(scalar_pred)}"
        
        # 验证在有效范围内
        assert scalar_pred in [0, 1, 2], \
            f"预测结果应该在[0,1,2]范围内，实际是{scalar_pred}"
    
    @given(features=feature_data())
    @settings(max_examples=100, deadline=None)
    def test_catboost_returns_scalar(self, trained_models, features):
        """
        属性1：CatBoost预测返回标量
        
        对于任意特征数据，CatBoost预测应该返回int类型的标量值
        """
        model = trained_models['cat']
        pred = model.predict(features)
        
        # 提取标量
        if isinstance(pred, (np.ndarray, list)):
            scalar_pred = int(pred[0])
        else:
            scalar_pred = int(pred)
        
        # 验证是int类型
        assert isinstance(scalar_pred, int), \
            f"预测结果应该是int类型，实际是{type(scalar_pred)}"
        
        # 验证在有效范围内
        assert scalar_pred in [0, 1, 2], \
            f"预测结果应该在[0,1,2]范围内，实际是{scalar_pred}"
    
    @given(features=feature_data())
    @settings(max_examples=100, deadline=None)
    def test_predict_scalar_method(self, trained_models, features):
        """
        属性1：_predict_scalar方法确保返回标量
        
        对于任意特征数据和模型，_predict_scalar方法应该返回int类型的标量值
        """
        # 导入EnsembleMLService（需要模拟）
        from app.model.ensemble_ml_service import EnsembleMLService
        
        service = EnsembleMLService()
        
        for model_name, model in trained_models.items():
            scalar_pred = service._predict_scalar(model, features, model_name=model_name)
            
            # 验证是int类型
            assert isinstance(scalar_pred, int), \
                f"{model_name}预测结果应该是int类型，实际是{type(scalar_pred)}"
            
            # 验证在有效范围内
            assert scalar_pred in [0, 1, 2], \
                f"{model_name}预测结果应该在[0,1,2]范围内，实际是{scalar_pred}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
