"""
属性测试：元特征维度一致性

Feature: trading-system-prediction-fix
Property 2: 元特征维度一致性
验证需求：1.2, 1.4
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


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


class TestMetaFeaturesDimension:
    """测试元特征维度一致性"""
    
    @pytest.fixture(scope="class")
    def trained_models_without_informer(self):
        """创建训练好的模型（不含Informer-2）"""
        # 生成简单的训练数据
        np.random.seed(42)
        n_samples = 200
        n_features = 255
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, 3, size=n_samples)
        
        # 训练LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # 训练XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        # 训练CatBoost
        cat_model = CatBoostClassifier(
            iterations=10,
            depth=3,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        
        # 训练元学习器（40个特征，不含Informer-2）
        X_meta = np.random.randn(n_samples, 40).astype(np.float32)
        meta_model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        meta_model.fit(X_meta, y_train)
        
        return {
            'lgb': lgb_model,
            'xgb': xgb_model,
            'cat': cat_model,
            'meta': meta_model
        }
    
    @pytest.fixture(scope="class")
    def trained_models_with_informer(self):
        """创建训练好的模型（含Informer-2）"""
        # 生成简单的训练数据
        np.random.seed(42)
        n_samples = 200
        n_features = 255
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, 3, size=n_samples)
        
        # 训练LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # 训练XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        # 训练CatBoost
        cat_model = CatBoostClassifier(
            iterations=10,
            depth=3,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        
        # 创建模拟的Informer-2模型
        class MockInformerModel:
            """模拟Informer-2模型"""
            def predict_proba(self, X):
                return np.array([[0.3, 0.4, 0.3]] * len(X), dtype=np.float32)
            
            def predict(self, X):
                return np.array([1] * len(X), dtype=np.int32)
        
        inf_model = MockInformerModel()
        
        # 训练元学习器（45个特征，含Informer-2）
        X_meta = np.random.randn(n_samples, 45).astype(np.float32)
        meta_model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        meta_model.fit(X_meta, y_train)
        
        return {
            'lgb': lgb_model,
            'xgb': xgb_model,
            'cat': cat_model,
            'inf': inf_model,
            'meta': meta_model
        }
    
    @given(features=feature_data())
    @settings(max_examples=100, deadline=None)
    def test_meta_features_shape_without_informer(self, trained_models_without_informer, features):
        """
        属性2：元特征形状一致性（不含Informer-2）
        
        对于任意特征数据，生成的元特征应该是形状为(1, 40)的numpy数组
        """
        from app.model.ensemble_ml_service import EnsembleMLService
        
        service = EnsembleMLService()
        
        try:
            # 生成元特征（不含Informer-2）
            meta_features = service._generate_enhanced_meta_features(
                features,
                trained_models_without_informer,
                inf_proba=None,
                inf_pred=None
            )
            
            # 验证是numpy数组
            assert isinstance(meta_features, np.ndarray), \
                f"元特征应该是numpy数组，实际是{type(meta_features)}"
            
            # 验证形状
            assert meta_features.shape == (1, 40), \
                f"元特征形状应该是(1, 40)，实际是{meta_features.shape}"
            
            # 验证没有NaN或Inf
            assert not np.isnan(meta_features).any(), \
                "元特征不应包含NaN值"
            assert not np.isinf(meta_features).any(), \
                "元特征不应包含Inf值"
            
        except ValueError as e:
            # 不应该抛出ValueError
            pytest.fail(f"生成元特征时不应抛出ValueError: {e}")
    
    @given(features=feature_data())
    @settings(max_examples=100, deadline=None)
    def test_meta_features_shape_with_informer(self, trained_models_with_informer, features):
        """
        属性2：元特征形状一致性（含Informer-2）
        
        对于任意特征数据，当提供Informer-2预测时，生成的元特征应该是形状为(1, 45)的numpy数组
        """
        from app.model.ensemble_ml_service import EnsembleMLService
        
        service = EnsembleMLService()
        
        # 模拟Informer-2预测
        inf_proba = np.array([0.3, 0.3, 0.4], dtype=np.float32)
        inf_pred = 2
        
        try:
            # 生成元特征（含Informer-2）
            meta_features = service._generate_enhanced_meta_features(
                features,
                trained_models_with_informer,
                inf_proba=inf_proba,
                inf_pred=inf_pred
            )
            
            # 验证是numpy数组
            assert isinstance(meta_features, np.ndarray), \
                f"元特征应该是numpy数组，实际是{type(meta_features)}"
            
            # 验证形状
            assert meta_features.shape == (1, 45), \
                f"元特征形状应该是(1, 45)，实际是{meta_features.shape}"
            
            # 验证没有NaN或Inf
            assert not np.isnan(meta_features).any(), \
                "元特征不应包含NaN值"
            assert not np.isinf(meta_features).any(), \
                "元特征不应包含Inf值"
            
        except ValueError as e:
            # 不应该抛出ValueError
            pytest.fail(f"生成元特征时不应抛出ValueError: {e}")
    
    def test_meta_features_no_value_error(self, trained_models_without_informer):
        """
        属性2：元特征生成不抛出ValueError
        
        生成元特征时不应该抛出ValueError异常
        """
        from app.model.ensemble_ml_service import EnsembleMLService
        
        service = EnsembleMLService()
        
        # 生成多个随机特征数据
        np.random.seed(42)
        for i in range(50):
            features = np.random.randn(1, 255).astype(np.float32)
            
            try:
                # 生成元特征
                meta_features = service._generate_enhanced_meta_features(
                    features,
                    trained_models_without_informer,
                    inf_proba=None,
                    inf_pred=None
                )
                
                # 验证形状
                assert meta_features.shape[0] == 1, \
                    f"元特征第一维应该是1，实际是{meta_features.shape[0]}"
                assert meta_features.shape[1] in [40, 45], \
                    f"元特征第二维应该是40或45，实际是{meta_features.shape[1]}"
                
            except ValueError as e:
                pytest.fail(f"第{i+1}次生成元特征时抛出ValueError: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
