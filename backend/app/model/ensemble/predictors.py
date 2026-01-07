"""
集成模型预测器模块
"""
# StdLib
import json
import logging
import warnings

# Third-Party
import numpy as np
import pandas as pd
import xgboost as xgb
# 可选依赖：cupy（GPU加速，用于XGBoost GPU预测）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


def predict_xgboost(model: xgb.XGBClassifier, X: np.ndarray, return_single: bool = False) -> tuple:
    """
    XGBoost预测辅助方法（修复设备不匹配问题）
    
    Args:
        model: XGBoost模型
        X: 特征数据（numpy数组或DataFrame）
        return_single: 是否返回单个值（True=单样本预测，False=批量预测）
    
    Returns:
        tuple: 
            - return_single=True: (预测类别标量, 预测概率1D数组)
            - return_single=False: (预测类别数组, 预测概率2D数组)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Falling back to prediction using DMatrix.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*mismatched devices.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*XGBoost is running on.*while the input data is on.*')
        warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
        try:
            if isinstance(X, pd.DataFrame):
                X_pred = X.values.astype(np.float32)
            elif isinstance(X, np.ndarray):
                X_pred = X.astype(np.float32)
            else:
                X_pred = np.asarray(X, dtype=np.float32)
            
            if len(X_pred.shape) == 1:
                X_pred = X_pred.reshape(1, -1)
            
            if not X_pred.flags['C_CONTIGUOUS']:
                X_pred = np.ascontiguousarray(X_pred, dtype=np.float32)
            
            booster = model.get_booster()
            
            try:
                config = booster.save_config()
                config_dict = json.loads(config)
                device = config_dict.get('learner', {}).get('learner_train_param', {}).get('device', '')
                
                if device and 'cuda' in device.lower():
                    if CUPY_AVAILABLE and cp is not None:
                        try:
                            X_pred_gpu = cp.asarray(X_pred)
                            
                            try:
                                dmatrix_gpu = xgb.DMatrix(X_pred_gpu)
                                xgb_proba_raw = booster.predict(dmatrix_gpu, output_margin=False)
                                
                                if hasattr(xgb_proba_raw, 'get'):
                                    xgb_proba_raw = xgb_proba_raw.get()
                                elif isinstance(xgb_proba_raw, cp.ndarray):
                                    xgb_proba_raw = cp.asnumpy(xgb_proba_raw)
                                
                                if len(xgb_proba_raw.shape) == 1:
                                    n_classes = len(xgb_proba_raw)
                                    xgb_proba = xgb_proba_raw.reshape(1, n_classes)
                                else:
                                    xgb_proba = xgb_proba_raw
                                
                                xgb_pred = np.argmax(xgb_proba, axis=1)
                                
                                if return_single and len(xgb_pred) == 1:
                                    return xgb_pred[0], xgb_proba[0]
                                else:
                                    return xgb_pred, xgb_proba
                                    
                            except Exception as e:
                                logger.warning(f"XGBoost GPU DMatrix预测失败，回退到标准方式: {e}")
                                xgb_proba = model.predict_proba(X_pred)
                                xgb_pred = model.predict(X_pred)
                                
                                if return_single and len(xgb_pred) == 1:
                                    return xgb_pred[0], xgb_proba[0]
                                else:
                                    return xgb_pred, xgb_proba
                                    
                        except Exception as e:
                            logger.warning(f"XGBoost GPU预测失败，回退到标准方式: {e}")
                            xgb_proba = model.predict_proba(X_pred)
                            xgb_pred = model.predict(X_pred)
                            
                            if return_single and len(xgb_pred) == 1:
                                return xgb_pred[0], xgb_proba[0]
                            else:
                                return xgb_pred, xgb_proba
                    else:
                        logger.warning("cupy未安装，XGBoost GPU预测将产生设备不匹配警告。建议安装: pip install cupy-cuda12x")
                        dmatrix = xgb.DMatrix(X_pred)
                        xgb_proba_raw = booster.predict(dmatrix, output_margin=False)
                        
                        if len(xgb_proba_raw.shape) == 1:
                            n_classes = len(xgb_proba_raw)
                            xgb_proba = xgb_proba_raw.reshape(1, n_classes)
                        else:
                            xgb_proba = xgb_proba_raw
                        
                        xgb_pred = np.argmax(xgb_proba, axis=1)
                        
                        if return_single and len(xgb_pred) == 1:
                            return xgb_pred[0], xgb_proba[0]
                        else:
                            return xgb_pred, xgb_proba
                else:
                    xgb_proba = model.predict_proba(X_pred)
                    xgb_pred = model.predict(X_pred)
                    
                    if return_single and len(xgb_pred) == 1:
                        return xgb_pred[0], xgb_proba[0]
                    else:
                        return xgb_pred, xgb_proba
            except Exception as e:
                logger.debug(f"XGBoost设备检测失败，使用标准方式: {e}")
                xgb_proba = model.predict_proba(X_pred)
                xgb_pred = model.predict(X_pred)
                
                if return_single and len(xgb_pred) == 1:
                    return xgb_pred[0], xgb_proba[0]
                else:
                    return xgb_pred, xgb_proba
        except Exception as e:
            logger.error(f"XGBoost预测失败: {e}")
            raise

