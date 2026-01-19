"""
集成模型管理器模块（保存/加载）
"""
# StdLib
import logging
import pickle
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Third-Party
# 深度学习模型（PyTorch）- 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local App
from app.core.config import settings

logger = logging.getLogger(__name__)


def save_ensemble_models(
    models: Dict[str, Any],
    timeframe: str,
    model_dir: str,
    scalers: Dict[str, Any],
    feature_columns_dict: Dict[str, list]
) -> bool:
    """
    保存集成模型（原子性保存 + 热部署）
    
    Args:
        models: 模型字典 {lgb, xgb, cat, inf, meta}
        timeframe: 时间框架
        model_dir: 模型目录
        scalers: 缩放器字典
        feature_columns_dict: 特征列字典
    
    Returns:
        是否保存成功
    """
    try:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        safe_symbol = settings.SYMBOL.replace('/', '_')
        
        old_model_files = []
        backup_dir = None
        if model_path.exists():
            pattern = f"{safe_symbol}_{timeframe}_*"
            existing_files = list(model_path.glob(pattern))
            if existing_files:
                # ✅ 修复：使用日期文件夹格式 old/2026-01-17/，而不是时间戳
                date_str = datetime.now().strftime('%Y-%m-%d')
                old_dir = model_path.parent / 'old' / date_str
                old_dir.mkdir(parents=True, exist_ok=True)
                backup_dir = old_dir
                
                for file_path in existing_files:
                    backup_path = old_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    old_model_files.append(file_path.name)
                
                if old_model_files:
                    logger.info(f"{timeframe} 旧模型已备份到: {old_dir} ({len(old_model_files)}个文件)")
        
        with tempfile.TemporaryDirectory(dir=model_path) as temp_dir:
            temp_path = Path(temp_dir)
            saved_count = 0
            
            model_mapping = {
                'lgb': 'lgb',
                'xgb': 'xgb',
                'cat': 'cat',
                'meta': 'meta'
            }
            
            for short_name in model_mapping:
                if short_name in models:
                    temp_file = temp_path / f"{safe_symbol}_{timeframe}_{short_name}_model.pkl"
                    temp_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_file, 'wb') as f:
                        pickle.dump(models[short_name], f)
                    saved_count += 1
            
            if 'inf' in models and TORCH_AVAILABLE:
                temp_file = temp_path / f"{safe_symbol}_{timeframe}_inf_model.pt"
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_file, 'wb') as f:
                    pickle.dump(models['inf'], f)
                saved_count += 1
            
            if timeframe in scalers:
                temp_file = temp_path / f"{safe_symbol}_{timeframe}_scaler.pkl"
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_file, 'wb') as f:
                    pickle.dump(scalers[timeframe], f)
                saved_count += 1
            
            if timeframe in feature_columns_dict:
                temp_file = temp_path / f"{safe_symbol}_{timeframe}_features.pkl"
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_file, 'wb') as f:
                    pickle.dump(feature_columns_dict[timeframe], f)
                saved_count += 1
            
            for temp_file in temp_path.glob(f"{safe_symbol}_{timeframe}_*"):
                target_file = model_path / temp_file.name
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_file), str(target_file))
            
            logger.info(f"{timeframe} 集成模型保存完成（{saved_count}个文件，原子性更新）")
        
        return True
        
    except Exception as e:
        logger.error(f"保存集成模型失败: {e}")
        return False


def load_ensemble_models(
    timeframe: str,
    model_dir: str,
    scalers: Dict[str, Any],
    feature_columns_dict: Dict[str, list]
) -> Optional[Dict[str, Any]]:
    """
    加载集成模型
    
    Args:
        timeframe: 时间框架
        model_dir: 模型目录
        scalers: 缩放器字典（用于存储加载的scaler）
        feature_columns_dict: 特征列字典（用于存储加载的features）
    
    Returns:
        模型字典，如果加载失败返回None
    """
    try:
        model_path = Path(model_dir)
        models = {}
        
        safe_symbol = settings.SYMBOL.replace('/', '_')
        
        model_mapping = {
            'lgb': 'lgb',
            'xgb': 'xgb',
            'cat': 'cat',
            'meta': 'meta'
        }
        
        for short_name in model_mapping:
            filepath = model_path / f"{safe_symbol}_{timeframe}_{short_name}_model.pkl"
            if not filepath.exists():
                logger.warning(f"{timeframe} {short_name}模型文件不存在: {filepath}")
                existing_files = list(model_path.glob(f"*_{timeframe}_{short_name}_model.pkl"))
                if existing_files:
                    logger.info(f"发现类似文件: {existing_files}")
                return None
        
        for short_name in model_mapping:
            filepath = model_path / f"{safe_symbol}_{timeframe}_{short_name}_model.pkl"
            with open(filepath, 'rb') as f:
                models[short_name] = pickle.load(f)
        
        if TORCH_AVAILABLE:
            inf_filepath = model_path / f"{safe_symbol}_{timeframe}_inf_model.pt"
            if inf_filepath.exists():
                with open(inf_filepath, 'rb') as f:
                    models['inf'] = pickle.load(f)
                logger.info(f"Informer-2模型已加载")
        
        scaler_filepath = model_path / f"{safe_symbol}_{timeframe}_scaler.pkl"
        if scaler_filepath.exists():
            with open(scaler_filepath, 'rb') as f:
                scalers[timeframe] = pickle.load(f)
        
        features_filepath = model_path / f"{safe_symbol}_{timeframe}_features.pkl"
        if features_filepath.exists():
            with open(features_filepath, 'rb') as f:
                feature_columns_dict[timeframe] = pickle.load(f)
        
        logger.info(f"{timeframe} 集成模型加载完成")
        
        return models
        
    except Exception as e:
        logger.error(f"加载集成模型失败: {e}")
        return None

