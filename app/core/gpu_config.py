"""
GPU配置和优化
确保充分利用GPU并行能力
"""
# StdLib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 🎮 GPU并发配置
GPU_CONCURRENT_CONFIG = {
    # LightGBM GPU配置（支持多任务并发）
    'lightgbm': {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'gpu_use_dp': False,  # 使用单精度（更快）
        # 🔑 关键：不设置 max_bin 限制，让LightGBM自动优化
    },
    
    # XGBoost GPU配置（支持多任务并发）
    'xgboost': {
        'tree_method': 'hist',  # GPU加速的hist方法
        'device': 'cuda',
        'gpu_id': 0,
        # 🔑 关键：XGBoost会自动管理GPU内存
    },
    
    # CatBoost GPU配置（支持多任务并发）
    'catboost': {
        'task_type': 'GPU',
        'devices': '0',
        'gpu_ram_part': 0.5,  # 🔑 使用50%显存（允许多任务并发）
        # 🔑 关键：通过 gpu_ram_part 控制显存使用，支持并发
    },
    
    # PyTorch (Informer-2) GPU配置
    'pytorch': {
        'device': 'cuda:0',
        'allow_tf32': True,  # 启用TF32加速（Ampere架构）
        'cudnn_benchmark': True,  # 启用cuDNN自动优化
        'memory_fraction': 0.3,  # 🔑 限制显存使用（允许其他任务并发）
    }
}

# 🔑 GPU显存分配策略（16GB总量）
GPU_MEMORY_ALLOCATION = {
    'training': {
        'lightgbm': 2.0,  # GB
        'xgboost': 2.0,   # GB
        'catboost': 2.0,  # GB
        'informer2': 6.0,  # GB
        'total': 12.0,    # GB（训练时总占用）
    },
    'backtest': {
        'per_task': 2.5,  # GB/任务
        'max_concurrent': 4,  # 最多4个并发回测（10GB）
    },
    'prediction': {
        'per_task': 0.5,  # GB/任务
        'max_concurrent': 8,  # 最多8个并发预测（4GB）
    },
    'reserved': 2.0,  # GB（系统保留）
}


def get_gpu_config(model_type: str) -> Dict[str, Any]:
    """
    获取指定模型类型的GPU配置
    
    Args:
        model_type: 模型类型（lightgbm/xgboost/catboost/pytorch）
    
    Returns:
        GPU配置字典
    """
    config = GPU_CONCURRENT_CONFIG.get(model_type, {})
    logger.debug(f"获取 {model_type} GPU配置: {config}")
    return config.copy()


def estimate_gpu_memory_usage(task_type: str, num_tasks: int = 1) -> float:
    """
    估算GPU显存使用量
    
    Args:
        task_type: 任务类型（training/backtest/prediction）
        num_tasks: 任务数量
    
    Returns:
        估算的显存使用量（GB）
    """
    if task_type == 'training':
        return GPU_MEMORY_ALLOCATION['training']['total']
    elif task_type == 'backtest':
        return GPU_MEMORY_ALLOCATION['backtest']['per_task'] * num_tasks
    elif task_type == 'prediction':
        return GPU_MEMORY_ALLOCATION['prediction']['per_task'] * num_tasks
    else:
        return 0.0


def check_gpu_capacity(task_type: str, num_tasks: int = 1) -> bool:
    """
    检查GPU是否有足够容量执行任务
    
    Args:
        task_type: 任务类型
        num_tasks: 任务数量
    
    Returns:
        是否有足够容量
    """
    estimated_usage = estimate_gpu_memory_usage(task_type, num_tasks)
    reserved = GPU_MEMORY_ALLOCATION['reserved']
    total_available = 16.0  # 16GB总显存
    
    available = total_available - reserved
    can_execute = estimated_usage <= available
    
    if not can_execute:
        logger.warning(
            f"⚠️ GPU显存不足: 需要{estimated_usage:.1f}GB, "
            f"可用{available:.1f}GB (总{total_available:.1f}GB - 保留{reserved:.1f}GB)"
        )
    
    return can_execute


def log_gpu_config():
    """记录GPU配置信息"""
    logger.info("=" * 70)
    logger.info("🎮 GPU并发配置")
    logger.info("=" * 70)
    logger.info(f"总显存: 16GB")
    logger.info(f"保留显存: {GPU_MEMORY_ALLOCATION['reserved']}GB")
    logger.info("")
    logger.info("训练任务显存分配:")
    for model, mem in GPU_MEMORY_ALLOCATION['training'].items():
        if model != 'total':
            logger.info(f"  - {model}: {mem}GB")
    logger.info(f"  总计: {GPU_MEMORY_ALLOCATION['training']['total']}GB")
    logger.info("")
    logger.info(f"回测任务: {GPU_MEMORY_ALLOCATION['backtest']['per_task']}GB/任务, "
               f"最多{GPU_MEMORY_ALLOCATION['backtest']['max_concurrent']}个并发")
    logger.info(f"预测任务: {GPU_MEMORY_ALLOCATION['prediction']['per_task']}GB/任务, "
               f"最多{GPU_MEMORY_ALLOCATION['prediction']['max_concurrent']}个并发")
    logger.info("")
    logger.info("GPU并发策略:")
    logger.info("  ✅ LightGBM: 支持多任务并发（独立CUDA流）")
    logger.info("  ✅ XGBoost: 支持多任务并发（自动内存管理）")
    logger.info("  ✅ CatBoost: 支持多任务并发（gpu_ram_part=0.5）")
    logger.info("  ⚠️ Informer-2: 有限并发（显存池管理）")
    logger.info("=" * 70)
