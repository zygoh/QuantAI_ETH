"""
全局线程池执行器
用于处理计算密集型任务（预测、训练、回测），避免阻塞主事件循环
"""
# StdLib
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Optional, Coroutine

# Local App
from app.core.constants import EXECUTOR_MAX_WORKERS

logger = logging.getLogger(__name__)


class GlobalExecutor:
    """全局线程池执行器（单例模式）"""
    
    _instance: Optional['GlobalExecutor'] = None
    _executor: Optional[ThreadPoolExecutor] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化线程池（只初始化一次）"""
        if self._executor is None:
            # 🔑 线程池配置（充分利用GPU并行能力）：
            # 硬件：8核CPU + 16GB GPU
            # - max_workers: 从配置常量读取（默认8，充分利用8核CPU）
            # - GPU并发策略：
            #   * LightGBM/XGBoost/CatBoost 支持多任务并发（独立CUDA流）
            #   * 1个训练任务（8-12GB显存）
            #   * 3-4个回测任务（2-3GB显存/任务，GPU并行）
            #   * 3-4个预测任务（<1GB显存/任务，GPU轻量）
            # - thread_name_prefix: 便于调试
            self._executor = ThreadPoolExecutor(
                max_workers=EXECUTOR_MAX_WORKERS,
                thread_name_prefix='compute_'
            )
            logger.info(f"✅ 全局线程池初始化完成: max_workers={EXECUTOR_MAX_WORKERS} (8核CPU + 16GB GPU)")
            logger.info("   GPU并发策略: LightGBM/XGBoost/CatBoost 支持多任务并发")
            logger.info("   支持并发: 1个训练 + 3-4个回测 + 3-4个预测（充分利用GPU）")
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """获取线程池实例"""
        if self._executor is None:
            raise RuntimeError("线程池未初始化")
        return self._executor
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """
        在线程池中执行同步函数（异步接口）
        
        Args:
            func: 要执行的同步函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            函数执行结果
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )
    
    async def run_async_in_thread(self, coro_func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """
        在独立线程中执行异步函数（创建新的事件循环）
        
        用于需要数据库操作的计算密集型任务（训练、回测）
        
        Args:
            coro_func: 异步函数（返回 coroutine）
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            函数执行结果
        """
        def run_in_new_loop():
            """在新线程中创建新的事件循环并执行异步函数"""
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 执行异步函数
                return loop.run_until_complete(coro_func(*args, **kwargs))
            finally:
                # 关闭事件循环
                loop.close()
        
        # 在线程池中执行
        main_loop = asyncio.get_event_loop()
        return await main_loop.run_in_executor(
            self._executor,
            run_in_new_loop
        )
    
    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        if self._executor:
            logger.info("🛑 关闭全局线程池...")
            self._executor.shutdown(wait=wait)
            self._executor = None
            logger.info("✅ 全局线程池已关闭")


# 全局单例实例
global_executor = GlobalExecutor()
