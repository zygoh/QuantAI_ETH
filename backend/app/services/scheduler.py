"""
任务调度器
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta, time as dt_time
import schedule
from dataclasses import dataclass
import pytz

from app.core.config import settings
from app.services.ml_service import MLService
from app.services.data_service import DataService
from app.services.historical_data import historical_data_manager

logger = logging.getLogger(__name__)

@dataclass
class ScheduledTask:
    """调度任务"""
    name: str
    func: Callable
    interval_hours: int = None  # 间隔小时（如果使用间隔模式）
    scheduled_time: Optional[dt_time] = None  # 固定时间（如果使用固定时间模式）
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    is_running: bool = False
    run_count: int = 0
    error_count: int = 0

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, ml_service: MLService, data_service: DataService, signal_generator=None):
        self.ml_service = ml_service
        self.data_service = data_service
        self.signal_generator = signal_generator  # 🔥 添加signal_generator引用
        self.is_running = False
        self.tasks: Dict[str, ScheduledTask] = {}
        self.scheduler_task = None
        
        # 上海时区
        self.shanghai_tz = pytz.timezone('Asia/Shanghai')
        
        # 初始化调度任务
        self._init_scheduled_tasks()
    
    def _init_scheduled_tasks(self):
        """初始化调度任务"""
        try:
            # 模型训练任务（每天00:01执行）
            self.tasks['model_training'] = ScheduledTask(
                name='模型训练',
                func=self._run_model_training,
                scheduled_time=dt_time(0, 1)  # 每天00:01
            )
            
            # 数据更新任务
            self.tasks['data_update'] = ScheduledTask(
                name='数据更新',
                func=self._run_data_update,
                interval_hours=1  # 每小时更新一次数据
            )
            
            # 数据完整性检查任务
            self.tasks['data_integrity_check'] = ScheduledTask(
                name='数据完整性检查',
                func=self._run_data_integrity_check,
                interval_hours=6  # 每6小时检查一次
            )
            
            # 系统健康检查已由 health_monitor 服务独立负责（每5分钟检查）
            # 不在scheduler中重复设置
            
            # 数据清理任务（禁用：只在系统启动时清理，不在运行中清理）
            # self.tasks['data_cleanup'] = ScheduledTask(
            #     name='数据清理',
            #     func=self._run_data_cleanup,
            #     interval_hours=24
            # )
            
            logger.info(f"初始化了{len(self.tasks)}个调度任务（健康检查由health_monitor服务负责，数据清理仅在启动时执行）")
            
        except Exception as e:
            logger.error(f"初始化调度任务失败: {e}")
    
    async def start(self):
        """启动调度器"""
        try:
            logger.info("启动任务调度器...")
            
            self.is_running = True
            
            # 检查是否需要立即训练模型（首次部署或模型不存在）
            await self._check_initial_model_training()
            
            # 计算下次运行时间
            self._calculate_next_run_times()
            
            # 启动调度循环
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            logger.info("任务调度器启动完成")
            
        except Exception as e:
            logger.error(f"启动调度器失败: {e}")
            raise
    
    async def stop(self):
        """停止调度器"""
        try:
            logger.info("停止任务调度器...")
            
            self.is_running = False
            
            # 取消调度任务
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("任务调度器已停止")
            
        except Exception as e:
            logger.error(f"停止调度器失败: {e}")
    
    async def _check_initial_model_training(self):
        """检查是否需要立即进行首次模型训练"""
        try:
            import os
            from app.core.config import settings
            
            # 检查是否存在至少一个时间框架的模型文件
            model_dir = "models"
            has_model = False
            
            if os.path.exists(model_dir):
                for timeframe in settings.TIMEFRAMES:
                    model_file = os.path.join(model_dir, f"{settings.SYMBOL}_{timeframe}_model.pkl")
                    if os.path.exists(model_file):
                        has_model = True
                        break
            
            if not has_model:
                logger.warning("⚠️ 未找到已保存的模型文件，开始首次训练...")
                logger.info("🎓 首次部署：立即执行模型训练（后续将在每天00:01自动训练）")
                
                # 立即执行模型训练
                task = self.tasks.get('model_training')
                if task:
                    await self._execute_task('model_training', task)
                    logger.info("✅ 首次模型训练完成")
            else:
                logger.info("✅ 检测到已保存的模型，跳过首次训练")
                
        except Exception as e:
            logger.error(f"检查初始模型失败: {e}")
    
    def _calculate_next_run_times(self):
        """计算下次运行时间（支持固定时间和间隔时间两种模式）"""
        try:
            current_time = datetime.now(self.shanghai_tz)
            
            for task_name, task in self.tasks.items():
                if task.scheduled_time is not None:
                    # 固定时间模式（如每天00:01）
                    if task.last_run is None:
                        # 首次运行：计算下一个00:01时刻
                        next_scheduled = current_time.replace(
                            hour=task.scheduled_time.hour,
                            minute=task.scheduled_time.minute,
                            second=0,
                            microsecond=0
                        )
                        
                        # 如果今天的时间已过，设为明天
                        if next_scheduled <= current_time:
                            next_scheduled += timedelta(days=1)
                        
                        task.next_run = next_scheduled
                        logger.info(f"任务 [{task.name}] 计划于 {next_scheduled.strftime('%Y-%m-%d %H:%M:%S')} 执行")
                    else:
                        # 已运行过：计算下一天的同一时间
                        next_scheduled = task.last_run.replace(
                            hour=task.scheduled_time.hour,
                            minute=task.scheduled_time.minute,
                            second=0,
                            microsecond=0
                        ) + timedelta(days=1)
                        task.next_run = next_scheduled
                
                elif task.interval_hours is not None:
                    # 间隔时间模式
                    if task.last_run is None:
                        # 首次运行策略（延迟执行，避免与模型训练冲突）
                        if task_name == 'data_integrity_check':
                            # 数据完整性检查：延迟1小时
                            task.next_run = current_time + timedelta(hours=1)
                            logger.info(f"任务 [{task.name}] 计划于 {task.next_run.strftime('%Y-%m-%d %H:%M:%S')} 首次执行（延迟1小时）")
                        elif task_name == 'data_update':
                            # 数据更新：延迟5分钟（等待模型训练完成）
                            task.next_run = current_time + timedelta(minutes=5)
                            logger.info(f"任务 [{task.name}] 计划于 {task.next_run.strftime('%Y-%m-%d %H:%M:%S')} 首次执行（延迟5分钟）")
                        else:
                            # 其他任务：立即执行
                            task.next_run = current_time
                    else:
                        # 根据间隔计算下次运行时间
                        task.next_run = task.last_run + timedelta(hours=task.interval_hours)
            
        except Exception as e:
            logger.error(f"计算下次运行时间失败: {e}")
    
    async def _scheduler_loop(self):
        """调度循环"""
        try:
            while self.is_running:
                try:
                    # 使用上海时区（与_calculate_next_run_times保持一致）
                    current_time = datetime.now(self.shanghai_tz)
                    
                    # 检查需要运行的任务
                    for task_name, task in self.tasks.items():
                        if (task.next_run and 
                            current_time >= task.next_run and 
                            not task.is_running):
                            
                            # 异步执行任务
                            asyncio.create_task(self._execute_task(task_name, task))
                    
                    # 等待一分钟后再次检查
                    await asyncio.sleep(60)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"调度循环错误: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("调度循环已取消")
    
    async def _execute_task(self, task_name: str, task: ScheduledTask):
        """执行任务"""
        try:
            logger.info(f"开始执行任务: {task.name}")
            
            task.is_running = True
            task.last_run = datetime.now(self.shanghai_tz)
            
            # 执行任务函数
            await task.func()
            
            task.run_count += 1
            
            # 计算下次运行时间
            self._calculate_next_run_times()
            
            logger.info(f"任务执行完成: {task.name}")
            
        except Exception as e:
            logger.error(f"执行任务失败: {task.name} - {e}")
            task.error_count += 1
            
            # 错误后延迟重试（1小时后）
            task.next_run = datetime.now(self.shanghai_tz) + timedelta(hours=1)
            
        finally:
            task.is_running = False
    
    async def _run_model_training(self):
        """运行模型训练任务"""
        try:
            logger.info("开始自动模型训练")
            
            # 检查是否有足够的新数据
            if await self._should_retrain_model():
                metrics = await self.ml_service.train_model()
                
                if metrics:
                    logger.info(f"模型训练完成，准确率: {metrics.get('accuracy', 0):.4f}")
                    logger.info("💡 定期训练完成，模型已自动更新（预热状态不变，继续正常交易）")
                else:
                    logger.warning("模型训练失败")
            else:
                logger.info("数据不足，跳过模型训练")
                
        except Exception as e:
            logger.error(f"自动模型训练失败: {e}")
            raise
    
    async def _run_data_update(self):
        """运行数据更新任务"""
        try:
            logger.info("开始数据更新")
            
            symbol = settings.SYMBOL
            
            # 更新最近24小时的数据
            await historical_data_manager.update_recent_data(symbol, hours=24)
            
            logger.info("数据更新完成")
            
        except Exception as e:
            logger.error(f"数据更新失败: {e}")
            raise
    
    async def _run_data_integrity_check(self):
        """运行数据完整性检查"""
        try:
            logger.info("开始数据完整性检查")
            
            symbol = settings.SYMBOL
            issues = []
            
            for interval in settings.TIMEFRAMES:
                is_valid = await historical_data_manager.validate_data_integrity(
                    symbol, interval, days=7
                )
                
                if not is_valid:
                    issues.append(f"{symbol} {interval}")
            
            if issues:
                logger.warning(f"数据完整性问题: {', '.join(issues)}")
                
                # 自动修复数据
                for issue in issues:
                    parts = issue.split()
                    if len(parts) == 2:
                        symbol, interval = parts
                        await historical_data_manager.fetch_historical_klines(
                            symbol, interval, days=7
                        )
            else:
                logger.info("数据完整性检查通过")
            
        except Exception as e:
            logger.error(f"数据完整性检查失败: {e}")
            raise
    
    # 系统健康检查已由 health_monitor 服务独立负责（每5分钟自动检查）
    # 不在scheduler中重复实现
    
    async def _run_data_cleanup(self):
        """运行数据清理任务"""
        try:
            logger.info("开始数据清理")
            
            # 清理30天前的数据
            await historical_data_manager.cleanup_old_data(days=30)
            
            # 清理缓存
            from app.core.cache import cache_manager
            await cache_manager.clear_cache_pattern("market_data:*")
            
            logger.info("数据清理完成")
            
        except Exception as e:
            logger.error(f"数据清理失败: {e}")
            raise
    
    async def _should_retrain_model(self) -> bool:
        """检查是否应该重新训练模型（多时间框架版本）"""
        try:
            # 检查是否有任何一个时间框架的模型缺失
            if not self.ml_service.models or len(self.ml_service.models) == 0:
                logger.info("📋 模型不存在，需要训练")
                return True
            
            # 检查是否所有时间框架都有模型
            missing_timeframes = []
            for timeframe in settings.TIMEFRAMES:
                if timeframe not in self.ml_service.models or self.ml_service.models[timeframe] is None:
                    missing_timeframes.append(timeframe)
            
            if missing_timeframes:
                logger.info(f"📋 部分时间框架模型缺失: {missing_timeframes}，需要训练")
                return True
            
            # 所有模型都存在，按计划重新训练（保持模型更新）
            logger.info("📋 所有模型已加载，执行定期重新训练")
            return True
            
        except Exception as e:
            logger.error(f"检查重训练条件失败: {e}")
            # 发生错误时保守处理：执行训练
            return True
    
    async def run_task_now(self, task_name: str) -> bool:
        """立即运行指定任务"""
        try:
            task = self.tasks.get(task_name)
            
            if not task:
                logger.error(f"任务不存在: {task_name}")
                return False
            
            if task.is_running:
                logger.warning(f"任务正在运行: {task_name}")
                return False
            
            logger.info(f"手动执行任务: {task.name}")
            
            # 异步执行任务
            asyncio.create_task(self._execute_task(task_name, task))
            
            return True
            
        except Exception as e:
            logger.error(f"手动执行任务失败: {e}")
            return False
    
    def get_task_status(self) -> Dict[str, Any]:
        """获取任务状态"""
        try:
            status = {}
            
            for task_name, task in self.tasks.items():
                status[task_name] = {
                    'name': task.name,
                    'interval_hours': task.interval_hours,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'next_run': task.next_run.isoformat() if task.next_run else None,
                    'is_running': task.is_running,
                    'run_count': task.run_count,
                    'error_count': task.error_count
                }
            
            return status
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            return {}
    
    def update_task_interval(self, task_name: str, interval_hours: int) -> bool:
        """更新任务间隔"""
        try:
            task = self.tasks.get(task_name)
            
            if not task:
                return False
            
            task.interval_hours = interval_hours
            
            # 重新计算下次运行时间
            if task.last_run:
                task.next_run = task.last_run + timedelta(hours=interval_hours)
            
            logger.info(f"更新任务间隔: {task.name} -> {interval_hours}小时")
            
            return True
            
        except Exception as e:
            logger.error(f"更新任务间隔失败: {e}")
            return False