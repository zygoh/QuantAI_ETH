"""
回撤监控器
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.database import postgresql_manager
from app.core.cache import cache_manager
from app.trading.position_manager import position_manager

logger = logging.getLogger(__name__)

@dataclass
class DrawdownEvent:
    """回撤事件"""
    start_time: datetime
    end_time: Optional[datetime]
    peak_value: float
    trough_value: float
    max_drawdown: float
    duration_days: int
    recovery_time: Optional[int]  # 恢复时间（天）
    is_active: bool

@dataclass
class DrawdownAlert:
    """回撤警报"""
    timestamp: datetime
    alert_type: str  # WARNING, CRITICAL, EMERGENCY
    current_drawdown: float
    threshold: float
    message: str
    recommended_action: str

class DrawdownMonitor:
    """回撤监控器"""
    
    def __init__(self):
        self.is_running = False
        self.monitor_task = None
        
        # 回撤阈值设置
        self.warning_threshold = 0.05    # 5%警告
        self.critical_threshold = 0.10   # 10%严重
        self.emergency_threshold = 0.15  # 15%紧急
        
        # 监控参数
        self.check_interval = 60  # 检查间隔（秒）
        self.lookback_days = 30   # 回看天数
        
        # 回撤历史
        self.drawdown_events: List[DrawdownEvent] = []
        self.current_drawdown_event: Optional[DrawdownEvent] = None
        
        # 警报回调
        self.alert_callbacks: List[callable] = []
    
    async def start(self):
        """启动回撤监控"""
        try:
            logger.info("启动回撤监控器...")
            
            self.is_running = True
            
            # 加载历史回撤事件
            await self._load_historical_drawdowns()
            
            # 启动监控任务
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            
            logger.info("回撤监控器启动完成")
            
        except Exception as e:
            logger.error(f"启动回撤监控器失败: {e}")
            raise
    
    async def stop(self):
        """停止回撤监控"""
        try:
            logger.info("停止回撤监控器...")
            
            self.is_running = False
            
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("回撤监控器已停止")
            
        except Exception as e:
            logger.error(f"停止回撤监控器失败: {e}")
    
    async def _monitor_loop(self):
        """监控循环"""
        try:
            while self.is_running:
                try:
                    # 检查当前回撤
                    await self._check_current_drawdown()
                    
                    # 等待下次检查
                    await asyncio.sleep(self.check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"回撤监控循环错误: {e}")
                    await asyncio.sleep(self.check_interval)
                    
        except asyncio.CancelledError:
            logger.info("回撤监控循环已取消")
    
    async def _check_current_drawdown(self):
        """检查当前回撤"""
        try:
            # 获取账户权益曲线
            equity_curve = await self._get_equity_curve()
            
            if equity_curve.empty:
                return
            
            # 计算回撤
            drawdown_series = self._calculate_drawdown_series(equity_curve)
            
            if drawdown_series.empty:
                return
            
            current_drawdown = abs(drawdown_series.iloc[-1])
            
            # 检查是否需要发出警报
            await self._check_drawdown_alerts(current_drawdown)
            
            # 更新回撤事件
            await self._update_drawdown_events(equity_curve, drawdown_series)
            
            # 缓存当前回撤
            await cache_manager.set_cache(
                'current_drawdown',
                str(current_drawdown),
                expire=300
            )
            
        except Exception as e:
            logger.error(f"检查当前回撤失败: {e}")
    
    async def _get_equity_curve(self) -> pd.DataFrame:
        """获取账户权益曲线"""
        try:
            # 获取账户历史数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.lookback_days)
            
            # 这里应该从数据库获取账户权益历史
            # 简化处理：使用ETH价格作为权益代理
            symbol = settings.SYMBOL
            
            df = await postgresql_manager.query_kline_data(
                symbol, '1h', start_time, end_time, limit=self.lookback_days * 24
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 使用收盘价作为权益代理
            equity_df = pd.DataFrame({
                'timestamp': df['timestamp'],
                'equity': df['close']
            })
            
            return equity_df
            
        except Exception as e:
            logger.error(f"获取权益曲线失败: {e}")
            return pd.DataFrame()
    
    def _calculate_drawdown_series(self, equity_curve: pd.DataFrame) -> pd.Series:
        """计算回撤序列"""
        try:
            if equity_curve.empty:
                return pd.Series()
            
            equity = equity_curve['equity']
            
            # 计算累积最高点
            peak = equity.expanding().max()
            
            # 计算回撤
            drawdown = (equity - peak) / peak * 100
            
            return drawdown
            
        except Exception as e:
            logger.error(f"计算回撤序列失败: {e}")
            return pd.Series()
    
    async def _check_drawdown_alerts(self, current_drawdown: float):
        """检查回撤警报"""
        try:
            alert_type = None
            threshold = 0
            message = ""
            recommended_action = ""
            
            if current_drawdown >= self.emergency_threshold * 100:
                alert_type = "EMERGENCY"
                threshold = self.emergency_threshold * 100
                message = f"紧急回撤警报：当前回撤{current_drawdown:.2f}%，超过紧急阈值{threshold:.2f}%"
                recommended_action = "立即停止交易，检查策略和风险管理"
                
            elif current_drawdown >= self.critical_threshold * 100:
                alert_type = "CRITICAL"
                threshold = self.critical_threshold * 100
                message = f"严重回撤警报：当前回撤{current_drawdown:.2f}%，超过严重阈值{threshold:.2f}%"
                recommended_action = "减少仓位，审查交易策略"
                
            elif current_drawdown >= self.warning_threshold * 100:
                alert_type = "WARNING"
                threshold = self.warning_threshold * 100
                message = f"回撤警告：当前回撤{current_drawdown:.2f}%，超过警告阈值{threshold:.2f}%"
                recommended_action = "密切监控，考虑降低风险"
            
            if alert_type:
                alert = DrawdownAlert(
                    timestamp=datetime.now(),
                    alert_type=alert_type,
                    current_drawdown=current_drawdown,
                    threshold=threshold,
                    message=message,
                    recommended_action=recommended_action
                )
                
                await self._send_alert(alert)
            
        except Exception as e:
            logger.error(f"检查回撤警报失败: {e}")
    
    async def _send_alert(self, alert: DrawdownAlert):
        """发送警报"""
        try:
            logger.warning(f"回撤警报: {alert.message}")
            
            # 缓存警报
            alert_data = {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'current_drawdown': alert.current_drawdown,
                'threshold': alert.threshold,
                'message': alert.message,
                'recommended_action': alert.recommended_action
            }
            
            await cache_manager.set_cache(
                f'drawdown_alert_{int(alert.timestamp.timestamp())}',
                str(alert_data),
                expire=3600
            )
            
            # 通知回调函数
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"警报回调失败: {e}")
            
        except Exception as e:
            logger.error(f"发送警报失败: {e}")
    
    async def _update_drawdown_events(self, equity_curve: pd.DataFrame, drawdown_series: pd.Series):
        """更新回撤事件"""
        try:
            if equity_curve.empty or drawdown_series.empty:
                return
            
            current_drawdown = abs(drawdown_series.iloc[-1])
            current_time = equity_curve['timestamp'].iloc[-1]
            current_equity = equity_curve['equity'].iloc[-1]
            
            # 检查是否开始新的回撤事件
            if current_drawdown > 0.01:  # 回撤超过1%
                if not self.current_drawdown_event:
                    # 开始新的回撤事件
                    peak_idx = drawdown_series.idxmax()  # 找到峰值
                    peak_time = equity_curve['timestamp'].iloc[peak_idx]
                    peak_value = equity_curve['equity'].iloc[peak_idx]
                    
                    self.current_drawdown_event = DrawdownEvent(
                        start_time=peak_time,
                        end_time=None,
                        peak_value=peak_value,
                        trough_value=current_equity,
                        max_drawdown=current_drawdown,
                        duration_days=0,
                        recovery_time=None,
                        is_active=True
                    )
                    
                    logger.info(f"开始新的回撤事件，峰值: {peak_value:.2f}")
                
                else:
                    # 更新现有回撤事件
                    self.current_drawdown_event.trough_value = min(
                        self.current_drawdown_event.trough_value, current_equity
                    )
                    self.current_drawdown_event.max_drawdown = max(
                        self.current_drawdown_event.max_drawdown, current_drawdown
                    )
                    
                    # 计算持续时间
                    duration = (current_time - self.current_drawdown_event.start_time).days
                    self.current_drawdown_event.duration_days = duration
            
            else:
                # 回撤结束
                if self.current_drawdown_event and self.current_drawdown_event.is_active:
                    self.current_drawdown_event.end_time = current_time
                    self.current_drawdown_event.is_active = False
                    
                    # 计算恢复时间
                    if self.current_drawdown_event.end_time:
                        recovery_days = (
                            self.current_drawdown_event.end_time - 
                            self.current_drawdown_event.start_time
                        ).days
                        self.current_drawdown_event.recovery_time = recovery_days
                    
                    # 保存到历史记录
                    self.drawdown_events.append(self.current_drawdown_event)
                    
                    logger.info(f"回撤事件结束，最大回撤: {self.current_drawdown_event.max_drawdown:.2f}%")
                    
                    self.current_drawdown_event = None
            
        except Exception as e:
            logger.error(f"更新回撤事件失败: {e}")
    
    async def _load_historical_drawdowns(self):
        """加载历史回撤事件"""
        try:
            # 这里可以从数据库加载历史回撤事件
            # 简化处理：初始化为空
            self.drawdown_events = []
            
            logger.info("历史回撤事件加载完成")
            
        except Exception as e:
            logger.error(f"加载历史回撤事件失败: {e}")
    
    async def get_drawdown_statistics(self) -> Dict[str, Any]:
        """获取回撤统计"""
        try:
            if not self.drawdown_events:
                return {
                    'total_events': 0,
                    'avg_max_drawdown': 0,
                    'avg_duration': 0,
                    'avg_recovery_time': 0,
                    'worst_drawdown': 0,
                    'longest_duration': 0
                }
            
            # 只统计已结束的事件
            completed_events = [e for e in self.drawdown_events if not e.is_active]
            
            if not completed_events:
                return {
                    'total_events': len(self.drawdown_events),
                    'active_events': len([e for e in self.drawdown_events if e.is_active])
                }
            
            max_drawdowns = [e.max_drawdown for e in completed_events]
            durations = [e.duration_days for e in completed_events]
            recovery_times = [e.recovery_time for e in completed_events if e.recovery_time]
            
            statistics = {
                'total_events': len(completed_events),
                'active_events': len([e for e in self.drawdown_events if e.is_active]),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'avg_duration': np.mean(durations),
                'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0,
                'worst_drawdown': max(max_drawdowns),
                'longest_duration': max(durations),
                'median_drawdown': np.median(max_drawdowns),
                'drawdown_frequency': len(completed_events) / max(self.lookback_days / 30, 1)  # 每月频率
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"获取回撤统计失败: {e}")
            return {}
    
    async def get_current_drawdown_status(self) -> Dict[str, Any]:
        """获取当前回撤状态"""
        try:
            # 获取当前回撤
            cached_drawdown = await cache_manager.get_cache('current_drawdown')
            current_drawdown = float(cached_drawdown) if cached_drawdown else 0
            
            # 风险等级
            risk_level = "LOW"
            if current_drawdown >= self.emergency_threshold * 100:
                risk_level = "EMERGENCY"
            elif current_drawdown >= self.critical_threshold * 100:
                risk_level = "CRITICAL"
            elif current_drawdown >= self.warning_threshold * 100:
                risk_level = "WARNING"
            
            status = {
                'current_drawdown': current_drawdown,
                'risk_level': risk_level,
                'thresholds': {
                    'warning': self.warning_threshold * 100,
                    'critical': self.critical_threshold * 100,
                    'emergency': self.emergency_threshold * 100
                },
                'active_event': None,
                'monitoring_status': 'ACTIVE' if self.is_running else 'INACTIVE'
            }
            
            # 添加活跃回撤事件信息
            if self.current_drawdown_event:
                status['active_event'] = {
                    'start_time': self.current_drawdown_event.start_time.isoformat(),
                    'duration_days': self.current_drawdown_event.duration_days,
                    'max_drawdown': self.current_drawdown_event.max_drawdown,
                    'peak_value': self.current_drawdown_event.peak_value,
                    'trough_value': self.current_drawdown_event.trough_value
                }
            
            return status
            
        except Exception as e:
            logger.error(f"获取当前回撤状态失败: {e}")
            return {
                'error': str(e),
                'monitoring_status': 'ERROR'
            }
    
    def add_alert_callback(self, callback: callable):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: callable):
        """移除警报回调函数"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def update_thresholds(
        self, 
        warning: Optional[float] = None,
        critical: Optional[float] = None,
        emergency: Optional[float] = None
    ):
        """更新回撤阈值"""
        try:
            if warning is not None:
                self.warning_threshold = warning
            if critical is not None:
                self.critical_threshold = critical
            if emergency is not None:
                self.emergency_threshold = emergency
            
            logger.info(f"回撤阈值已更新: 警告={self.warning_threshold:.2%}, "
                       f"严重={self.critical_threshold:.2%}, 紧急={self.emergency_threshold:.2%}")
            
        except Exception as e:
            logger.error(f"更新回撤阈值失败: {e}")

# 全局回撤监控器实例
drawdown_monitor = DrawdownMonitor()