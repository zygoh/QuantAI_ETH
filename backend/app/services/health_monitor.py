"""
系统健康监控服务
每5分钟自动检查系统健康状态
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd

from app.core.config import settings
from app.core.database import postgresql_manager
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)

class HealthMonitor:
    """健康监控器"""
    
    def __init__(self):
        self.is_running = False
        self.check_interval = 300  # 5分钟 = 300秒
        self.monitor_task = None
        self.last_check_time = None
        self.signal_generator = None  # 将由主程序设置
        self.health_status = {
            'overall': 'UNKNOWN',
            'timestamp': None,
            'services': {},
            'details': {}
        }
        # 🔥 保存上次状态，用于检测变化
        self.last_overall_status = 'UNKNOWN'
    
    def set_signal_generator(self, sg):
        """设置信号生成器引用"""
        self.signal_generator = sg
    
    async def start(self):
        """启动健康监控"""
        try:
            logger.info(f"启动健康监控服务（检查间隔: {self.check_interval}秒）")
            
            self.is_running = True
            
            # 不在启动时立即检查，等待第一个定时周期
            # await self.check_system_health()  # 注释掉，避免启动时数据未完成
            
            # 启动定期检查任务（第一次检查在5分钟后）
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            
            logger.info("健康监控服务启动完成")
            
        except Exception as e:
            logger.error(f"启动健康监控服务失败: {e}")
            raise
    
    async def stop(self):
        """停止健康监控"""
        try:
            logger.info("停止健康监控服务...")
            
            self.is_running = False
            
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("健康监控服务已停止")
            
        except Exception as e:
            logger.error(f"停止健康监控服务失败: {e}")
    
    async def _monitor_loop(self):
        """监控循环"""
        try:
            while self.is_running:
                try:
                    # 等待检查间隔
                    await asyncio.sleep(self.check_interval)
                    
                    if not self.is_running:
                        break
                    
                    # 执行健康检查
                    await self.check_system_health()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"健康检查失败: {e}")
                    # 继续运行，不中断监控
                    
        except asyncio.CancelledError:
            logger.info("健康监控循环已取消")
    
    async def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        try:
            check_time = datetime.now()
            self.last_check_time = check_time
            
            # 检查各个组件
            db_status = await self._check_database()
            websocket_status = await self._check_websocket_data()
            model_status = await self._check_model()
            cache_status = await self._check_cache()
            
            # 汇总状态
            services = {
                'database': db_status['healthy'],
                'websocket': websocket_status['healthy'],
                'model': model_status['healthy'],
                'cache': cache_status['healthy'],
                'postgresql': db_status['connected'],
                'redis': cache_status['connected']
            }
            
            # 判断整体状态（🔥 优化：以WebSocket缓冲区为核心）
            # 关键服务：WebSocket缓冲区（预测数据源）、缓存（系统通信）
            critical_services = ['websocket', 'cache']
            all_critical_ok = all(services[s] for s in critical_services)
            
            # 数据库和模型是辅助服务，不影响核心功能
            auxiliary_ok = services['database'] and services['model']
            
            if all_critical_ok and auxiliary_ok:
                overall = 'HEALTHY'
            elif all_critical_ok:
                overall = 'DEGRADED'  # 核心正常，辅助有问题
            else:
                overall = 'UNHEALTHY'  # 核心服务有问题
            
            # 组装健康状态
            self.health_status = {
                'overall': overall,
                'timestamp': check_time.isoformat(),
                'services': services,
                'details': {
                    'database': db_status,
                    'websocket': websocket_status,
                    'model': model_status,
                    'cache': cache_status
                },
                'last_check': check_time.isoformat(),
                'next_check': (check_time + timedelta(seconds=self.check_interval)).isoformat()
            }
            
            # 缓存健康状态
            await cache_manager.set_system_status(self.health_status)
            
            # 🔥 优化日志：只在状态变化时输出详细信息
            status_changed = (overall != self.last_overall_status)
            
            if status_changed:
                # 状态变化，输出详细信息
                status_icon = "✅" if overall == "HEALTHY" else "⚠️" if overall == "DEGRADED" else "❌"
                logger.info(f"{status_icon} 系统健康检查完成: {overall} (状态变化: {self.last_overall_status} → {overall})")
                logger.info(f"   数据库: {'✅' if db_status['healthy'] else '❌'} (总记录: {db_status.get('total_records', 0)}条)")
                
                # 打印各时间框架的数据量
                if db_status.get('timeframes'):
                    for interval, tf_status in db_status['timeframes'].items():
                        records = tf_status.get('records', 0)
                        age = tf_status.get('age_minutes')
                        freshness = '🟢' if tf_status.get('fresh') else '🟡'
                        age_str = f"{age:.1f}分钟前" if age is not None else "无数据"
                        logger.info(f"      {interval}: {records}条 {freshness} (最新: {age_str})")
                
                logger.info(f"   WebSocket: {'✅' if websocket_status['healthy'] else '❌'} (缓冲区: {websocket_status.get('buffer_count', 0)}个)")
                
                # 打印缓冲区详细信息
                if websocket_status.get('buffers'):
                    logger.info(f"   缓冲区详情:")
                    for timeframe, buf_info in websocket_status['buffers'].items():
                        size = buf_info.get('size', 0)
                        healthy_icon = '✅' if buf_info.get('healthy') else '⚠️'
                        logger.info(f"      {timeframe}: {size}条数据 {healthy_icon}")
                
                # 安全处理可能为 None 的 accuracy 值
                accuracy = model_status.get('accuracy')
                accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
                logger.info(f"   模型: {'✅' if model_status['healthy'] else '❌'} (准确率: {accuracy_str})")
                logger.info(f"   缓存: {'✅' if cache_status['healthy'] else '❌'}")
                
                # 更新上次状态
                self.last_overall_status = overall
            else:
                # 状态未变化，只输出简洁摘要
                logger.debug(f"✅ 健康检查: {overall} (WebSocket: {'✅' if websocket_status['healthy'] else '❌'}, 缓冲区: {websocket_status.get('buffer_count', 0)}个)")
            
            return self.health_status
            
        except Exception as e:
            logger.error(f"系统健康检查失败: {e}")
            return {
                'overall': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """检查数据库状态"""
        try:
            # 🔥 修复：使用health_check而不是connect，避免重复创建连接
            try:
                connected = await postgresql_manager.health_check()
            except:
                connected = False
            
            if not connected:
                return {
                    'healthy': False,
                    'connected': False,
                    'message': 'PostgreSQL 连接失败'
                }
            
            # 检查数据新鲜度
            symbol = settings.SYMBOL
            timeframes = settings.TIMEFRAMES
            
            has_recent_data = False
            total_records = 0
            timeframe_status = {}
            
            for interval in timeframes:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=1)
                
                df = await postgresql_manager.query_kline_data(
                    symbol, interval, start_time, end_time, limit=100
                )
                
                if not df.empty:
                    latest_time = df['timestamp'].max()
                    
                    # 统一时区处理：转换为 naive datetime
                    if hasattr(latest_time, 'tz_localize'):
                        latest_time_dt = latest_time.tz_localize(None)
                    else:
                        latest_time_dt = latest_time.to_pydatetime()
                        if latest_time_dt.tzinfo is not None:
                            latest_time_dt = latest_time_dt.replace(tzinfo=None)
                    
                    time_diff = (datetime.now() - latest_time_dt).total_seconds() / 60
                    
                    timeframe_status[interval] = {
                        'records': len(df),
                        'latest': latest_time.isoformat(),
                        'age_minutes': round(time_diff, 1),
                        'fresh': time_diff < 60  # 1小时内算新鲜
                    }
                    
                    total_records += len(df)
                    
                    if time_diff < 60:
                        has_recent_data = True
                else:
                    timeframe_status[interval] = {
                        'records': 0,
                        'latest': None,
                        'age_minutes': None,
                        'fresh': False
                    }
            
            # 🔥 修复：不应该关闭全局连接！这会影响整个系统
            # await postgresql_manager.close()  # ❌ 删除：导致系统数据库连接断开
            
            # 🔥 优化判断标准：数据库健康不再是关键指标
            # 原因：禁用首次写入后，数据库只有实时数据，数据少是正常的
            # 真正关键的是WebSocket缓冲区（预测数据源）
            return {
                'healthy': connected,  # 只要连接正常就算健康
                'connected': True,
                'total_records': total_records,
                'timeframes': timeframe_status,
                'has_recent_data': has_recent_data,  # 作为信息，不作为健康判断
                'message': 'PostgreSQL连接正常（数据库仅供前端展示，预测使用WebSocket缓冲区）'
            }
            
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return {
                'healthy': False,
                'connected': False,
                'error': str(e),
                'message': '数据库检查失败'
            }
    
    async def _check_websocket_data(self) -> Dict[str, Any]:
        """检查 WebSocket 数据接收状态（增强版：检查连接状态和数据新鲜度）"""
        try:
            # 检查WebSocket连接状态
            from app.services.binance_client import binance_ws_client
            is_ws_connected = binance_ws_client.is_connected
            
            # 通过检查缓冲区更新判断 WebSocket 是否正常
            if not self.signal_generator:
                return {
                    'healthy': False,
                    'connected': is_ws_connected,
                    'message': '信号生成器未初始化'
                }
            
            # 检查缓冲区
            buffers = self.signal_generator.kline_buffers if hasattr(self.signal_generator, 'kline_buffers') else {}
            
            if not buffers:
                return {
                    'healthy': False,
                    'connected': is_ws_connected,
                    'buffer_count': 0,
                    'message': f'WebSocket缓冲区为空 (连接状态: {is_ws_connected})'
                }
            
            buffer_status = {}
            all_buffers_ok = True
            
            for timeframe, df in buffers.items():
                buffer_size = len(df) if isinstance(df, pd.DataFrame) else 0
                buffer_ok = buffer_size >= 200  # 至少需要200条数据
                
                buffer_status[timeframe] = {
                    'size': buffer_size,
                    'healthy': buffer_ok
                }
                
                if not buffer_ok:
                    all_buffers_ok = False
            
            # 综合判断：连接正常且数据充足
            overall_healthy = is_ws_connected and all_buffers_ok
            
            # 构建消息
            if not is_ws_connected:
                message = '⚠️ WebSocket连接已断开'
            elif not all_buffers_ok:
                message = '部分缓冲区数据不足'
            else:
                message = 'OK'
            
            return {
                'healthy': overall_healthy,
                'connected': is_ws_connected,
                'buffer_count': len(buffers),
                'buffers': buffer_status,
                'message': message
            }
            
        except Exception as e:
            logger.error(f"WebSocket数据检查失败: {e}")
            return {
                'healthy': False,
                'connected': False,
                'error': str(e),
                'message': 'WebSocket检查失败'
            }
    
    async def _check_model(self) -> Dict[str, Any]:
        """检查模型状态"""
        try:
            import os
            
            # 🔥 检查集成模型文件（新格式）
            all_files_exist = True
            missing_files = []
            
            for timeframe in settings.TIMEFRAMES:
                # 检查Stacking集成模型的4个文件
                lgb_path = f"models/{settings.SYMBOL}_{timeframe}_lgb_model.pkl"
                xgb_path = f"models/{settings.SYMBOL}_{timeframe}_xgb_model.pkl"
                cat_path = f"models/{settings.SYMBOL}_{timeframe}_cat_model.pkl"
                meta_path = f"models/{settings.SYMBOL}_{timeframe}_meta_model.pkl"
                scaler_path = f"models/{settings.SYMBOL}_{timeframe}_scaler.pkl"
                features_path = f"models/{settings.SYMBOL}_{timeframe}_features.pkl"
                
                required_files = [
                    (lgb_path, f"{timeframe}_lgb"),
                    (xgb_path, f"{timeframe}_xgb"),
                    (cat_path, f"{timeframe}_cat"),
                    (meta_path, f"{timeframe}_meta"),
                    (scaler_path, f"{timeframe}_scaler"),
                    (features_path, f"{timeframe}_features")
                ]
                
                for file_path, file_name in required_files:
                    if not os.path.exists(file_path):
                        all_files_exist = False
                        missing_files.append(file_name)
            
            # 检查最近的信号
            last_signal = await cache_manager.get_trading_signal(settings.SYMBOL)
            
            if last_signal:
                signal_time = last_signal.get('timestamp')
                has_recent_signal = True
            else:
                signal_time = None
                has_recent_signal = False
            
            # 检查模型指标
            metrics = await cache_manager.get_model_metrics(settings.SYMBOL)
            
            return {
                'healthy': all_files_exist,
                'all_files_exist': all_files_exist,
                'missing_files': missing_files,
                'has_recent_signal': has_recent_signal,
                'last_signal_time': signal_time,
                'has_metrics': metrics is not None,
                'accuracy': metrics.get('accuracy') if metrics else None,
                'message': 'OK' if all_files_exist else f'缺失模型文件: {", ".join(missing_files)}'
            }
            
        except Exception as e:
            logger.error(f"模型健康检查失败: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'message': '模型检查失败'
            }
    
    async def _check_cache(self) -> Dict[str, Any]:
        """检查缓存状态"""
        try:
            # 尝试读写缓存
            test_key = "_health_check_test"
            test_value = datetime.now().isoformat()
            
            # 写入测试
            await cache_manager.redis.client.set(test_key, test_value, ex=10)
            
            # 读取测试
            result = await cache_manager.redis.client.get(test_key)
            
            # 解码结果（Redis返回bytes）
            if result:
                result = result.decode('utf-8') if isinstance(result, bytes) else result
            
            connected = result == test_value
            
            # 清理测试键
            await cache_manager.redis.client.delete(test_key)
            
            # 获取缓存统计
            try:
                stats = await cache_manager.get_cache_stats()
            except:
                stats = {}
            
            return {
                'healthy': connected,
                'connected': connected,
                'stats': stats,
                'message': 'OK' if connected else 'Redis 读写测试失败'
            }
            
        except Exception as e:
            logger.error(f"缓存健康检查失败: {e}")
            return {
                'healthy': False,
                'connected': False,
                'error': str(e),
                'message': 'Redis 连接失败'
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取当前健康状态"""
        return self.health_status

# 全局健康监控器实例
health_monitor = HealthMonitor()

