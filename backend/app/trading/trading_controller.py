"""
交易控制器
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from app.core.config import settings
from app.core.cache import cache_manager
from app.trading.trading_engine import TradingEngine, TradingMode
from app.trading.signal_generator import SignalGenerator, TradingSignal
from app.trading.position_manager import position_manager
from app.model.ml_service import MLService
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """系统状态"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"

@dataclass
class TradingSession:
    """交易会话"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    mode: TradingMode
    signals_generated: int
    trades_executed: int
    total_pnl: float
    status: str

class TradingController:
    """交易控制器"""
    
    def __init__(
        self, 
        trading_engine: TradingEngine,
        signal_generator: SignalGenerator,
        ml_service: MLService,
        data_service: DataService
    ):
        self.trading_engine = trading_engine
        self.signal_generator = signal_generator
        self.ml_service = ml_service
        self.data_service = data_service
        
        self.system_state = SystemState.STOPPED
        self.current_session: Optional[TradingSession] = None
        self.signal_callbacks: List[callable] = []
        
        # 注册信号回调
        self.signal_generator.add_signal_callback(self._on_signal_generated)
    
    async def start_system(self) -> Dict[str, Any]:
        """启动交易系统"""
        try:
            logger.info("启动交易系统...")
            
            self.system_state = SystemState.STARTING
            
            # 启动各个服务
            if not self.data_service.is_running:
                await self.data_service.start()
            
            if not self.ml_service.is_running:
                await self.ml_service.start()
            
            if not self.trading_engine.is_running:
                await self.trading_engine.start()
            
            if not self.signal_generator.is_running:
                await self.signal_generator.start()
            
            # 初始化仓位管理器
            await position_manager.initialize()
            
            # 创建新的交易会话
            await self._start_new_session()
            
            self.system_state = SystemState.RUNNING
            
            logger.info("交易系统启动完成")
            
            return {
                'success': True,
                'message': '交易系统启动成功',
                'system_state': self.system_state.value,
                'session_id': self.current_session.session_id if self.current_session else None
            }
            
        except Exception as e:
            logger.error(f"启动交易系统失败: {e}")
            self.system_state = SystemState.ERROR
            
            return {
                'success': False,
                'message': f'启动失败: {str(e)}',
                'system_state': self.system_state.value
            }
    
    async def stop_system(self) -> Dict[str, Any]:
        """停止交易系统"""
        try:
            logger.info("停止交易系统...")
            
            # 结束当前会话
            if self.current_session:
                await self._end_current_session()
            
            # 停止各个服务
            if self.signal_generator.is_running:
                await self.signal_generator.stop()
            
            if self.trading_engine.is_running:
                await self.trading_engine.stop()
            
            # 注意：不停止数据服务和ML服务，它们可能被其他组件使用
            
            self.system_state = SystemState.STOPPED
            
            logger.info("交易系统已停止")
            
            return {
                'success': True,
                'message': '交易系统已停止',
                'system_state': self.system_state.value
            }
            
        except Exception as e:
            logger.error(f"停止交易系统失败: {e}")
            
            return {
                'success': False,
                'message': f'停止失败: {str(e)}'
            }
    
    async def pause_system(self) -> Dict[str, Any]:
        """暂停交易系统"""
        try:
            if self.system_state != SystemState.RUNNING:
                return {
                    'success': False,
                    'message': '系统未在运行状态'
                }
            
            self.system_state = SystemState.PAUSED
            
            logger.info("交易系统已暂停")
            
            return {
                'success': True,
                'message': '交易系统已暂停',
                'system_state': self.system_state.value
            }
            
        except Exception as e:
            logger.error(f"暂停交易系统失败: {e}")
            return {
                'success': False,
                'message': f'暂停失败: {str(e)}'
            }
    
    async def resume_system(self) -> Dict[str, Any]:
        """恢复交易系统"""
        try:
            if self.system_state != SystemState.PAUSED:
                return {
                    'success': False,
                    'message': '系统未在暂停状态'
                }
            
            self.system_state = SystemState.RUNNING
            
            logger.info("交易系统已恢复")
            
            return {
                'success': True,
                'message': '交易系统已恢复',
                'system_state': self.system_state.value
            }
            
        except Exception as e:
            logger.error(f"恢复交易系统失败: {e}")
            return {
                'success': False,
                'message': f'恢复失败: {str(e)}'
            }
    
    async def set_trading_mode(self, mode: str) -> Dict[str, Any]:
        """设置交易模式"""
        try:
            if mode not in ['AUTO', 'SIGNAL_ONLY']:
                return {
                    'success': False,
                    'message': '无效的交易模式'
                }
            
            trading_mode = TradingMode.AUTO if mode == 'AUTO' else TradingMode.SIGNAL_ONLY
            
            self.trading_engine.set_trading_mode(trading_mode)
            
            # 更新会话信息
            if self.current_session:
                self.current_session.mode = trading_mode
            
            logger.info(f"交易模式已设置为: {mode} ({'实盘自动交易' if mode == 'AUTO' else '虚拟信号模式'})")
            
            return {
                'success': True,
                'message': f'交易模式已设置为: {mode}',
                'trading_mode': mode,
                'is_auto': (mode == 'AUTO')  # 返回是否为自动模式（供前端使用）
            }
            
        except Exception as e:
            logger.error(f"设置交易模式失败: {e}")
            return {
                'success': False,
                'message': f'设置失败: {str(e)}'
            }
    
    async def manual_trade(
        self, 
        symbol: str, 
        action: str, 
        quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """手动交易"""
        try:
            if self.system_state not in [SystemState.RUNNING, SystemState.PAUSED]:
                return {
                    'success': False,
                    'message': '系统未运行'
                }
            
            if action == 'CLOSE':
                # 平仓
                result = await self.trading_engine._close_position(symbol)
            elif action in ['LONG', 'SHORT']:
                # 开仓
                if not quantity:
                    return {
                        'success': False,
                        'message': '开仓需要指定数量'
                    }
                
                # 创建手动信号
                current_price = await self._get_current_price(symbol)
                if not current_price:
                    return {
                        'success': False,
                        'message': '无法获取当前价格'
                    }
                
                manual_signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=action,
                    confidence=1.0,  # 手动交易置信度为100%
                    entry_price=current_price,
                    stop_loss=0,  # 手动交易不设置止损止盈
                    take_profit=0,
                    position_size=quantity,
                    timeframe='manual',
                    model_version='manual',
                    metadata={'manual_trade': True}
                )
                
                result = await self.trading_engine.execute_signal(manual_signal)
            else:
                return {
                    'success': False,
                    'message': '无效的交易动作'
                }
            
            # 更新会话统计
            if result.get('success') and self.current_session:
                self.current_session.trades_executed += 1
            
            return result
            
        except Exception as e:
            logger.error(f"手动交易失败: {e}")
            return {
                'success': False,
                'message': f'手动交易失败: {str(e)}'
            }
    
    async def force_generate_signal(self, symbol: str) -> Dict[str, Any]:
        """强制生成信号"""
        try:
            if self.system_state not in [SystemState.RUNNING, SystemState.PAUSED]:
                return {
                    'success': False,
                    'message': '系统未运行'
                }
            
            signal = await self.signal_generator.force_generate_signal(symbol)
            
            if signal:
                return {
                    'success': True,
                    'message': '信号生成成功',
                    'signal': {
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type,
                        'confidence': signal.confidence,
                        'entry_price': signal.entry_price,
                        'timestamp': signal.timestamp.isoformat()
                    }
                }
            else:
                return {
                    'success': False,
                    'message': '未生成有效信号'
                }
            
        except Exception as e:
            logger.error(f"强制生成信号失败: {e}")
            return {
                'success': False,
                'message': f'生成信号失败: {str(e)}'
            }
    
    async def _on_signal_generated(self, signal: TradingSignal):
        """处理生成的信号"""
        try:
            logger.info(f"收到交易信号: {signal.signal_type} {signal.symbol} (置信度: {signal.confidence:.4f})")
            
            # 🔥 自动创建会话（如果不存在）
            if not self.current_session:
                await self._start_new_session()
            
            # 更新会话统计
            if self.current_session:
                self.current_session.signals_generated += 1
            
            # 🔥 直接执行信号（不检查系统状态）
            # trading_engine 内部会根据 TRADING_MODE 决定虚拟/实盘交易
            result = await self.trading_engine.execute_signal(signal)
            
            if result.get('success'):
                logger.info(f"信号执行成功: {signal.symbol}")
                
                # 更新会话统计
                if self.current_session:
                    self.current_session.trades_executed += 1
            else:
                logger.warning(f"信号执行失败: {result.get('message', 'Unknown error')}")
            
            # 通知回调函数
            for callback in self.signal_callbacks:
                try:
                    await callback(signal)
                except Exception as e:
                    logger.error(f"信号回调失败: {e}")
            
        except Exception as e:
            logger.error(f"处理信号失败: {e}")
    
    async def _start_new_session(self):
        """开始新的交易会话"""
        try:
            session_id = f"session_{int(datetime.now().timestamp())}"
            
            self.current_session = TradingSession(
                session_id=session_id,
                start_time=datetime.now(),
                end_time=None,
                mode=self.trading_engine.trading_mode,
                signals_generated=0,
                trades_executed=0,
                total_pnl=0.0,
                status='ACTIVE'
            )
            
            logger.info(f"新交易会话开始: {session_id}")
            
        except Exception as e:
            logger.error(f"开始新会话失败: {e}")
    
    async def _end_current_session(self):
        """结束当前交易会话"""
        try:
            if not self.current_session:
                return
            
            self.current_session.end_time = datetime.now()
            self.current_session.status = 'COMPLETED'
            
            # 计算总盈亏
            positions = await position_manager.get_all_positions()
            total_pnl = sum(p.unrealized_pnl for p in positions)
            self.current_session.total_pnl = total_pnl
            
            logger.info(f"交易会话结束: {self.current_session.session_id}")
            
            # 这里可以保存会话数据到数据库
            
        except Exception as e:
            logger.error(f"结束会话失败: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            # 从缓存获取
            ticker_data = await cache_manager.get_market_data(symbol, "ticker")
            
            if ticker_data:
                return float(ticker_data.get('price', 0))
            
            # 从数据服务获取
            latest_klines = await self.data_service.get_latest_klines(symbol, '1m', limit=1)
            
            if latest_klines:
                # 🔥 UnifiedKlineData是对象，使用属性访问而不是索引
                kline = latest_klines[0]
                if isinstance(kline, dict):
                    return float(kline.get('close', 0))
                else:
                    return float(kline.close)
            
            return None
            
        except Exception as e:
            logger.error(f"获取当前价格失败: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # 从 trading_mode 判断是否为自动交易
            is_auto = (self.trading_engine.trading_mode == TradingMode.AUTO)
            
            status = {
                'system_state': self.system_state.value,
                'trading_mode': self.trading_engine.trading_mode.value,
                'auto_trading_enabled': is_auto,  # 从 trading_mode 派生，保持向后兼容
                'services': {
                    'data_service': self.data_service.is_running,
                    'ml_service': self.ml_service.is_running,
                    'trading_engine': self.trading_engine.is_running,
                    'signal_generator': self.signal_generator.is_running
                }
            }
            
            # 添加会话信息
            if self.current_session:
                status['current_session'] = {
                    'session_id': self.current_session.session_id,
                    'start_time': self.current_session.start_time.isoformat(),
                    'signals_generated': self.current_session.signals_generated,
                    'trades_executed': self.current_session.trades_executed,
                    'total_pnl': self.current_session.total_pnl,
                    'status': self.current_session.status
                }
            
            return status
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                'system_state': 'ERROR',
                'error': str(e)
            }
    
    def add_signal_callback(self, callback: callable):
        """添加信号回调函数"""
        self.signal_callbacks.append(callback)
    
    def remove_signal_callback(self, callback: callable):
        """移除信号回调函数"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    async def get_trading_performance(self) -> Dict[str, Any]:
        """获取交易表现"""
        try:
            # 获取持仓摘要
            position_summary = await position_manager.get_position_summary()
            
            # 获取信号表现
            signal_performance = await self.signal_generator.get_signal_performance(
                settings.SYMBOL, days=7
            )
            
            # 获取交易状态
            trading_status = self.trading_engine.get_trading_status()
            
            performance = {
                'position_summary': position_summary,
                'signal_performance': signal_performance,
                'trading_status': trading_status,
                'system_status': self.get_system_status()
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"获取交易表现失败: {e}")
            return {}