"""
交易信号生成器

职责：
1. 🎯 多时间框架信号生成（3m/5m/15m）
2. 🔄 信号缓存与合成
3. 🔒 预热信号保护（前5个信号仅记录）
4. 📊 WebSocket实时数据处理

注意：
- 仓位计算已委托给 position_manager（避免重复代码）
- 信号生成基于缓存的预测结果，避免重复预测
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.database import postgresql_manager
from app.core.cache import cache_manager
from app.model.ml_service import MLService
from app.services.data_service import DataService, KlineData
from app.exchange.exchange_factory import ExchangeFactory
from app.utils.helpers import format_signal_type
from app.services.risk_service import RiskService
from app.trading.position_manager import position_manager
import pytz

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """交易信号数据类"""
    timestamp: datetime
    symbol: str
    signal_type: str  # LONG, SHORT, CLOSE
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timeframe: str
    model_version: str
    metadata: Dict[str, Any]

class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self, ml_service: MLService, data_service: DataService):
        self.ml_service = ml_service
        self.data_service = data_service
        self.is_running = False
        
        # 🔑 获取交易所客户端（使用工厂模式）
        self.exchange_client = ExchangeFactory.get_current_client()
        self.signal_callbacks: List[callable] = []
        self.last_signals: Dict[str, TradingSignal] = {}
        
        # 信号生成参数
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.min_signal_interval = 180  # 短线策略：3分钟最小信号间隔（180秒，与3m K线周期一致）
        # 🔥 移除预测间隔限制：K线完成时立即预测（实时响应）
        # 原设计：min_prediction_interval = 30秒（防抖）
        # 新设计：K线完成是明确事件，应该立即响应，不需要防抖
        
        # 🔑 预测频率控制（新增）
        self.last_prediction_time: Dict[str, float] = {}  # {timeframe: timestamp}
        
        # 🔑 特征缓存（新增）
        self.feature_cache: Dict[str, pd.DataFrame] = {}  # {cache_key: features}
        self.feature_cache_time: Dict[str, float] = {}  # {cache_key: timestamp}
        self.feature_cache_ttl = 300  # 特征缓存5分钟
        
        # 止损止盈参数（中频交易策略：更紧的止损，保持止盈）
        self.stop_loss_pct = 0.015  # 1.5%止损（中频快速止损，减少风险）
        self.take_profit_pct = 0.04   # 4%止盈（让利润奔跑）
        
        # WebSocket 数据缓冲区（存储实时K线数据）
        self.kline_buffers: Dict[str, pd.DataFrame] = {}  # {timeframe: DataFrame}
        
        # 🔥 信号缓存：每个时间框架独立缓存预测结果
        self.cached_predictions: Dict[str, Dict[str, Any]] = {}  # {timeframe: prediction}
        
        # 🔒 安全保护：前5个信号仅记录，不交易（仅首次部署时启用）
        self.warmup_signals = 5  # 预热信号数量
        self.signal_counter = 0  # 信号计数器（启动时会从Redis加载）
        
        # 缓冲区设计：按天数统一（所有时间框架覆盖相同天数）
        self.buffer_days = 30  # 统一30天覆盖范围（超短线策略，较短缓冲）
        
        # 根据时间框架计算实际需要的K线数量
        self.buffer_sizes = {
            '3m': int(self.buffer_days * 24 * 20),   # 30天 = 14400条（3分钟）
            '5m': int(self.buffer_days * 24 * 12),   # 30天 = 8640条（5分钟）
            '15m': int(self.buffer_days * 24 * 4)    # 30天 = 2880条（15分钟）
        }
        
        # 注册数据回调
        self.data_service.add_data_callback(self._on_new_data)
        
        # 注册WebSocket重连回调
        self.data_service.add_reconnect_callback(self._on_websocket_reconnect)
    
    async def start(self):
        """启动信号生成器"""
        try:
            logger.info("启动交易信号生成器...")
            
            # 初始化WebSocket数据缓冲区
            await self._initialize_kline_buffers()
            
            self.is_running = True
            
            # 🔒 从Redis加载预热状态（持久化，避免重启后重新预热）
            await self._load_warmup_state()
            
            # 🔒 启动安全保护提示
            if self.signal_counter < self.warmup_signals:
                logger.warning(f"🔒 安全保护已启用：前 {self.warmup_signals} 个信号仅记录，不执行交易")
                logger.info(f"   当前已完成: {self.signal_counter}/{self.warmup_signals} 个信号")
                logger.info(f"   目的：观察模型稳定性，确保资金安全")
            else:
                logger.info(f"✅ 预热已完成（{self.signal_counter}个信号），系统处于正常交易模式")
            
            # 🔥 首次启动：立即对所有时间框架进行预测，填充信号缓存
            await self._initial_predictions()
            
            logger.info("✅ 交易信号生成器启动完成")
        except Exception as e:
            logger.error(f"启动信号生成器失败: {e}")
            raise
    
    async def _initialize_kline_buffers(self):
        """初始化K线数据缓冲区 - 从API获取初始数据"""
        try:
            symbol = settings.SYMBOL
            logger.info(f"初始化WebSocket数据缓冲区: {symbol}")
            
            for timeframe in settings.TIMEFRAMES:
                try:
                    # 获取该时间框架需要的K线数量
                    buffer_size = self.buffer_sizes.get(timeframe, 500)
                    
                    # Binance API limit 最大1500条，需要分批获取
                    max_limit = 1500
                    all_klines = []
                    
                    # ✅ 统一使用分页方法（自动处理超过1500的情况）
                    logger.info(f"获取 {timeframe} 初始数据（{buffer_size}条，覆盖{self.buffer_days}天）...")
                    all_klines = self.exchange_client.get_klines_paginated(
                            symbol=symbol,
                            interval=timeframe,
                        limit=buffer_size,
                        rate_limit_delay=0.2
                    )
                    
                    if all_klines:
                        # 初始化缓冲区
                        df = pd.DataFrame(all_klines)
                        
                        # ✅ timestamp 保持为整数（毫秒时间戳），不转换
                        # 与 WebSocket 新数据保持一致（KlineData.open_time 现在是 int 类型）
                        
                        self.kline_buffers[timeframe] = df
                        days_covered = len(all_klines) / self.buffer_sizes.get(timeframe, 1) * self.buffer_days
                        logger.info(f"✓ {timeframe} 缓冲区初始化完成: {len(all_klines)}条数据（约{days_covered:.1f}天）")
                    else:
                        logger.warning(f"⚠️ {timeframe} 初始数据获取失败")
                    
                    # API限流延迟
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"初始化 {timeframe} 缓冲区失败: {e}")
            
            logger.info(f"WebSocket数据缓冲区初始化完成: {len(self.kline_buffers)}个时间框架")
            
        except Exception as e:
            logger.error(f"初始化K线缓冲区失败: {e}")
    
    async def stop(self):
        """停止信号生成器"""
        try:
            logger.info("停止交易信号生成器...")
            self.is_running = False
            logger.info("交易信号生成器已停止")
        except Exception as e:
            logger.error(f"停止信号生成器失败: {e}")
    
    async def _on_websocket_reconnect(self):
        """WebSocket重连回调 - 重置缓冲区"""
        try:
            logger.warning("⚠️ WebSocket已重连，开始重置缓冲区...")
            logger.info("🔄 原因：重连期间数据可能有缺口，重新获取完整数据以确保质量")
            
            # 清空现有缓冲区
            self.kline_buffers.clear()
            logger.info("✓ 已清空旧缓冲区数据")
            
            # 重新初始化缓冲区（从API获取最新的完整数据）
            await self._initialize_kline_buffers()
            
            logger.info("✅ 缓冲区重置完成，数据质量已恢复")
            
        except Exception as e:
            logger.error(f"WebSocket重连回调失败: {e}")
    
    async def _on_new_data(self, kline_data: KlineData):
        """处理新的K线数据 - 更新缓冲区并预测该时间框架"""
        try:
            logger.info(f"📊 信号生成器收到新K线: {kline_data.symbol} {kline_data.interval} is_closed={kline_data.is_closed}")
            
            if not self.is_running:
                logger.warning("⚠️ 信号生成器未运行，跳过处理")
                return
            
            # 1. 将WebSocket数据添加到缓冲区
            await self._update_kline_buffer(kline_data)
            
            # 2. 🔥 实时预测：K线完成时立即预测（移除时间间隔限制）
            timeframe = kline_data.interval
            logger.info(f"🎯 {timeframe} K线完成，立即触发预测")
            
            # 3. 对该时间框架进行预测并缓存（每个时间框架独立预测）
            prediction = await self._predict_single_timeframe(kline_data.symbol, timeframe)
            
            if prediction:
                # 缓存该时间框架的预测结果
                self.cached_predictions[timeframe] = prediction
                
                # 更新最后预测时间（用于统计）
                self.last_prediction_time[timeframe] = time.time()
                
                logger.info(f"✅ {timeframe} 预测完成并缓存: {prediction.get('signal_type')} (置信度={prediction.get('confidence'):.4f})")
            else:
                # 预测失败或模型训练中（详细信息已在ml_service层记录）
                logger.warning(f"⏸️ {timeframe} 预测暂不可用（模型可能正在训练中）")
                return
            
            # 4. 🔥 只有5m信号更新时才触发合成（5m作为主时间框架）
            if timeframe != '5m':
                logger.debug(f"⏭️ {timeframe} 信号已缓存，等待5m触发合成")
                return
            
            logger.info(f"🔄 5m信号更新，触发合成 (当前已缓存: {list(self.cached_predictions.keys())})")
            
            # 🔥 预热计数应该在尝试合成前就+1（不管是否HOLD）
            self.signal_counter += 1
            # 💾 保存预热状态到Redis（持久化）
            await self._save_warmup_state()
            
            signal = await self._try_synthesize_cached_signals(kline_data.symbol)
            
            if signal:
                logger.info(f"✅ 生成合成信号: {format_signal_type(signal.signal_type)} 置信度={signal.confidence:.4f} 入场={signal.entry_price:.2f}")
                await self._process_signal(signal)
            else:
                # HOLD或置信度不足
                logger.info(f"⏸️ 未生成交易信号（可能是HOLD或置信度不足）")
                
                # 如果在预热期，也应该记录
                if self.signal_counter <= self.warmup_signals:
                    logger.info(f"ℹ️ 预热期观望 [{self.signal_counter}/{self.warmup_signals}]（信号为HOLD或低置信度）")
                    logger.info(f"   剩余{self.warmup_signals - self.signal_counter}个预热信号")
                
        except Exception as e:
            logger.error(f"❌ 处理新数据失败: {e}", exc_info=True)
    
    async def _update_kline_buffer(self, kline_data: KlineData):
        """更新K线数据缓冲区（同时写入数据库持久化）"""
        try:
            timeframe = kline_data.interval
            
            # 转换为DataFrame行
            new_row = pd.DataFrame([{
                'timestamp': kline_data.open_time,
                'open': kline_data.open_price,
                'high': kline_data.high_price,
                'low': kline_data.low_price,
                'close': kline_data.close_price,
                'volume': kline_data.volume,
                'quote_volume': kline_data.quote_volume
            }])
            
            # 如果缓冲区不存在，初始化
            if timeframe not in self.kline_buffers:
                logger.info(f"初始化 {timeframe} 数据缓冲区")
                self.kline_buffers[timeframe] = new_row
            else:
                # 记录追加前的缓冲区大小
                old_size = len(self.kline_buffers[timeframe])
                old_last_close = self.kline_buffers[timeframe]['close'].iloc[-1]
                
                # 追加新数据
                self.kline_buffers[timeframe] = pd.concat(
                    [self.kline_buffers[timeframe], new_row],
                    ignore_index=True
                )
                
                # 限制缓冲区大小（根据时间框架保持统一天数）
                buffer_size = self.buffer_sizes.get(timeframe, 500)
                if len(self.kline_buffers[timeframe]) > buffer_size:
                    self.kline_buffers[timeframe] = self.kline_buffers[timeframe].tail(buffer_size)
                
                # ✅ 调试日志：验证缓冲区更新
                new_size = len(self.kline_buffers[timeframe])
                new_last_close = self.kline_buffers[timeframe]['close'].iloc[-1]
                logger.debug(f"📈 {timeframe} 缓冲区更新: {old_size}→{new_size}条, 最新收盘价: {old_last_close:.2f}→{new_last_close:.2f}")
            
            # ✅ 写入数据库持久化（PostgreSQL + TimescaleDB）
            try:
                # 直接使用 Binance 的时间戳（毫秒），不做任何转换
                kline_dict = {
                    'symbol': kline_data.symbol,
                    'interval': timeframe,
                    'timestamp': kline_data.open_time,  # ✅ Binance原始时间戳（毫秒）
                    'open': kline_data.open_price,
                    'high': kline_data.high_price,
                    'low': kline_data.low_price,
                    'close': kline_data.close_price,
                    'volume': kline_data.volume,
                    'close_time': kline_data.close_time,  # ✅ Binance原始时间戳（毫秒）
                    'quote_volume': kline_data.quote_volume,
                    'trades': kline_data.trades,  # ✅ 使用真实的trades数据
                    'taker_buy_base_volume': kline_data.taker_buy_base_volume,  # ✅ 主动买入量
                    'taker_buy_quote_volume': kline_data.taker_buy_quote_volume  # ✅ 主动买入额
                }
                
                # 🚀 异步写入数据库（不等待完成，避免阻塞信号生成）
                asyncio.create_task(postgresql_manager.write_kline_data([kline_dict]))
                
                # ✅ 简化日志输出（改为DEBUG级别，减少日志量）
                logger.debug(f"💾 WebSocket数据已提交写入: {timeframe} | trades={kline_data.trades}")
            except Exception as db_error:
                logger.error(f"❌ 写入数据库失败: {db_error}")
                logger.error(f"   K线详情: symbol={kline_dict.get('symbol')} interval={kline_dict.get('interval')} timestamp={kline_dict.get('timestamp')}")
            
            logger.debug(f"📈 更新 {timeframe} 缓冲区完成: 当前{len(self.kline_buffers[timeframe])}条数据")
            
        except Exception as e:
            logger.error(f"更新K线缓冲区失败: {e}")
    
    async def _initial_predictions(self):
        """首次启动时预测所有时间框架并填充缓存（如果模型可用）"""
        try:
            symbol = settings.SYMBOL
            
            # 🔒 检查模型是否可用（可能正在训练中）
            # 兼容EnsembleMLService（使用ensemble_models）和MLService（使用models）
            models_dict = getattr(self.ml_service, 'ensemble_models', None) or getattr(self.ml_service, 'models', None)
            
            if not models_dict or len(models_dict) == 0:
                logger.warning("⚠️ 模型尚未训练完成，跳过首次预测（等待模型训练完成后首次WebSocket触发）")
                return
            
            logger.info(f"🎯 开始首次预测所有时间框架: {settings.TIMEFRAMES}")
            
            for timeframe in settings.TIMEFRAMES:
                try:
                    # 再次确认该时间框架的模型存在
                    if timeframe not in models_dict:
                        logger.warning(f"⚠️ {timeframe} 模型不可用，跳过首次预测")
                        continue
                    
                    prediction = await self._predict_single_timeframe(symbol, timeframe)
                    if prediction:
                        self.cached_predictions[timeframe] = prediction
                        logger.info(f"✅ {timeframe} 首次预测完成: {format_signal_type(prediction.get('signal_type'))} (置信度={prediction.get('confidence'):.4f})")
                    else:
                        logger.warning(f"⚠️ {timeframe} 首次预测返回空结果")
                except Exception as e:
                    logger.warning(f"⚠️ {timeframe} 首次预测异常（不影响系统运行）: {e}")
            
            logger.info(f"✅ 首次预测完成，已缓存 {len(self.cached_predictions)}/{len(settings.TIMEFRAMES)} 个时间框架")
            
            # 🔥 首次预测只填充缓存，不生成信号
            # 首次信号应该等待实时K线完成后再生成（由5m K线完成触发）
            logger.info("💡 首次预测仅用于初始化缓存，等待实时K线完成后再生成信号")
            
        except Exception as e:
            logger.warning(f"⚠️ 首次预测失败（不影响系统运行）: {e}")
    
    async def _predict_single_timeframe(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """预测单个时间框架"""
        try:
            # 确定需要的数据量（超短线策略：较短周期）
            prediction_days_config = {
                '3m': 10,    # 10天=4800条（超短期）
                '5m': 10,    # 10天=2880条（主时间框架）
                '15m': 10    # 10天=960条（趋势确认）
            }
            prediction_days = prediction_days_config.get(timeframe, 10)
            
            interval_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '6h': 360, '8h': 480,
                '12h': 720, '1d': 1440
            }
            minutes = interval_minutes.get(timeframe, 60)
            required_klines = int((prediction_days * 24 * 60) / minutes)
            
            # 优先使用WebSocket缓冲区
            if timeframe in self.kline_buffers and len(self.kline_buffers[timeframe]) >= required_klines:
                df = self.kline_buffers[timeframe].tail(required_klines).copy()
                logger.debug(f"✓ 使用缓冲区: {timeframe} ({len(df)}条)")
            else:
                # 从API获取（使用统一的分页方法）
                logger.debug(f"⚠️ 缓冲区不足，从API获取: {timeframe} (需要{required_klines}条)")
                klines = self.exchange_client.get_klines_paginated(
                    symbol=symbol,
                    interval=timeframe,
                    limit=required_klines
                )
                if not klines:
                    logger.warning(f"❌ {timeframe} 数据获取失败")
                    return None
                
                df = pd.DataFrame(klines)
                # 🔥 确保timestamp是datetime类型
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if df is None or len(df) < 50:
                logger.warning(f"❌ {timeframe} 数据不足")
                return None
            
            # 调用ML服务预测
            prediction = await self.ml_service.predict(df, timeframe=timeframe)
            return prediction
            
        except Exception as e:
            logger.error(f"预测{timeframe}失败: {e}")
            return None
    
    async def _try_synthesize_cached_signals(self, symbol: str) -> Optional[TradingSignal]:
        """尝试合成所有缓存的信号"""
        try:
            # 检查是否所有时间框架都有缓存
            if not self.cached_predictions:
                logger.debug("❌ 信号缓存为空")
                return None
            
            # 如果不是所有时间框架都有预测，可以继续（使用已有的）
            # ✅ 5m是主时间框架，必须要有；其他时间框架（3m、15m）是辅助的，有则用，无则忽略
            if '5m' not in self.cached_predictions:
                logger.warning("❌ 缺少5m主信号，无法合成")
                return None
            
            # 合成信号（合成过程中的日志已在_synthesize_signal中输出）
            # 只要有5m信号就可以合成，其他时间框架（3m、15m）有则用，无则忽略（辅助作用）
            signal = await self._synthesize_signal(symbol, self.cached_predictions)
            
            # 如果没有信号（HOLD或其他原因），直接返回
            # _synthesize_signal 内部已经记录了详细日志
            if not signal:
                return None
            
            # 检查置信度
            if signal.confidence < self.confidence_threshold:
                logger.info(f"❌ 置信度不足: {signal.confidence:.4f} < {self.confidence_threshold}")
                return None
            
            # 检查信号去重（去重检查中的日志已在_should_send_signal中输出）
            if not await self._should_send_signal(symbol, signal.signal_type):
                return None
            
            return signal
            
        except Exception as e:
            logger.error(f"合成信号失败: {e}")
            return None
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """生成交易信号（基于WebSocket实时数据）"""
        try:
            logger.info(f"🔮 开始生成交易信号: {symbol}")
            logger.debug(f"数据源: WebSocket 实时缓冲区 (优先) / API (备用)")
            
            # 获取多时间框架预测
            predictions = await self._get_multi_timeframe_predictions(symbol)
            
            if not predictions:
                logger.warning(f"未获取到有效预测数据")
                return None
            
            # 合成信号
            signal = await self._synthesize_signal(symbol, predictions)
            
            if not signal or signal.confidence < self.confidence_threshold:
                logger.info(f"❌ 信号置信度不足: {signal.confidence if signal else 0:.4f} < {self.confidence_threshold}")
                return None
            
            # 检查信号去重（从缓存中获取上一次的信号）
            if not await self._should_send_signal(symbol, signal.signal_type):
                logger.info(f"✗ 信号已存在，拒绝重复: {signal.signal_type} {signal.confidence:.4f}")
                return None
            
            logger.info(f"✅ 生成新交易信号: {format_signal_type(signal.signal_type)} 置信度:{signal.confidence:.4f}")
            return signal
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return None
    
    async def _get_multi_timeframe_predictions(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """获取多时间框架预测 - 使用固定天数确保时间对齐"""
        try:
            predictions = {}
            
            # ✅ 差异化预测天数：每个时间框架使用最优配置
            # 原则：确保特征完整（最长窗口200期）+ 适合时间框架特性
            prediction_days_config = {
                '3m': 10,    # 10天=4800条（超短期，高频样本）
                '5m': 10,    # 10天=2880条（主时间框架，快速响应）
                '15m': 10    # 10天=960条（趋势确认）
            }
            
            # 时间周期对应的分钟数
            interval_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '6h': 360, '8h': 480,
                '12h': 720, '1d': 1440
            }
            
            for timeframe in settings.TIMEFRAMES:
                df = None
                data_source = ""
                
                # 根据时间框架使用差异化的预测天数
                prediction_days = prediction_days_config.get(timeframe, 35)
                minutes = interval_minutes.get(timeframe, 60)
                required_klines = int((prediction_days * 24 * 60) / minutes)
                
                # 优先使用WebSocket缓冲区数据
                if timeframe in self.kline_buffers and len(self.kline_buffers[timeframe]) >= required_klines:
                    df = self.kline_buffers[timeframe].tail(required_klines).copy()
                    data_source = "WebSocket缓冲区"
                    logger.debug(f"✓ 使用WebSocket缓冲区: {timeframe} (需要{required_klines}条, 当前{len(df)}条, {prediction_days}天)")
                else:
                    # 缓冲区数据不足，从API获取（使用统一的分页方法）
                    logger.debug(f"⚠️ 缓冲区数据不足({len(self.kline_buffers.get(timeframe, []))}条 < {required_klines}条)，从API获取: {timeframe} (需要{required_klines}条)")
                    klines = self.exchange_client.get_klines_paginated(
                        symbol=symbol,
                        interval=timeframe,
                        limit=required_klines
                    )
                    
                    if not klines:
                        logger.warning(f"❌ 未获取到{timeframe}数据")
                        continue
                    
                    df = pd.DataFrame(klines)
                    # 🔥 确保timestamp是datetime类型
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    data_source = "API备用"
                
                # 🔧 修复：确保有足够数据计算长周期指标（如long_vol需要100周期）
                # 至少需要250条数据，确保所有技术指标都能正常计算
                min_required_klines = 250
                if df is None or len(df) < min_required_klines:
                    logger.warning(f"❌ {timeframe}数据不足: {len(df) if df is not None else 0}条 < {min_required_klines}条（需要足够数据计算长周期指标）")
                    continue
                
                logger.debug(f"🤖 开始{timeframe}模型预测 (数据源: {data_source}, {len(df)}条K线)...")
                
                # 模型预测（传入timeframe使用对应的模型）
                prediction = await self.ml_service.predict(df, timeframe=timeframe)
                
                if prediction:
                    predictions[timeframe] = prediction
                    logger.debug(f"✅ {timeframe}预测完成: {prediction.get('signal_type')} (置信度={prediction.get('confidence'):.4f})")
                else:
                    # 预测失败或模型训练中（详细信息已在ml_service层记录）
                    logger.debug(f"⏸️ {timeframe}预测暂不可用")
            
            return predictions
            
        except Exception as e:
            logger.error(f"获取多时间框架预测失败: {e}")
            return {}
    
    async def _synthesize_signal(
        self, 
        symbol: str, 
        predictions: Dict[str, Dict[str, Any]]
    ) -> Optional[TradingSignal]:
        """合成多时间框架信号"""
        try:
            if not predictions:
                logger.warning("⚠️ 没有可用的预测数据，无法合成信号")
                return None
            
            # 时间框架权重（超短线交易策略：以5m为主导）
            # 差异化训练天数后的数据量：
            # 3m:  57,600条/120天 (训练46k)  ✅ 超短期辅助，快速反应
            # 5m:  34,560条/120天 (训练27.6k) ✅ 主导，捕捉短期入场点
            # 15m: 11,520条/120天 (训练9.2k)  ✅ 中期辅助，趋势过滤
            timeframe_weights = {
                '3m': 0.15,    # 超短期辅助：捕捉极短期波动
                '5m': 0.70,    # 🎯 短线主导：提高权重，快速捕捉入场点
                '15m': 0.15    # 中期辅助：趋势确认，避免逆势交易
            }
            
            # 计算加权信号（动态权重：长周期HOLD时降权）
            weighted_scores = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
            total_weight = 0
            
            for timeframe, prediction in predictions.items():
                base_weight = timeframe_weights.get(timeframe, 0.2)
                probabilities = prediction.get('probabilities', {})
                signal = prediction.get('signal_type')
                
                # 🔑 动态权重调整：如果长周期（15m）是HOLD且置信度高，大幅降低权重
                if timeframe in ['15m'] and signal == 'HOLD':
                    hold_confidence = prediction.get('confidence', 0)
                    if hold_confidence > 0.65:
                        # HOLD置信度很高时，权重减半（避免压制5m）
                        weight = base_weight * 0.5
                        logger.debug(f"   {timeframe} HOLD高置信度({hold_confidence:.2f})，权重{base_weight}→{weight}")
                    else:
                        weight = base_weight
                else:
                    weight = base_weight
                
                weighted_scores['LONG'] += probabilities.get('long', 0) * weight
                weighted_scores['SHORT'] += probabilities.get('short', 0) * weight
                weighted_scores['HOLD'] += probabilities.get('hold', 0) * weight
                
                total_weight += weight
            
            # 归一化
            if total_weight > 0:
                for key in weighted_scores:
                    weighted_scores[key] /= total_weight
            
            # 确定最终信号
            signal_type = max(weighted_scores, key=weighted_scores.get)
            confidence = weighted_scores[signal_type]
            
            # 记录合成过程
            logger.info(f"🔄 信号合成: {len(predictions)}个时间框架")
            for tf, pred in predictions.items():
                logger.info(f"  • {tf}: {format_signal_type(pred['signal_type'])} (置信度={pred['confidence']:.4f})")
            logger.info(f"  ➜ 最终: {format_signal_type(signal_type)} (加权置信度={confidence:.4f})")
            
            # 过滤HOLD信号
            if signal_type == 'HOLD':
                logger.info(f"⊗ 最终信号为HOLD，不发出交易信号")
                return None
            
            # 🆕 信号增强过滤（预期胜率+5-10%）
            filter_result = await self._enhanced_signal_filter(
                signal_type=signal_type,
                confidence=confidence,
                predictions=predictions,
                symbol=symbol
            )
            
            if not filter_result['pass']:
                logger.info(f"❌ 信号被过滤: {filter_result['reason']}")
                return None
            
            # 获取当前价格
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.warning("⚠️ 无法获取当前价格，放弃本次信号")
                return None
            
            # 🆕 使用动态止损止盈（基于ATR）
            stop_levels = await RiskService.calculate_dynamic_stop_levels(
                symbol=symbol,
                entry_price=current_price,
                signal_type=signal_type,
                confidence=confidence
            )
            
            if not stop_levels:
                logger.warning("⚠️ 止损止盈计算失败，使用固定百分比")
                # 降级方案
                if signal_type == 'LONG':
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                else:  # SHORT
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
            else:
                stop_loss = stop_levels['stop_loss']
                take_profit = stop_levels['take_profit']
                logger.debug(f"✅ 使用动态止损: 盈亏比 1:{stop_levels.get('risk_reward_ratio', 0):.2f}")
            
            # 🆕 统一使用 position_manager 计算仓位大小（USDT价值）
            # 从 Redis 读取当前交易模式（支持动态切换）
            current_mode = await cache_manager.get("system:trading_mode")
            is_virtual_mode = (current_mode != "AUTO")  # 默认虚拟模式，只有明确是 AUTO 才用实盘
            
            # 🔑 获取仓位大小（直接使用USDT价值，不换算张数）
            position_size = await position_manager.calculate_position_size(
                symbol, signal_type, confidence, current_price,
                is_virtual=is_virtual_mode  # 动态根据 Redis 中的模式决定
            )
            
            logger.debug(f"💰 仓位大小: {position_size:.2f} USDT @ {current_price:.2f}")
            
            # 创建信号对象
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timeframe='multi',
                model_version='1.0',
                metadata={
                    'timeframe_predictions': predictions,
                    'weighted_scores': weighted_scores,
                    'generation_method': 'multi_timeframe_synthesis'
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"合成信号失败: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        获取当前实时价格 - 严格模式，拒绝过时数据
        
        安全原则：宁可放弃信号，也不使用过时价格导致亏损
        """
        try:
            # 🔧 修复：将标准格式转换为交易所格式（如BTC/USDT -> BTCUSDT）
            from app.exchange.mappers import SymbolMapper
            exchange_symbol = SymbolMapper.to_exchange_format(symbol, settings.EXCHANGE_TYPE)
            
            # 方法1：从ticker缓存获取（WebSocket实时推送，有时间戳验证）
            ticker_data = await cache_manager.get_market_data(exchange_symbol, "ticker")
            if ticker_data and ticker_data.get('price'):
                # 检查数据时效性（如果有时间戳）
                ticker_time = ticker_data.get('timestamp')
                if ticker_time:
                    from datetime import datetime, timezone
                    try:
                        if isinstance(ticker_time, (int, float)):
                            data_time = datetime.fromtimestamp(ticker_time / 1000, tz=timezone.utc)
                        else:
                            data_time = datetime.fromisoformat(str(ticker_time).replace('Z', '+00:00'))
                        age_seconds = (datetime.now(timezone.utc) - data_time).total_seconds()
                        if age_seconds > 60:  # 超过60秒视为过时
                            logger.warning(f"⚠️ ticker数据过时 ({age_seconds:.0f}秒)，拒绝使用")
                        else:
                            price = float(ticker_data.get('price'))
                            if price > 0:
                                logger.debug(f"✓ ticker价格: {price} (延迟{age_seconds:.1f}秒)")
                                return price
                    except Exception:
                        pass  # 时间戳解析失败，继续尝试其他方法
                else:
                    # 无时间戳但有价格，谨慎使用
                    price = float(ticker_data.get('price'))
                    if price > 0:
                        logger.debug(f"✓ ticker价格: {price} (无时间戳)")
                        return price
            
            # 方法2：从API获取最新价格（同步请求，确保实时）
            logger.debug(f"从API获取实时价格: {symbol}")
            klines = self.exchange_client.get_klines_paginated(exchange_symbol, '1m', limit=1)
            
            if klines and len(klines) > 0:
                kline = klines[0]
                if isinstance(kline, dict):
                    price = float(kline.get('close', 0))
                else:
                    price = float(kline.close)
                if price > 0:
                    logger.debug(f"✓ API价格: {price}")
                    return price
            
            # 所有方法失败 - 严格模式：不使用任何回退，直接放弃信号
            logger.error(f"❌ 无法获取{symbol}实时价格，放弃本次信号（保护资金安全）")
            return None
            
        except Exception as e:
            logger.error(f"❌ 获取实时价格异常: {e}，放弃本次信号")
            return None
    
    # ✅ 已移除重复的 _calculate_position_size 方法
    # 现在统一使用 position_manager.calculate_position_size()
    
    async def _should_send_signal(self, symbol: str, signal_type: str) -> bool:
        """检查是否应该发送信号 - 基于缓存的上一次信号去重"""
        try:
            # 从缓存获取上一次的信号
            last_signal = await cache_manager.get_trading_signal(symbol)
            
            # 如果没有缓存的信号，直接发送
            if not last_signal:
                logger.info(f"✓ 无缓存信号，允许发送 {signal_type}")
                return True
            
            # 获取上一次的信号类型
            last_signal_type = last_signal.get('signal_type')
            
            # 如果信号类型相同，拒绝（去重）- 相同方向不需要重复开仓
            if last_signal_type == signal_type:
                logger.warning(f"✗ 信号重复: 上次={last_signal_type}, 本次={signal_type}")
                return False
            
            # 信号类型不同，允许发送（方向改变）
            logger.info(f"✓ 信号方向改变: {last_signal_type} → {signal_type}")
            return True
            
        except Exception as e:
            logger.error(f"检查信号去重失败: {e}")
            # 发生错误时保守处理，允许发送信号
            return True
    
    async def _process_signal(self, signal: TradingSignal):
        """处理生成的信号（注意：signal_counter已在调用前+1）"""
        try:
            # 🔒 安全保护：前5个信号仅记录不交易
            # 注意：signal_counter已在_on_new_data中+1，这里不再重复
            
            if self.signal_counter <= self.warmup_signals:
                logger.warning(f"⚠️ 预热信号 [{self.signal_counter}/{self.warmup_signals}]：仅记录，不执行交易")
                logger.info(f"   信号详情: {format_signal_type(signal.signal_type)} 置信度={signal.confidence:.4f} 入场={signal.entry_price:.2f}")
                
                # 只保存到数据库用于观察，不发送给交易引擎
                await self._save_signal(signal)
                
                logger.info(f"✅ 预热信号已记录到数据库 (剩余{self.warmup_signals - self.signal_counter}个预热信号)")
                return  # 🔒 直接返回，不执行后续交易逻辑
            
            # ✅ 预热完成，正式交易信号
            logger.info(f"🚀 正式交易信号 (第{self.signal_counter}个): {format_signal_type(signal.signal_type)} 置信度={signal.confidence:.4f}")
            
            # 更新最后信号记录
            self.last_signals[signal.symbol] = signal
            
            # 存储信号到数据库
            await self._save_signal(signal)
            
            # 缓存信号（不设置过期时间，用于信号去重）
            await cache_manager.set_trading_signal(
                signal.symbol,
                {
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'position_size': signal.position_size,
                    'timestamp': signal.timestamp.isoformat()
                },
                expire=None  # 不过期，只在新信号产生时覆盖
            )
            
            # 通知回调函数（发送给交易引擎）
            if not self.signal_callbacks:
                logger.warning(f"⚠️ 没有注册的信号回调函数，信号将不会被执行！")
                logger.warning(f"   请检查trading_controller是否正确注册了回调")
            else:
                logger.info(f"📤 准备发送信号到{len(self.signal_callbacks)}个回调函数")
                for idx, callback in enumerate(self.signal_callbacks):
                    try:
                        logger.debug(f"   调用回调函数 {idx+1}/{len(self.signal_callbacks)}: {callback.__name__ if hasattr(callback, '__name__') else type(callback).__name__}")
                        await callback(signal)
                        logger.debug(f"   ✅ 回调函数 {idx+1} 执行成功")
                    except Exception as e:
                        logger.error(f"   ❌ 回调函数 {idx+1} 执行失败: {e}", exc_info=True)
            
            logger.info(f"✅ 交易信号已发送: {signal.symbol} {signal.signal_type}")
            
        except Exception as e:
            logger.error(f"处理信号失败: {e}")
    
    async def _save_signal(self, signal: TradingSignal):
        """保存信号到数据库（只保存一个合成信号，predictions保留原始预测详情）"""
        try:
            # 处理 predictions 中的 datetime 对象（转换为 ISO 格式字符串）
            predictions = signal.metadata.get('timeframe_predictions', {})
            cleaned_predictions = {}
            for tf, pred in predictions.items():
                cleaned_pred = pred.copy()
                # 将 datetime 对象转换为字符串
                if 'timestamp' in cleaned_pred and hasattr(cleaned_pred['timestamp'], 'isoformat'):
                    cleaned_pred['timestamp'] = cleaned_pred['timestamp'].isoformat()
                cleaned_predictions[tf] = cleaned_pred
            
            signal_data = {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                # 保存3个时间框架的预测信息到predictions字段（已清理 datetime）
                'predictions': cleaned_predictions
            }
            
            await postgresql_manager.write_signal_data(signal_data)
            
        except Exception as e:
            logger.error(f"保存信号失败: {e}")
    
    async def get_recent_signals(
        self, 
        symbol: str, 
        hours: int = 24, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取最近的信号"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            signals = await postgresql_manager.query_signals(
                symbol, start_time, end_time, limit
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"获取最近信号失败: {e}")
            return []
    
    async def get_signal_performance(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """获取信号表现统计"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            signals = await postgresql_manager.query_signals(
                symbol, start_time, end_time
            )
            
            if not signals:
                return {}
            
            # 统计信号数量
            total_signals = len(signals)
            long_signals = len([s for s in signals if s['signal_type'] == 'LONG'])
            short_signals = len([s for s in signals if s['signal_type'] == 'SHORT'])
            
            # 平均置信度
            avg_confidence = np.mean([s['confidence'] for s in signals])
            
            # 信号频率（每天）
            signal_frequency = total_signals / days
            
            performance = {
                'total_signals': total_signals,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'long_ratio': long_signals / total_signals if total_signals > 0 else 0,
                'short_ratio': short_signals / total_signals if total_signals > 0 else 0,
                'avg_confidence': avg_confidence,
                'signal_frequency': signal_frequency,
                'period_days': days
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"获取信号表现失败: {e}")
            return {}
    
    async def _enhanced_signal_filter(
        self,
        signal_type: str,
        confidence: float,
        predictions: Dict[str, Dict[str, Any]],
        symbol: str
    ) -> Dict[str, Any]:
        """增强的信号过滤（优化目标：胜率+5-10%）
        
        多维度过滤低质量信号：
        1. 趋势一致性过滤
        2. 量能确认
        3. 波动率过滤
        4. 时间过滤
        
        Returns:
            {'pass': bool, 'reason': str}
        """
        try:
            # 1. 置信度基础过滤（已有）
            if confidence < self.confidence_threshold:
                return {'pass': False, 'reason': f'置信度过低 ({confidence:.4f} < {self.confidence_threshold})'}
            
            # 2. 趋势一致性过滤（已通过权重合成，不应再过滤）
            # 🔧 修复：权重信号已经考虑了多时间框架的加权结果
            # 如果最终信号是通过权重计算得出的，说明已经考虑了各时间框架的贡献
            # 因此不应该因为某个时间框架有反向信号就过滤掉
            # 只有在权重信号置信度很低时才需要额外检查
            if confidence < 0.5:  # 只有置信度很低时才检查趋势一致性
                if len(predictions) >= 2:
                    signal_types = [pred['signal_type'] for pred in predictions.values()]
                    # 如果所有时间框架都是反向信号，才过滤
                    if signal_type == 'LONG' and all(s == 'SHORT' for s in signal_types):
                        return {'pass': False, 'reason': '所有时间框架都是SHORT信号，与LONG冲突'}
                    elif signal_type == 'SHORT' and all(s == 'LONG' for s in signal_types):
                        return {'pass': False, 'reason': '所有时间框架都是LONG信号，与SHORT冲突'}
            
            # 3. 波动率过滤（避免在极端波动时交易）
            try:
                # 获取最新5m K线数据来计算波动率（主时间框架）
                buffer_data = self.kline_buffers.get('5m', [])
                if len(buffer_data) >= 60:  # 5m需要更多样本（60个=5小时）
                    recent_closes = [k['close'] for k in buffer_data[-60:]]
                    returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1] 
                              for i in range(1, len(recent_closes))]
                    current_volatility = np.std(returns)
                    
                    # 日波动率估算（5分钟 → 日，假设288个5分钟周期）
                    daily_volatility = current_volatility * np.sqrt(288)
                    
                    if daily_volatility > 0.08:  # 日波动率>8%
                        return {'pass': False, 'reason': f'市场波动过大 (日波动率={daily_volatility*100:.2f}%)'}
                    
                    if daily_volatility < 0.005:  # 日波动率<0.5%
                        return {'pass': False, 'reason': f'市场波动过小 (日波动率={daily_volatility*100:.2f}%)'}
            except Exception as e:
                logger.debug(f"波动率计算失败（跳过此过滤）: {e}")
            
            # 4. 量能确认（高置信度信号需要量能配合）
            if confidence > 0.6:  # 高置信度信号
                try:
                    buffer_data = self.kline_buffers.get('5m', [])
                    if len(buffer_data) >= 60:  # 5m需要更多样本
                        recent_volumes = [k['volume'] for k in buffer_data[-20:]]
                        current_volume = buffer_data[-1]['volume']
                        avg_volume = np.mean(recent_volumes)
                        
                        # 高置信度信号需要量能至少达到平均的70%
                        if current_volume < avg_volume * 0.7:
                            return {'pass': False, 'reason': f'量能不足（当前={current_volume:.0f}, 平均={avg_volume:.0f}）'}
                except Exception as e:
                    logger.debug(f"量能检查失败（跳过此过滤）: {e}")
            
            # 5. 信号频率限制（动态限制，基于主时间框架）
            # 超短线策略需要在保持灵活性的同时避免过度交易
            try:
                # 🔑 基于时间框架的动态限制策略
                # 计算逻辑：1小时内的K线数 × 合理触发率
                timeframe_limits = {
                    '3m': 8,   # 3m: 1小时20个K线 × 40% = 8个信号
                    '5m': 6,   # 5m: 1小时12个K线 × 50% = 6个信号（主时间框架）
                    '15m': 3   # 15m: 1小时4个K线 × 75% = 3个信号
                }
                
                # 使用主时间框架（5m）的限制
                main_timeframe = '5m'  # 当前系统主时间框架
                max_signals_per_hour = timeframe_limits.get(main_timeframe, 6)
                
                # 查询最近1小时的信号
                recent_signals = await self.get_recent_signals(
                    symbol, 
                    hours=1, 
                    limit=max_signals_per_hour + 2  # 多查2个用于准确判断
                )
                
                if len(recent_signals) >= max_signals_per_hour:
                    return {
                        'pass': False, 
                        'reason': f'信号频率过高（1小时内已有{len(recent_signals)}个，{main_timeframe}限制{max_signals_per_hour}个）'
                    }
                
                # 调试日志：显示当前信号频率状态
                logger.debug(f"✓ 信号频率检查通过: {len(recent_signals)}/{max_signals_per_hour} ({main_timeframe})")
                
            except Exception as e:
                logger.debug(f"信号频率检查失败（跳过此过滤）: {e}")
            
            # 6. 🆕 市场状态过滤（Market Regime Filtering）
            # 根据ADX指标判断市场状态，过滤不匹配的信号
            try:
                # 获取最新5m K线数据（主时间框架）来计算ADX
                buffer_data = self.kline_buffers.get('5m', [])
                if len(buffer_data) >= 14:  # ADX需要至少14个周期
                    # 转换为DataFrame以便计算ADX
                    df_buffer = pd.DataFrame(buffer_data[-50:])  # 取最近50条用于计算
                    if 'timestamp' in df_buffer.columns:
                        df_buffer['timestamp'] = pd.to_datetime(df_buffer['timestamp'], unit='ms')
                    df_buffer = df_buffer.set_index('timestamp') if 'timestamp' in df_buffer.columns else df_buffer
                    
                    # 计算ADX（使用ta库，与feature_engineering保持一致）
                    try:
                        import ta  # 与feature_engineering.py保持一致
                        adx_indicator = ta.trend.ADXIndicator(
                            df_buffer['high'], 
                            df_buffer['low'], 
                            df_buffer['close']
                        )
                        current_adx = adx_indicator.adx().iloc[-1]
                        
                        # 验证ADX值有效性
                        if pd.isna(current_adx) or np.isinf(current_adx):
                            raise ValueError("ADX计算结果无效")
                        
                        # 市场状态判断阈值
                        TRENDING_THRESHOLD = 25  # ADX >= 25: 趋势市场
                        RANGING_THRESHOLD = 20   # ADX <= 20: 震荡市场
                        
                        # 判断市场状态
                        if current_adx >= TRENDING_THRESHOLD:
                            market_regime = 'TRENDING'
                        elif current_adx <= RANGING_THRESHOLD:
                            market_regime = 'RANGING'
                        else:
                            market_regime = 'MIXED'  # 20 < ADX < 25: 混合状态
                        
                        logger.debug(f"📊 市场状态: {market_regime} (ADX={current_adx:.2f})")
                        
                        # 根据市场状态过滤信号
                        if market_regime == 'TRENDING':
                            # 趋势市场：只接受LONG/SHORT信号，拒绝HOLD
                            if signal_type == 'HOLD':
                                return {
                                    'pass': False, 
                                    'reason': f'趋势市场(ADX={current_adx:.2f})拒绝HOLD信号'
                                }
                            logger.debug(f"✅ 趋势市场信号匹配: {signal_type}")
                            
                        elif market_regime == 'RANGING':
                            # 震荡市场：只接受HOLD信号，拒绝LONG/SHORT
                            if signal_type in ['LONG', 'SHORT']:
                                return {
                                    'pass': False, 
                                    'reason': f'震荡市场(ADX={current_adx:.2f})拒绝{signal_type}信号'
                                }
                            logger.debug(f"✅ 震荡市场信号匹配: {signal_type}")
                            
                        else:  # MIXED
                            # 混合状态：接受所有信号，但降低置信度要求
                            logger.debug(f"⚠️ 混合市场状态(ADX={current_adx:.2f})，接受所有信号")
                            
                    except ImportError:
                        logger.warning("pandas_ta未安装，跳过ADX计算（市场状态过滤失效）")
                    except Exception as e:
                        logger.debug(f"ADX计算失败（跳过市场状态过滤）: {e}")
                else:
                    logger.debug(f"缓冲区数据不足({len(buffer_data)}条 < 14条)，跳过市场状态过滤")
            except Exception as e:
                logger.debug(f"市场状态过滤失败（跳过此过滤）: {e}")
            
            # 7. 所有过滤器通过
            logger.info(f"✅ 信号通过所有增强过滤器（包括市场状态过滤）")
            return {'pass': True, 'reason': '通过所有过滤条件'}
            
        except Exception as e:
            logger.error(f"信号过滤失败: {e}")
            # 过滤失败时保守处理：通过信号（避免错失机会）
            return {'pass': True, 'reason': '过滤器异常，默认通过'}
    
    def add_signal_callback(self, callback: callable):
        """添加信号回调函数"""
        self.signal_callbacks.append(callback)
    
    def remove_signal_callback(self, callback: callable):
        """移除信号回调函数"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    async def force_generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """强制生成信号（用于手动触发）"""
        try:
            logger.info(f"强制生成信号: {symbol}")
            
            # 临时移除时间间隔限制
            original_interval = self.min_signal_interval
            self.min_signal_interval = 0
            
            signal = await self.generate_signal(symbol)
            
            # 恢复时间间隔限制
            self.min_signal_interval = original_interval
            
            if signal:
                await self._process_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"强制生成信号失败: {e}")
            return None
    
    async def _load_warmup_state(self):
        """从Redis加载预热状态（持久化，避免重启/重训练后重新预热）"""
        try:
            # 从Redis加载信号计数器
            cached_counter = await cache_manager.get(f"warmup:signal_counter:{settings.SYMBOL}")
            
            if cached_counter is not None:
                self.signal_counter = int(cached_counter)
                logger.info(f"📂 已加载预热状态: {self.signal_counter}/{self.warmup_signals} 个信号")
                
                if self.signal_counter >= self.warmup_signals:
                    logger.info(f"✅ 预热已在之前完成，系统处于正常交易模式")
            else:
                logger.info(f"📂 首次部署，初始化预热状态: 0/{self.warmup_signals}")
                await self._save_warmup_state()
                
        except Exception as e:
            logger.warning(f"加载预热状态失败（使用默认值0）: {e}")
            self.signal_counter = 0
    
    async def _save_warmup_state(self):
        """保存预热状态到Redis（无过期时间，永久保存）"""
        try:
            # 保存信号计数器到Redis（不过期）
            await cache_manager.set(
                f"warmup:signal_counter:{settings.SYMBOL}",
                self.signal_counter,
                expire=None  # 永不过期
            )
            logger.debug(f"💾 预热状态已保存: {self.signal_counter}/{self.warmup_signals}")
            
        except Exception as e:
            logger.warning(f"保存预热状态失败: {e}")