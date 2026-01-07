"""
交易执行引擎
"""
# StdLib
import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Any, Optional

# Local App
from app.core.cache import cache_manager
from app.core.config import settings
from app.core.constants import VIRTUAL_OPEN_FEE_RATE, VIRTUAL_CLOSE_FEE_RATE
from app.core.database import postgresql_manager
from app.exchange.exchange_factory import ExchangeFactory
from app.trading.position_manager import position_manager
from app.trading.signal_generator import TradingSignal

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

class OrderStatus(Enum):
    """订单状态"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class TradingMode(Enum):
    """交易模式"""
    AUTO = "AUTO"  # 自动交易
    SIGNAL_ONLY = "SIGNAL_ONLY"  # 仅信号提醒

@dataclass
class Order:
    """订单数据类"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    avg_price: float
    commission: float
    created_at: int  # ✅ 毫秒时间戳（Binance原始）
    updated_at: int  # ✅ 毫秒时间戳（Binance原始）
    metadata: Dict[str, Any]

@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    side: str  # LONG, SHORT
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    margin_type: str
    leverage: int
    created_at: int  # ✅ 毫秒时间戳
    updated_at: int  # ✅ 毫秒时间戳

class TradingEngine:
    """交易执行引擎"""
    
    def __init__(self, data_service=None):
        self.is_running = False
        
        # 🔑 保存 data_service 引用（用于注册价格回调）
        self.data_service = data_service
        
        # 🔑 获取交易所客户端（使用工厂模式）
        self.exchange_client = ExchangeFactory.get_current_client()
        
        # 从配置文件读取默认交易模式
        default_mode = settings.TRADING_MODE
        self.trading_mode = TradingMode.AUTO if default_mode == "AUTO" else TradingMode.SIGNAL_ONLY
        
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_monitor_task = None
        
        # 🆕 虚拟仓位缓存（内存）
        self.virtual_positions_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # 风险控制参数
        self.max_position_size = 1000  # 最大持仓数量
        self.max_daily_trades = 50     # 每日最大交易次数
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
    async def start(self):
        """启动交易引擎"""
        try:
            logger.info("启动交易执行引擎...")
            
            # 加载现有订单和持仓
            await self._load_orders_and_positions()
            
            # 同步交易模式到 Redis（供其他模块读取）
            await self._sync_trading_mode_to_cache()
            
            # 启动订单监控任务
            self.order_monitor_task = asyncio.create_task(self._monitor_orders())
            
            # 🆕 加载虚拟仓位到缓存
            await self._load_virtual_positions_cache()
            
            # 🆕 注册价格更新回调（用于虚拟仓位止损止盈监控）
            if self.data_service:
                self.data_service.add_price_callback(self._on_price_update)
                logger.info("✅ 已注册虚拟仓位止损止盈监控（使用内存缓存，零数据库查询）")
            else:
                logger.warning("⚠️ data_service未传入，虚拟仓位止损止盈监控未启用")
            
            self.is_running = True
            logger.info(f"交易执行引擎启动完成 (模式: {self.trading_mode.value})")
            
        except Exception as e:
            logger.error(f"启动交易引擎失败: {e}")
            raise
    
    async def stop(self):
        """停止交易引擎"""
        try:
            logger.info("停止交易执行引擎...")
            
            self.is_running = False
            
            # 取消订单监控任务
            if self.order_monitor_task:
                self.order_monitor_task.cancel()
                try:
                    await self.order_monitor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("交易执行引擎已停止")
            
        except Exception as e:
            logger.error(f"停止交易引擎失败: {e}")
    
    async def execute_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """执行交易信号"""
        try:
            logger.info(f"执行交易信号: {signal.signal_type} {signal.symbol}")
            
            # 检查交易模式
            if self.trading_mode == TradingMode.SIGNAL_ONLY:
                logger.info(f"📊 信号模式 - 执行虚拟交易: {signal.signal_type} {signal.symbol} (置信度={signal.confidence:.4f})")
                # 在信号模式下执行虚拟交易
                result = await self._execute_virtual_trade(signal)
                if result.get('success'):
                    logger.info(f"✅ 虚拟交易执行成功: {result.get('message', '')}")
                else:
                    logger.warning(f"⚠️ 虚拟交易执行失败: {result.get('message', '')}")
                return result
            
            # 风险检查
            risk_check = await self._check_trading_risks(signal)
            if not risk_check['allowed']:
                logger.warning(f"风险检查失败: {risk_check['reason']}")
                return {
                    'success': False,
                    'message': f"风险检查失败: {risk_check['reason']}"
                }
            
            # 处理不同类型的信号
            if signal.signal_type == 'CLOSE':
                return await self._close_position(signal.symbol)
            else:
                return await self._open_position(signal)
            
        except Exception as e:
            logger.error(f"执行交易信号失败: {e}")
            return {
                'success': False,
                'message': f"执行失败: {str(e)}"
            }
    
    async def _open_position(self, signal: TradingSignal) -> Dict[str, Any]:
        """开仓"""
        try:
            symbol = signal.symbol
            
            # 检查是否已有持仓
            existing_position = await self._get_position(symbol)
            if existing_position and existing_position.size != 0:
                logger.warning(f"已有持仓，先平仓: {symbol}")
                await self._close_position(symbol)
            
            # 确定订单方向
            side = OrderSide.BUY if signal.signal_type == 'LONG' else OrderSide.SELL
            
            # 下市价单开仓
            order_result = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=signal.position_size,
                metadata={
                    'signal_id': str(uuid.uuid4()),
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'entry_reason': 'signal_execution'
                }
            )
            
            if order_result['success']:
                # 设置止损止盈订单
                await self._set_stop_loss_take_profit(
                    symbol, signal.stop_loss, signal.take_profit, signal.signal_type
                )
                
                logger.info(f"开仓成功: {symbol} {signal.signal_type}")
                
                return {
                    'success': True,
                    'message': '开仓成功',
                    'order': order_result['order']
                }
            else:
                return order_result
            
        except Exception as e:
            logger.error(f"开仓失败: {e}")
            return {
                'success': False,
                'message': f"开仓失败: {str(e)}"
            }
    
    async def _close_position(self, symbol: str) -> Dict[str, Any]:
        """平仓"""
        try:
            position = await self._get_position(symbol)
            
            if not position or position.size == 0:
                return {
                    'success': True,
                    'message': '无持仓需要平仓'
                }
            
            # 确定平仓方向
            side = OrderSide.SELL if position.side == 'LONG' else OrderSide.BUY
            
            # 下市价单平仓
            order_result = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(position.size),
                reduce_only=True,
                metadata={
                    'action': 'close_position',
                    'original_side': position.side
                }
            )
            
            if order_result['success']:
                # 取消相关的止损止盈订单
                await self._cancel_stop_orders(symbol)
                
                logger.info(f"平仓成功: {symbol}")
                
                return {
                    'success': True,
                    'message': '平仓成功',
                    'order': order_result['order']
                }
            else:
                return order_result
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return {
                'success': False,
                'message': f"平仓失败: {str(e)}"
            }
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        下单（信号系统：仅支持虚拟交易）
        
        注意：本系统为信号系统，不进行实际交易
        此方法仅用于兼容性，实际下单应通过虚拟交易实现
        """
        logger.warning(f"信号系统：place_order被调用（不支持实际交易）symbol={symbol}, side={side.value}")
        logger.warning("   提示：请使用_execute_virtual_trade进行虚拟交易")
        return {
            'success': False,
            'message': '信号系统不支持实际交易，请使用虚拟交易功能'
        }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        撤销订单（信号系统：仅支持虚拟交易）
        
        注意：本系统为信号系统，不进行实际交易
        """
        logger.warning(f"信号系统：cancel_order被调用（不支持实际交易）order_id={order_id}")
        return {
            'success': False,
            'message': '信号系统不支持实际交易，请使用虚拟交易功能'
        }
    
    async def _set_stop_loss_take_profit(
        self, 
        symbol: str, 
        stop_loss: float, 
        take_profit: float, 
        position_side: str
    ):
        """设置止损止盈"""
        try:
            position = await self._get_position(symbol)
            
            if not position or position.size == 0:
                return
            
            # 止损单
            if stop_loss > 0:
                stop_side = OrderSide.SELL if position_side == 'LONG' else OrderSide.BUY
                
                await self.place_order(
                    symbol=symbol,
                    side=stop_side,
                    order_type=OrderType.STOP_MARKET,
                    quantity=abs(position.size),
                    stop_price=stop_loss,
                    reduce_only=True,
                    metadata={
                        'order_purpose': 'stop_loss',
                        'position_side': position_side
                    }
                )
            
            # 止盈单
            if take_profit > 0:
                tp_side = OrderSide.SELL if position_side == 'LONG' else OrderSide.BUY
                
                await self.place_order(
                    symbol=symbol,
                    side=tp_side,
                    order_type=OrderType.TAKE_PROFIT_MARKET,
                    quantity=abs(position.size),
                    stop_price=take_profit,
                    reduce_only=True,
                    metadata={
                        'order_purpose': 'take_profit',
                        'position_side': position_side
                    }
                )
            
            logger.info(f"止损止盈设置完成: {symbol}")
            
        except Exception as e:
            logger.error(f"设置止损止盈失败: {e}")
    
    async def _cancel_stop_orders(self, symbol: str):
        """
        取消止损止盈订单（信号系统：虚拟订单无需取消）
        
        注意：信号系统使用虚拟订单，止损止盈通过价格监控自动触发平仓
        无需手动取消订单
        """
        logger.debug(f"信号系统：_cancel_stop_orders被调用（虚拟订单无需取消）symbol={symbol}")
        # 信号系统：虚拟订单的止损止盈通过_on_price_update自动处理，无需手动取消
    
    async def _execute_virtual_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """执行虚拟交易（信号模式）"""
        try:
            symbol = signal.symbol
            signal_id = str(uuid.uuid4())
            
            logger.info(f"💰 虚拟交易: {signal.signal_type} {symbol}")
            
            # 获取当前价格（用于虚拟成交）
            try:
                ticker = self.exchange_client.get_ticker_price(symbol)
                if ticker:
                    current_price = float(ticker.price)
                else:
                    current_price = signal.entry_price
                    logger.warning(f"无法获取实时价格，使用信号价格: {current_price}")
            except:
                current_price = signal.entry_price
                logger.warning(f"无法获取实时价格，使用信号价格: {current_price}")
            
            # 处理不同类型的信号
            if signal.signal_type == 'CLOSE':
                # 平掉所有虚拟仓位
                return await self._close_virtual_positions(symbol, current_price, signal_id)
            else:
                # 开虚拟仓位（LONG/SHORT）
                return await self._open_virtual_position(signal, current_price, signal_id)
            
        except Exception as e:
            logger.error(f"执行虚拟交易失败: {e}")
            return {
                'success': False,
                'message': f"虚拟交易失败: {str(e)}"
            }
    
    async def _open_virtual_position(self, signal: TradingSignal, current_price: float, signal_id: str) -> Dict[str, Any]:
        """开虚拟仓位"""
        try:
            symbol = signal.symbol
            
            # 🔑 先平掉现有虚拟仓位（使用缓存，避免查询数据库）
            existing_positions = self.virtual_positions_cache.get(symbol, [])
            total_closed_pnl = Decimal('0')
            
            if existing_positions:
                # 🔥 过滤掉已经关闭的仓位（可能已被止损/止盈触发）
                open_positions = [pos for pos in existing_positions if pos.get('status') == 'OPEN']
                if open_positions:
                    logger.info("=" * 70)
                    logger.info(f"📊 检测到{len(open_positions)}个现有虚拟仓位，执行平仓...")
                    logger.info("=" * 70)
                    
                    for pos in open_positions:
                        # 🔥 再次验证仓位状态（防止重复平仓）
                        pos_check = await postgresql_manager.get_virtual_position_by_id(pos['id'])
                        if not pos_check or pos_check['status'] != 'OPEN':
                            logger.debug(f"⚠️ 仓位{pos['id']}已被关闭，跳过（可能已被止损/止盈触发）")
                            continue
                        
                        # 🔑 计算价差盈亏（quantity现在是USDT价值，需要转换成币的数量）
                        # 🔥 转换为Decimal确保精度，避免float和Decimal混用
                        quantity_decimal = Decimal(str(pos['quantity']))
                        entry_price_decimal = Decimal(str(pos['entry_price']))
                        current_price_decimal = Decimal(str(current_price))
                        
                        coin_amount = quantity_decimal / entry_price_decimal  # 币的数量
                        if pos['side'] == 'LONG':
                            price_pnl = (current_price_decimal - entry_price_decimal) * coin_amount
                        else:  # SHORT
                            price_pnl = (entry_price_decimal - current_price_decimal) * coin_amount
                        
                        # 🔑 计算手续费（quantity已经是USDT价值）
                        open_position_value = Decimal(str(pos['quantity']))  # 开仓时的USDT价值
                        open_commission = open_position_value * VIRTUAL_OPEN_FEE_RATE
                        
                        close_position_value = Decimal(str(coin_amount)) * current_price_decimal  # 平仓时的USDT价值
                        close_commission = close_position_value * VIRTUAL_CLOSE_FEE_RATE
                        
                        # 净盈亏（转换为Decimal进行计算）
                        price_pnl_decimal = Decimal(str(price_pnl))
                        net_pnl = price_pnl_decimal - open_commission - close_commission
                        total_closed_pnl += net_pnl
                        
                        # 🔑 更新虚拟账户余额（平仓后）
                        net_pnl_float = float(net_pnl.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
                        await position_manager.update_virtual_balance(net_pnl_float)
                        
                        # 创建平仓虚拟订单
                        close_order = {
                            'order_id': None,
                            'symbol': symbol,
                            'side': 'SELL' if pos['side'] == 'LONG' else 'BUY',
                            'type': 'MARKET',
                            'status': 'FILLED',
                            'quantity': float(quantity_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                            'price': float(current_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                            'filled_quantity': float(quantity_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                            'commission': float(close_commission.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'is_virtual': True,
                            'signal_id': signal_id,
                            'position_id': pos['id'],
                            'order_action': 'CLOSE',
                            'entry_price': float(entry_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                            'exit_price': float(current_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                            'pnl': float(net_pnl.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                            'pnl_percent': float((net_pnl / open_position_value * Decimal('100')).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
                        }
                        # 执行数据库平仓操作
                        await postgresql_manager.close_virtual_position(pos['id'], current_price)
                        
                        # 🔑 修复：订单表只在平仓时创建记录（包含完整的开仓和平仓信息）
                        await postgresql_manager.write_order_data(close_order)
                        
                        # 📊 详细日志输出
                        logger.info(f"📉 平仓订单 #{pos['id']}:")
                        logger.info(f"   方向: {pos['side']}")
                        logger.info(f"   开仓金额: {float(open_position_value):.2f} USDT")
                        logger.info(f"   开仓价格: {float(entry_price_decimal):.2f}")
                        logger.info(f"   平仓价格: {float(current_price_decimal):.2f}")
                        logger.info(f"   平仓金额: {float(close_position_value):.2f} USDT")
                        logger.info(f"   价差盈亏: {float(price_pnl_decimal):+.2f} USDT")
                        logger.info(f"   开仓手续费: {float(open_commission):.4f} USDT (0.02%)")
                        logger.info(f"   平仓手续费: {float(close_commission):.4f} USDT (0.05%)")
                        logger.info(f"   净盈亏: {net_pnl_float:+.2f} USDT ({float(close_order['pnl_percent']):+.2f}%)")
                        logger.info("-" * 70)
                    
                    logger.info(f"✅ 平仓完成: 共平仓{len(open_positions)}个仓位，总盈亏: {float(total_closed_pnl):+.2f} USDT")
                    logger.info("=" * 70)
            
            # 创建新的虚拟仓位
            # 🔑 position_size 现在直接是USDT价值
            position_data = {
                'symbol': symbol,
                'side': signal.signal_type,  # LONG or SHORT
                'entry_price': current_price,
                'quantity': signal.position_size,  # USDT价值
                'entry_time': int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'signal_id': signal_id
            }
            
            position_id = await postgresql_manager.create_virtual_position(position_data)
            
            # 🔑 计算开仓手续费（0.02%），position_size已经是USDT价值
            # 🔥 转换为Decimal确保精度，避免float和Decimal混用
            position_value = Decimal(str(signal.position_size))
            open_commission = position_value * VIRTUAL_OPEN_FEE_RATE
            
            # 🔑 注意：开仓手续费不在开仓时扣除，而是在平仓时与平仓手续费一起结算
            # 平仓时的净盈亏 = 价差盈亏 - 开仓手续费 - 平仓手续费
            # 这样避免重复扣除手续费
            
            # 🔑 修复：订单表只在平仓时创建，不在开仓时创建（避免开仓订单记录）
            
            # 📊 详细日志输出
            logger.info("=" * 70)
            logger.info(f"📈 开仓订单:")
            logger.info(f"   方向: {signal.signal_type}")
            logger.info(f"   开仓金额: {float(position_value):.2f} USDT")
            logger.info(f"   开仓价格: {float(current_price_decimal):.2f}")
            logger.info(f"   开仓手续费: {float(open_commission):.4f} USDT (0.02%，平仓时结算)")
            logger.info(f"   止损价格: {signal.stop_loss:.2f}")
            logger.info(f"   止盈价格: {signal.take_profit:.2f}")
            logger.info(f"   信号置信度: {signal.confidence:.4f}")
            logger.info("=" * 70)
            
            # 🔑 刷新虚拟仓位缓存
            await self._refresh_virtual_positions_cache(symbol)
            
            # 🔑 修复：在开仓后打印虚拟仓位历史统计
            await self._print_virtual_positions_statistics(symbol)
            
            return {
                'success': True,
                'message': f'虚拟开仓成功',
                'virtual_position_id': position_id,
                'entry_price': current_price,
                'quantity': signal.position_size
            }
            
        except Exception as e:
            logger.error(f"开虚拟仓位失败: {e}")
            return {
                'success': False,
                'message': f"开虚拟仓位失败: {str(e)}"
            }
    
    async def _close_virtual_positions(self, symbol: str, current_price: float, signal_id: str) -> Dict[str, Any]:
        """平掉所有虚拟仓位"""
        try:
            existing_positions = await postgresql_manager.get_open_virtual_positions(symbol)
            
            if not existing_positions:
                logger.info(f"无虚拟仓位需要平仓: {symbol}")
                return {
                    'success': True,
                    'message': '无虚拟仓位需要平仓'
                }
            
            closed_count = 0
            total_pnl = 0
            
            for pos in existing_positions:
                # 平仓
                await postgresql_manager.close_virtual_position(pos['id'], current_price)
                
                # 🔑 计算价差盈亏（quantity现在是USDT价值，需要转换成币的数量）- 使用Decimal确保精度
                entry_price = Decimal(str(pos['entry_price']))
                quantity = Decimal(str(pos['quantity']))
                current_price_decimal = Decimal(str(current_price))
                
                coin_amount = quantity / entry_price  # 币的数量
                if pos['side'] == 'LONG':
                    price_pnl = (current_price_decimal - entry_price) * coin_amount
                else:  # SHORT
                    price_pnl = (entry_price - current_price_decimal) * coin_amount
                
                # 🔑 计算手续费（quantity已经是USDT价值）
                open_position_value = quantity  # 开仓时的USDT价值
                open_commission = open_position_value * VIRTUAL_OPEN_FEE_RATE
                
                close_position_value = coin_amount * current_price_decimal  # 平仓时的USDT价值
                close_commission = close_position_value * VIRTUAL_CLOSE_FEE_RATE
                
                # 净盈亏 = 价差盈亏 - 开仓手续费 - 平仓手续费
                net_pnl = price_pnl - open_commission - close_commission
                
                net_pnl_float = float(net_pnl.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
                total_pnl += net_pnl_float
                
                # 🔑 更新虚拟账户余额（平仓后）
                await position_manager.update_virtual_balance(net_pnl_float)
                
                # 创建平仓虚拟订单
                # 🔥 将Decimal转换为float用于存储
                pnl_percent_float = float((net_pnl / open_position_value * Decimal('100')).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
                
                close_order = {
                    'order_id': None,
                    'symbol': symbol,
                    'side': 'SELL' if pos['side'] == 'LONG' else 'BUY',
                    'type': 'MARKET',
                    'status': 'FILLED',
                    'quantity': float(quantity.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                    'price': float(current_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                    'filled_quantity': float(quantity.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                    'commission': float(close_commission.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),  # 平仓手续费 0.05%
                    'timestamp': int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                    'is_virtual': True,
                    'signal_id': signal_id,
                    'position_id': pos['id'],  # 🔑 关联虚拟仓位ID
                    'order_action': 'CLOSE',  # 🔑 明确标识为平仓订单
                    'entry_price': float(entry_price.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                    'exit_price': float(current_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                    'pnl': net_pnl_float,  # 已扣除手续费的净盈亏
                    'pnl_percent': pnl_percent_float
                }
                
                await postgresql_manager.write_order_data(close_order)
                closed_count += 1
            
            logger.info(f"✅ 虚拟平仓: {symbol} 平仓{closed_count}个仓位 @{current_price}")
            logger.info(f"   净盈亏: ${total_pnl:+.2f} (已扣除开仓0.02%+平仓0.05%手续费)")
            
            # 🔑 刷新虚拟仓位缓存
            await self._refresh_virtual_positions_cache(symbol)
            
            return {
                'success': True,
                'message': f'虚拟平仓成功',
                'closed_count': closed_count,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            logger.error(f"平虚拟仓位失败: {e}")
            return {
                'success': False,
                'message': f"平虚拟仓位失败: {str(e)}"
            }
    
    async def _close_virtual_position_by_trigger(
        self,
        pos_id: int,
        current_price: float,
        reason: str,
        trigger_type: str = None
    ):
        """
        因止损止盈触发而平仓（WebSocket实时监控触发）
        
        Args:
            pos_id: 仓位ID
            current_price: 当前价格
            reason: 平仓原因
            trigger_type: 触发类型 ('STOP_LOSS' 或 'TAKE_PROFIT')
        """
        try:
            # 获取仓位信息
            pos = await postgresql_manager.get_virtual_position_by_id(pos_id)
            if not pos:
                logger.warning(f"⚠️ 仓位不存在: {pos_id}")
                return {
                    'success': False,
                    'message': f'仓位不存在: {pos_id}'
                }
            
            if pos['status'] != 'OPEN':
                logger.debug(f"⚠️ 仓位已关闭，跳过: {pos_id} (状态: {pos['status']})")
                return {
                    'success': False,
                    'message': f'仓位已关闭: {pos_id}'
                }
            
            symbol = pos['symbol']
            
            # 平仓
            await postgresql_manager.close_virtual_position(pos_id, current_price)
            
            # 🔑 计算盈亏（quantity是USDT价值）- 使用Decimal确保金融计算精度
            entry_price = Decimal(str(pos['entry_price']))
            quantity = Decimal(str(pos['quantity']))
            current_price_decimal = Decimal(str(current_price))
            
            coin_amount = quantity / entry_price  # 币的数量
            
            if pos['side'] == 'LONG':
                price_pnl = (current_price_decimal - entry_price) * coin_amount
            else:  # SHORT
                price_pnl = (entry_price - current_price_decimal) * coin_amount
            
            # 手续费
            open_commission = quantity * VIRTUAL_OPEN_FEE_RATE  # 0.02%
            close_commission = coin_amount * current_price_decimal * VIRTUAL_CLOSE_FEE_RATE  # 0.05%
            
            # 净盈亏
            net_pnl = price_pnl - open_commission - close_commission
            pnl_percent = (net_pnl / quantity) * Decimal('100')
            
            # 转换为float用于日志和返回（保持API兼容性）
            net_pnl_float = float(net_pnl.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
            pnl_percent_float = float(pnl_percent.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
            
            # 🔑 更新虚拟账户余额（平仓后）
            await position_manager.update_virtual_balance(net_pnl_float)
            
            # 记录平仓订单
            order_data = {
                'order_id': None,
                'symbol': symbol,
                'side': 'SELL' if pos['side'] == 'LONG' else 'BUY',
                'type': 'MARKET',
                'status': 'FILLED',
                'quantity': float(quantity.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                'price': float(current_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                'filled_quantity': float(quantity.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                'commission': float(close_commission.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                'timestamp': int(datetime.now().timestamp() * 1000),
                'is_virtual': True,
                'signal_id': pos.get('signal_id', ''),
                'position_id': pos_id,  # 🔑 关联虚拟仓位ID
                'order_action': 'CLOSE',  # 🔑 明确标识为平仓订单
                'entry_price': float(entry_price.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                'exit_price': float(current_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
                'pnl': net_pnl_float,  # 🔥 已转换为float
                'pnl_percent': pnl_percent_float  # 🔥 已转换为float
            }
            
            await postgresql_manager.write_order_data(order_data)
            
            # 🔑 确保止损/止盈订单状态已更新（如果存在相关订单）
            # 信号系统：虚拟订单状态已通过write_order_data记录，无需额外更新
            
            logger.info(f"✅ 虚拟平仓: {symbol} {pos['side']} {pos['quantity']:.2f} USDT @{current_price:.2f}")
            logger.info(f"   {reason}")
            logger.info(f"   触发类型: {trigger_type or 'UNKNOWN'}")
            logger.info(f"   开仓价: {pos['entry_price']:.2f} → 平仓价: {current_price:.2f}")
            logger.info(f"   净盈亏: ${net_pnl:+.2f} ({pnl_percent:+.2f}%)")
            logger.info(f"   订单已记录: position_id={pos_id}, order_action=CLOSE, status=FILLED")
            
            # 🔑 缓存刷新由调用方统一处理（避免多次刷新）
            
            return {
                'success': True,
                'reason': reason,
                'pnl': net_pnl
            }
            
        except Exception as e:
            logger.error(f"止损止盈触发平仓失败: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    async def _check_trading_risks(self, signal: TradingSignal) -> Dict[str, Any]:
        """检查交易风险"""
        try:
            # 检查每日交易次数
            if self.daily_trade_count >= self.max_daily_trades:
                return {
                    'allowed': False,
                    'reason': f'已达到每日最大交易次数限制: {self.max_daily_trades}'
                }
            
            # 检查持仓大小
            if signal.position_size > self.max_position_size:
                return {
                    'allowed': False,
                    'reason': f'持仓大小超过限制: {signal.position_size} > {self.max_position_size}'
                }
            
            # 🔥 模拟交易模式：不检查真实账户余额（使用虚拟余额）
            # 虚拟余额检查在虚拟交易执行时进行
            
            # 检查置信度
            if signal.confidence < settings.CONFIDENCE_THRESHOLD:
                return {
                    'allowed': False,
                    'reason': f'信号置信度不足: {signal.confidence} < {settings.CONFIDENCE_THRESHOLD}'
                }
            
            return {
                'allowed': True,
                'reason': '风险检查通过'
            }
            
        except Exception as e:
            logger.error(f"风险检查失败: {e}")
            return {
                'allowed': False,
                'reason': f'风险检查异常: {str(e)}'
            }
    
    async def _get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓信息"""
        try:
            positions = self.exchange_client.get_position_info(symbol)
            
            if positions:
                pos_data = positions[0]
                
                position = Position(
                    symbol=symbol,
                    side='LONG' if float(pos_data['position_amt']) > 0 else 'SHORT',
                    size=float(pos_data['position_amt']),
                    entry_price=float(pos_data['entry_price']),
                    mark_price=float(pos_data['mark_price']),
                    unrealized_pnl=float(pos_data['pnl']),
                    percentage=float(pos_data['percentage']),
                    margin_type=pos_data['margin_type'],
                    leverage=int(pos_data['leverage']),
                    created_at=int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                    updated_at=int(datetime.now().timestamp() * 1000)   # ✅ 毫秒时间戳
                )
                
                return position
            
            return None
            
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
            return None
    
    async def _load_orders_and_positions(self):
        """
        加载订单和持仓（信号系统：仅加载虚拟订单）
        
        注意：信号系统不加载实际订单，所有订单都是虚拟的
        """
        try:
            # 信号系统：不加载实际订单，所有订单都是虚拟的
            # 虚拟订单从数据库加载（通过虚拟仓位管理）
            logger.info("信号系统：跳过实际订单加载（仅使用虚拟订单）")
            self.orders = {}  # 清空实际订单缓存
            
        except Exception as e:
            logger.error(f"加载订单和持仓失败: {e}")
    
    async def _monitor_orders(self):
        """监控订单状态"""
        try:
            while self.is_running:
                try:
                    # 检查未成交订单状态
                    for order_id, order in list(self.orders.items()):
                        if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                            await self._update_order_status(order)
                    
                    # 等待30秒后再次检查
                    await asyncio.sleep(30)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"订单监控错误: {e}")
                    await asyncio.sleep(30)
                    
        except asyncio.CancelledError:
            logger.info("订单监控任务已取消")
    
    async def _on_price_update(self, symbol: str, price: float):
        """
        处理价格更新（WebSocket实时推送）
        用于检查虚拟仓位的止损止盈
        
        🔑 性能优化：使用内存缓存，避免频繁查询数据库
        """
        try:
            # 只在虚拟交易模式下检查
            if self.trading_mode != TradingMode.SIGNAL_ONLY:
                return
            
            # 🔑 从内存缓存获取虚拟仓位（零数据库查询！）
            positions = self.virtual_positions_cache.get(symbol, [])
            if not positions:
                return
            
            # 🔥 添加调试日志（每100次价格更新记录一次，避免日志过多）
            if random.random() < 0.01:  # 1%的概率记录调试日志
                logger.debug(f"📊 价格更新检查: {symbol} @{price:.2f}, 虚拟仓位数: {len(positions)}")
            
            # 记录触发平仓的仓位ID
            closed_position_ids = []
            
            # 检查每个仓位的止损止盈
            for pos in positions:
                # 🔥 验证仓位状态（防止重复平仓）
                if pos.get('status') != 'OPEN':
                    continue
                
                should_close = False
                reason = ""
                trigger_type = None  # STOP_LOSS 或 TAKE_PROFIT
                
                # 检查止损
                if pos['side'] == 'LONG':
                    if price <= pos['stop_loss']:
                        should_close = True
                        trigger_type = 'STOP_LOSS'
                        reason = f"止损触发 ({price:.2f} <= {pos['stop_loss']:.2f})"
                    elif price >= pos['take_profit']:
                        should_close = True
                        trigger_type = 'TAKE_PROFIT'
                        reason = f"止盈触发 ({price:.2f} >= {pos['take_profit']:.2f})"
                else:  # SHORT
                    if price >= pos['stop_loss']:
                        should_close = True
                        trigger_type = 'STOP_LOSS'
                        reason = f"止损触发 ({price:.2f} >= {pos['stop_loss']:.2f})"
                    elif price <= pos['take_profit']:
                        should_close = True
                        trigger_type = 'TAKE_PROFIT'
                        reason = f"止盈触发 ({price:.2f} <= {pos['take_profit']:.2f})"
                
                # 触发平仓
                if should_close:
                    logger.info(f"🎯 {symbol} {pos['side']} {reason}")
                    logger.info(f"   仓位ID: {pos['id']}, 开仓价: {pos['entry_price']:.2f}, 触发价: {price:.2f}")
                    result = await self._close_virtual_position_by_trigger(
                        pos_id=pos['id'],
                        current_price=price,
                        reason=reason,
                        trigger_type=trigger_type
                    )
                    if result.get('success'):
                        closed_position_ids.append(pos['id'])
                        logger.info(f"✅ 止损/止盈平仓成功: {symbol} {pos['side']} 仓位ID={pos['id']}")
                    else:
                        logger.error(f"❌ 止损/止盈平仓失败: {result.get('message', 'Unknown error')}")
            
            # 🔑 如果有仓位被平掉，统一刷新缓存（避免循环中多次刷新）
            if closed_position_ids:
                await self._refresh_virtual_positions_cache(symbol)
                logger.info(f"🔄 已平仓{len(closed_position_ids)}个仓位，缓存已刷新")
            
        except Exception as e:
            logger.error(f"处理价格更新失败: {e}", exc_info=True)
    
    async def _load_virtual_positions_cache(self):
        """加载虚拟仓位到内存缓存"""
        try:
            # 获取所有开仓的虚拟仓位
            positions = await postgresql_manager.get_open_virtual_positions(settings.SYMBOL)
            self.virtual_positions_cache[settings.SYMBOL] = positions
            
            logger.info(f"✅ 虚拟仓位缓存已加载: {len(positions)}个仓位")
            
        except Exception as e:
            logger.error(f"加载虚拟仓位缓存失败: {e}")
            self.virtual_positions_cache[settings.SYMBOL] = []
    
    async def _refresh_virtual_positions_cache(self, symbol: str):
        """刷新虚拟仓位缓存（开仓/平仓后调用）"""
        try:
            positions = await postgresql_manager.get_open_virtual_positions(symbol)
            self.virtual_positions_cache[symbol] = positions
            logger.debug(f"🔄 虚拟仓位缓存已刷新: {len(positions)}个仓位")
            
        except Exception as e:
            logger.error(f"刷新虚拟仓位缓存失败: {e}")
    
    async def _print_virtual_positions_statistics(self, symbol: str):
        """
        打印虚拟仓位历史统计（平仓后调用）
        
        统计内容：
        - 总交易次数、胜率
        - 总盈亏、平均盈亏
        - 最大盈利、最大亏损
        - 平均持仓时间
        - 信号产生到开仓的平均延迟
        """
        try:
            stats = await postgresql_manager.get_virtual_positions_statistics(symbol)
            
            if stats['total_trades'] == 0:
                logger.info("📊 虚拟仓位统计: 暂无历史交易数据")
                return
            
            logger.info("=" * 70)
            logger.info("📊 虚拟仓位历史统计")
            logger.info("=" * 70)
            logger.info(f"   交易对: {symbol}")
            logger.info(f"   总交易次数: {stats['total_trades']}")
            logger.info(f"   盈利次数: {stats['win_count']} | 亏损次数: {stats['loss_count']}")
            logger.info(f"   胜率: {stats['win_rate']:.2f}%")
            logger.info("-" * 70)
            logger.info(f"   总盈亏: ${stats['total_pnl']:+.2f}")
            logger.info(f"   平均盈亏: ${stats['avg_pnl']:+.2f}")
            logger.info(f"   最大单笔盈利: ${stats['max_profit']:+.2f}")
            logger.info(f"   最大单笔亏损: ${stats['max_loss']:+.2f}")
            logger.info("-" * 70)
            logger.info(f"   平均持仓时间: {stats['avg_hold_time_minutes']:.2f} 分钟")
            logger.info(f"   信号→开仓平均延迟: {stats['avg_signal_delay_seconds']:.2f} 秒")
            
            if stats['recent_trades']:
                logger.info("-" * 70)
                logger.info("   最近交易:")
                for i, trade in enumerate(stats['recent_trades'][:5], 1):
                    delay_str = f"{trade['signal_delay_seconds']:.1f}s" if trade['signal_delay_seconds'] is not None else "N/A"
                    logger.info(
                        f"   {i}. {trade['side']} | "
                        f"入{trade['entry_price']:.2f}→出{trade['exit_price']:.2f} | "
                        f"PnL: ${trade['pnl']:+.2f} ({trade['pnl_percent']:+.2f}%) | "
                        f"延迟: {delay_str}"
                    )
            
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"打印虚拟仓位统计失败: {e}")
    
    async def _update_order_status(self, order: Order):
        """更新订单状态"""
        try:
            # 从API获取订单状态
            # 这里简化处理，实际应该调用具体的API
            pass
            
        except Exception as e:
            logger.error(f"更新订单状态失败: {e}")
    
    async def _save_order(self, order: Order):
        """保存订单到数据库"""
        try:
            order_data = {
                'timestamp': order.created_at,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.type.value,
                'status': order.status.value,
                'quantity': order.quantity,
                'price': order.price or 0,
                'filled_quantity': order.filled_quantity,
                'commission': order.commission
            }
            
            await postgresql_manager.write_order_data(order_data)
            
        except Exception as e:
            logger.error(f"保存订单失败: {e}")
    
    def _update_trade_count(self):
        """更新交易计数"""
        try:
            current_date = datetime.now().date()
            
            if current_date != self.last_trade_date:
                # 新的一天，重置计数
                self.daily_trade_count = 0
                self.last_trade_date = current_date
            
            self.daily_trade_count += 1
            
        except Exception as e:
            logger.error(f"更新交易计数失败: {e}")
    
    async def set_trading_mode(self, mode: TradingMode):
        """设置交易模式（同步到 Redis 供其他模块读取）"""
        self.trading_mode = mode
        
        # 同步到 Redis
        await self._sync_trading_mode_to_cache()
        
        logger.info(f"交易模式已设置为: {mode.value} ({'实盘自动交易' if mode == TradingMode.AUTO else '虚拟信号模式'})")
    
    async def _sync_trading_mode_to_cache(self):
        """将当前交易模式同步到 Redis（供其他模块动态读取）"""
        try:
            await cache_manager.set(
                "system:trading_mode",
                self.trading_mode.value,
                expire=None  # 永不过期
            )
            logger.debug(f"💾 交易模式已同步到缓存: {self.trading_mode.value}")
        except Exception as e:
            logger.warning(f"同步交易模式到缓存失败: {e}")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """获取交易状态"""
        try:
            return {
                'is_running': self.is_running,
                'trading_mode': self.trading_mode.value,
                'daily_trade_count': self.daily_trade_count,
                'max_daily_trades': self.max_daily_trades,
                'active_orders': len([o for o in self.orders.values() 
                                    if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]),
                'total_orders': len(self.orders)
            }
            
        except Exception as e:
            logger.error(f"获取交易状态失败: {e}")
            return {}