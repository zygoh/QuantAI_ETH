"""
交易执行引擎
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from app.core.config import settings
from app.core.database import postgresql_manager
from app.core.cache import cache_manager
from app.exchange.exchange_factory import ExchangeFactory
from app.trading.signal_generator import TradingSignal

logger = logging.getLogger(__name__)

# 🎯 虚拟交易手续费配置（模拟实际交易所费率）
VIRTUAL_OPEN_FEE_RATE = 0.0002   # 开仓手续费：0.02% (Maker)
VIRTUAL_CLOSE_FEE_RATE = 0.0005  # 平仓手续费：0.05% (Taker)

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
                logger.info("📊 信号模式 - 执行虚拟交易")
                # 在信号模式下执行虚拟交易
                return await self._execute_virtual_trade(signal)
            
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
        """下单"""
        try:
            # 生成客户端订单ID
            client_order_id = f"ETH_TRADING_{int(datetime.now().timestamp() * 1000)}"
            
            # 调用交易所API下单
            api_result = self.exchange_client.place_order(
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                quantity=quantity,
                price=price,
                reduce_only=reduce_only,
                stop_price=stop_price
            )
            
            if not api_result:
                return {
                    'success': False,
                    'message': 'API下单失败'
                }
            
            # 创建订单对象
            order = Order(
                order_id=str(api_result.get('orderId', '')),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus(api_result.get('status', 'NEW')),
                filled_quantity=float(api_result.get('executedQty', 0)),
                remaining_quantity=quantity - float(api_result.get('executedQty', 0)),
                avg_price=float(api_result.get('avgPrice', 0)),
                commission=0.0,  # 手续费稍后计算
                created_at=int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                updated_at=int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                metadata=metadata or {}
            )
            
            # 保存订单
            self.orders[order.order_id] = order
            await self._save_order(order)
            
            # 更新交易计数
            self._update_trade_count()
            
            logger.info(f"下单成功: {symbol} {side.value} {quantity}")
            
            return {
                'success': True,
                'message': '下单成功',
                'order': order
            }
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {
                'success': False,
                'message': f"下单失败: {str(e)}"
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """撤销订单"""
        try:
            order = self.orders.get(order_id)
            
            if not order:
                return {
                    'success': False,
                    'message': '订单不存在'
                }
            
            # 调用API撤销订单
            api_result = self.exchange_client.cancel_order(order.symbol, order.order_id)
            
            if api_result:
                # 更新订单状态
                order.status = OrderStatus.CANCELED
                order.updated_at = int(datetime.now().timestamp() * 1000)  # ✅ 毫秒时间戳
                
                await self._save_order(order)
                
                logger.info(f"撤销订单成功: {order_id}")
                
                return {
                    'success': True,
                    'message': '撤销订单成功'
                }
            else:
                return {
                    'success': False,
                    'message': 'API撤销订单失败'
                }
            
        except Exception as e:
            logger.error(f"撤销订单失败: {e}")
            return {
                'success': False,
                'message': f"撤销订单失败: {str(e)}"
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
        """取消止损止盈订单"""
        try:
            # 获取未成交订单
            open_orders = self.exchange_client.get_open_orders(symbol)
            
            for order_data in open_orders:
                order_type = order_data.get('type', '')
                
                if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                    order_id = order_data.get('orderId')
                    if order_id:
                        self.exchange_client.cancel_order(symbol, str(order_id))
            
            logger.info(f"止损止盈订单已取消: {symbol}")
            
        except Exception as e:
            logger.error(f"取消止损止盈订单失败: {e}")
    
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
            if existing_positions:
                logger.info(f"检测到现有虚拟仓位，先平仓...")
                for pos in existing_positions:
                    await postgresql_manager.close_virtual_position(pos['id'], current_price)
                    
                    # 🔑 计算价差盈亏（quantity现在是USDT价值，需要转换成币的数量）
                    coin_amount = pos['quantity'] / pos['entry_price']  # 币的数量
                    if pos['side'] == 'LONG':
                        price_pnl = (current_price - pos['entry_price']) * coin_amount
                    else:  # SHORT
                        price_pnl = (pos['entry_price'] - current_price) * coin_amount
                    
                    # 🔑 计算手续费（quantity已经是USDT价值）
                    open_position_value = pos['quantity']  # 开仓时的USDT价值
                    open_commission = open_position_value * VIRTUAL_OPEN_FEE_RATE
                    
                    close_position_value = coin_amount * current_price  # 平仓时的USDT价值
                    close_commission = close_position_value * VIRTUAL_CLOSE_FEE_RATE
                    
                    # 净盈亏
                    net_pnl = price_pnl - open_commission - close_commission
                    
                    # 创建平仓虚拟订单
                    close_order = {
                        'order_id': None,
                        'symbol': symbol,
                        'side': 'SELL' if pos['side'] == 'LONG' else 'BUY',
                        'type': 'MARKET',
                        'status': 'FILLED',
                        'quantity': pos['quantity'],
                        'price': current_price,
                        'filled_quantity': pos['quantity'],
                        'commission': close_commission,  # 平仓手续费 0.05%
                        'timestamp': int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                        'is_virtual': True,
                        'signal_id': signal_id,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'pnl': net_pnl,
                        'pnl_percent': (net_pnl / open_position_value) * 100
                    }
                    await postgresql_manager.write_order_data(close_order)
            
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
            position_value = signal.position_size
            open_commission = position_value * VIRTUAL_OPEN_FEE_RATE
            
            # 创建虚拟开仓订单
            order_data = {
                'order_id': None,
                'symbol': symbol,
                'side': 'BUY' if signal.signal_type == 'LONG' else 'SELL',
                'type': 'MARKET',
                'status': 'FILLED',
                'quantity': signal.position_size,
                'price': current_price,
                'filled_quantity': signal.position_size,
                'commission': open_commission,  # 开仓手续费 0.02%
                'timestamp': int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                'is_virtual': True,
                'signal_id': signal_id,
                'entry_price': current_price
            }
            
            await postgresql_manager.write_order_data(order_data)
            
            logger.info(f"✅ 虚拟开仓: {symbol} {signal.signal_type} {signal.position_size:.2f} USDT @{current_price:.2f}")
            logger.info(f"   止损: {signal.stop_loss:.2f} | 止盈: {signal.take_profit:.2f}")
            logger.info(f"   开仓手续费: ${open_commission:.4f} (0.02%)")
            
            # 🔑 刷新虚拟仓位缓存
            await self._refresh_virtual_positions_cache(symbol)
            
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
                
                # 🔑 计算价差盈亏（quantity现在是USDT价值，需要转换成币的数量）
                coin_amount = pos['quantity'] / pos['entry_price']  # 币的数量
                if pos['side'] == 'LONG':
                    price_pnl = (current_price - pos['entry_price']) * coin_amount
                else:  # SHORT
                    price_pnl = (pos['entry_price'] - current_price) * coin_amount
                
                # 🔑 计算手续费（quantity已经是USDT价值）
                open_position_value = pos['quantity']  # 开仓时的USDT价值
                open_commission = open_position_value * VIRTUAL_OPEN_FEE_RATE
                
                close_position_value = coin_amount * current_price  # 平仓时的USDT价值
                close_commission = close_position_value * VIRTUAL_CLOSE_FEE_RATE
                
                # 净盈亏 = 价差盈亏 - 开仓手续费 - 平仓手续费
                net_pnl = price_pnl - open_commission - close_commission
                
                total_pnl += net_pnl
                
                # 创建平仓虚拟订单
                close_order = {
                    'order_id': None,
                    'symbol': symbol,
                    'side': 'SELL' if pos['side'] == 'LONG' else 'BUY',
                    'type': 'MARKET',
                    'status': 'FILLED',
                    'quantity': pos['quantity'],
                    'price': current_price,
                    'filled_quantity': pos['quantity'],
                    'commission': close_commission,  # 平仓手续费 0.05%
                    'timestamp': int(datetime.now().timestamp() * 1000),  # ✅ 毫秒时间戳
                    'is_virtual': True,
                    'signal_id': signal_id,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'pnl': net_pnl,  # 已扣除手续费的净盈亏
                    'pnl_percent': (net_pnl / open_position_value) * 100
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
        reason: str
    ):
        """
        因止损止盈触发而平仓（WebSocket实时监控触发）
        
        Args:
            pos_id: 仓位ID
            current_price: 当前价格
            reason: 平仓原因
        """
        try:
            # 获取仓位信息
            pos = await postgresql_manager.get_virtual_position_by_id(pos_id)
            if not pos or pos['status'] != 'OPEN':
                return
            
            symbol = pos['symbol']
            
            # 平仓
            await postgresql_manager.close_virtual_position(pos_id, current_price)
            
            # 🔑 计算盈亏（quantity是USDT价值）
            coin_amount = pos['quantity'] / pos['entry_price']  # 币的数量
            
            if pos['side'] == 'LONG':
                price_pnl = (current_price - pos['entry_price']) * coin_amount
            else:  # SHORT
                price_pnl = (pos['entry_price'] - current_price) * coin_amount
            
            # 手续费
            open_commission = pos['quantity'] * VIRTUAL_OPEN_FEE_RATE  # 0.02%
            close_commission = coin_amount * current_price * VIRTUAL_CLOSE_FEE_RATE  # 0.05%
            
            # 净盈亏
            net_pnl = price_pnl - open_commission - close_commission
            pnl_percent = (net_pnl / pos['quantity']) * 100
            
            # 记录平仓订单
            order_data = {
                'order_id': None,
                'symbol': symbol,
                'side': 'SELL' if pos['side'] == 'LONG' else 'BUY',
                'type': 'MARKET',
                'status': 'FILLED',
                'quantity': pos['quantity'],
                'price': current_price,
                'filled_quantity': pos['quantity'],
                'commission': close_commission,
                'timestamp': int(datetime.now().timestamp() * 1000),
                'is_virtual': True,
                'signal_id': pos.get('signal_id', ''),
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'pnl': net_pnl,
                'pnl_percent': pnl_percent
            }
            
            await postgresql_manager.write_order_data(order_data)
            
            logger.info(f"✅ 虚拟平仓: {symbol} {pos['side']} {pos['quantity']:.2f} USDT @{current_price:.2f}")
            logger.info(f"   {reason}")
            logger.info(f"   净盈亏: ${net_pnl:+.2f} ({pnl_percent:+.2f}%)")
            
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
            
            # 检查账户余额
            account_info = self.exchange_client.get_account_info()
            available_balance = float(account_info.get('available_balance', 0))
            
            if available_balance <= 0:
                return {
                    'allowed': False,
                    'reason': '账户余额不足'
                }
            
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
        """加载订单和持仓"""
        try:
            # 从API获取未成交订单
            open_orders = self.exchange_client.get_open_orders()
            
            for order_data in open_orders:
                order = Order(
                    order_id=str(order_data['orderId']),
                    client_order_id=order_data.get('clientOrderId', ''),
                    symbol=order_data['symbol'],
                    side=OrderSide(order_data['side']),
                    type=OrderType(order_data['type']),
                    quantity=float(order_data['origQty']),
                    price=float(order_data['price']) if order_data['price'] else None,
                    stop_price=float(order_data['stopPrice']) if order_data.get('stopPrice') else None,
                    status=OrderStatus(order_data['status']),
                    filled_quantity=float(order_data['executedQty']),
                    remaining_quantity=float(order_data['origQty']) - float(order_data['executedQty']),
                    avg_price=float(order_data.get('avgPrice', 0)),
                    commission=0.0,
                    created_at=order_data['time'],        # ✅ Binance原始时间戳
                    updated_at=order_data['updateTime'],  # ✅ Binance原始时间戳
                    metadata={}
                )
                
                self.orders[order.order_id] = order
            
            logger.info(f"加载了{len(self.orders)}个未成交订单")
            
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
            
            # 记录触发平仓的仓位ID
            closed_position_ids = []
            
            # 检查每个仓位的止损止盈
            for pos in positions:
                should_close = False
                reason = ""
                
                # 检查止损
                if pos['side'] == 'LONG' and price <= pos['stop_loss']:
                    should_close = True
                    reason = f"止损触发 ({price:.2f} <= {pos['stop_loss']:.2f})"
                elif pos['side'] == 'SHORT' and price >= pos['stop_loss']:
                    should_close = True
                    reason = f"止损触发 ({price:.2f} >= {pos['stop_loss']:.2f})"
                
                # 检查止盈
                if not should_close:
                    if pos['side'] == 'LONG' and price >= pos['take_profit']:
                        should_close = True
                        reason = f"止盈触发 ({price:.2f} >= {pos['take_profit']:.2f})"
                    elif pos['side'] == 'SHORT' and price <= pos['take_profit']:
                        should_close = True
                        reason = f"止盈触发 ({price:.2f} <= {pos['take_profit']:.2f})"
                
                # 触发平仓
                if should_close:
                    logger.info(f"🎯 {symbol} {pos['side']} {reason}")
                    await self._close_virtual_position_by_trigger(
                        pos_id=pos['id'],
                        current_price=price,
                        reason=reason
                    )
                    closed_position_ids.append(pos['id'])
            
            # 🔑 如果有仓位被平掉，统一刷新缓存（避免循环中多次刷新）
            if closed_position_ids:
                await self._refresh_virtual_positions_cache(symbol)
                logger.debug(f"🔄 已平仓{len(closed_position_ids)}个仓位，缓存已刷新")
            
        except Exception as e:
            logger.error(f"处理价格更新失败: {e}")
    
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