"""
仓位管理器

职责：
1. 🎯 计算仓位大小（统一的仓位计算逻辑）
2. 📊 查询持仓信息（通过 Binance API，用于展示）
3. ⚙️ 初始化杠杆设置

注意：
- 本模块不负责持仓状态管理（依赖 Binance API 实时查询）
- 仓位计算已统一到此模块，其他模块不应重复实现
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from app.core.config import settings
from app.core.database import postgresql_manager
from app.core.cache import cache_manager
from app.exchange.exchange_factory import ExchangeFactory

logger = logging.getLogger(__name__)

# 🎯 虚拟账户配置（用于 SIGNAL_ONLY 模式）
VIRTUAL_ACCOUNT_BALANCE = 100.0  # 虚拟账户初始余额（USDT）
VIRTUAL_BALANCE_KEY = "virtual_account:balance"  # Redis键名

@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    side: str  # LONG, SHORT
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    margin_type: str
    leverage: int
    liquidation_price: float
    margin_ratio: float
    created_at: datetime
    updated_at: datetime

@dataclass
class RiskMetrics:
    """风险指标"""
    total_margin: float
    free_margin: float
    margin_level: float
    total_unrealized_pnl: float
    total_wallet_balance: float
    max_withdraw_amount: float

class PositionManager:
    """仓位管理器"""
    
    def __init__(self):
        self.positions: Dict[str, PositionInfo] = {}
        self.leverage = settings.LEVERAGE
        self.max_position_value = 500000  # 最大持仓价值（USDT）- 全仓模式需要较大值
        self.min_position_value = 20  # ✅ U本位最小仓位价值（币安要求）
        # 🔑 获取交易所客户端（使用工厂模式，支持多交易所）
        self.exchange_client = ExchangeFactory.get_current_client()
        
    async def initialize(self):
        """初始化仓位管理器（仅使用公共接口，不调用需要API key的功能）"""
        try:
            logger.info("初始化仓位管理器（模拟交易模式）...")
            
            # 🔥 模拟交易模式：不设置杠杆，不加载真实持仓
            # 所有持仓信息都来自虚拟交易引擎
            logger.info("✅ 仓位管理器初始化完成（使用虚拟持仓）")
            
        except Exception as e:
            logger.error(f"初始化仓位管理器失败: {e}")
            raise
            
        except Exception as e:
            logger.error(f"加载持仓失败: {e}")
    
    async def get_virtual_balance(self) -> float:
        """
        获取虚拟账户余额
        
        Returns:
            当前虚拟账户余额（USDT）
        """
        try:
            cached_balance = await cache_manager.get(VIRTUAL_BALANCE_KEY)
            if cached_balance is not None:
                try:
                    balance = float(cached_balance)
                    return max(0.0, balance)  # 确保不为负
                except (ValueError, TypeError):
                    pass
            
            # 如果不存在或格式错误，返回初始值
            balance = VIRTUAL_ACCOUNT_BALANCE
            await cache_manager.set(VIRTUAL_BALANCE_KEY, balance)
            return balance
        except Exception as e:
            logger.error(f"获取虚拟账户余额失败: {e}")
            return VIRTUAL_ACCOUNT_BALANCE
    
    async def update_virtual_balance(self, pnl: float) -> float:
        """
        更新虚拟账户余额（平仓后调用）
        
        Args:
            pnl: 净盈亏（USDT，已扣除手续费）
            
        Returns:
            更新后的余额
        """
        try:
            current_balance = await self.get_virtual_balance()
            new_balance = current_balance + pnl
            new_balance = max(0.0, new_balance)  # 确保不为负
            
            await cache_manager.set(VIRTUAL_BALANCE_KEY, new_balance)
            logger.info(f"💰 虚拟账户余额更新: {current_balance:.2f} → {new_balance:.2f} USDT (盈亏: {pnl:+.2f})")
            return new_balance
        except Exception as e:
            logger.error(f"更新虚拟账户余额失败: {e}")
            return await self.get_virtual_balance()
    
    async def reset_virtual_balance(self) -> float:
        """
        重置虚拟账户余额为初始值
        
        Returns:
            重置后的余额
        """
        try:
            await cache_manager.set(VIRTUAL_BALANCE_KEY, VIRTUAL_ACCOUNT_BALANCE)
            logger.info(f"🔄 虚拟账户余额已重置: {VIRTUAL_ACCOUNT_BALANCE} USDT")
            return VIRTUAL_ACCOUNT_BALANCE
        except Exception as e:
            logger.error(f"重置虚拟账户余额失败: {e}")
            return VIRTUAL_ACCOUNT_BALANCE
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        signal_type: str, 
        confidence: float,
        current_price: float,
        is_virtual: bool = True,  # 默认使用虚拟余额
        use_full_position: bool = True  # 🔥 是否使用全仓策略（默认全仓）
    ) -> float:
        """
        仓位计算（支持全仓和动态两种模式）
        
        全仓模式（默认）：
        - 仓位价值 = 全部可用余额 × 杠杆
        - 适合：中频交易、高置信度策略
        
        动态模式（可选）：
        - 基础仓位：10% × 置信度
        - 波动率调整：波动大→降仓位
        - 持仓调整：避免过度集中
        - 连续亏损保护：3连亏→减半
        - 最终限制：2%-15%
        
        Args:
            symbol: 交易对
            signal_type: 信号类型（LONG/SHORT）
            confidence: 信号置信度
            current_price: 当前价格
            is_virtual: 是否使用虚拟余额
            use_full_position: 是否使用全仓策略（默认True）
            
        Returns:
            仓位大小（USDT）
        """
        try:
            # 1. 获取可用余额
            if is_virtual:
                # 🔑 从Redis获取实际虚拟账户余额（如果不存在则使用初始值）
                cached_balance = await cache_manager.get(VIRTUAL_BALANCE_KEY)
                if cached_balance is not None:
                    try:
                        available_balance = float(cached_balance)
                        if available_balance <= 0:
                            logger.warning(f"⚠️ 虚拟账户余额不足: {available_balance}，重置为初始值")
                            available_balance = VIRTUAL_ACCOUNT_BALANCE
                            await cache_manager.set(VIRTUAL_BALANCE_KEY, available_balance)
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ 虚拟账户余额格式错误: {cached_balance}，重置为初始值")
                        available_balance = VIRTUAL_ACCOUNT_BALANCE
                        await cache_manager.set(VIRTUAL_BALANCE_KEY, available_balance)
                else:
                    # 首次使用，设置初始余额
                    available_balance = VIRTUAL_ACCOUNT_BALANCE
                    await cache_manager.set(VIRTUAL_BALANCE_KEY, available_balance)
                    logger.info(f"💰 初始化虚拟账户余额: {available_balance} USDT")
                
                logger.debug(f"📊 使用虚拟余额: {available_balance} USDT")
            else:
                # 🔥 模拟交易模式：即使is_virtual=False，也使用虚拟余额（不调用需要API key的接口）
                cached_balance = await cache_manager.get(VIRTUAL_BALANCE_KEY)
                if cached_balance is not None:
                    try:
                        available_balance = float(cached_balance)
                        if available_balance <= 0:
                            logger.warning(f"⚠️ 虚拟账户余额不足: {available_balance}，重置为初始值")
                            available_balance = VIRTUAL_ACCOUNT_BALANCE
                            await cache_manager.set(VIRTUAL_BALANCE_KEY, available_balance)
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ 虚拟账户余额格式错误: {cached_balance}，重置为初始值")
                        available_balance = VIRTUAL_ACCOUNT_BALANCE
                        await cache_manager.set(VIRTUAL_BALANCE_KEY, available_balance)
                else:
                    available_balance = VIRTUAL_ACCOUNT_BALANCE
                    await cache_manager.set(VIRTUAL_BALANCE_KEY, available_balance)
                    logger.info(f"💰 初始化虚拟账户余额: {available_balance} USDT")
                
                logger.debug(f"📊 使用虚拟余额（模拟模式）: {available_balance} USDT")
            
            # 2. 根据策略计算仓位
            if use_full_position:
                # ✅ 全仓策略：使用全部可用余额
                position_value = available_balance * self.leverage
                original_value = position_value
                
                # 限制最大仓位价值（安全保护）
                position_value = min(position_value, self.max_position_value)
                
                logger.debug(f"💰 全仓仓位计算: {symbol} | 余额: {available_balance:.2f} USDT | 杠杆: {self.leverage}x | 仓位价值: {position_value:.2f} USDT" + 
                           (f" (已限制，原始: {original_value:.2f})" if original_value > self.max_position_value else ""))
                
            else:
                # 动态仓位策略（可选）
                base_ratio = 0.10 * confidence
                logger.debug(f"  📌 基础仓位比例: {base_ratio*100:.1f}%")
                
                # 市场波动率调整
                volatility_adj = await self._get_volatility_adjustment(symbol)
                logger.debug(f"  📊 波动率调整: {volatility_adj:.2f}x")
                
                # 持仓调整
                exposure_adj = await self._get_exposure_adjustment(symbol, available_balance)
                logger.debug(f"  📊 持仓调整: {exposure_adj:.2f}x")
                
                # 连续亏损保护
                loss_adj = await self._get_loss_adjustment()
                logger.debug(f"  📊 亏损保护: {loss_adj:.2f}x")
                
                # 计算最终比例
                final_ratio = base_ratio * volatility_adj * exposure_adj * loss_adj
                final_ratio = max(0.02, min(final_ratio, 0.15))
                
                position_value = available_balance * final_ratio * self.leverage
                
                logger.info(f"💰 动态仓位计算: {symbol} {position_value:.2f} USDT")
                logger.info(f"  余额={available_balance:.2f} | 杠杆={self.leverage}x | 比例={final_ratio*100:.1f}%")
            
            # 3. 检查最小仓位要求
            if position_value < self.min_position_value:
                logger.warning(f"⚠️ 仓位不足最小要求: {position_value:.2f} < {self.min_position_value} USDT")
                return 0.0
            
            return position_value
            
        except Exception as e:
            logger.error(f"计算仓位失败: {e}")
            return 0.0
    
    async def _get_volatility_adjustment(self, symbol: str) -> float:
        """获取波动率调整系数（波动大→降仓位）"""
        try:
            # 从交易所API获取最近24小时价格变化
            # 注意：如果交易所不支持24h ticker，可以使用get_ticker_price
            ticker = self.exchange_client.get_ticker_price(symbol)
            if not ticker:
                return 1.0
            
            # 如果交易所不支持24h ticker，使用默认值
            # 这里需要根据实际交易所API调整
            price_change_percent = 0.0
            if hasattr(ticker, 'price_change_percent'):
                price_change_percent = abs(float(ticker.price_change_percent))
            elif isinstance(ticker, dict):
                price_change_percent = abs(float(ticker.get('priceChangePercent', 0)))
            
            # 波动率映射到调整系数
            if price_change_percent > 8.0:  # 日波动>8%
                return 0.5  # 减半仓位
            elif price_change_percent > 5.0:  # 5%-8%
                return 0.7  # 降低30%
            elif price_change_percent < 2.0:  # <2%
                return 1.3  # 增加30%
            else:
                return 1.0  # 正常
            
        except Exception as e:
            logger.warning(f"获取波动率调整失败: {e}")
            return 1.0
    
    async def _get_exposure_adjustment(self, symbol: str, available_balance: float) -> float:
        """获取持仓暴露调整系数（持仓多→降仓位）"""
        try:
            # 获取当前持仓
            positions = self.exchange_client.get_position_info(symbol)
            if not positions:
                return 1.0
            
            # 计算当前持仓占用的保证金比例
            total_position_value = 0.0
            for pos in positions:
                position_amt = abs(float(pos.get('positionAmt', 0)))
                if position_amt > 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    total_position_value += position_amt * entry_price
            
            exposure_ratio = total_position_value / (available_balance * self.leverage + 1e-10)
            
            # 暴露比例映射到调整系数
            if exposure_ratio > 0.5:  # 持仓>50%
                return 0.5  # 减半仓位
            elif exposure_ratio > 0.3:  # 30%-50%
                return 0.75  # 降低25%
            else:
                return 1.0  # 正常
            
        except Exception as e:
            logger.warning(f"获取持仓调整失败: {e}")
            return 1.0
    
    async def _get_loss_adjustment(self) -> float:
        """获取连续亏损调整系数（3连亏→减半）"""
        try:
            # 从Redis缓存获取最近交易记录
            recent_trades_key = f"recent_trades:{settings.SYMBOL}"
            recent_trades = await cache_manager.get(recent_trades_key)
            
            if not recent_trades:
                return 1.0
            
            # 统计最近5笔交易的盈亏
            if isinstance(recent_trades, list):
                recent_pnl = [trade.get('pnl', 0) for trade in recent_trades[-5:]]
                
                # 计算连续亏损次数
                consecutive_losses = 0
                for pnl in reversed(recent_pnl):
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        break
                
                # 连续亏损映射到调整系数
                if consecutive_losses >= 3:
                    logger.warning(f"⚠️ 检测到{consecutive_losses}连亏，降低仓位")
                    return 0.5  # 减半
                elif consecutive_losses >= 2:
                    return 0.75  # 降低25%
                else:
                    return 1.0  # 正常
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"获取亏损调整失败: {e}")
            return 1.0
    
    
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """获取指定持仓"""
        try:
            # 先从缓存获取
            position = self.positions.get(symbol)
            
            if position:
                # 更新持仓信息
                await self._update_position(position)
                return position
            
            # 从API获取
            positions = self.exchange_client.get_position_info(symbol)
            
            if positions:
                pos_data = positions[0]
                position_amt = float(pos_data.get('positionAmt', 0))
                
                if position_amt != 0:
                    position = PositionInfo(
                        symbol=symbol,
                        side='LONG' if position_amt > 0 else 'SHORT',
                        size=abs(position_amt),
                        entry_price=float(pos_data.get('entryPrice', 0)),
                        mark_price=float(pos_data.get('markPrice', 0)),
                        unrealized_pnl=float(pos_data.get('unRealizedProfit', 0)),
                        percentage=float(pos_data.get('percentage', 0)),
                        margin_type=pos_data.get('marginType', 'cross'),
                        leverage=int(pos_data.get('leverage', 1)),
                        liquidation_price=float(pos_data.get('liquidationPrice', 0)),
                        margin_ratio=0.0,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    self.positions[symbol] = position
                    return position
            
            return None
            
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return None
    
    async def _update_position(self, position: PositionInfo):
        """更新持仓信息"""
        try:
            # 从API获取最新信息
            positions = self.exchange_client.get_position_info(position.symbol)
            
            if positions:
                pos_data = positions[0]
                
                position.mark_price = float(pos_data.get('markPrice', 0))
                position.unrealized_pnl = float(pos_data.get('unRealizedProfit', 0))
                position.percentage = float(pos_data.get('percentage', 0))
                position.liquidation_price = float(pos_data.get('liquidationPrice', 0))
                position.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"更新持仓信息失败: {e}")
    
    async def get_all_positions(self) -> List[PositionInfo]:
        """获取所有持仓"""
        try:
            await self._load_positions()
            return list(self.positions.values())
            
        except Exception as e:
            logger.error(f"获取所有持仓失败: {e}")
            return []
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """计算风险指标（使用虚拟余额，不调用需要API key的接口）"""
        try:
            # 🔥 模拟交易模式：使用虚拟余额
            cached_balance = await cache_manager.get(VIRTUAL_BALANCE_KEY)
            if cached_balance is not None:
                try:
                    total_wallet_balance = float(cached_balance)
                except (ValueError, TypeError):
                    total_wallet_balance = VIRTUAL_ACCOUNT_BALANCE
            else:
                total_wallet_balance = VIRTUAL_ACCOUNT_BALANCE
            
            # 🔥 模拟交易模式：从虚拟持仓计算风险指标
            total_unrealized_pnl = 0.0
            total_position_initial_margin = 0.0
            available_balance = total_wallet_balance
            
            # 计算所有虚拟持仓的保证金和未实现盈亏
            for symbol, position in self.positions.items():
                if position.size > 0:
                    # 计算持仓保证金（简化计算：持仓价值 / 杠杆）
                    position_value = position.size * position.mark_price
                    position_margin = position_value / self.leverage
                    total_position_initial_margin += position_margin
                    total_unrealized_pnl += position.unrealized_pnl
            
            total_margin_balance = total_wallet_balance + total_unrealized_pnl
            available_balance = total_margin_balance - total_position_initial_margin
            
            # 计算保证金水平
            margin_level = 0
            if total_position_initial_margin > 0:
                margin_level = total_margin_balance / total_position_initial_margin
            else:
                margin_level = 999.0  # 无持仓时，保证金水平设为很高
            
            risk_metrics = RiskMetrics(
                total_margin=total_position_initial_margin,
                free_margin=max(0, available_balance),
                margin_level=margin_level,
                total_unrealized_pnl=total_unrealized_pnl,
                total_wallet_balance=total_wallet_balance,
                max_withdraw_amount=max(0, available_balance)
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0)
    
    async def check_margin_call_risk(self) -> Dict[str, Any]:
        """检查强平风险"""
        try:
            risk_metrics = await self.calculate_risk_metrics()
            
            # 强平风险阈值
            margin_call_threshold = 1.1  # 保证金水平低于110%时警告
            liquidation_threshold = 1.05  # 保证金水平低于105%时危险
            
            risk_level = "LOW"
            message = "保证金充足"
            
            if risk_metrics.margin_level < liquidation_threshold:
                risk_level = "CRITICAL"
                message = "强平风险极高，请立即减仓或追加保证金"
            elif risk_metrics.margin_level < margin_call_threshold:
                risk_level = "HIGH"
                message = "保证金不足，建议减仓或追加保证金"
            
            return {
                'risk_level': risk_level,
                'message': message,
                'margin_level': risk_metrics.margin_level,
                'free_margin': risk_metrics.free_margin,
                'total_margin': risk_metrics.total_margin
            }
            
        except Exception as e:
            logger.error(f"检查强平风险失败: {e}")
            return {
                'risk_level': 'UNKNOWN',
                'message': '无法获取风险信息',
                'margin_level': 0,
                'free_margin': 0,
                'total_margin': 0
            }
    
    async def calculate_position_value(self, symbol: str) -> float:
        """计算持仓价值"""
        try:
            position = await self.get_position(symbol)
            
            if not position:
                return 0.0
            
            return position.size * position.mark_price
            
        except Exception as e:
            logger.error(f"计算持仓价值失败: {e}")
            return 0.0
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """获取持仓摘要"""
        try:
            positions = await self.get_all_positions()
            risk_metrics = await self.calculate_risk_metrics()
            margin_risk = await self.check_margin_call_risk()
            
            total_positions = len(positions)
            long_positions = len([p for p in positions if p.side == 'LONG'])
            short_positions = len([p for p in positions if p.side == 'SHORT'])
            
            total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            
            summary = {
                'total_positions': total_positions,
                'long_positions': long_positions,
                'short_positions': short_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_wallet_balance': risk_metrics.total_wallet_balance,
                'available_balance': risk_metrics.free_margin,
                'margin_level': risk_metrics.margin_level,
                'risk_level': margin_risk['risk_level'],
                'risk_message': margin_risk['message'],
                'positions': [
                    {
                        'symbol': p.symbol,
                        'side': p.side,
                        'size': p.size,
                        'entry_price': p.entry_price,
                        'mark_price': p.mark_price,
                        'unrealized_pnl': p.unrealized_pnl,
                        'percentage': p.percentage
                    }
                    for p in positions
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取持仓摘要失败: {e}")
            return {}
    
    async def save_position_snapshot(self):
        """保存持仓快照"""
        try:
            positions = await self.get_all_positions()
            
            for position in positions:
                # 保存到数据库
                position_data = {
                    'timestamp': position.updated_at,
                    'symbol': position.symbol,
                    'side': position.side,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'mark_price': position.mark_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'percentage': position.percentage,
                    'leverage': position.leverage
                }
                
                # 这里可以扩展保存到PostgreSQL的逻辑（持仓历史记录）
                
            logger.debug(f"保存了{len(positions)}个持仓快照")
            
        except Exception as e:
            logger.error(f"保存持仓快照失败: {e}")

# 全局仓位管理器实例
position_manager = PositionManager()