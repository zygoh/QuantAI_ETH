"""
仓位管理器

职责：
1. 🎯 计算仓位大小（统一的仓位计算逻辑）
2. 💰 虚拟账户余额管理
3. 📊 获取持仓摘要和风险指标

注意：
- 本模块用于虚拟交易模式，不调用需要API key的接口
- 实际持仓信息从数据库获取（virtual_trades表）
"""
# StdLib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

# Local App
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)

# 🎯 虚拟账户配置（用于 SIGNAL_ONLY 模式）
VIRTUAL_ACCOUNT_BALANCE = 20.0  # 🔥 虚拟账户初始余额（USDT）- 调整为20U
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
        self.leverage = settings.LEVERAGE
        self.max_position_value = 500000  # 最大持仓价值（USDT）- 全仓模式需要较大值
        self.min_position_value = 20  # ✅ U本位最小仓位价值（币安要求）
                
    async def initialize(self):
        """初始化仓位管理器（模拟交易模式）"""
        try:
            logger.info("初始化仓位管理器（模拟交易模式）...")
            logger.info("✅ 仓位管理器初始化完成（使用虚拟持仓）")
        except Exception as e:
            logger.error(f"初始化仓位管理器失败: {e}")
            raise
    
    # ========== 虚拟账户余额管理 ==========
    
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
    
    # ========== 仓位计算 ==========
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        signal_type: str, 
        confidence: float,
        current_price: float,
        is_virtual: bool = True,
        use_full_position: bool = True  # 🔥 默认使用全仓策略
    ) -> float:
        """
        仓位计算（全仓策略）
        
        全仓模式：
        - 仓位价值 = 全部可用余额 × 杠杆
        - 适合：中频交易、高置信度策略
        
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
            # 1. 获取可用余额（虚拟账户）
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
            
            logger.info(f"📊 使用虚拟余额: {available_balance:.2f} USDT")
            
            # 2. 全仓策略计算
            position_value = available_balance * self.leverage
            original_value = position_value
            
            # 限制最大仓位价值（安全保护）
            position_value = min(position_value, self.max_position_value)
            
            logger.info(f"💰 全仓仓位计算: {symbol} | 余额: {available_balance:.2f} USDT | 杠杆: {self.leverage}x | 仓位价值: {position_value:.2f} USDT" + 
                       (f" (已限制，原始: {original_value:.2f})" if original_value > self.max_position_value else ""))
            
            # 3. 检查最小仓位要求
            if position_value < self.min_position_value:
                logger.warning(f"⚠️ 仓位不足最小要求: {position_value:.2f} < {self.min_position_value} USDT")
                return 0.0
            
            return position_value
            
        except Exception as e:
            logger.error(f"计算仓位失败: {e}")
            return 0.0
    
    # ========== 持仓信息查询 ==========
    
    async def get_all_positions(self) -> List[PositionInfo]:
        """
        获取所有持仓
        
        注意：虚拟交易模式下，持仓信息应从数据库获取
        本方法返回空列表，实际持仓查询应使用 postgresql_manager.get_open_virtual_positions()
        """
        # 虚拟交易模式不维护内存中的持仓状态
        return []
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """计算风险指标（使用虚拟余额）"""
        try:
            # 获取虚拟余额
            total_wallet_balance = await self.get_virtual_balance()
            
            # 虚拟交易模式：返回基本的风险指标
            risk_metrics = RiskMetrics(
                total_margin=0.0,
                free_margin=total_wallet_balance,
                margin_level=999.0,  # 无持仓时，保证金水平设为很高
                total_unrealized_pnl=0.0,
                total_wallet_balance=total_wallet_balance,
                max_withdraw_amount=total_wallet_balance
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0)
    
    async def check_margin_call_risk(self) -> Dict[str, Any]:
        """检查强平风险"""
        try:
            risk_metrics = await self.calculate_risk_metrics()
            
            # 虚拟交易模式下，强平风险始终为低
            return {
                'risk_level': 'LOW',
                'message': '保证金充足',
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
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """获取持仓摘要"""
        try:
            risk_metrics = await self.calculate_risk_metrics()
            margin_risk = await self.check_margin_call_risk()
            
            summary = {
                'total_positions': 0,
                'long_positions': 0,
                'short_positions': 0,
                'total_unrealized_pnl': 0.0,
                'total_wallet_balance': risk_metrics.total_wallet_balance,
                'available_balance': risk_metrics.free_margin,
                'margin_level': risk_metrics.margin_level,
                'risk_level': margin_risk['risk_level'],
                'risk_message': margin_risk['message'],
                'positions': []
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取持仓摘要失败: {e}")
            return {}

# 全局仓位管理器实例
position_manager = PositionManager()
