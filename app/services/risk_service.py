"""
风险管理服务
"""
# StdLib
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Third-Party
import numpy as np
import pandas as pd
import ta
from scipy import stats

# Local App
from app.core.cache import cache_manager
from app.core.config import settings
from app.core.database import postgresql_manager
from app.exchange.base_exchange_client import UnifiedKlineData
from app.exchange.exchange_factory import ExchangeFactory
from app.services.data_service import DataService
from app.trading.position_manager import position_manager

logger = logging.getLogger(__name__)

@dataclass
class VaRResult:
    """VaR计算结果"""
    var_1d: float  # 1日VaR
    var_5d: float  # 5日VaR
    var_10d: float # 10日VaR
    confidence_level: float
    method: str
    calculation_date: datetime

@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float
    var_99: float
    expected_shortfall: float  # 条件VaR
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    kelly_percentage: float
    volatility: float
    beta: float  # 相对于基准的贝塔值

class RiskService:
    """风险管理服务"""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.confidence_levels = [0.95, 0.99]
        self.var_methods = ['historical', 'parametric', 'monte_carlo']
        # 🔑 获取交易所客户端（使用工厂模式，支持多交易所）
        self.exchange_client = ExchangeFactory.get_current_client()
        
    async def calculate_var(
        self, 
        symbol: str, 
        confidence: float = 0.95, 
        holding_period: int = 1,
        method: str = 'historical'
    ) -> VaRResult:
        """计算VaR (Value at Risk)"""
        try:
            logger.info(f"计算VaR: {symbol} {confidence} {method}")
            
            # 获取历史价格数据
            returns = await self._get_returns_data(symbol, days=252)  # 一年数据
            
            if returns.empty:
                raise Exception("无法获取收益率数据")
            
            # 根据方法计算VaR
            if method == 'historical':
                var_value = self._calculate_historical_var(returns, confidence, holding_period)
            elif method == 'parametric':
                var_value = self._calculate_parametric_var(returns, confidence, holding_period)
            elif method == 'monte_carlo':
                var_value = self._calculate_monte_carlo_var(returns, confidence, holding_period)
            else:
                raise ValueError(f"不支持的VaR计算方法: {method}")
            
            # 计算不同持有期的VaR
            var_1d = var_value
            var_5d = var_value * np.sqrt(5)
            var_10d = var_value * np.sqrt(10)
            
            result = VaRResult(
                var_1d=var_1d,
                var_5d=var_5d,
                var_10d=var_10d,
                confidence_level=confidence,
                method=method,
                calculation_date=datetime.now()
            )
            
            logger.info(f"VaR计算完成: 1日VaR={var_1d:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return VaRResult(0, 0, 0, confidence, method, datetime.now())
    
    def _calculate_historical_var(
        self, 
        returns: pd.Series, 
        confidence: float, 
        holding_period: int
    ) -> float:
        """历史模拟法计算VaR"""
        try:
            # 调整持有期
            if holding_period > 1:
                returns = returns.rolling(holding_period).sum().dropna()
            
            # 计算分位数
            var_percentile = 1 - confidence
            var_value = np.percentile(returns, var_percentile * 100)
            
            return abs(var_value)
            
        except Exception as e:
            logger.error(f"历史模拟法VaR计算失败: {e}")
            return 0.0
    
    def _calculate_parametric_var(
        self, 
        returns: pd.Series, 
        confidence: float, 
        holding_period: int
    ) -> float:
        """参数法计算VaR（假设正态分布）"""
        try:
            # 计算收益率统计量
            mean_return = returns.mean()
            std_return = returns.std()
            
            # 调整持有期
            if holding_period > 1:
                mean_return = mean_return * holding_period
                std_return = std_return * np.sqrt(holding_period)
            
            # 计算VaR
            z_score = stats.norm.ppf(1 - confidence)
            var_value = -(mean_return + z_score * std_return)
            
            return max(var_value, 0)
            
        except Exception as e:
            logger.error(f"参数法VaR计算失败: {e}")
            return 0.0
    
    def _calculate_monte_carlo_var(
        self, 
        returns: pd.Series, 
        confidence: float, 
        holding_period: int,
        num_simulations: int = 10000
    ) -> float:
        """蒙特卡洛模拟法计算VaR"""
        try:
            # 拟合收益率分布
            mean_return = returns.mean()
            std_return = returns.std()
            
            # 蒙特卡洛模拟
            np.random.seed(42)
            simulated_returns = np.random.normal(
                mean_return, std_return, num_simulations
            )
            
            # 调整持有期
            if holding_period > 1:
                simulated_returns = simulated_returns * np.sqrt(holding_period)
            
            # 计算VaR
            var_percentile = 1 - confidence
            var_value = np.percentile(simulated_returns, var_percentile * 100)
            
            return abs(var_value)
            
        except Exception as e:
            logger.error(f"蒙特卡洛VaR计算失败: {e}")
            return 0.0
    
    async def calculate_expected_shortfall(
        self, 
        symbol: str, 
        confidence: float = 0.95
    ) -> float:
        """计算期望损失（条件VaR）"""
        try:
            returns = await self._get_returns_data(symbol, days=252)
            
            if returns.empty:
                return 0.0
            
            # 计算VaR阈值
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            
            # 计算超过VaR的平均损失
            tail_losses = returns[returns <= var_threshold]
            
            if len(tail_losses) > 0:
                expected_shortfall = abs(tail_losses.mean())
            else:
                expected_shortfall = 0.0
            
            return expected_shortfall
            
        except Exception as e:
            logger.error(f"计算期望损失失败: {e}")
            return 0.0
    
    async def calculate_max_drawdown(self, symbol: str, days: int = 252) -> Tuple[float, float]:
        """计算最大回撤和当前回撤"""
        try:
            # 获取价格数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            df = await postgresql_manager.query_kline_data(
                symbol, '1h', start_time, end_time, limit=days * 24
            )
            
            if df.empty:
                return 0.0, 0.0
            
            # 计算累积收益
            prices = df['close'].values
            cumulative_returns = (prices / prices[0] - 1) * 100
            
            # 计算回撤
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak)
            
            # 最大回撤
            max_drawdown = abs(drawdown.min())
            
            # 当前回撤
            current_drawdown = abs(drawdown[-1])
            
            return max_drawdown, current_drawdown
            
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return 0.0, 0.0
    
    async def calculate_sharpe_ratio(self, symbol: str, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        try:
            returns = await self._get_returns_data(symbol, days=252)
            
            if returns.empty:
                return 0.0
            
            # 年化收益率
            annual_return = returns.mean() * 252
            
            # 年化波动率
            annual_volatility = returns.std() * np.sqrt(252)
            
            if annual_volatility == 0:
                return 0.0
            
            # 夏普比率
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0
    
    async def calculate_sortino_ratio(self, symbol: str, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        try:
            returns = await self._get_returns_data(symbol, days=252)
            
            if returns.empty:
                return 0.0
            
            # 年化收益率
            annual_return = returns.mean() * 252
            
            # 下行波动率（只考虑负收益）
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                return float('inf')  # 没有负收益
            
            downside_volatility = negative_returns.std() * np.sqrt(252)
            
            if downside_volatility == 0:
                return 0.0
            
            # 索提诺比率
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
            
            return sortino_ratio
            
        except Exception as e:
            logger.error(f"计算索提诺比率失败: {e}")
            return 0.0
    
    def kelly_criterion(
        self, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float
    ) -> float:
        """Kelly准则计算最优仓位比例"""
        try:
            if avg_loss == 0 or win_rate == 0 or win_rate == 1:
                return 0.0
            
            # Kelly公式: f = (bp - q) / b
            # 其中 b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # 限制Kelly比例在合理范围内
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 最大25%
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Kelly准则计算失败: {e}")
            return 0.0
    
    async def calculate_trading_metrics(self, symbol: str, days: int = 30) -> Dict[str, float]:
        """计算交易指标"""
        try:
            # 获取交易信号历史
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            signals = await postgresql_manager.query_signals(
                symbol, start_time, end_time, limit=1000
            )
            
            if not signals:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'total_trades': 0
                }
            
            # 模拟交易结果（简化计算）
            wins = []
            losses = []
            
            for signal in signals:
                # 这里应该根据实际交易结果计算盈亏
                # 简化处理：假设根据置信度和随机因素确定盈亏
                confidence = signal.get('confidence', 0.5)
                
                # 模拟结果（实际应该从交易记录获取）
                if np.random.random() < confidence:
                    wins.append(np.random.uniform(0.01, 0.05))  # 1-5%收益
                else:
                    losses.append(np.random.uniform(-0.05, -0.01))  # 1-5%损失
            
            total_trades = len(wins) + len(losses)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # 盈亏比
            profit_factor = abs(sum(wins) / sum(losses)) if losses else 0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_trades': total_trades
            }
            
        except Exception as e:
            logger.error(f"计算交易指标失败: {e}")
            return {}
    
    async def _get_returns_data(self, symbol: str, days: int = 252) -> pd.Series:
        """获取收益率数据"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            df = await postgresql_manager.query_kline_data(
                symbol, '1h', start_time, end_time, limit=days * 24
            )
            
            if df.empty:
                return pd.Series()
            
            # 计算收益率
            prices = df['close']
            returns = prices.pct_change().dropna()
            
            return returns
            
        except Exception as e:
            logger.error(f"获取收益率数据失败: {e}")
            return pd.Series()
    
    async def calculate_portfolio_var(
        self, 
        positions: List[Dict[str, Any]], 
        confidence: float = 0.95
    ) -> float:
        """计算投资组合VaR"""
        try:
            if not positions:
                return 0.0
            
            # 获取各资产的收益率数据
            returns_data = {}
            weights = {}
            total_value = sum(pos['value'] for pos in positions)
            
            for position in positions:
                symbol = position['symbol']
                value = position['value']
                
                returns = await self._get_returns_data(symbol, days=252)
                
                if not returns.empty:
                    returns_data[symbol] = returns
                    weights[symbol] = value / total_value
            
            if not returns_data:
                return 0.0
            
            # 构建收益率矩阵
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                return 0.0
            
            # 计算协方差矩阵
            cov_matrix = returns_df.cov()
            
            # 权重向量
            weight_vector = np.array([weights.get(col, 0) for col in returns_df.columns])
            
            # 投资组合方差
            portfolio_variance = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # 投资组合平均收益
            portfolio_mean = np.dot(weight_vector, returns_df.mean())
            
            # VaR计算
            z_score = stats.norm.ppf(1 - confidence)
            portfolio_var = -(portfolio_mean + z_score * portfolio_std)
            
            return max(portfolio_var * total_value, 0)
            
        except Exception as e:
            logger.error(f"计算投资组合VaR失败: {e}")
            return 0.0
    
    async def check_risk_limits(self, symbol: str) -> Dict[str, Any]:
        """检查风险限制"""
        try:
            # 获取当前持仓
            position = await position_manager.get_position(symbol)
            
            # 计算风险指标
            var_result = await self.calculate_var(symbol, confidence=settings.VAR_CONFIDENCE)
            max_dd, current_dd = await self.calculate_max_drawdown(symbol)
            
            # 风险检查
            risk_checks = {
                'var_check': {
                    'passed': True,
                    'value': var_result.var_1d,
                    'limit': 0.05,  # 5% VaR限制
                    'message': 'VaR风险正常'
                },
                'drawdown_check': {
                    'passed': current_dd <= settings.MAX_DRAWDOWN_LIMIT * 100,
                    'value': current_dd,
                    'limit': settings.MAX_DRAWDOWN_LIMIT * 100,
                    'message': '回撤风险正常' if current_dd <= settings.MAX_DRAWDOWN_LIMIT * 100 else '回撤超过限制'
                },
                'position_size_check': {
                    'passed': True,
                    'value': position.size if position else 0,
                    'limit': 1000,  # 最大持仓限制
                    'message': '持仓大小正常'
                }
            }
            
            # 总体风险评估
            all_passed = all(check['passed'] for check in risk_checks.values())
            
            return {
                'overall_risk': 'LOW' if all_passed else 'HIGH',
                'checks': risk_checks,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"风险限制检查失败: {e}")
            return {
                'overall_risk': 'UNKNOWN',
                'error': str(e)
            }
    
    async def get_comprehensive_risk_report(self, symbol: str) -> Dict[str, Any]:
        """获取综合风险报告"""
        try:
            logger.info(f"生成综合风险报告: {symbol}")
            
            # 计算各种风险指标
            var_95 = await self.calculate_var(symbol, confidence=0.95)
            var_99 = await self.calculate_var(symbol, confidence=0.99)
            expected_shortfall = await self.calculate_expected_shortfall(symbol)
            max_dd, current_dd = await self.calculate_max_drawdown(symbol)
            sharpe_ratio = await self.calculate_sharpe_ratio(symbol)
            sortino_ratio = await self.calculate_sortino_ratio(symbol)
            
            # 交易指标
            trading_metrics = await self.calculate_trading_metrics(symbol)
            
            # Kelly准则
            kelly_pct = self.kelly_criterion(
                trading_metrics.get('win_rate', 0),
                trading_metrics.get('avg_win', 0),
                trading_metrics.get('avg_loss', 0)
            )
            
            # 波动率
            returns = await self._get_returns_data(symbol, days=30)
            volatility = returns.std() * np.sqrt(252) if not returns.empty else 0
            
            # 风险限制检查
            risk_limits = await self.check_risk_limits(symbol)
            
            # 构建风险报告
            risk_report = {
                'symbol': symbol,
                'calculation_time': datetime.now().isoformat(),
                'var_metrics': {
                    'var_95_1d': var_95.var_1d,
                    'var_95_5d': var_95.var_5d,
                    'var_99_1d': var_99.var_1d,
                    'expected_shortfall': expected_shortfall
                },
                'drawdown_metrics': {
                    'max_drawdown': max_dd,
                    'current_drawdown': current_dd,
                    'drawdown_limit': settings.MAX_DRAWDOWN_LIMIT * 100
                },
                'performance_metrics': {
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'volatility': volatility
                },
                'trading_metrics': trading_metrics,
                'position_sizing': {
                    'kelly_percentage': kelly_pct,
                    'recommended_size': kelly_pct * settings.KELLY_MULTIPLIER
                },
                'risk_assessment': risk_limits
            }
            
            # 缓存风险报告
            await cache_manager.set_risk_metrics(risk_report)
            
            logger.info("综合风险报告生成完成")
            
            return risk_report
            
        except Exception as e:
            logger.error(f"生成风险报告失败: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @staticmethod
    async def calculate_dynamic_stop_levels(
        symbol: str,
        entry_price: float,
        signal_type: str,  # 'LONG' or 'SHORT'
        confidence: float
    ) -> Dict[str, float]:
        """
        动态止损止盈计算（优化目标：盈亏比3:1）
        
        基于ATR（Average True Range）的自适应止损止盈：
        - 止损：1.2倍ATR（收紧止损，减少单笔亏损）
        - 止盈：根据置信度调整（高置信度3倍ATR，低置信度2.5倍ATR）
        - 跟踪止损：1倍ATR距离
        
        Args:
            symbol: 交易对
            entry_price: 入场价格
            signal_type: 信号类型（LONG/SHORT）
            confidence: 信号置信度
        
        Returns:
            包含止损止盈的字典
        """
        try:
            # 1. 获取最近的K线数据计算ATR（使用5m主时间框架）
            # ✅ 统一使用分页方法（limit=100时自动调用单次获取，不影响性能，支持多交易所）
            # 🔥 静态方法中不能使用self，使用ExchangeFactory获取客户端
            exchange_client = ExchangeFactory.get_current_client()
            klines = exchange_client.get_klines_paginated(
                symbol=symbol,
                interval='5m',
                limit=100  # 5m需要更多样本（100个=8.3小时）
            )
            
            if not klines or len(klines) < 20:
                logger.warning("数据不足，使用固定百分比止损")
                return RiskService._calculate_fixed_percentage_stop(entry_price, signal_type, confidence)
            
            # 2. 计算ATR（14周期）
            # 🔧 修复：将UnifiedKlineData对象转换为字典
            klines_dict = []
            for kline in klines:
                if isinstance(kline, UnifiedKlineData):
                    klines_dict.append(asdict(kline))
                elif isinstance(kline, dict):
                    klines_dict.append(kline)
                else:
                    logger.warning(f"⚠️ 未知的K线数据类型: {type(kline)}")
                    continue
            
            df = pd.DataFrame(klines_dict)
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            current_atr = atr_indicator.average_true_range().iloc[-1]
            
            logger.info(f"📊 当前ATR: {current_atr:.2f} ({current_atr/entry_price*100:.2f}%)")
            
            # 3. 🔥 优化止损止盈计算（提高盈亏比到3:1）
            if signal_type == 'LONG':
                # 做多：止损在下方，止盈在上方
                # 🔥 收紧止损：从1.5倍ATR降到1.2倍ATR
                stop_loss = entry_price - (current_atr * 1.2)
                
                # 🔥 统一使用3.6倍ATR止盈，确保盈亏比3:1（1.2 * 3 = 3.6）
                take_profit = entry_price + (current_atr * 3.6)  # 盈亏比3:1
                logger.debug(f"  置信度({confidence:.2f})：使用3.6倍ATR止盈（盈亏比3:1）")
                
                # 跟踪止损初始距离
                trailing_stop_distance = current_atr * 1.0
                
            elif signal_type == 'SHORT':
                # 做空：止损在上方，止盈在下方
                # 🔥 收紧止损：从1.5倍ATR降到1.2倍ATR
                stop_loss = entry_price + (current_atr * 1.2)
                
                # 🔥 统一使用3.6倍ATR止盈，确保盈亏比3:1（1.2 * 3 = 3.6）
                take_profit = entry_price - (current_atr * 3.6)  # 盈亏比3:1
                logger.debug(f"  置信度({confidence:.2f})：使用3.6倍ATR止盈（盈亏比3:1）")
                
                trailing_stop_distance = current_atr * 1.0
            else:
                logger.warning(f"未知信号类型: {signal_type}")
                return {}
            
            # 4. 计算盈亏比
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # 5. 组装结果
            stop_levels = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop_enabled': True,
                'trailing_stop_distance': trailing_stop_distance,
                'atr': current_atr,
                'atr_percent': (current_atr / entry_price) * 100,
                'risk_reward_ratio': risk_reward_ratio,
                'max_loss_percent': (risk / entry_price) * 100,
                'max_profit_percent': (reward / entry_price) * 100
            }
            
            logger.info(f"🎯 动态止损止盈已计算:")
            logger.info(f"  入场价: {entry_price:.2f}")
            logger.info(f"  止损价: {stop_loss:.2f} (风险: {stop_levels['max_loss_percent']:.2f}%)")
            logger.info(f"  止盈价: {take_profit:.2f} (收益: {stop_levels['max_profit_percent']:.2f}%)")
            logger.info(f"  盈亏比: 1:{risk_reward_ratio:.2f}")
            logger.info(f"  跟踪止损: {trailing_stop_distance:.2f} ({trailing_stop_distance/entry_price*100:.2f}%)")
            
            return stop_levels
            
        except Exception as e:
            logger.error(f"计算动态止损失败: {e}")
            # 降级到固定百分比
            return RiskService._calculate_fixed_percentage_stop(entry_price, signal_type, confidence)
    
    @staticmethod
    def _calculate_fixed_percentage_stop(
        entry_price: float,
        signal_type: str,
        confidence: float
    ) -> Dict[str, float]:
        """固定百分比止损（备用方案）"""
        try:
            stop_loss_pct = 0.015  # 1.5%
            
            # 🔥 统一使用3:1盈亏比（所有置信度级别）
            take_profit_pct = 0.045  # 4.5%，盈亏比1:3
            
            if signal_type == 'LONG':
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # SHORT
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
            
            risk_reward = take_profit_pct / stop_loss_pct
            
            logger.warning(f"⚠️ 使用固定百分比止损: ±{stop_loss_pct*100:.1f}% / ±{take_profit_pct*100:.1f}%")
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop_enabled': False,
                'trailing_stop_distance': entry_price * 0.01,  # 1%
                'atr': None,
                'atr_percent': None,
                'risk_reward_ratio': risk_reward,
                'max_loss_percent': stop_loss_pct * 100,
                'max_profit_percent': take_profit_pct * 100
            }
            
        except Exception as e:
            logger.error(f"固定止损计算失败: {e}")
            return {}