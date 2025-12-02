"""
模拟交易所客户端

用于测试的模拟交易所客户端，不发送真实请求
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.exchange.base_exchange_client import (
    BaseExchangeClient,
    UnifiedKlineData,
    UnifiedTickerData,
    UnifiedOrderData
)

logger = logging.getLogger(__name__)


class MockExchangeClient(BaseExchangeClient):
    """
    模拟交易所客户端（用于测试）
    
    不发送真实API请求，返回预定义的测试数据
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Mock客户端
        
        Args:
            config: 可选配置参数
        """
        self.call_history: List[tuple] = []
        self.mock_responses: Dict[str, Any] = {}
        self.error_mode: Optional[str] = None
        
        logger.info("✅ Mock交易所客户端初始化完成")
    
    def set_mock_response(self, method: str, response: Any):
        """
        设置模拟响应
        
        Args:
            method: 方法名
            response: 模拟响应数据
        """
        self.mock_responses[method] = response
        logger.debug(f"📝 设置Mock响应: {method}")
    
    def set_error_mode(self, error_type: Optional[str]):
        """
        设置错误模式
        
        Args:
            error_type: 错误类型（network_error, auth_error等）
        """
        self.error_mode = error_type
        logger.debug(f"⚠️ 设置错误模式: {error_type}")
    
    def get_call_history(self) -> List[tuple]:
        """
        获取调用历史
        
        Returns:
            调用历史列表
        """
        return self.call_history
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        self.call_history.append(("test_connection",))
        
        if self.error_mode == "connection_error":
            return False
        
        return True
    
    def get_server_time(self) -> int:
        """获取服务器时间"""
        self.call_history.append(("get_server_time",))
        
        if "get_server_time" in self.mock_responses:
            return self.mock_responses["get_server_time"]
        
        return int(datetime.now().timestamp() * 1000)
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        self.call_history.append(("get_exchange_info",))
        
        if "get_exchange_info" in self.mock_responses:
            return self.mock_responses["get_exchange_info"]
        
        return {"exchange": "MOCK", "status": "ok"}
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取交易对信息"""
        self.call_history.append(("get_symbol_info", symbol))
        
        if "get_symbol_info" in self.mock_responses:
            return self.mock_responses["get_symbol_info"]
        
        return {"symbol": symbol, "status": "TRADING"}
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[UnifiedKlineData]:
        """获取K线数据"""
        self.call_history.append(("get_klines", symbol, interval, limit))
        
        if self.error_mode == "network_error":
            return []
        
        if "get_klines" in self.mock_responses:
            return self.mock_responses["get_klines"]
        
        # 返回默认测试数据
        return [
            UnifiedKlineData(
                timestamp=1609459200000,
                open=1000.0,
                high=1100.0,
                low=900.0,
                close=1050.0,
                volume=10000.0,
                close_time=1609459259999,
                quote_volume=10500000.0,
                trades=1000,
                taker_buy_base_volume=5000.0,
                taker_buy_quote_volume=5250000.0
            )
        ]
    
    def get_klines_paginated(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        rate_limit_delay: float = 0.1
    ) -> List[UnifiedKlineData]:
        """分页获取K线数据"""
        self.call_history.append(("get_klines_paginated", symbol, interval, limit))
        return self.get_klines(symbol, interval, limit, start_time, end_time)
    
    def get_ticker_price(self, symbol: str) -> Optional[UnifiedTickerData]:
        """获取实时价格"""
        self.call_history.append(("get_ticker_price", symbol))
        
        if "get_ticker_price" in self.mock_responses:
            return self.mock_responses["get_ticker_price"]
        
        return UnifiedTickerData(
            symbol=symbol,
            price=1000.0,
            timestamp=int(datetime.now().timestamp() * 1000)
        )
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        self.call_history.append(("get_account_info",))
        
        if "get_account_info" in self.mock_responses:
            return self.mock_responses["get_account_info"]
        
        return {
            'total_wallet_balance': 10000.0,
            'available_balance': 8000.0,
            'can_trade': True
        }
    
    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        self.call_history.append(("get_position_info", symbol))
        
        if "get_position_info" in self.mock_responses:
            return self.mock_responses["get_position_info"]
        
        return []
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_position: bool = False,
        stop_price: Optional[float] = None,
        callback_rate: Optional[float] = None,
        working_type: str = "MARK_PRICE"
    ) -> Dict[str, Any]:
        """下单"""
        self.call_history.append(("place_order", symbol, side, order_type, quantity, price))
        
        if "place_order" in self.mock_responses:
            return self.mock_responses["place_order"]
        
        return {
            'orderId': 'MOCK_ORDER_123',
            'status': 'FILLED',
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'price': price
        }
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """取消订单"""
        self.call_history.append(("cancel_order", symbol, order_id))
        
        if "cancel_order" in self.mock_responses:
            return self.mock_responses["cancel_order"]
        
        return {'success': True, 'orderId': order_id}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        self.call_history.append(("get_open_orders", symbol))
        
        if "get_open_orders" in self.mock_responses:
            return self.mock_responses["get_open_orders"]
        
        return []
    
    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """修改杠杆倍数"""
        self.call_history.append(("change_leverage", symbol, leverage))
        
        if "change_leverage" in self.mock_responses:
            return self.mock_responses["change_leverage"]
        
        return {'success': True, 'leverage': leverage}
    
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """修改保证金模式"""
        self.call_history.append(("change_margin_type", symbol, margin_type))
        
        if "change_margin_type" in self.mock_responses:
            return self.mock_responses["change_margin_type"]
        
        return {'success': True, 'margin_type': margin_type}
