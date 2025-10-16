"""
WebSocket相关API端点
"""
import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.api.models import WSMessage, WSPriceUpdate, WSSignalUpdate, WSOrderUpdate, WSRiskAlert

logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket连接管理
class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"WebSocket连接已建立，当前连接数: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info(f"WebSocket连接已断开，当前连接数: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"发送个人消息失败: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, channel: str = None):
        """广播消息"""
        disconnected = set()
        
        for websocket in self.active_connections:
            try:
                # 检查订阅
                if channel and channel not in self.subscriptions.get(websocket, set()):
                    continue
                
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected.add(websocket)
        
        # 清理断开的连接
        for websocket in disconnected:
            self.disconnect(websocket)
    
    def subscribe(self, websocket: WebSocket, channel: str):
        """订阅频道"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(channel)
            logger.info(f"WebSocket订阅频道: {channel}")
    
    def unsubscribe(self, websocket: WebSocket, channel: str):
        """取消订阅频道"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(channel)
            logger.info(f"WebSocket取消订阅频道: {channel}")

# 全局连接管理器
manager = ConnectionManager()

# 全局服务实例
data_service = None
signal_generator = None
trading_controller = None

def set_services(ds, sg, tc):
    """设置服务实例"""
    global data_service, signal_generator, trading_controller
    data_service = ds
    signal_generator = sg
    trading_controller = tc

@router.websocket("/connect")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    await manager.connect(websocket)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, message)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "无效的JSON格式"
                    }),
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket连接异常: {e}")
        manager.disconnect(websocket)

async def handle_websocket_message(websocket: WebSocket, message: dict):
    """处理WebSocket消息"""
    try:
        msg_type = message.get("type")
        
        if msg_type == "subscribe":
            # 订阅频道
            channel = message.get("channel")
            if channel:
                manager.subscribe(websocket, channel)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscribed",
                        "channel": channel,
                        "message": f"已订阅频道: {channel}"
                    }),
                    websocket
                )
        
        elif msg_type == "unsubscribe":
            # 取消订阅
            channel = message.get("channel")
            if channel:
                manager.unsubscribe(websocket, channel)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "unsubscribed",
                        "channel": channel,
                        "message": f"已取消订阅频道: {channel}"
                    }),
                    websocket
                )
        
        elif msg_type == "ping":
            # 心跳检测
            await manager.send_personal_message(
                json.dumps({
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                }),
                websocket
            )
        
        elif msg_type == "get_status":
            # 获取系统状态
            if trading_controller:
                status = trading_controller.get_system_status()
                await manager.send_personal_message(
                    json.dumps({
                        "type": "status",
                        "data": status
                    }),
                    websocket
                )
        
        else:
            await manager.send_personal_message(
                json.dumps({
                    "type": "error",
                    "message": f"未知消息类型: {msg_type}"
                }),
                websocket
            )
    
    except Exception as e:
        logger.error(f"处理WebSocket消息失败: {e}")
        await manager.send_personal_message(
            json.dumps({
                "type": "error",
                "message": "消息处理失败"
            }),
            websocket
        )

# WebSocket广播函数
async def broadcast_price_update(price_data: WSPriceUpdate):
    """广播价格更新"""
    message = WSMessage(
        type="price_update",
        data=price_data.dict()
    )
    
    await manager.broadcast(
        json.dumps(message.dict(), default=str),
        channel="price"
    )

async def broadcast_signal_update(signal_data: WSSignalUpdate):
    """广播信号更新"""
    message = WSMessage(
        type="signal_update",
        data=signal_data.dict()
    )
    
    await manager.broadcast(
        json.dumps(message.dict(), default=str),
        channel="signals"
    )

async def broadcast_order_update(order_data: WSOrderUpdate):
    """广播订单更新"""
    message = WSMessage(
        type="order_update",
        data=order_data.dict()
    )
    
    await manager.broadcast(
        json.dumps(message.dict(), default=str),
        channel="orders"
    )

async def broadcast_risk_alert(alert_data: WSRiskAlert):
    """广播风险警报"""
    message = WSMessage(
        type="risk_alert",
        data=alert_data.dict()
    )
    
    await manager.broadcast(
        json.dumps(message.dict(), default=str),
        channel="risk"
    )

async def broadcast_system_status(status_data: dict):
    """广播系统状态"""
    message = WSMessage(
        type="system_status",
        data=status_data
    )
    
    await manager.broadcast(
        json.dumps(message.dict(), default=str),
        channel="system"
    )

# 启动WebSocket数据推送任务
async def start_websocket_tasks():
    """启动WebSocket数据推送任务"""
    try:
        # 启动价格推送任务
        asyncio.create_task(price_push_task())
        
        # 启动状态推送任务
        asyncio.create_task(status_push_task())
        
        logger.info("WebSocket推送任务已启动")
        
    except Exception as e:
        logger.error(f"启动WebSocket推送任务失败: {e}")

async def price_push_task():
    """价格推送任务"""
    while True:
        try:
            if data_service and len(manager.active_connections) > 0:
                # 获取最新价格
                from app.core.cache import cache_manager
                ticker_data = await cache_manager.get_market_data("ETHUSDT", "ticker")
                
                if ticker_data:
                    price_update = WSPriceUpdate(
                        symbol="ETHUSDT",
                        price=float(ticker_data.get('price', 0)),
                        change=0,  # 需要计算
                        change_percent=0,  # 需要计算
                        timestamp=ticker_data.get('timestamp')
                    )
                    
                    await broadcast_price_update(price_update)
            
            # 每5秒推送一次
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"价格推送任务异常: {e}")
            await asyncio.sleep(5)

async def status_push_task():
    """状态推送任务"""
    while True:
        try:
            if trading_controller and len(manager.active_connections) > 0:
                # 获取系统状态
                status = trading_controller.get_system_status()
                await broadcast_system_status(status)
            
            # 每30秒推送一次
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"状态推送任务异常: {e}")
            await asyncio.sleep(30)

# 信号回调函数
async def on_signal_generated(signal):
    """信号生成回调"""
    try:
        from app.api.models import TradingSignal
        
        signal_update = WSSignalUpdate(
            signal=TradingSignal(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size=signal.position_size
            )
        )
        
        await broadcast_signal_update(signal_update)
        
    except Exception as e:
        logger.error(f"信号回调失败: {e}")

# 风险警报回调函数
async def on_risk_alert(alert):
    """风险警报回调"""
    try:
        risk_alert = WSRiskAlert(
            alert_type=alert.alert_type,
            message=alert.message,
            current_drawdown=alert.current_drawdown,
            threshold=alert.threshold,
            recommended_action=alert.recommended_action
        )
        
        await broadcast_risk_alert(risk_alert)
        
    except Exception as e:
        logger.error(f"风险警报回调失败: {e}")

# 获取连接统计
@router.get("/stats")
async def get_websocket_stats():
    """获取WebSocket连接统计"""
    try:
        stats = {
            'active_connections': len(manager.active_connections),
            'total_subscriptions': sum(len(subs) for subs in manager.subscriptions.values()),
            'channels': {}
        }
        
        # 统计各频道订阅数
        for websocket, channels in manager.subscriptions.items():
            for channel in channels:
                stats['channels'][channel] = stats['channels'].get(channel, 0) + 1
        
        return {
            'success': True,
            'message': 'WebSocket统计获取成功',
            'data': stats
        }
        
    except Exception as e:
        logger.error(f"获取WebSocket统计失败: {e}")
        return {
            'success': False,
            'message': f'获取WebSocket统计失败: {str(e)}'
        }