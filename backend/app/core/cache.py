"""
缓存管理工具类
"""
import json
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import pickle
import asyncio

from app.core.database import redis_manager

logger = logging.getLogger(__name__)

class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.redis = redis_manager
    
    # ========================================
    # 通用缓存方法（基础）
    # ========================================
    
    async def get(self, key: str) -> Optional[Any]:
        """通用获取缓存方法"""
        try:
            value = await self.redis.get_cache(key)
            if value:
                # 尝试解析 JSON，如果失败则返回原始字符串
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return None
        except Exception as e:
            logger.error(f"获取缓存失败 [{key}]: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        """通用设置缓存方法"""
        try:
            # 处理 None 值
            if value is None:
                logger.debug(f"跳过设置缓存（值为None）[{key}]")
                return
            
            # 如果是字符串或数字，直接存储；否则 JSON 序列化
            if isinstance(value, (str, int, float, bool)):
                cache_value = str(value)
            else:
                cache_value = json.dumps(value, default=str)
            
            await self.redis.set_cache(key, cache_value, expire)
            logger.debug(f"设置缓存成功 [{key}]")
        except Exception as e:
            logger.error(f"设置缓存失败 [{key}]: {e}")
    
    # ========================================
    # 专用缓存方法（业务相关）
    # ========================================
    
    async def set_market_data(self, symbol: str, interval: str, data: Dict[str, Any], expire: int = 60):
        """缓存市场数据"""
        try:
            key = f"market_data:{symbol}:{interval}"
            value = json.dumps(data, default=str)
            await self.redis.set_cache(key, value, expire)
            logger.debug(f"缓存市场数据: {key}")
        except Exception as e:
            logger.error(f"缓存市场数据失败: {e}")
    
    async def get_market_data(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """获取缓存的市场数据"""
        try:
            key = f"market_data:{symbol}:{interval}"
            value = await self.redis.get_cache(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取缓存市场数据失败: {e}")
            return None
    
    async def set_model_prediction(self, symbol: str, prediction: Dict[str, Any], expire: int = 300):
        """缓存模型预测结果"""
        try:
            key = f"prediction:{symbol}"
            value = json.dumps(prediction, default=str)
            await self.redis.set_cache(key, value, expire)
            logger.debug(f"缓存模型预测: {key}")
        except Exception as e:
            logger.error(f"缓存模型预测失败: {e}")
    
    async def get_model_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取缓存的模型预测"""
        try:
            key = f"prediction:{symbol}"
            value = await self.redis.get_cache(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取缓存模型预测失败: {e}")
            return None
    
    async def set_account_info(self, account_data: Dict[str, Any], expire: int = 30):
        """缓存账户信息"""
        try:
            key = "account_info"
            await self.redis.set_hash(key, account_data)
            # 设置过期时间
            await self.redis.client.expire(key, expire)
            logger.debug("缓存账户信息")
        except Exception as e:
            logger.error(f"缓存账户信息失败: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取缓存的账户信息"""
        try:
            key = "account_info"
            return await self.redis.get_hash(key)
        except Exception as e:
            logger.error(f"获取缓存账户信息失败: {e}")
            return {}
    
    async def set_position_info(self, positions: List[Dict[str, Any]], expire: int = 30):
        """缓存持仓信息"""
        try:
            key = "position_info"
            value = json.dumps(positions, default=str)
            await self.redis.set_cache(key, value, expire)
            logger.debug("缓存持仓信息")
        except Exception as e:
            logger.error(f"缓存持仓信息失败: {e}")
    
    async def get_position_info(self) -> List[Dict[str, Any]]:
        """获取缓存的持仓信息"""
        try:
            key = "position_info"
            value = await self.redis.get_cache(key)
            if value:
                return json.loads(value)
            return []
        except Exception as e:
            logger.error(f"获取缓存持仓信息失败: {e}")
            return []
    
    async def set_trading_signal(self, symbol: str, signal: Dict[str, Any], expire: int = None):
        """缓存交易信号
        
        Args:
            symbol: 交易对符号
            signal: 信号数据
            expire: 过期时间（秒），None表示永不过期
        """
        try:
            key = f"signal:{symbol}"
            value = json.dumps(signal, default=str)
            
            if expire is None:
                # 不设置过期时间，永久保存（直到被新信号覆盖）
                await self.redis.client.set(key, value)
                logger.debug(f"缓存交易信号（永久）: {key}")
            else:
                # 设置过期时间
                await self.redis.set_cache(key, value, expire)
                logger.debug(f"缓存交易信号（{expire}秒）: {key}")
        except Exception as e:
            logger.error(f"缓存交易信号失败: {e}")
    
    async def get_trading_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取缓存的交易信号"""
        try:
            key = f"signal:{symbol}"
            value = await self.redis.get_cache(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取缓存交易信号失败: {e}")
            return None
    
    async def set_risk_metrics(self, metrics: Dict[str, Any], expire: int = 300):
        """缓存风险指标"""
        try:
            key = "risk_metrics"
            value = json.dumps(metrics, default=str)
            await self.redis.set_cache(key, value, expire)
            logger.debug("缓存风险指标")
        except Exception as e:
            logger.error(f"缓存风险指标失败: {e}")
    
    async def get_risk_metrics(self) -> Optional[Dict[str, Any]]:
        """获取缓存的风险指标"""
        try:
            key = "risk_metrics"
            value = await self.redis.get_cache(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取缓存风险指标失败: {e}")
            return None
    
    async def set_system_status(self, status: Dict[str, Any], expire: int = 60):
        """缓存系统状态"""
        try:
            key = "system_status"
            await self.redis.set_hash(key, status)
            await self.redis.client.expire(key, expire)
            logger.debug("缓存系统状态")
        except Exception as e:
            logger.error(f"缓存系统状态失败: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取缓存的系统状态"""
        try:
            key = "system_status"
            return await self.redis.get_hash(key)
        except Exception as e:
            logger.error(f"获取缓存系统状态失败: {e}")
            return {}
    
    async def set_model_metrics(self, symbol: str, metrics: Dict[str, Any], expire: int = None):
        """缓存模型性能指标（永久保存，直到下次训练更新）"""
        try:
            key = f"model_metrics:{symbol}"
            value = json.dumps(metrics, default=str)
            
            if expire is None:
                # 永久保存，不设置过期时间
                await self.redis.client.set(key, value)
                logger.debug(f"缓存模型指标（永久）: {key}")
            else:
                # 设置过期时间
                await self.redis.set_cache(key, value, expire)
                logger.debug(f"缓存模型指标（{expire}秒）: {key}")
        except Exception as e:
            logger.error(f"缓存模型指标失败: {e}")
    
    async def get_model_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取缓存的模型性能指标"""
        try:
            key = f"model_metrics:{symbol}"
            value = await self.redis.get_cache(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取缓存模型指标失败: {e}")
            return None
    
    async def increment_counter(self, key: str, expire: int = 3600) -> int:
        """递增计数器"""
        try:
            count = await self.redis.client.incr(key)
            await self.redis.client.expire(key, expire)
            return count
        except Exception as e:
            logger.error(f"递增计数器失败: {e}")
            return 0
    
    async def set_rate_limit(self, key: str, limit: int, window: int = 60):
        """设置速率限制"""
        try:
            current = await self.redis.client.incr(key)
            if current == 1:
                await self.redis.client.expire(key, window)
            return current <= limit
        except Exception as e:
            logger.error(f"设置速率限制失败: {e}")
            return True
    
    async def lock_resource(self, resource: str, timeout: int = 30) -> bool:
        """资源锁定"""
        try:
            key = f"lock:{resource}"
            result = await self.redis.client.set(key, "locked", nx=True, ex=timeout)
            return result is not None
        except Exception as e:
            logger.error(f"资源锁定失败: {e}")
            return False
    
    async def unlock_resource(self, resource: str):
        """释放资源锁"""
        try:
            key = f"lock:{resource}"
            await self.redis.delete_cache(key)
        except Exception as e:
            logger.error(f"释放资源锁失败: {e}")
    
    async def clear_cache_pattern(self, pattern: str):
        """清理匹配模式的缓存"""
        try:
            keys = []
            async for key in self.redis.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis.client.delete(*keys)
                logger.info(f"清理了{len(keys)}个缓存项: {pattern}")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            info = await self.redis.client.info()
            return {
                "used_memory": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}

# 全局缓存管理器实例
cache_manager = CacheManager()