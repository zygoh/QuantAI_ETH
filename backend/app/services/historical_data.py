"""
历史数据获取和管理
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from app.core.config import settings
from app.core.database import postgresql_manager
from app.services.binance_client import binance_client

logger = logging.getLogger(__name__)

class HistoricalDataManager:
    """历史数据管理器"""
    
    def __init__(self):
        self.batch_size = 1000  # 每批获取的数据量
        self.rate_limit_delay = 0.1  # API调用间隔（秒）
    
    async def fetch_all_historical_data(self, symbol: str, days: int = 30):
        """获取所有时间框架的历史数据"""
        try:
            logger.info(f"开始获取历史数据: {symbol} {days}天")
            
            timeframes = settings.TIMEFRAMES
            
            for interval in timeframes:
                await self.fetch_historical_klines(symbol, interval, days)
                await asyncio.sleep(self.rate_limit_delay)
            
            logger.info(f"历史数据获取完成: {symbol}")
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            raise
    
    async def fetch_historical_klines(
        self, 
        symbol: str, 
        interval: str, 
        days: int = 30
    ):
        """获取指定时间框架的历史K线数据"""
        try:
            # 计算时间范围（只获取已完成的K线）
            now = datetime.now()
            interval_minutes = self._get_interval_minutes(interval)
            
            # ✅ 计算最后一根已完成K线的开始时间
            # 例如：当前16:17，15分钟K线，正在进行的是16:15-16:30，最后完成的是16:00-16:15
            current_minute = now.hour * 60 + now.minute
            current_period_start = (current_minute // interval_minutes) * interval_minutes
            # 减去一个周期得到最后已完成K线的开始时间
            last_completed_start = current_period_start - interval_minutes
            
            # 处理跨天的情况
            if last_completed_start < 0:
                # 如果是负数，说明跨天了，从前一天算
                end_time = (now - timedelta(days=1)).replace(
                    hour=23, minute=(1440 + last_completed_start) % 60, second=0, microsecond=0
                )
            else:
                end_time = now.replace(
                    minute=last_completed_start % 60, 
                    hour=last_completed_start // 60,
                    second=0, microsecond=0
                )
            
            start_time = end_time - timedelta(days=days)
            
            # 计算需要获取的批次
            total_klines = int((days * 24 * 60) / interval_minutes)
            batches = (total_klines + self.batch_size - 1) // self.batch_size
            
            logger.info(f"获取历史K线: {symbol} {interval} {days}天 {batches}批次（截止到{end_time.strftime('%H:%M')}）")
            
            all_klines = []
            current_end_time = int(end_time.timestamp() * 1000)
            
            for batch in range(batches):
                try:
                    # 获取一批数据
                    klines = binance_client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=self.batch_size,
                        end_time=current_end_time
                    )
                    
                    if not klines:
                        break
                    
                    # 添加到总列表
                    all_klines.extend(klines)
                    
                    # 更新时间范围
                    current_end_time = klines[0]['timestamp'] - 1
                    
                    # 检查是否已经获取到足够的数据
                    if klines[0]['timestamp'] < int(start_time.timestamp() * 1000):
                        break
                    
                    logger.debug(f"批次 {batch + 1}/{batches} 完成: {len(klines)}条")
                    
                    # API限流
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"获取批次数据失败: {e}")
                    continue
            
            # 过滤时间范围内的数据
            filtered_klines = [
                kline for kline in all_klines
                if kline['timestamp'] >= int(start_time.timestamp() * 1000)
            ]
            
            # 按时间排序
            filtered_klines.sort(key=lambda x: x['timestamp'])
            
            # 批量写入数据库
            if filtered_klines:
                await self._batch_write_klines(symbol, interval, filtered_klines)
            
            logger.info(f"历史K线获取完成: {symbol} {interval} {len(filtered_klines)}条")
            
        except Exception as e:
            logger.error(f"获取历史K线失败: {e}")
            raise
    
    async def _batch_write_klines(
        self, 
        symbol: str, 
        interval: str, 
        klines: List[Dict[str, Any]]
    ):
        """批量写入K线数据（优化：一次性写入，避免循环调用）"""
        try:
            # 🔥 优化：给所有数据添加metadata后一次性写入
            klines_with_meta = [
                {**kline, 'symbol': symbol, 'interval': interval}
                for kline in klines
            ]
            
            # 一次性写入（内部会自动分批）
            await postgresql_manager.write_kline_data(klines_with_meta)
            
            logger.debug(f"批量写入完成: {symbol} {interval} {len(klines)}条")
            
        except Exception as e:
            logger.error(f"批量写入K线数据失败: {e}")
            raise
    
    def _get_interval_minutes(self, interval: str) -> int:
        """获取时间间隔的分钟数"""
        interval_map = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440,
            '3d': 4320,
            '1w': 10080,
            '1M': 43200
        }
        return interval_map.get(interval, 60)
    
    async def update_recent_data(self, symbol: str, hours: int = 24):
        """更新最近的数据"""
        try:
            logger.info(f"更新最近数据: {symbol} {hours}小时")
            
            timeframes = settings.TIMEFRAMES
            
            for interval in timeframes:
                await self._update_recent_klines(symbol, interval, hours)
                await asyncio.sleep(self.rate_limit_delay)
            
            logger.info(f"最近数据更新完成: {symbol}")
            
        except Exception as e:
            logger.error(f"更新最近数据失败: {e}")
    
    async def _update_recent_klines(
        self, 
        symbol: str, 
        interval: str, 
        hours: int = 24
    ):
        """更新最近的K线数据（只获取已完成的K线）"""
        try:
            # 计算需要的数据量
            interval_minutes = self._get_interval_minutes(interval)
            limit = min(int((hours * 60) / interval_minutes), 1000)
            
            # ✅ 计算最后一根已完成K线的开始时间
            now = datetime.now()
            current_minute = now.hour * 60 + now.minute
            current_period_start = (current_minute // interval_minutes) * interval_minutes
            # 减去一个周期得到最后已完成K线的开始时间
            last_completed_start = current_period_start - interval_minutes
            
            # 处理跨天的情况
            if last_completed_start < 0:
                end_time = (now - timedelta(days=1)).replace(
                    hour=23, minute=(1440 + last_completed_start) % 60, second=0, microsecond=0
                )
            else:
                end_time = now.replace(
                    minute=last_completed_start % 60, 
                    hour=last_completed_start // 60,
                    second=0, microsecond=0
                )
            
            end_time_ms = int(end_time.timestamp() * 1000)
            
            # 获取最新数据（只到最后已完成的K线）
            klines = binance_client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                end_time=end_time_ms  # ✅ 只获取已完成的K线
            )
            
            if klines:
                # 给每条数据添加 symbol 和 interval
                klines_with_meta = [
                    {**kline, 'symbol': symbol, 'interval': interval}
                    for kline in klines
                ]
                await postgresql_manager.write_kline_data(klines_with_meta)
                logger.debug(f"更新K线数据: {symbol} {interval} {len(klines)}条")
            
        except Exception as e:
            logger.error(f"更新K线数据失败: {e}")
    
    async def validate_data_integrity(self, symbol: str, interval: str, days: int = 7):
        """验证数据完整性"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # 从数据库查询数据
            df = await postgresql_manager.query_kline_data(
                symbol, interval, start_time, end_time
            )
            
            if df.empty:
                logger.warning(f"数据库中没有数据: {symbol} {interval}")
                return False
            
            # 检查数据连续性
            interval_minutes = self._get_interval_minutes(interval)
            expected_count = int((days * 24 * 60) / interval_minutes)
            actual_count = len(df)
            
            completeness = actual_count / expected_count
            
            logger.info(f"数据完整性: {symbol} {interval} {completeness:.2%} ({actual_count}/{expected_count})")
            
            # 如果完整性低于90%，建议重新获取
            if completeness < 0.9:
                logger.warning(f"数据完整性较低，建议重新获取: {symbol} {interval}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证数据完整性失败: {e}")
            return False
    
    async def get_data_summary(self, symbol: str) -> Dict[str, Any]:
        """获取数据摘要"""
        try:
            summary = {
                'symbol': symbol,
                'timeframes': {},
                'total_records': 0,
                'date_range': {}
            }
            
            for interval in settings.TIMEFRAMES:
                # 查询最近7天的数据
                end_time = datetime.now()
                start_time = end_time - timedelta(days=7)
                
                df = await postgresql_manager.query_kline_data(
                    symbol, interval, start_time, end_time
                )
                
                if not df.empty:
                    summary['timeframes'][interval] = {
                        'count': len(df),
                        'start_time': df['timestamp'].min().isoformat(),
                        'end_time': df['timestamp'].max().isoformat(),
                        'latest_price': float(df['close'].iloc[-1])
                    }
                    summary['total_records'] += len(df)
                else:
                    summary['timeframes'][interval] = {
                        'count': 0,
                        'start_time': None,
                        'end_time': None,
                        'latest_price': None
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取数据摘要失败: {e}")
            return {}
    
    async def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        try:
            await postgresql_manager.cleanup_old_data(days)
            logger.info(f"清理了{days}天前的旧数据")
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")

# 全局历史数据管理器实例
historical_data_manager = HistoricalDataManager()