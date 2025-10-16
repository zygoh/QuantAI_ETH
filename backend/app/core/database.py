"""
PostgreSQL + TimescaleDB 数据库管理
完全替换 InfluxDB，保持接口兼容性
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import redis.asyncio as redis
import json
import pytz

from app.core.config import settings

logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """PostgreSQL + TimescaleDB 管理器"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        
    async def connect(self):
        """连接到PostgreSQL"""
        try:
            # 创建异步引擎
            database_url = (
                f"postgresql+asyncpg://{settings.PG_USER}:{settings.PG_PASSWORD}"
                f"@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DATABASE}"
            )
            
            self.engine = create_async_engine(
                database_url,
                echo=False,
                pool_size=settings.PG_POOL_SIZE,
                max_overflow=settings.PG_MAX_OVERFLOW,
                pool_pre_ping=True,  # 连接池健康检查
                pool_recycle=3600    # 1小时回收连接
            )
            
            # 创建会话工厂
            self.SessionLocal = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # 测试连接
            await self.health_check()
            
            # 初始化数据库结构
            await self._init_schema()
            
            logger.info("PostgreSQL连接成功")
            
        except Exception as e:
            logger.error(f"PostgreSQL连接失败: {e}")
            raise
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
            return True
        except Exception as e:
            logger.debug(f"PostgreSQL健康检查失败: {e}")
            return False
    
    async def _init_schema(self):
        """初始化数据库表结构（如果不存在）"""
        try:
            async with self.engine.begin() as conn:
                # 1. 启用 TimescaleDB 扩展
                await conn.execute(text("""
                    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE
                """))
                
                # 2. 创建 klines 表
                # ✅ 使用 BIGINT 存储 Binance 原始毫秒时间戳
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS klines (
                        time BIGINT NOT NULL,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        open NUMERIC(20, 8) NOT NULL,
                        high NUMERIC(20, 8) NOT NULL,
                        low NUMERIC(20, 8) NOT NULL,
                        close NUMERIC(20, 8) NOT NULL,
                        volume NUMERIC(30, 8) NOT NULL,
                        close_time BIGINT NOT NULL,
                        quote_volume NUMERIC(30, 8),
                        trades INTEGER DEFAULT 0,
                        taker_buy_base_volume NUMERIC(30, 8) DEFAULT 0,
                        taker_buy_quote_volume NUMERIC(30, 8) DEFAULT 0,
                        PRIMARY KEY (symbol, interval, time)
                    )
                """))
                
                # 3. TimescaleDB hypertable（由于time改为BIGINT，不使用hypertable）
                # 注意：TimescaleDB的hypertable要求时间列为TIMESTAMP类型
                # 由于我们使用BIGINT存储原始时间戳，不启用hypertable
                # PostgreSQL 的 B-tree 索引对于我们的查询已经足够快
                logger.debug("跳过 hypertable 创建（time列为BIGINT类型）")
                
                # 4. 创建索引
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_time 
                        ON klines (symbol, interval, time DESC)
                """))
                
                # 5. 创建交易信号表
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id BIGSERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence NUMERIC(5, 4) NOT NULL,
                        entry_price NUMERIC(20, 8) NOT NULL,
                        stop_loss NUMERIC(20, 8) DEFAULT 0,
                        take_profit NUMERIC(20, 8) DEFAULT 0,
                        position_size NUMERIC(20, 8) DEFAULT 0,
                        timestamp TIMESTAMPTZ NOT NULL,
                        predictions JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
                        ON trading_signals (symbol, timestamp DESC)
                """))
                
                # 6. 创建订单表
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id BIGSERIAL PRIMARY KEY,
                        order_id BIGINT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        quantity NUMERIC(20, 8) NOT NULL,
                        price NUMERIC(20, 8) DEFAULT 0,
                        filled_quantity NUMERIC(20, 8) DEFAULT 0,
                        commission NUMERIC(20, 8) DEFAULT 0,
                        timestamp TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        is_virtual BOOLEAN DEFAULT FALSE,
                        signal_id TEXT,
                        entry_price NUMERIC(20, 8),
                        exit_price NUMERIC(20, 8),
                        pnl NUMERIC(20, 8),
                        pnl_percent NUMERIC(10, 4)
                    )
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_orders_symbol_time 
                        ON orders (symbol, timestamp DESC)
                """))
                
                # 7. 创建虚拟仓位表（用于信号模式）
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS virtual_positions (
                        id BIGSERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price NUMERIC(20, 8) NOT NULL,
                        quantity NUMERIC(20, 8) NOT NULL,
                        entry_time TIMESTAMPTZ NOT NULL,
                        exit_price NUMERIC(20, 8),
                        exit_time TIMESTAMPTZ,
                        stop_loss NUMERIC(20, 8),
                        take_profit NUMERIC(20, 8),
                        pnl NUMERIC(20, 8) DEFAULT 0,
                        pnl_percent NUMERIC(10, 4) DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'OPEN',
                        signal_id TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_virtual_positions_symbol_status 
                        ON virtual_positions (symbol, status)
                """))
                
                logger.info("数据库表结构初始化完成")
                
        except Exception as e:
            logger.warning(f"初始化数据库结构失败（可能已存在）: {e}")
    
    async def write_kline_data(self, data: List[Dict[str, Any]]):
        """批量写入K线数据（🚀 一条 VALUES 列表 SQL，最快速度）"""
        try:
            if not data:
                return

            # 工具函数：转义单引号，防止SQL注入
            def esc(s: str) -> str:
                return s.replace("'", "''")
            
            # 工具函数：确保时间戳为整数（毫秒）
            def to_timestamp_int(ts) -> int:
                if isinstance(ts, (int, float)):
                    return int(ts)  # ✅ 直接返回整数时间戳
                elif isinstance(ts, datetime):
                    # datetime 转回毫秒时间戳
                    return int(ts.timestamp() * 1000)
                else:
                    # 未知类型，返回当前时间戳
                    return int(datetime.now().timestamp() * 1000)

            # 拼接 VALUES 列表（一条SQL搞定所有数据）
            values_list = []
            for k in data:
                ts = to_timestamp_int(k['timestamp'])
                close_ts = to_timestamp_int(k.get('close_time', k['timestamp']))
                
                values_list.append(
                    f"({ts},'{esc(k['symbol'])}','{esc(k['interval'])}',"
                    f"{k['open']},{k['high']},{k['low']},{k['close']},{k['volume']},"
                    f"{close_ts},{k.get('quote_volume', 0)},"
                    f"{k.get('trades', 0)},{k.get('taker_buy_base_volume', 0)},"
                    f"{k.get('taker_buy_quote_volume', 0)})"
                )

            if len(data) > 1000:
                logger.info(f"📊 准备写入{len(data)}条数据（批量INSERT）...")

            # 构造完整SQL
            sql = (
                "INSERT INTO klines "
                "(time, symbol, interval, open, high, low, close, volume, "
                "close_time, quote_volume, trades, taker_buy_base_volume, taker_buy_quote_volume) "
                "VALUES " + ",".join(values_list) +
                " ON CONFLICT (symbol, interval, time) DO UPDATE SET "
                "open = EXCLUDED.open, "
                "high = EXCLUDED.high, "
                "low = EXCLUDED.low, "
                "close = EXCLUDED.close, "
                "volume = EXCLUDED.volume, "
                "close_time = EXCLUDED.close_time, "
                "quote_volume = EXCLUDED.quote_volume, "
                "trades = EXCLUDED.trades, "
                "taker_buy_base_volume = EXCLUDED.taker_buy_base_volume, "
                "taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume"
            )

            # 一条SQL发送，PostgreSQL自己处理冲突
            async with self.SessionLocal() as session:
                connection = await session.connection()
                raw_connection = await connection.get_raw_connection()
                pg_conn = raw_connection.driver_connection
                
                async with pg_conn.transaction():
                    await pg_conn.execute(sql)
                    
                    if len(data) > 1000:
                        logger.info(f"   ✓ 批量插入完成: {len(data)}条")

            logger.debug(f"写入{len(data)}条K线数据")

        except Exception as e:
            logger.error(f"写入K线数据失败: {e}")
            logger.warning("数据库写入失败不影响系统运行（数据在WebSocket缓冲区中）")
    
    async def write_signal_data(self, signal: Dict[str, Any]):
        """写入交易信号数据"""
        try:
            async with self.SessionLocal() as session:
                async with session.begin():
                    stmt = text("""
                        INSERT INTO trading_signals (
                            symbol, signal_type, confidence, entry_price,
                            stop_loss, take_profit, position_size, timestamp, predictions
                        ) VALUES (
                            :symbol, :signal_type, :confidence, :entry_price,
                            :stop_loss, :take_profit, :position_size, :timestamp, :predictions
                        )
                    """)
                    
                    # 处理时间戳（统一使用UTC）
                    timestamp_val = signal.get('timestamp')
                    if isinstance(timestamp_val, (int, float)):
                        import pytz
                        timestamp_val = datetime.fromtimestamp(timestamp_val / 1000, tz=pytz.UTC)
                    elif not isinstance(timestamp_val, datetime):
                        import pytz
                        timestamp_val = datetime.now(pytz.UTC)
                    
                    # 预测数据转为 JSON
                    predictions_json = None
                    if 'predictions' in signal:
                        predictions_json = json.dumps(signal['predictions'])
                    
                    await session.execute(stmt, {
                        'symbol': signal['symbol'],
                        'signal_type': signal['signal_type'],
                        'confidence': float(signal['confidence']),
                        'entry_price': float(signal['entry_price']),
                        'stop_loss': float(signal.get('stop_loss', 0)),
                        'take_profit': float(signal.get('take_profit', 0)),
                        'position_size': float(signal.get('position_size', 0)),
                        'timestamp': timestamp_val,
                        'predictions': predictions_json
                    })
            
            logger.debug(f"写入交易信号: {signal['symbol']} {signal['signal_type']}")
            
        except Exception as e:
            logger.error(f"写入交易信号失败: {e}")
            raise
    
    async def write_order_data(self, order: Dict[str, Any]):
        """写入订单数据（支持虚拟订单）"""
        try:
            async with self.SessionLocal() as session:
                async with session.begin():
                    stmt = text("""
                        INSERT INTO orders (
                            order_id, symbol, side, order_type, status,
                            quantity, price, filled_quantity, commission, timestamp,
                            is_virtual, signal_id, entry_price, exit_price, pnl, pnl_percent
                        ) VALUES (
                            :order_id, :symbol, :side, :order_type, :status,
                            :quantity, :price, :filled_quantity, :commission, :timestamp,
                            :is_virtual, :signal_id, :entry_price, :exit_price, :pnl, :pnl_percent
                        )
                    """)
                    
                    # 处理时间戳（统一使用UTC）
                    timestamp_val = order.get('timestamp')
                    if isinstance(timestamp_val, (int, float)):
                        import pytz
                        timestamp_val = datetime.fromtimestamp(timestamp_val / 1000, tz=pytz.UTC)
                    elif not isinstance(timestamp_val, datetime):
                        import pytz
                        timestamp_val = datetime.now(pytz.UTC)
                    
                    await session.execute(stmt, {
                        'order_id': order.get('order_id'),
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'order_type': order['type'],
                        'status': order['status'],
                        'quantity': float(order['quantity']),
                        'price': float(order.get('price', 0)),
                        'filled_quantity': float(order.get('filled_quantity', 0)),
                        'commission': float(order.get('commission', 0)),
                        'timestamp': timestamp_val,
                        'is_virtual': order.get('is_virtual', False),
                        'signal_id': order.get('signal_id'),
                        'entry_price': float(order.get('entry_price', 0)) if order.get('entry_price') else None,
                        'exit_price': float(order.get('exit_price', 0)) if order.get('exit_price') else None,
                        'pnl': float(order.get('pnl', 0)) if order.get('pnl') else None,
                        'pnl_percent': float(order.get('pnl_percent', 0)) if order.get('pnl_percent') else None
                    })
            
            order_type_str = "虚拟订单" if order.get('is_virtual') else "实盘订单"
            logger.debug(f"写入{order_type_str}: {order['symbol']} {order['side']}")
            
        except Exception as e:
            logger.error(f"写入订单数据失败: {e}")
            raise
    
    async def query_kline_data(
        self, 
        symbol: str, 
        interval: str, 
        start_time: datetime, 
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """查询K线数据"""
        try:
            if end_time is None:
                import pytz
                end_time = datetime.now(pytz.UTC)
            
            # 将 datetime 转换为毫秒时间戳（用于查询）
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            async with self.SessionLocal() as session:
                # ✅ time 已经是 BIGINT 类型，直接查询
                query = """
                    SELECT 
                        time as timestamp,
                        open, high, low, close, volume,
                        close_time,
                        quote_volume, trades,
                        taker_buy_base_volume, taker_buy_quote_volume
                    FROM klines
                    WHERE symbol = :symbol
                      AND interval = :interval
                      AND time >= :start_time
                      AND time <= :end_time
                    ORDER BY time DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                result = await session.execute(text(query), {
                    'symbol': symbol,
                    'interval': interval,
                    'start_time': start_ts,
                    'end_time': end_ts
                })
                
                # 🔥 关键修复：在session内部完成fetchall()
                rows = result.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                # 转换为 DataFrame
                df = pd.DataFrame(rows, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades',
                    'taker_buy_base_volume', 'taker_buy_quote_volume'
                ])
                
                # 转换时间戳为 datetime（现在是BIGINT，可以正常转换）
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 按时间升序排列（与 InfluxDB 保持一致）
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.debug(f"查询到{len(df)}条K线数据: {symbol} {interval}")
                return df
            
        except Exception as e:
            logger.error(f"查询K线数据失败: {e}")
            return pd.DataFrame()
    
    async def query_signals(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: Optional[datetime] = None,
        limit: Optional[int] = 100
    ) -> List[Dict[str, Any]]:
        """查询交易信号"""
        try:
            if end_time is None:
                import pytz
                end_time = datetime.now(pytz.UTC)
            
            async with self.SessionLocal() as session:
                query = """
                    SELECT 
                        symbol, signal_type, confidence, entry_price,
                        stop_loss, take_profit, position_size,
                        timestamp, predictions
                    FROM trading_signals
                    WHERE symbol = :symbol
                      AND timestamp >= :start_time
                      AND timestamp <= :end_time
                    ORDER BY timestamp DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                result = await session.execute(text(query), {
                    'symbol': symbol,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                # 🔥 关键修复：在session内部完成fetchall()和数据处理
                rows = result.fetchall()
                
                signals = []
                for row in rows:
                    signal = {
                        'symbol': row[0],
                        'signal_type': row[1],
                        'confidence': float(row[2]),
                        'entry_price': float(row[3]),
                        'stop_loss': float(row[4]) if row[4] else 0,
                        'take_profit': float(row[5]) if row[5] else 0,
                        'position_size': float(row[6]) if row[6] else 0,
                        'timestamp': row[7],
                        'predictions': json.loads(row[8]) if row[8] else None
                    }
                    signals.append(signal)
                
                logger.debug(f"查询到{len(signals)}条交易信号: {symbol}")
                return signals
            
        except Exception as e:
            logger.error(f"查询交易信号失败: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 30):
        """清理旧数据
        
        Args:
            days: 保留天数，0表示清空所有数据
        """
        try:
            async with self.SessionLocal() as session:
                async with session.begin():
                    if days == 0:
                        # 清空所有K线数据
                        logger.info("🧹 开始清空所有K线数据...")
                        await session.execute(text("TRUNCATE TABLE klines"))
                        logger.info("✅ 已清空所有K线数据")
                    else:
                        # 清理指定天数前的数据
                        logger.info(f"🧹 开始清理{days}天前的旧数据（保留最近{days}天）...")
                        import pytz
                        cutoff_time = datetime.now(pytz.UTC) - timedelta(days=days)
                        cutoff_ts = int(cutoff_time.timestamp() * 1000)  # ✅ 转为毫秒时间戳
                        
                        result = await session.execute(
                            text("DELETE FROM klines WHERE time < :cutoff"),
                            {'cutoff': cutoff_ts}
                        )
                        
                        deleted_count = result.rowcount
                        logger.info(f"✅ 已清理{days}天前的旧数据（删除{deleted_count}条）")
                        
        except Exception as e:
            logger.warning(f"⚠️ 数据清理失败（不影响系统运行）: {e}")
    
    async def create_virtual_position(self, position: Dict[str, Any]):
        """创建虚拟仓位"""
        try:
            async with self.SessionLocal() as session:
                async with session.begin():
                    stmt = text("""
                        INSERT INTO virtual_positions (
                            symbol, side, entry_price, quantity, entry_time,
                            stop_loss, take_profit, status, signal_id
                        ) VALUES (
                            :symbol, :side, :entry_price, :quantity, :entry_time,
                            :stop_loss, :take_profit, :status, :signal_id
                        )
                        RETURNING id
                    """)
                    
                    # 处理入场时间（统一使用UTC）
                    import pytz
                    entry_time = position.get('entry_time', datetime.now(pytz.UTC))
                    if isinstance(entry_time, (int, float)):
                        entry_time = datetime.fromtimestamp(entry_time / 1000, tz=pytz.UTC)
                    
                    result = await session.execute(stmt, {
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'entry_price': float(position['entry_price']),
                        'quantity': float(position['quantity']),
                        'entry_time': entry_time,
                        'stop_loss': float(position.get('stop_loss', 0)),
                        'take_profit': float(position.get('take_profit', 0)),
                        'status': 'OPEN',
                        'signal_id': position.get('signal_id')
                    })
                    
                    position_id = result.scalar()
                    logger.info(f"创建虚拟仓位: {position['symbol']} {position['side']} @{position['entry_price']}")
                    return position_id
            
        except Exception as e:
            logger.error(f"创建虚拟仓位失败: {e}")
            raise
    
    async def close_virtual_position(self, position_id: int, exit_price: float, exit_time: datetime = None):
        """平掉虚拟仓位"""
        try:
            if exit_time is None:
                import pytz
                exit_time = datetime.now(pytz.UTC)
            
            async with self.SessionLocal() as session:
                async with session.begin():
                    # 查询仓位信息
                    query_stmt = text("""
                        SELECT id, symbol, side, entry_price, quantity
                        FROM virtual_positions
                        WHERE id = :position_id AND status = 'OPEN'
                    """)
                    
                    result = await session.execute(query_stmt, {'position_id': position_id})
                    row = result.first()
                    
                    if not row:
                        logger.warning(f"虚拟仓位不存在或已关闭: {position_id}")
                        return
                    
                    # 计算盈亏
                    entry_price = float(row[3])
                    quantity = float(row[4])
                    side = row[2]
                    
                    if side == 'LONG':
                        pnl = (exit_price - entry_price) * quantity
                    else:  # SHORT
                        pnl = (entry_price - exit_price) * quantity
                    
                    pnl_percent = (pnl / (entry_price * quantity)) * 100
                    
                    # 更新仓位
                    update_stmt = text("""
                        UPDATE virtual_positions
                        SET exit_price = :exit_price,
                            exit_time = :exit_time,
                            pnl = :pnl,
                            pnl_percent = :pnl_percent,
                            status = 'CLOSED',
                            updated_at = NOW()
                        WHERE id = :position_id
                    """)
                    
                    await session.execute(update_stmt, {
                        'position_id': position_id,
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent
                    })
                    
                    logger.info(f"平掉虚拟仓位 #{position_id}: {row[1]} PnL={pnl:.2f} ({pnl_percent:+.2f}%)")
                    
        except Exception as e:
            logger.error(f"平掉虚拟仓位失败: {e}")
            raise
    
    async def get_open_virtual_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取所有未平仓的虚拟仓位"""
        try:
            async with self.SessionLocal() as session:
                if symbol:
                    query = text("""
                        SELECT id, symbol, side, entry_price, quantity, entry_time,
                               stop_loss, take_profit, signal_id
                        FROM virtual_positions
                        WHERE status = 'OPEN' AND symbol = :symbol
                        ORDER BY entry_time DESC
                    """)
                    result = await session.execute(query, {'symbol': symbol})
                else:
                    query = text("""
                        SELECT id, symbol, side, entry_price, quantity, entry_time,
                               stop_loss, take_profit, signal_id
                        FROM virtual_positions
                        WHERE status = 'OPEN'
                        ORDER BY entry_time DESC
                    """)
                    result = await session.execute(query)
                
                rows = result.fetchall()
                positions = []
                for row in rows:
                    positions.append({
                        'id': row[0],
                        'symbol': row[1],
                        'side': row[2],
                        'entry_price': float(row[3]),
                        'quantity': float(row[4]),
                        'entry_time': row[5],
                        'stop_loss': float(row[6]) if row[6] else 0,
                        'take_profit': float(row[7]) if row[7] else 0,
                        'signal_id': row[8]
                    })
                
                return positions
            
        except Exception as e:
            logger.error(f"获取虚拟仓位失败: {e}")
            return []
    
    async def close(self):
        """关闭连接"""
        if self.engine:
            await self.engine.dispose()
            logger.info("PostgreSQL连接已关闭")


class RedisManager:
    """Redis管理器（保持不变）"""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
    
    async def connect(self):
        """连接到Redis"""
        try:
            self.client = redis.from_url(
                settings.REDIS_URL,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            
            # 测试连接
            await self.client.ping()
            logger.info("Redis连接成功")
            
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    async def set_cache(self, key: str, value: str, expire: Optional[int] = 3600):
        """设置缓存（支持永久缓存）"""
        try:
            if expire is None:
                # 永久缓存（不设置过期时间）
                await self.client.set(key, value)
            else:
                # 带过期时间的缓存
                await self.client.setex(key, expire, value)
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
    
    async def get_cache(self, key: str) -> Optional[str]:
        """获取缓存"""
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    async def delete_cache(self, key: str):
        """删除缓存"""
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
    
    async def set_hash(self, name: str, mapping: Dict[str, Any]):
        """设置哈希"""
        try:
            # 转换所有值为字符串（Redis 只支持 string）
            str_mapping = {}
            for key, value in mapping.items():
                if isinstance(value, (bool, int, float)):
                    str_mapping[key] = str(value)
                elif isinstance(value, datetime):
                    str_mapping[key] = value.isoformat()
                else:
                    str_mapping[key] = str(value)
            
            await self.client.hset(name, mapping=str_mapping)
        except Exception as e:
            logger.error(f"设置哈希失败: {e}")
    
    async def get_hash(self, name: str) -> Dict[str, str]:
        """获取哈希"""
        try:
            return await self.client.hgetall(name)
        except Exception as e:
            logger.error(f"获取哈希失败: {e}")
            return {}
    
    async def close(self):
        """关闭连接"""
        if self.client:
            await self.client.close()
            logger.info("Redis连接已关闭")


# 全局数据库管理器实例
postgresql_manager = PostgreSQLManager()
redis_manager = RedisManager()


async def init_database():
    """初始化数据库连接"""
    await postgresql_manager.connect()
    await redis_manager.connect()


async def cleanup_database():
    """清理数据库（清空所有数据，确保启动后数据完全是最新的）"""
    # days=0 表示清空所有K线数据
    await postgresql_manager.cleanup_old_data(days=0)
    logger.info("数据库清理完成（已清空所有数据）")


async def close_database():
    """关闭数据库连接"""
    await postgresql_manager.close()
    await redis_manager.close()
