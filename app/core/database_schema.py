"""
PostgreSQL 数据库表结构定义
从 database.py 中抽离，保持职责单一
"""
# StdLib
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)


async def init_database_schema(conn: AsyncConnection) -> None:
    """
    初始化数据库表结构（如果不存在）
    
    Args:
        conn: SQLAlchemy 异步连接对象
    """
    try:
        # 1. 启用 TimescaleDB 扩展
        await conn.execute(text("""
            CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE
        """))
        
        # 2. 创建 klines 表
        await _create_klines_table(conn)
        
        # 3. 创建交易信号表
        await _create_trading_signals_table(conn)
        
        # 4. 创建虚拟仓位表
        await _create_virtual_positions_table(conn)
        
        # 5. 创建订单表
        await _create_orders_table(conn)
        
        # 6. 创建回测结果表
        await _create_backtest_tables(conn)
        
        logger.info("数据库表结构初始化完成")
        
    except Exception as e:
        logger.error(f"初始化数据库结构失败: {e}")
        logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
        raise


async def _create_klines_table(conn: AsyncConnection) -> None:
    """创建 K线数据表"""
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
    
    # 创建索引
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_time 
            ON klines (symbol, interval, time DESC)
    """))
    
    # 添加表和字段注释
    await conn.execute(text("COMMENT ON TABLE klines IS 'K线数据表：存储历史K线数据，用于模型训练和特征工程。使用BIGINT存储毫秒时间戳（不使用hypertable）'"))
    await conn.execute(text("COMMENT ON COLUMN klines.time IS '开盘时间（毫秒时间戳）'"))
    await conn.execute(text("COMMENT ON COLUMN klines.symbol IS '交易对（如：BTC/USDT）'"))
    await conn.execute(text("COMMENT ON COLUMN klines.interval IS '时间周期（3m, 5m, 15m等）'"))
    await conn.execute(text("COMMENT ON COLUMN klines.open IS '开盘价'"))
    await conn.execute(text("COMMENT ON COLUMN klines.high IS '最高价'"))
    await conn.execute(text("COMMENT ON COLUMN klines.low IS '最低价'"))
    await conn.execute(text("COMMENT ON COLUMN klines.close IS '收盘价'"))
    await conn.execute(text("COMMENT ON COLUMN klines.volume IS '成交量（基础货币）'"))
    await conn.execute(text("COMMENT ON COLUMN klines.close_time IS '收盘时间（毫秒时间戳）'"))
    await conn.execute(text("COMMENT ON COLUMN klines.quote_volume IS '成交额（计价货币）'"))
    await conn.execute(text("COMMENT ON COLUMN klines.trades IS '成交笔数'"))
    await conn.execute(text("COMMENT ON COLUMN klines.taker_buy_base_volume IS '主动买入成交量'"))
    await conn.execute(text("COMMENT ON COLUMN klines.taker_buy_quote_volume IS '主动买入成交额'"))
    
    logger.debug("跳过 hypertable 创建（time列为BIGINT类型）")


async def _create_trading_signals_table(conn: AsyncConnection) -> None:
    """创建交易信号表"""
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
            created_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'Asia/Shanghai')::TIMESTAMPTZ
        )
    """))
    
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
            ON trading_signals (symbol, timestamp DESC)
    """))
    
    # 添加表和字段注释
    await conn.execute(text("COMMENT ON TABLE trading_signals IS '交易信号表：存储生成的交易信号，包含多时间框架预测结果（JSONB格式）'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.id IS '主键ID'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.symbol IS '交易对'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.signal_type IS '信号类型：LONG, SHORT, HOLD, CLOSE'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.confidence IS '置信度（0.0000-1.0000）'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.entry_price IS '入场价格'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.stop_loss IS '止损价格'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.take_profit IS '止盈价格'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.position_size IS '仓位大小（USDT价值）'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.timestamp IS '信号生成时间'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.predictions IS '多时间框架预测详情（3m/5m/15m），JSONB格式'"))
    await conn.execute(text("COMMENT ON COLUMN trading_signals.created_at IS '记录创建时间'"))


async def _create_virtual_positions_table(conn: AsyncConnection) -> None:
    """创建虚拟仓位表"""
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
            created_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'Asia/Shanghai')::TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'Asia/Shanghai')::TIMESTAMPTZ
        )
    """))
    
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_virtual_positions_symbol_status 
            ON virtual_positions (symbol, status)
    """))
    
    # 添加表和字段注释
    await conn.execute(text("COMMENT ON TABLE virtual_positions IS '虚拟仓位表：存储SIGNAL_ONLY模式下的虚拟仓位，支持止损止盈监控、盈亏计算'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.id IS '主键ID'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.symbol IS '交易对'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.side IS '仓位方向：LONG, SHORT'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.entry_price IS '开仓价格'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.quantity IS '仓位数量（USDT价值）'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.entry_time IS '开仓时间'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.exit_price IS '平仓价格'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.exit_time IS '平仓时间'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.stop_loss IS '止损价格'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.take_profit IS '止盈价格'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.pnl IS '盈亏金额'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.pnl_percent IS '盈亏百分比'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.status IS '仓位状态：OPEN, CLOSED'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.signal_id IS '关联的信号ID'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.created_at IS '记录创建时间'"))
    await conn.execute(text("COMMENT ON COLUMN virtual_positions.updated_at IS '记录更新时间'"))


async def _create_orders_table(conn: AsyncConnection) -> None:
    """创建订单表"""
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
            created_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'Asia/Shanghai')::TIMESTAMPTZ,
            is_virtual BOOLEAN DEFAULT FALSE,
            signal_id TEXT,
            position_id BIGINT,
            order_action TEXT,
            entry_price NUMERIC(20, 8),
            exit_price NUMERIC(20, 8),
            pnl NUMERIC(20, 8),
            pnl_percent NUMERIC(10, 4),
            FOREIGN KEY (position_id) REFERENCES virtual_positions(id) ON DELETE SET NULL
        )
    """))
    
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_orders_symbol_time 
            ON orders (symbol, timestamp DESC)
    """))
    
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_orders_position_id 
            ON orders (position_id)
    """))
    
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_orders_order_action 
            ON orders (order_action)
    """))
    
    # 添加表和字段注释
    await conn.execute(text("COMMENT ON TABLE orders IS '订单表：存储所有订单（包括虚拟订单和实盘订单），支持虚拟交易模式（SIGNAL_ONLY）和实盘交易模式（AUTO）。注意：虚拟订单只在平仓时创建（order_action=CLOSE），开仓时只创建virtual_positions记录'"))
    await conn.execute(text("COMMENT ON COLUMN orders.id IS '主键ID'"))
    await conn.execute(text("COMMENT ON COLUMN orders.order_id IS '交易所订单ID（实盘订单）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.symbol IS '交易对'"))
    await conn.execute(text("COMMENT ON COLUMN orders.side IS '订单方向：BUY, SELL'"))
    await conn.execute(text("COMMENT ON COLUMN orders.order_type IS '订单类型：MARKET, LIMIT, STOP_MARKET等'"))
    await conn.execute(text("COMMENT ON COLUMN orders.status IS '订单状态：NEW, FILLED, PARTIALLY_FILLED, CANCELED等'"))
    await conn.execute(text("COMMENT ON COLUMN orders.quantity IS '订单数量（虚拟订单：USDT价值；实盘订单：币的数量）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.price IS '订单价格（限价单）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.filled_quantity IS '已成交数量（虚拟订单：USDT价值；实盘订单：币的数量）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.commission IS '手续费'"))
    await conn.execute(text("COMMENT ON COLUMN orders.timestamp IS '订单时间'"))
    await conn.execute(text("COMMENT ON COLUMN orders.created_at IS '记录创建时间'"))
    await conn.execute(text("COMMENT ON COLUMN orders.is_virtual IS '是否为虚拟订单（SIGNAL_ONLY模式）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.signal_id IS '关联的信号ID'"))
    await conn.execute(text("COMMENT ON COLUMN orders.position_id IS '关联的虚拟仓位ID（用于关联同一仓位的开仓和平仓订单）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.order_action IS '订单动作：OPEN（开仓，仅实盘订单）, CLOSE（平仓，虚拟订单只在平仓时创建）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.entry_price IS '开仓价格（虚拟订单）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.exit_price IS '平仓价格（虚拟订单）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.pnl IS '盈亏金额（虚拟订单）'"))
    await conn.execute(text("COMMENT ON COLUMN orders.pnl_percent IS '盈亏百分比（虚拟订单）'"))


async def _create_backtest_tables(conn: AsyncConnection) -> None:
    """创建回测相关表"""
    # 创建回测结果表
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            days INTEGER NOT NULL,
            initial_balance NUMERIC(20, 8) NOT NULL,
            final_balance NUMERIC(20, 8) NOT NULL,
            total_return NUMERIC(20, 10) NOT NULL,
            win_rate NUMERIC(10, 6) NOT NULL,
            profit_factor NUMERIC(20, 8) NOT NULL,
            max_drawdown NUMERIC(10, 6) NOT NULL,
            total_trades INTEGER NOT NULL,
            avg_trade_return NUMERIC(20, 8) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'Asia/Shanghai')::TIMESTAMPTZ
        )
    """))

    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbol_time
            ON backtest_runs (symbol, created_at DESC)
    """))

    await conn.execute(text("COMMENT ON TABLE backtest_runs IS '回测结果表：存储每次回测的汇总指标'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.id IS '主键ID'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.symbol IS '交易对'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.days IS '回测天数'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.initial_balance IS '初始资金'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.final_balance IS '最终资金'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.total_return IS '总收益率'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.win_rate IS '胜率'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.profit_factor IS '盈亏因子'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.max_drawdown IS '最大回撤'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.total_trades IS '总交易次数'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.avg_trade_return IS '平均单笔回报'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_runs.created_at IS '记录创建时间'"))

    # 🔧 迁移：修改 total_return 字段精度（从 NUMERIC(10,6) 到 NUMERIC(20,10)）
    try:
        await conn.execute(text("""
            ALTER TABLE backtest_runs 
            ALTER COLUMN total_return TYPE NUMERIC(20, 10)
        """))
        logger.info("✅ backtest_runs.total_return 字段精度已更新: NUMERIC(10,6) → NUMERIC(20,10)")
    except Exception as e:
        # 如果字段已经是正确类型，忽略错误
        if "cannot be cast automatically" not in str(e):
            logger.debug(f"⚠️ total_return 字段迁移跳过（可能已更新）: {e}")

    # 创建回测交易明细表（🎯 检查并迁移旧表结构）
    check_table_result = await conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'backtest_trades' 
        AND column_name IN ('position_value', 'balance_after')
    """))
    existing_columns = [row[0] for row in check_table_result.fetchall()]
    
    # 如果表存在但缺少新字段，删除旧表（CASCADE 会同时删除依赖的外键）
    if len(existing_columns) < 2:
        logger.warning("⚠️ 检测到旧版 backtest_trades 表结构，正在迁移...")
        await conn.execute(text("DROP TABLE IF EXISTS backtest_trades CASCADE"))
        logger.info("✅ 旧表已删除，准备创建新表结构")
    
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id BIGSERIAL PRIMARY KEY,
            run_id BIGINT NOT NULL,
            entry_time TIMESTAMPTZ NOT NULL,
            exit_time TIMESTAMPTZ NOT NULL,
            side TEXT NOT NULL,
            entry_price NUMERIC(20, 8) NOT NULL,
            exit_price NUMERIC(20, 8) NOT NULL,
            position_value NUMERIC(20, 8) NOT NULL,
            open_fee NUMERIC(20, 8) NOT NULL DEFAULT 0,
            close_fee NUMERIC(20, 8) NOT NULL DEFAULT 0,
            total_fee NUMERIC(20, 8) NOT NULL DEFAULT 0,
            pnl NUMERIC(20, 8) NOT NULL,
            pnl_percent NUMERIC(10, 4) NOT NULL,
            balance_after NUMERIC(20, 8) NOT NULL,
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT (NOW() AT TIME ZONE 'Asia/Shanghai')::TIMESTAMPTZ,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
        )
    """))

    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_id
            ON backtest_trades (run_id)
    """))

    await conn.execute(text("COMMENT ON TABLE backtest_trades IS '回测交易明细表：存储每次回测的交易记录'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.id IS '主键ID'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.run_id IS '关联回测ID'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.entry_time IS '开仓时间'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.exit_time IS '平仓时间'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.side IS '方向：LONG/SHORT'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.entry_price IS '开仓价格'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.exit_price IS '平仓价格'"))
    # 🔧 迁移：添加手续费字段（如果不存在）
    try:
        await conn.execute(text("""
            ALTER TABLE backtest_trades 
            ADD COLUMN IF NOT EXISTS open_fee NUMERIC(20, 8) NOT NULL DEFAULT 0
        """))
        await conn.execute(text("""
            ALTER TABLE backtest_trades 
            ADD COLUMN IF NOT EXISTS close_fee NUMERIC(20, 8) NOT NULL DEFAULT 0
        """))
        await conn.execute(text("""
            ALTER TABLE backtest_trades 
            ADD COLUMN IF NOT EXISTS total_fee NUMERIC(20, 8) NOT NULL DEFAULT 0
        """))
        logger.info("✅ backtest_trades 手续费字段已添加: open_fee, close_fee, total_fee")
    except Exception as e:
        logger.debug(f"⚠️ 手续费字段添加跳过（可能已存在）: {e}")
    
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.position_value IS '开仓金额（仓位价值 = 余额 × 杠杆 × 仓位比例）'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.open_fee IS '开仓手续费'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.close_fee IS '平仓手续费'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.total_fee IS '总手续费（开仓+平仓）'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.pnl IS '盈亏金额（已扣除手续费）'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.pnl_percent IS '盈亏百分比'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.balance_after IS '平仓后余额（用于验证复利增长路径）'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.reason IS '平仓原因'"))
    await conn.execute(text("COMMENT ON COLUMN backtest_trades.created_at IS '记录创建时间'"))
    
    # 🔧 迁移：修改 pnl_percent 字段精度（从 NUMERIC(20,8) 到 NUMERIC(10,4)）
    try:
        await conn.execute(text("""
            ALTER TABLE backtest_trades 
            ALTER COLUMN pnl_percent TYPE NUMERIC(10, 4)
        """))
        logger.info("✅ backtest_trades.pnl_percent 字段精度已更新: NUMERIC(20,8) → NUMERIC(10,4)")
    except Exception as e:
        # 如果字段已经是正确类型，忽略错误
        if "cannot be cast automatically" not in str(e):
            logger.debug(f"⚠️ pnl_percent 字段迁移跳过（可能已更新）: {e}")
