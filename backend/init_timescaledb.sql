-- =====================================================
-- PostgreSQL + TimescaleDB 初始化脚本
-- QuantAI-ETH 交易系统数据库表结构
-- =====================================================
-- 
-- 说明：
-- 1. 本脚本用于手动初始化数据库（可选）
-- 2. 系统启动时会自动创建表结构（database.py）
-- 3. 如果表已存在，不会重复创建
-- 
-- 执行方式：
-- psql -U postgres -d trading-data -f init_timescaledb.sql
-- =====================================================

-- =====================================================
-- 第一部分：扩展和基础设置
-- =====================================================

-- 1. 启用 TimescaleDB 扩展
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =====================================================
-- 第二部分：K线数据表（klines）
-- =====================================================

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
);

-- 表注释
COMMENT ON TABLE klines IS 'K线数据表：存储历史K线数据，用于模型训练和特征工程。使用BIGINT存储毫秒时间戳（不使用hypertable）';

-- 字段注释
COMMENT ON COLUMN klines.time IS '开盘时间（毫秒时间戳）';
COMMENT ON COLUMN klines.symbol IS '交易对（如：ETH/USDT）';
COMMENT ON COLUMN klines.interval IS '时间周期（3m, 5m, 15m等）';
COMMENT ON COLUMN klines.open IS '开盘价';
COMMENT ON COLUMN klines.high IS '最高价';
COMMENT ON COLUMN klines.low IS '最低价';
COMMENT ON COLUMN klines.close IS '收盘价';
COMMENT ON COLUMN klines.volume IS '成交量（基础货币）';
COMMENT ON COLUMN klines.close_time IS '收盘时间（毫秒时间戳）';
COMMENT ON COLUMN klines.quote_volume IS '成交额（计价货币）';
COMMENT ON COLUMN klines.trades IS '成交笔数';
COMMENT ON COLUMN klines.taker_buy_base_volume IS '主动买入成交量';
COMMENT ON COLUMN klines.taker_buy_quote_volume IS '主动买入成交额';

-- 索引
CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_time 
    ON klines (symbol, interval, time DESC);

-- =====================================================
-- 第三部分：交易信号表（trading_signals）
-- =====================================================

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
);

-- 表注释
COMMENT ON TABLE trading_signals IS '交易信号表：存储生成的交易信号，包含多时间框架预测结果（JSONB格式）';

-- 字段注释
COMMENT ON COLUMN trading_signals.id IS '主键ID';
COMMENT ON COLUMN trading_signals.symbol IS '交易对';
COMMENT ON COLUMN trading_signals.signal_type IS '信号类型：LONG, SHORT, HOLD, CLOSE';
COMMENT ON COLUMN trading_signals.confidence IS '置信度（0.0000-1.0000）';
COMMENT ON COLUMN trading_signals.entry_price IS '入场价格';
COMMENT ON COLUMN trading_signals.stop_loss IS '止损价格';
COMMENT ON COLUMN trading_signals.take_profit IS '止盈价格';
COMMENT ON COLUMN trading_signals.position_size IS '仓位大小（USDT价值）';
COMMENT ON COLUMN trading_signals.timestamp IS '信号生成时间';
COMMENT ON COLUMN trading_signals.predictions IS '多时间框架预测详情（3m/5m/15m），JSONB格式';
COMMENT ON COLUMN trading_signals.created_at IS '记录创建时间';

-- 索引
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
    ON trading_signals (symbol, timestamp DESC);

-- =====================================================
-- 第四部分：订单表（orders）
-- =====================================================

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
);

-- 表注释
COMMENT ON TABLE orders IS '订单表：存储所有订单（包括虚拟订单和实盘订单），支持虚拟交易模式（SIGNAL_ONLY）和实盘交易模式（AUTO）';

-- 字段注释
COMMENT ON COLUMN orders.id IS '主键ID';
COMMENT ON COLUMN orders.order_id IS '交易所订单ID（实盘订单）';
COMMENT ON COLUMN orders.symbol IS '交易对';
COMMENT ON COLUMN orders.side IS '订单方向：BUY, SELL';
COMMENT ON COLUMN orders.order_type IS '订单类型：MARKET, LIMIT, STOP_MARKET等';
COMMENT ON COLUMN orders.status IS '订单状态：NEW, FILLED, PARTIALLY_FILLED, CANCELED等';
COMMENT ON COLUMN orders.quantity IS '订单数量（虚拟订单：USDT价值；实盘订单：币的数量）';
COMMENT ON COLUMN orders.price IS '订单价格（限价单）';
COMMENT ON COLUMN orders.filled_quantity IS '已成交数量（虚拟订单：USDT价值；实盘订单：币的数量）';
COMMENT ON COLUMN orders.commission IS '手续费';
COMMENT ON COLUMN orders.timestamp IS '订单时间';
COMMENT ON COLUMN orders.created_at IS '记录创建时间';
COMMENT ON COLUMN orders.is_virtual IS '是否为虚拟订单（SIGNAL_ONLY模式）';
COMMENT ON COLUMN orders.signal_id IS '关联的信号ID';
COMMENT ON COLUMN orders.position_id IS '关联的虚拟仓位ID（用于关联同一仓位的开仓和平仓订单）';
COMMENT ON COLUMN orders.order_action IS '订单动作：OPEN（开仓）, CLOSE（平仓）';
COMMENT ON COLUMN orders.entry_price IS '开仓价格（虚拟订单）';
COMMENT ON COLUMN orders.exit_price IS '平仓价格（虚拟订单）';
COMMENT ON COLUMN orders.pnl IS '盈亏金额（虚拟订单）';
COMMENT ON COLUMN orders.pnl_percent IS '盈亏百分比（虚拟订单）';

-- 索引
CREATE INDEX IF NOT EXISTS idx_orders_symbol_time 
    ON orders (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_orders_status 
    ON orders (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orders_order_id 
    ON orders (order_id) WHERE order_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_orders_position_id 
    ON orders (position_id);

CREATE INDEX IF NOT EXISTS idx_orders_order_action 
    ON orders (order_action);

-- =====================================================
-- 第五部分：虚拟仓位表（virtual_positions）
-- =====================================================

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
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 表注释
COMMENT ON TABLE virtual_positions IS '虚拟仓位表：存储SIGNAL_ONLY模式下的虚拟仓位，支持止损止盈监控、盈亏计算';

-- 字段注释
COMMENT ON COLUMN virtual_positions.id IS '主键ID';
COMMENT ON COLUMN virtual_positions.symbol IS '交易对';
COMMENT ON COLUMN virtual_positions.side IS '仓位方向：LONG, SHORT';
COMMENT ON COLUMN virtual_positions.entry_price IS '开仓价格';
COMMENT ON COLUMN virtual_positions.quantity IS '仓位数量（USDT价值）';
COMMENT ON COLUMN virtual_positions.entry_time IS '开仓时间';
COMMENT ON COLUMN virtual_positions.exit_price IS '平仓价格';
COMMENT ON COLUMN virtual_positions.exit_time IS '平仓时间';
COMMENT ON COLUMN virtual_positions.stop_loss IS '止损价格';
COMMENT ON COLUMN virtual_positions.take_profit IS '止盈价格';
COMMENT ON COLUMN virtual_positions.pnl IS '盈亏金额';
COMMENT ON COLUMN virtual_positions.pnl_percent IS '盈亏百分比';
COMMENT ON COLUMN virtual_positions.status IS '仓位状态：OPEN, CLOSED';
COMMENT ON COLUMN virtual_positions.signal_id IS '关联的信号ID';
COMMENT ON COLUMN virtual_positions.created_at IS '记录创建时间';
COMMENT ON COLUMN virtual_positions.updated_at IS '记录更新时间';

-- 索引
CREATE INDEX IF NOT EXISTS idx_virtual_positions_symbol_status 
    ON virtual_positions (symbol, status);

-- =====================================================
-- 第六部分：初始化完成提示
-- =====================================================

DO $$ 
BEGIN
    RAISE NOTICE '====================================================';
    RAISE NOTICE '✅ QuantAI-ETH 数据库初始化完成！';
    RAISE NOTICE '====================================================';
    RAISE NOTICE '';
    RAISE NOTICE '📊 已创建的表：';
    RAISE NOTICE '   1. klines - K线数据表（BIGINT时间戳，不使用hypertable）';
    RAISE NOTICE '   2. trading_signals - 交易信号表';
    RAISE NOTICE '   3. orders - 订单表（支持虚拟和实盘订单）';
    RAISE NOTICE '   4. virtual_positions - 虚拟仓位表（SIGNAL_ONLY模式）';
    RAISE NOTICE '';
    RAISE NOTICE '📋 表结构说明：';
    RAISE NOTICE '   - klines: 存储历史K线数据，用于模型训练和特征工程';
    RAISE NOTICE '   - trading_signals: 存储生成的交易信号，包含多时间框架预测';
    RAISE NOTICE '   - orders: 存储所有订单，支持虚拟交易和实盘交易';
    RAISE NOTICE '   - virtual_positions: 存储虚拟仓位，支持止损止盈监控';
    RAISE NOTICE '';
    RAISE NOTICE '💡 提示：所有表和字段都已添加注释，可在数据库工具中查看';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 可以启动交易系统了！';
    RAISE NOTICE '====================================================';
END $$;
