-- =====================================================
-- PostgreSQL + TimescaleDB 初始化脚本
-- 交易系统数据库表结构
-- =====================================================

-- 1. 启用 TimescaleDB 扩展
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 2. 创建 K线数据表（时序优化）
CREATE TABLE IF NOT EXISTS klines (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(30, 8) NOT NULL,
    close_time TIMESTAMPTZ NOT NULL,
    quote_volume NUMERIC(30, 8),
    trades INTEGER DEFAULT 0,
    taker_buy_base_volume NUMERIC(30, 8) DEFAULT 0,
    taker_buy_quote_volume NUMERIC(30, 8) DEFAULT 0,
    PRIMARY KEY (symbol, interval, time)
);

-- 3. 转换为 hypertable（TimescaleDB 时序优化）
SELECT create_hypertable('klines', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- 4. 创建索引（提高查询性能）
CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_time 
    ON klines (symbol, interval, time DESC);

CREATE INDEX IF NOT EXISTS idx_klines_close_time 
    ON klines (close_time DESC);

-- 5. 添加压缩策略（7天前的数据自动压缩，节省存储空间）
SELECT add_compression_policy('klines', INTERVAL '7 days', if_not_exists => TRUE);

-- 6. 添加数据保留策略（90天，与模型训练数据量一致）
SELECT add_retention_policy('klines', INTERVAL '90 days', if_not_exists => TRUE);

-- 7. 创建交易信号表
CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- LONG, SHORT, CLOSE
    confidence NUMERIC(5, 4) NOT NULL,
    entry_price NUMERIC(20, 8) NOT NULL,
    stop_loss NUMERIC(20, 8) DEFAULT 0,
    take_profit NUMERIC(20, 8) DEFAULT 0,
    position_size NUMERIC(20, 8) DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL,
    predictions JSONB,  -- 多时间框架预测详情
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 8. 信号表索引
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
    ON trading_signals (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_signals_processed 
    ON trading_signals (processed, created_at);

CREATE INDEX IF NOT EXISTS idx_signals_type 
    ON trading_signals (signal_type, timestamp DESC);

-- 9. 创建订单表
CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT,  -- Binance 订单ID
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- BUY, SELL
    order_type TEXT NOT NULL,  -- MARKET, LIMIT, STOP
    status TEXT NOT NULL,  -- NEW, FILLED, PARTIALLY_FILLED, CANCELED
    quantity NUMERIC(20, 8) NOT NULL,
    price NUMERIC(20, 8) DEFAULT 0,
    filled_quantity NUMERIC(20, 8) DEFAULT 0,
    commission NUMERIC(20, 8) DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 10. 订单表索引
CREATE INDEX IF NOT EXISTS idx_orders_symbol_time 
    ON orders (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_orders_status 
    ON orders (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orders_binance_id 
    ON orders (order_id) WHERE order_id IS NOT NULL;

-- 11. 创建持仓表（可选）
CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    position_amt NUMERIC(20, 8) NOT NULL,
    entry_price NUMERIC(20, 8),
    mark_price NUMERIC(20, 8),
    unrealized_pnl NUMERIC(20, 8),
    leverage INTEGER,
    margin_type TEXT,  -- ISOLATED, CROSS
    position_side TEXT,  -- BOTH, LONG, SHORT
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 12. 创建系统配置表（可选）
CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- 初始化完成提示
-- =====================================================
DO $$ 
BEGIN
    RAISE NOTICE '✅ TimescaleDB 数据库初始化完成！';
    RAISE NOTICE '   - klines 表已创建（hypertable，自动压缩和保留）';
    RAISE NOTICE '   - trading_signals 表已创建';
    RAISE NOTICE '   - orders 表已创建';
    RAISE NOTICE '   - positions 表已创建';
    RAISE NOTICE '   - 数据保留策略：90天';
    RAISE NOTICE '   - 压缩策略：7天前数据自动压缩';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 可以启动交易系统了！';
END $$;


