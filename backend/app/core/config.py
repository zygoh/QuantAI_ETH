"""
系统配置管理
"""
import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """系统配置"""
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Binance API配置
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET_KEY: str = ""
    BINANCE_TESTNET: bool = True  # 使用生产环境
    
    # 代理配置（可选）
    USE_PROXY: bool = True  # 是否使用代理（REST API）
    USE_PROXY_WS: bool = True  # 是否为WebSocket使用代理（SOCKS5更稳定）
    PROXY_HOST: str = "127.0.0.1"  # 代理主机
    PROXY_PORT: int = 10808  # 代理端口
    PROXY_TYPE: str = "socks5"  # 代理类型：http, https, socks5（WebSocket推荐socks5）
    
    # 交易配置
    SYMBOL: str = "ETHUSDT"
    LEVERAGE: int = 50  # 50x杠杆（1.5%止损 × 50x = 75%单次风险，风险回报1:2.67）
    CONFIDENCE_THRESHOLD: float = 0.35  # 降低到0.35以增加信号数量（81%准确率下合理阈值）
    
    # 交易模式配置
    TRADING_MODE: str = "SIGNAL_ONLY"  # 默认交易模式：SIGNAL_ONLY（信号模式/虚拟交易）或 AUTO（自动交易/实盘）
    
    # 时间框架配置（以5m为主，3m和15m为辅助）
    TIMEFRAMES: list = ["3m", "5m", "15m"]
    
    # PostgreSQL + TimescaleDB 配置
    PG_HOST: str = "172.22.22.93"
    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "Kuan12345"
    PG_DATABASE: str = "trading-data"
    PG_POOL_SIZE: int = 20  # ✅ 修复：增加连接池大小（从10增加到20）
    PG_MAX_OVERFLOW: int = 40  # ✅ 修复：增加溢出连接数（从20增加到40）
    
    # Redis配置（缓存）
    REDIS_URL: str = "redis://172.22.22.93:6379"
    REDIS_DB: int = 0
    
    # 机器学习配置
    # 注意：模型训练由scheduler统一管理（每天00:01执行）
    TRAINING_SPLIT: float = 0.8  # 训练集/验证集分割比例
    USE_GMADL_LOSS: bool = False  # 生产默认使用稳定的交叉熵损失
    GMADL_ALPHA: float = 1.0
    GMADL_BETA: float = 0.5
    
    # GPU配置
    USE_GPU: bool = True
    GPU_DEVICE: str = "cuda:0"
    
    # 风险管理配置
    VAR_CONFIDENCE: float = 0.95
    MAX_DRAWDOWN_LIMIT: float = 0.15  # 15%
    KELLY_MULTIPLIER: float = 0.25  # Kelly系数乘数
    
    # 日志配置
    LOG_LEVEL: str = "DEBUG"  # ✅ 临时改为DEBUG查看详细日志
    LOG_FILE: str = "trading_system.log"
    
    # WebSocket重连配置
    WS_RECONNECT_INITIAL_DELAY: float = 1.0  # 初始重连延迟（秒）
    WS_RECONNECT_MAX_DELAY: float = 60.0  # 最大重连延迟（秒）
    WS_RECONNECT_BACKOFF_FACTOR: float = 2.0  # 退避因子
    WS_RECONNECT_MAX_RETRIES: int = 10  # 最大重试次数
    WS_PING_INTERVAL: int = 30  # 心跳ping间隔（秒）
    WS_PONG_TIMEOUT: int = 10  # pong超时时间（秒）
    WS_SSL_TIMEOUT: int = 30  # SSL握手超时（秒）
    WS_MESSAGE_TIMEOUT: int = 1200  # 消息超时（秒，20分钟）
    WS_MESSAGE_WARNING_TIMEOUT: int = 600  # 消息警告超时（秒，10分钟）
    
    # GradScaler配置
    GRAD_SCALER_GROWTH_FACTOR: float = 1.2  # 缩放增长因子（从1.5降低到1.2）
    GRAD_SCALER_GROWTH_INTERVAL: int = 2000  # 缩放增长间隔（从1000增加到2000）
    GRAD_SCALER_MAX_SCALE: float = 100000.0  # 最大scale阈值
    GRAD_SCALER_AUTO_RESET: bool = True  # 是否启用自动重置
    GRAD_SCALER_RESET_THRESHOLD_EPOCHS: int = 3  # 触发重置的epoch阈值
    GRAD_SCALER_MAX_CONSECUTIVE_OVERFLOW: int = 5  # 最大连续溢出次数
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def validate_config(self):
        """
        验证配置参数的合理性
        
        Raises:
            ValueError: 配置参数不合法时抛出
        """
        errors = []
        
        # WebSocket重连配置验证
        if self.WS_RECONNECT_INITIAL_DELAY <= 0:
            errors.append(f"WS_RECONNECT_INITIAL_DELAY必须大于0，当前值: {self.WS_RECONNECT_INITIAL_DELAY}")
        
        if self.WS_RECONNECT_MAX_DELAY <= 0:
            errors.append(f"WS_RECONNECT_MAX_DELAY必须大于0，当前值: {self.WS_RECONNECT_MAX_DELAY}")
        
        if self.WS_RECONNECT_MAX_DELAY < self.WS_RECONNECT_INITIAL_DELAY:
            errors.append(f"WS_RECONNECT_MAX_DELAY({self.WS_RECONNECT_MAX_DELAY})必须大于等于WS_RECONNECT_INITIAL_DELAY({self.WS_RECONNECT_INITIAL_DELAY})")
        
        if self.WS_RECONNECT_BACKOFF_FACTOR <= 1.0:
            errors.append(f"WS_RECONNECT_BACKOFF_FACTOR必须大于1.0，当前值: {self.WS_RECONNECT_BACKOFF_FACTOR}")
        
        if self.WS_RECONNECT_MAX_RETRIES <= 0:
            errors.append(f"WS_RECONNECT_MAX_RETRIES必须大于0，当前值: {self.WS_RECONNECT_MAX_RETRIES}")
        
        if self.WS_PING_INTERVAL <= 0:
            errors.append(f"WS_PING_INTERVAL必须大于0，当前值: {self.WS_PING_INTERVAL}")
        
        if self.WS_PONG_TIMEOUT <= 0:
            errors.append(f"WS_PONG_TIMEOUT必须大于0，当前值: {self.WS_PONG_TIMEOUT}")
        
        if self.WS_SSL_TIMEOUT <= 0:
            errors.append(f"WS_SSL_TIMEOUT必须大于0，当前值: {self.WS_SSL_TIMEOUT}")
        
        # GradScaler配置验证
        if self.GRAD_SCALER_GROWTH_FACTOR <= 1.0:
            errors.append(f"GRAD_SCALER_GROWTH_FACTOR必须大于1.0，当前值: {self.GRAD_SCALER_GROWTH_FACTOR}")
        
        if self.GRAD_SCALER_GROWTH_INTERVAL <= 0:
            errors.append(f"GRAD_SCALER_GROWTH_INTERVAL必须大于0，当前值: {self.GRAD_SCALER_GROWTH_INTERVAL}")
        
        if self.GRAD_SCALER_MAX_SCALE <= 0:
            errors.append(f"GRAD_SCALER_MAX_SCALE必须大于0，当前值: {self.GRAD_SCALER_MAX_SCALE}")
        
        if self.GRAD_SCALER_RESET_THRESHOLD_EPOCHS <= 0:
            errors.append(f"GRAD_SCALER_RESET_THRESHOLD_EPOCHS必须大于0，当前值: {self.GRAD_SCALER_RESET_THRESHOLD_EPOCHS}")
        
        if self.GRAD_SCALER_MAX_CONSECUTIVE_OVERFLOW <= 0:
            errors.append(f"GRAD_SCALER_MAX_CONSECUTIVE_OVERFLOW必须大于0，当前值: {self.GRAD_SCALER_MAX_CONSECUTIVE_OVERFLOW}")
        
        # 如果有错误，抛出异常
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

# 创建全局配置实例
settings = Settings()

# 验证配置
try:
    settings.validate_config()
except ValueError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"❌ 配置验证失败: {e}")
    raise