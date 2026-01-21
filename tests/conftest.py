"""
pytest配置文件

配置测试环境、日志输出等
"""
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def pytest_configure(config):
    """pytest启动时的配置"""
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建测试日志文件处理器
    test_log_file = os.path.join(log_dir, "test_results.log")
    file_handler = TimedRotatingFileHandler(
        test_log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除现有处理器（避免重复）
    root_logger.handlers.clear()
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 记录测试开始
    root_logger.info("=" * 80)
    root_logger.info(f"🧪 测试开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    root_logger.info("=" * 80)


def pytest_unconfigure(config):
    """pytest结束时的清理"""
    root_logger = logging.getLogger()
    root_logger.info("=" * 80)
    root_logger.info(f"✅ 测试结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    root_logger.info("=" * 80)
