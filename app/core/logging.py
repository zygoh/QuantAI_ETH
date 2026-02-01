"""
统一日志配置

提供统一的日志配置和工具函数。
"""
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from typing import Optional


# 默认日志格式
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    log_format: str = DEFAULT_LOG_FORMAT,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 30,
    use_time_rotation: bool = True
) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名（不含路径）
        log_dir: 日志目录
        log_format: 日志格式
        max_bytes: 单个日志文件最大字节数（仅size rotation时使用）
        backup_count: 保留的日志文件数量
        use_time_rotation: 是否使用时间轮转（按天）

    Returns:
        配置好的根日志器
    """
    # 创建日志目录
    if log_file:
        os.makedirs(log_dir, exist_ok=True)

    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # 清除现有处理器
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(log_format)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = os.path.join(log_dir, log_file)

        if use_time_rotation:
            # 按时间轮转（每天）
            file_handler = TimedRotatingFileHandler(
                log_path,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8',
                utc=False
            )
        else:
            # 按大小轮转
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )

        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器

    Args:
        name: 日志器名称（通常使用 __name__）

    Returns:
        日志器实例
    """
    return logging.getLogger(name)


def set_log_level(level: str, logger_name: Optional[str] = None):
    """
    动态设置日志级别

    Args:
        level: 日志级别
        logger_name: 日志器名称（None表示根日志器）
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))


class LoggerMixin:
    """
    日志混入类

    为类提供便捷的日志访问。

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Hello")
    """

    @property
    def logger(self) -> logging.Logger:
        """获取类专属的日志器"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger


def log_exception(logger: logging.Logger, exc: Exception, message: str = ""):
    """
    记录异常信息

    Args:
        logger: 日志器
        exc: 异常对象
        message: 附加消息
    """
    if message:
        logger.error(f"{message}: {exc}", exc_info=True)
    else:
        logger.error(f"Exception occurred: {exc}", exc_info=True)
