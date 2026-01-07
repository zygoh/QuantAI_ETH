"""
时间特征模块
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间特征"""
    try:
        # 确保 timestamp 是 datetime 类型
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            datetime_col = df['timestamp']
        else:
            logger.warning("timestamp 列不存在，跳过时间特征")
            return df
        
        new_features = {}
        
        # 基础时间特征
        hour = datetime_col.dt.hour
        day_of_week = datetime_col.dt.dayofweek
        month = datetime_col.dt.month
        
        new_features['hour'] = hour
        new_features['day_of_week'] = day_of_week
        new_features['day_of_month'] = datetime_col.dt.day
        new_features['month'] = month
        new_features['quarter'] = datetime_col.dt.quarter
        
        # 周期性编码
        new_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        new_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        new_features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        new_features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        new_features['month_sin'] = np.sin(2 * np.pi * month / 12)
        new_features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # 交易时段
        new_features['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
        new_features['is_european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
        new_features['is_american_session'] = ((hour >= 16) & (hour < 24)).astype(int)
        
        # 周末标识
        new_features['is_weekend'] = (day_of_week >= 5).astype(int)
        
        # 一次性添加所有特征
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"添加时间特征失败: {e}")
        return df

