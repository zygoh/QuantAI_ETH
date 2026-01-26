"""
特征工程模块
"""
# StdLib
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Third-Party
import numpy as np
import pandas as pd
import ta

# Local App
from app.exchange.exchange_factory import ExchangeFactory
from app.model.features import (
    add_momentum_features,
    add_multi_timeframe_features,
    add_order_flow_features,
    add_pattern_features,
    add_price_features,
    add_sentiment_features,
    add_support_resistance_features,
    add_swing_features,
    add_technical_indicators,
    add_time_features,
    add_trend_strength_features,
    add_volume_features,
    add_microstructure_features,
    add_volatility_features
)

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.feature_columns = []
    
    def create_features(self, df: pd.DataFrame, loop: Optional[asyncio.AbstractEventLoop] = None) -> pd.DataFrame:
        """创建所有特征"""
        try:
            self.loop = loop
            inf_tracker = {}
            if df.empty:
                return df
            
            logger.info(f"开始特征工程: {len(df)}行原始数据")
            
            rows_before_validation = len(df)
            
            if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
                # 验证价格范围合理性
                invalid_price_mask = (
                    (df['close'] <= 0) |
                    (df['high'] <= 0) |
                    (df['low'] <= 0) |
                    (df['open'] <= 0) |
                    (df['high'] < df['low']) |
                    (df['close'] < df['low']) |
                    (df['close'] > df['high']) |
                    (df['open'] < df['low']) |
                    (df['open'] > df['high'])
                )
                
                if invalid_price_mask.any():
                    invalid_count = invalid_price_mask.sum()
                    logger.error(f"检测到{invalid_count}条异常K线（价格范围不合理），将过滤")
                    df = df[~invalid_price_mask]
            
            if 'volume' in df.columns:
                # 过滤负数成交量
                negative_volume_mask = df['volume'] < 0
                if negative_volume_mask.any():
                    negative_count = negative_volume_mask.sum()
                    logger.error(f"检测到{negative_count}条异常K线（成交量为负数），将过滤")
                    df = df[~negative_volume_mask]
                
                # 过滤未完成K线（volume=0）
                zero_volume_mask = df['volume'] == 0
                if zero_volume_mask.any():
                    zero_count = zero_volume_mask.sum()
                    logger.warning(f"过滤{zero_count}条未完成K线（volume=0）")
                    df = df[~zero_volume_mask]
            
            # 验证异常波动（单根K线涨跌幅超过50%视为异常）
            if 'close' in df.columns and 'open' in df.columns:
                price_change_pct = (df['close'] - df['open']) / (df['open'] + 1e-10)
                extreme_volatility_mask = np.abs(price_change_pct) > 0.5
                if extreme_volatility_mask.any():
                    extreme_count = extreme_volatility_mask.sum()
                    logger.warning(f"检测到{extreme_count}条异常波动K线（涨跌幅>50%），将过滤")
                    df = df[~extreme_volatility_mask]
            
            filtered_count = rows_before_validation - len(df)
            if filtered_count > 0:
                logger.warning(f"数据验证完成：过滤{filtered_count}条异常K线，剩余{len(df)}条")
            
            if df.empty:
                logger.error("数据验证后为空，无法进行特征工程")
                return df
            
            # 统一清洗close价格，创建clean_close列
            if 'close' in df.columns:
                df['clean_close'] = df['close'].replace(0, np.nan)
                df['clean_close'] = df['clean_close'].ffill()
                df['clean_close'] = df['clean_close'].bfill()
                if df['clean_close'].isna().any():
                    logger.warning(f"clean_close仍有NaN，使用原始close值")
                    df['clean_close'] = df['clean_close'].fillna(df['close'])
            else:
                df['clean_close'] = np.nan
            
            if df.index.name == 'timestamp' or 'timestamp' not in df.columns:
                df = df.reset_index()
            
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df = add_price_features(df)
            df = add_technical_indicators(df)
            df = add_volume_features(df)
            df = add_time_features(df)
            df = add_microstructure_features(df)
            df = add_volatility_features(df)
            df = add_momentum_features(df)
            df = add_sentiment_features(df)
            df = add_multi_timeframe_features(df)
            df = add_trend_strength_features(df)
            df = add_support_resistance_features(df)
            df = add_pattern_features(df)
            df = add_order_flow_features(df)
            df = add_swing_features(df)
            
            # 处理无穷大值（inf）- 必须在NaN处理前完成
            inf_count = 0
            
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    inf_mask = np.isinf(df[col])
                    if inf_mask.any():
                        col_inf_count = inf_mask.sum()
                        inf_count += col_inf_count
                        df.loc[inf_mask, col] = np.nan
            
            if inf_count > 0:
                logger.warning(f"检测到{inf_count}个无穷大值（inf），已替换为NaN")
            
            # 处理过大值（可能导致缩放时溢出）
            large_value_threshold = 1e15
            large_count = 0
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    large_mask = np.abs(df[col]) > large_value_threshold
                    if large_mask.any():
                        large_count += large_mask.sum()
                        df.loc[large_mask, col] = np.nan
            
            if large_count > 0:
                logger.warning(f"检测到{large_count}个过大值（>1e15），已替换为NaN")
            
            # 处理NaN值（训练用dropna，预测用fillna）
            rows_before = len(df)
            
            nan_by_column = {}
            for col in df.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        nan_by_column[col] = nan_count
            
            # 先尝试删除NaN
            df_clean = df.dropna()
            
            # 如果删除后数据量<50行，说明是预测场景，改用填充
            if len(df_clean) < 50 and rows_before >= 100:
                logger.debug(f"预测场景检测：dropna会导致数据过少（{rows_before}→{len(df_clean)}），改用fillna")
                df = df.ffill()
                df = df.bfill()
                
                if 'close' in df.columns:
                    if df['close'].isna().any():
                        for idx in df[df['close'].isna()].index:
                            if not df.loc[idx, ['open', 'high', 'low']].isna().all():
                                df.loc[idx, 'close'] = df.loc[idx, ['open', 'high', 'low']].mean()
                
                for col in df.columns:
                    if col not in ['timestamp', 'close', 'volume']:
                        df[col] = df[col].fillna(0)
                
                if 'close' in df.columns:
                    zero_close = (df['close'] == 0).sum()
                    if zero_close > 0:
                        logger.warning(f"预测场景：仍有{zero_close}个close为0，将替换为NaN")
                        df.loc[df['close'] == 0, 'close'] = np.nan
                
                logger.debug(f"特征工程完成（预测模式）: {len(df)}行，特征数: {len(df.columns)}")
            else:
                # 训练场景，正常删除NaN
                df = df_clean
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    if nan_by_column:
                        top_nan_cols = sorted(nan_by_column.items(), key=lambda x: x[1], reverse=True)[:5]
                        nan_reason = f"（技术指标窗口导致：前{rows_dropped}行无法计算长周期指标如SMA_200/SMA_50等，NaN最多的列: {', '.join([f'{col}({count})' for col, count in top_nan_cols])}）"
                    else:
                        nan_reason = "（因NaN/Inf导致）"
                    
                    logger.info(f"特征工程完成: {len(df)}行（丢弃{rows_dropped}行{nan_reason}），特征数: {len(df.columns)}")
                else:
                    logger.info(f"特征工程完成: {len(df)}行，特征数: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"特征工程失败: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def _calculate_coefficient_of_variation(self, df: pd.DataFrame, col: str) -> float:
        """
        计算变异系数（Coefficient of Variation，尺度无关）
        
        变异系数 = 标准差 / |均值|，用于衡量不同尺度特征的相对变异性
        
        Args:
            df: 数据框
            col: 列名
            
        Returns:
            变异系数（均值接近零时返回标准化后的方差）
        """
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # 边界情况：均值接近零（如震荡指标中心化后）
            if abs(mean_val) < 1e-10:
                logger.debug(f"⚠️ 特征 {col} 均值接近零 ({mean_val:.2e})，使用标准化后的方差")
                # 使用标准化后的方差作为备选方案
                if std_val > 1e-10:
                    normalized = (df[col] - mean_val) / std_val
                    return normalized.var()
                else:
                    return 0.0
            
            # 正常情况：返回变异系数
            cv = std_val / abs(mean_val)
            return cv if not np.isnan(cv) else 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ 计算特征 {col} 的变异系数失败: {e}")
            return 0.0
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        获取特征重要性（基于变异系数，尺度无关）
        
        修复原因：
        - 原方法使用原始方差，对尺度敏感（RSI 0-100 vs EMA 90000）
        - 导致丢弃 RSI、MACD 等重要形态指标
        - 新方法使用变异系数（CV = std/mean），尺度无关
        
        Args:
            df: 数据框
            
        Returns:
            特征重要性字典（按重要性降序排列）
        """
        try:
            # 排除基础价格和时间列
            exclude_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'clean_close']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # 计算每个特征的变异系数
            feature_cv = {}
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    cv = self._calculate_coefficient_of_variation(df, col)
                    feature_cv[col] = cv
            
            # 归一化：转换为相对重要性（总和为1）
            total_cv = sum(feature_cv.values())
            if total_cv > 0:
                feature_importance = {k: v/total_cv for k, v in feature_cv.items()}
            else:
                feature_importance = feature_cv
            
            # 按重要性降序排列
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"❌ 计算特征重要性失败: {e}")
            return {}
    
    def select_features(self, df: pd.DataFrame, top_n: int = None) -> List[str]:
        """
        选择重要特征（基于变异系数）
        
        Args:
            df: 数据框
            top_n: 保留的特征数量（默认从 constants.py 读取 FEATURE_SELECTION_TOP_N）
            
        Returns:
            选中的特征列表
        """
        try:
            # 如果未指定 top_n，从 constants.py 读取配置
            if top_n is None:
                from app.core.constants import FEATURE_SELECTION_TOP_N
                top_n = FEATURE_SELECTION_TOP_N
            
            # 计算特征重要性
            feature_importance = self.get_feature_importance(df)
            
            if not feature_importance:
                logger.warning("⚠️ 特征重要性为空，返回空列表")
                return []
            
            # 选择 top-N 特征
            selected_features = list(feature_importance.keys())[:top_n]
            excluded_features = list(feature_importance.keys())[top_n:]
            
            # 📊 记录特征选择结果
            logger.info(f"✅ 特征选择完成：保留 {len(selected_features)}/{len(feature_importance)} 个特征")
            
            # 记录被选中的 top-10 特征及其重要性
            if len(selected_features) > 0:
                top_10_selected = selected_features[:10]
                top_10_scores = [f"{feat}({feature_importance[feat]:.4f})" for feat in top_10_selected]
                logger.info(f"📊 Top-10 选中特征: {', '.join(top_10_scores)}")
            
            # 记录被排除的特征（仅记录前5个）
            if len(excluded_features) > 0:
                excluded_sample = excluded_features[:5]
                excluded_scores = [f"{feat}({feature_importance[feat]:.4f})" for feat in excluded_sample]
                logger.debug(f"🚫 排除特征示例（前5个）: {', '.join(excluded_scores)}")
            
            # 检查重要形态指标是否被保留
            important_indicators = ['RSI', 'MACD', 'Stochastic', 'ADX', 'ATR', 'BB', 'CCI']
            preserved_indicators = []
            for indicator in important_indicators:
                matching_features = [f for f in selected_features if indicator.lower() in f.lower()]
                if matching_features:
                    preserved_indicators.append(f"{indicator}({len(matching_features)})")
            
            if preserved_indicators:
                logger.info(f"✅ 保留的形态指标: {', '.join(preserved_indicators)}")
            else:
                logger.warning("⚠️ 未检测到常见形态指标（RSI/MACD等），请检查特征工程")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"❌ 特征选择失败: {e}")
            logger.error(traceback.format_exc())
            return []


# 全局特征工程器实例
feature_engineer = FeatureEngineer()
