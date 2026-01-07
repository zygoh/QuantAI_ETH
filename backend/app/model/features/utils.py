"""
特征工程工具函数
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_fractal_dimension(series: pd.Series, period: int) -> pd.Series:
    """计算分形维度"""
    try:
        def fractal_dim(data):
            try:
                data = np.array(data)
                if len(data) < 10:
                    return np.nan
                
                # Higuchi方法计算分形维度
                N = len(data)
                L = []
                x = []
                
                for k in range(1, min(N//2, 10)):
                    Lk = 0
                    for m in range(k):
                        Lmk = 0
                        for i in range(1, int((N-m)/k)):
                            Lmk += abs(data[m+i*k] - data[m+(i-1)*k])
                        if ((N-m)/k) * k > 0:
                            Lmk = Lmk * (N-1) / (((N-m)/k) * k)
                        Lk += Lmk
                    
                    if k > 0:
                        L.append(Lk/k)
                        x.append(1.0/k)
                
                if len(L) < 2:
                    return np.nan
                
                # 线性回归计算斜率
                x = np.log(x)
                y = np.log(L)
                coeffs = np.polyfit(x, y, 1)
                return coeffs[0]
            except:
                return np.nan
        
        return series.rolling(period).apply(fractal_dim, raw=False)
        
    except Exception as e:
        logger.error(f"计算分形维度失败: {e}")
        return pd.Series(np.nan, index=series.index)


def calculate_hurst_exponent(series: pd.Series, period: int) -> pd.Series:
    """计算Hurst指数"""
    try:
        def hurst_exp(data):
            if len(data) < 10:
                return np.nan
            
            # R/S分析计算Hurst指数
            data = np.array(data)
            N = len(data)
            
            # 计算累积偏差
            mean_data = np.mean(data)
            cumulative_deviate = np.cumsum(data - mean_data)
            
            # 计算范围
            R = np.max(cumulative_deviate) - np.min(cumulative_deviate)
            
            # 计算标准差
            S = np.std(data)
            
            if S == 0:
                return np.nan
            
            # R/S比率
            rs = R / S
            
            if rs <= 0:
                return np.nan
            
            # Hurst指数
            return np.log(rs) / np.log(N)
        
        return series.rolling(period).apply(hurst_exp, raw=False)
        
    except Exception as e:
        logger.error(f"计算Hurst指数失败: {e}")
        return pd.Series(np.nan, index=series.index)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI指标"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series(50, index=prices.index)  # 默认值

