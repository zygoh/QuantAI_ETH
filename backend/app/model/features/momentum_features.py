"""
动量特征模块
"""
import logging
import pandas as pd
import numpy as np
import ta

logger = logging.getLogger(__name__)


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加动量特征"""
    try:
        new_features = {}
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            new_features[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()
        
        # Momentum
        for period in [5, 10, 20]:
            new_features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # TSI (True Strength Index)
        new_features['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()
        
        # Ultimate Oscillator
        new_features['uo'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
        
        # Awesome Oscillator
        new_features['ao'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
        
        # 动量加速度（捕捉动量变化）
        rsi_14 = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        new_features['rsi_acceleration'] = rsi_14 - rsi_14.shift(1)
        new_features['rsi_velocity'] = rsi_14.diff()
        
        # 多周期动量一致性（趋势确认）
        roc_5 = ta.momentum.ROCIndicator(df['close'], window=5).roc()
        roc_10 = ta.momentum.ROCIndicator(df['close'], window=10).roc()
        roc_20 = ta.momentum.ROCIndicator(df['close'], window=20).roc()
        # 如果多个周期都是正/负，则趋势更可靠
        new_features['momentum_alignment'] = ((roc_5 > 0).astype(int) + 
                                              (roc_10 > 0).astype(int) + 
                                              (roc_20 > 0).astype(int)) - 1.5
        
        # 一次性添加所有特征
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"添加动量特征失败: {e}")
        return df

