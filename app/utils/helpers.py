"""
辅助工具函数（精简版 - 仅保留实际使用的函数）
"""

def format_signal_type(signal_type: str) -> str:
    """
    格式化信号类型显示（图标+中文）
    
    Args:
        signal_type: 信号类型（LONG/SHORT/HOLD）
    
    Returns:
        格式化后的信号字符串
    
    Examples:
        >>> format_signal_type('LONG')
        '📈 做多'
        >>> format_signal_type('SHORT')
        '📉 做空'
        >>> format_signal_type('HOLD')
        '⏸️ 持有'
    """
    signal_map = {
        'LONG': '📈 做多',
        'SHORT': '📉 做空',
        'HOLD': '⏸️ 持有'
    }
    return signal_map.get(signal_type.upper(), signal_type)