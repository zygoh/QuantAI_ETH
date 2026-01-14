"""
系统常量定义
"""
# StdLib
from decimal import Decimal

# 🎯 虚拟交易手续费配置（模拟实际交易所费率）- 使用Decimal确保精度
VIRTUAL_OPEN_FEE_RATE = Decimal('0.0002')   # 开仓手续费：0.02% (Maker)
VIRTUAL_CLOSE_FEE_RATE = Decimal('0.0005')  # 平仓手续费：0.05% (Taker)

