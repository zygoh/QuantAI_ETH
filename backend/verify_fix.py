"""
验证OKX修复
"""
import sys
sys.path.insert(0, '.')

from app.exchange.mappers import SymbolMapper
from app.core.config import settings

print("=" * 70)
print("OKX修复验证")
print("=" * 70)

# 1. 验证配置
print(f"\n1. 配置验证:")
print(f"   SYMBOL配置: {settings.SYMBOL}")
print(f"   EXCHANGE_TYPE: {settings.EXCHANGE_TYPE}")

# 2. 验证symbol转换
print(f"\n2. Symbol格式转换:")
okx_symbol = SymbolMapper.to_exchange_format(settings.SYMBOL, "OKX")
print(f"   输入 (标准格式): {settings.SYMBOL}")
print(f"   输出 (OKX格式):  {okx_symbol}")

# 3. 验证预期结果
expected = "ETH-USDT-SWAP"
if okx_symbol == expected:
    print(f"   ✅ 转换正确! 期望: {expected}, 实际: {okx_symbol}")
else:
    print(f"   ❌ 转换错误! 期望: {expected}, 实际: {okx_symbol}")

# 4. 测试其他格式
print(f"\n3. 其他格式测试:")
test_cases = [
    ("ETH/USDT", "ETH-USDT-SWAP"),
    ("BTC/USDT", "BTC-USDT-SWAP"),
]

for input_symbol, expected_output in test_cases:
    output = SymbolMapper.to_exchange_format(input_symbol, "OKX")
    status = "✅" if output == expected_output else "❌"
    print(f"   {status} {input_symbol:12} -> {output:20} (期望: {expected_output})")

print("\n" + "=" * 70)
print("验证完成!")
print("=" * 70)
