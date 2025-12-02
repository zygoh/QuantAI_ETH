"""
测试交易对格式转换
"""
from app.exchange.mappers import SymbolMapper

# 测试标准格式转OKX格式
test_symbols = ["ETH/USDT", "BTC/USDT", "ETHUSDT"]

print("=" * 60)
print("测试交易对格式转换")
print("=" * 60)

for symbol in test_symbols:
    okx_format = SymbolMapper.to_exchange_format(symbol, "OKX")
    print(f"标准格式: {symbol:15} -> OKX格式: {okx_format}")

print("\n" + "=" * 60)
print("测试反向转换")
print("=" * 60)

okx_symbols = ["ETH-USDT-SWAP", "BTC-USDT-SWAP"]
for symbol in okx_symbols:
    std_format = SymbolMapper.to_standard_format(symbol, "OKX")
    print(f"OKX格式: {symbol:20} -> 标准格式: {std_format}")
