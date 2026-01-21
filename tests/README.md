# 测试说明

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_prediction_scalar.py -v

# 运行并显示详细日志
pytest tests/ -v --log-cli-level=DEBUG
```

## 查看测试日志

测试日志会输出到两个地方：

1. **控制台**：实时显示测试进度和结果
2. **日志文件**：`logs/test_results.log`

```bash
# 查看测试日志
cat logs/test_results.log

# 实时监控测试日志
tail -f logs/test_results.log

# 搜索特定内容
grep "✅" logs/test_results.log
grep "❌" logs/test_results.log
```

## 测试文件说明

- `test_prediction_scalar.py`: 预测结果标量化测试
- `test_fatal_error_penalty.py`: 致命错误惩罚权重测试
- `test_hold_weight_multiplier.py`: HOLD类别权重倍数测试
- `test_meta_features_dimension.py`: 元特征维度一致性测试
- `test_array_dimension_fix.py`: 数组维度修复验证测试

## 配置文件

- `conftest.py`: pytest配置，设置日志输出
- `pytest.ini`: pytest全局配置（项目根目录）
