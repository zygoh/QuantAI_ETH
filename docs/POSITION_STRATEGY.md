# 📊 仓位管理策略说明

**版本**: v1.0  
**更新时间**: 2025-10-16  
**当前策略**: ✅ 全仓交易（默认）

---

## 🎯 核心策略：全仓交易

### 策略定义

**每次开仓使用全部可用余额**

```python
仓位价值 = 全部可用余额 × 杠杆倍数

# 示例计算
虚拟交易: 10,000 USDT × 20倍 = 200,000 USDT
实盘交易: 实际余额 USDT × 20倍 = 仓位价值
```

---

## ✅ 为什么使用全仓？

### 1. 适合中频交易策略

- **信号频率**: 每天5-8个信号
- **持仓时间**: 短期（几小时到1天）
- **快速进出**: 不需要保留资金等待其他机会

### 2. 最大化收益潜力

- 每个信号都能充分利用资金
- 在高置信度信号上获取最大收益
- 符合中频交易的特点

### 3. 简化仓位管理

- 无需复杂计算
- 避免因仓位过小错失机会
- 减少决策复杂度

### 4. 配合动态止损

- 全仓 + 智能止损 = 平衡风险收益
- 止损保护资金（1.5×ATR）
- 止盈锁定利润（3-4×ATR）

---

## 🔧 代码实现

### 默认行为（全仓）

```python
# backend/app/services/position_manager.py

async def calculate_position_size(
    symbol, signal_type, confidence, current_price,
    is_virtual=True,
    use_full_position=True  # ✅ 默认全仓
):
    if use_full_position:
        # 全仓策略
        position_value = available_balance * leverage
        
        logger.info(f"💰 全仓仓位计算: {symbol} {position_value:.2f} USDT")
        logger.info(f"  余额: {available_balance} USDT | 杠杆: {leverage}x | 策略: 全仓")
    
    return position_value
```

### 调用示例

```python
# signal_generator.py - 自动使用全仓策略
position_size = await position_manager.calculate_position_size(
    symbol, signal_type, confidence, current_price,
    is_virtual=is_virtual_mode
    # use_full_position 默认为 True（全仓）
)
```

---

## 🔄 可选：动态仓位模式

**如需降低风险**，可以切换到动态模式：

```python
position_size = await position_manager.calculate_position_size(
    symbol, signal_type, confidence, current_price,
    is_virtual=is_virtual_mode,
    use_full_position=False  # 启用动态调整
)
```

**动态模式特性**：
- 基础仓位：10% × 置信度
- 波动率调整：0.5x - 1.3x
- 持仓调整：0.5x - 1.0x
- 连续亏损保护：0.5x - 0.75x
- 最终范围：2% - 15%

**适用场景**：
- 市场极度波动（日波动>10%）
- 连续多次亏损（3连亏+）
- 需要测试保守策略

---

## ⚖️ 全仓 vs 动态对比

| 维度 | 全仓策略 | 动态策略 |
|------|---------|---------|
| **收益潜力** | ⭐⭐⭐⭐⭐ 最高 | ⭐⭐⭐ 中等 |
| **风险控制** | ⭐⭐ 依赖止损 | ⭐⭐⭐⭐ 多重保护 |
| **适用场景** | 中频交易 | 保守策略 |
| **复杂度** | ⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| **计算开销** | ⭐ 极小 | ⭐⭐⭐ 中等 |
| **当前使用** | ✅ 默认启用 | ⚪ 备用方案 |

---

## 📊 风险说明

### 全仓策略的风险

1. **单次亏损影响大**
   - 一次错误信号可能损失较大
   - **缓解**: 动态ATR止损（限制单次亏损）

2. **连续亏损累积**
   - 多次亏损会快速消耗本金
   - **缓解**: 高置信度阈值（0.40+），提高信号质量

3. **波动率影响**
   - 高波动市场风险增加
   - **缓解**: 止损位自适应（ATR自动扩大）

### 风险缓解措施

✅ **已实施**:
1. 动态ATR止损（1.5倍ATR，约1-2%）
2. 智能止盈（3-4倍ATR，盈亏比1.8:1-2.67:1）
3. 置信度过滤（≥0.40）
4. 信号去重（避免频繁交易）
5. 预热保护（前5个信号仅记录）

✅ **备用方案**:
- 动态仓位调整（可随时启用）
- 固定百分比止损（降级备用）

---

## 🎯 实际表现监控

### 关键指标

**必须监控**：
1. **单次最大亏损**: 应<2%（1.5×ATR止损）
2. **连续亏损**: 观察3连亏频率
3. **最大回撤**: 目标<15%
4. **胜率**: 目标≥55%

**告警阈值**：
```python
if single_loss > 3%: 检查止损是否生效
if consecutive_losses >= 3: 考虑暂停或启用动态模式
if max_drawdown > 15%: 暂停交易
```

### 性能评估周期

- **短期**（7天）: 观察信号质量和止损效果
- **中期**（30天）: 评估胜率、盈亏比、夏普比率
- **长期**（90天）: 决定是否保持全仓或切换动态

---

## 🔄 策略切换指南

### 何时考虑切换到动态模式？

**建议切换条件**（同时满足任意2项）：
1. 最大回撤超过12%
2. 连续3次以上亏损
3. 市场波动率持续>8%
4. 30日胜率<48%

### 如何切换？

**方法1: 代码修改**（推荐）
```python
# signal_generator.py:720
position_size = await position_manager.calculate_position_size(
    symbol, signal_type, confidence, current_price,
    is_virtual=is_virtual_mode,
    use_full_position=False  # 改为False启用动态模式
)
```

**方法2: 配置参数**（未来可实现）
```python
# config.py
USE_FULL_POSITION: bool = True  # 全仓开关
```

---

## 📈 预期表现

### 虚拟交易示例（10,000 USDT本金）

**单笔交易**:
```
信号: LONG, 置信度=0.50
入场: 4000 USDT
仓位: 10,000 × 20 = 200,000 USDT

止损: 3940 (1.5%), 亏损 12,000 USDT
止盈: 4120 (3%), 盈利 24,000 USDT

盈亏比: 1:2.0
```

**连续5笔交易**（假设胜率55%）:
```
信号1: LONG ✅ +24,000 (余额: 34,000)
信号2: SHORT ❌ -13,000 (余额: 21,000)  
信号3: LONG ✅ +25,000 (余额: 46,000)
信号4: LONG ✅ +28,000 (余额: 74,000)
信号5: SHORT ❌ -11,000 (余额: 63,000)

最终: +63,000 (+530%)
胜率: 60% (3/5)
```

**风险警示**: 
- ⚠️ 全仓高风险高收益
- ⚠️ 单次亏损可达10-15%
- ⚠️ 务必严格执行止损

---

## 📚 相关文档

1. **优化路线图**: `OPTIMIZATION_ROADMAP.md`
2. **优化应用**: `docs/OPTIMIZATION_APPLIED.md`
3. **Phase 1报告**: `docs/PHASE1_OPTIMIZATION_COMPLETE.md`
4. **后端规则**: `backend/.cursor/rules/backend.mdc`

---

## ✅ 总结

**当前仓位策略**: ✅ **全仓交易**

**特点**:
- 每次开仓使用100%可用余额
- 配合20倍杠杆
- 通过动态ATR止损保护资金
- 最大化收益潜力

**风险控制**:
- 动态止损（1.5×ATR）
- 智能止盈（3-4×ATR）
- 置信度过滤（≥0.40）
- 预热保护（前5个信号）

**适用性**: ✅ 适合当前中频交易系统

**备用方案**: 动态仓位调整（默认关闭，如需可启用）

---

**文档创建**: 2025-10-16  
**策略状态**: ✅ 已确认，全仓交易

