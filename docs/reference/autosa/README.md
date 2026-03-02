# AutoSA 相关文档

> **目录**: `docs/autosa/`  
> **目的**: 收集所有与 AutoSA 工具相关的分析、对比和参考文档

---

## 📚 文档列表

### 核心分析文档

1. **[AUTOSA_ANALYSIS.md](AUTOSA_ANALYSIS.md)**
   - AutoSA 架构、算法、参数影响分析
   - 详细的功能和实现分析

2. **[AUTOSA_ARCHITECTURE.md](AUTOSA_ARCHITECTURE.md)**
   - AutoSA 架构详细分析
   - 系统设计和组件说明

3. **[AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md)**
   - AutoSA 源码逻辑与编译流水线
   - 脉动阵列性能瓶颈与可改进空间（含 MTTKRP 随机/跨步访存案例）
   - 在 MLIR 框架下的优化机会与落地建议

### 对比与对照

4. **[comparison_with_autosa.md](comparison_with_autosa.md)**
   - mlir-systolic 与 AutoSA 的功能对比
   - 差异分析和兼容性说明

5. **[AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md)** ⭐
   - AutoSA 与 mlir-systolic 源码/阶段逐项对照
   - 结合 AUTOSA_SOURCE_PERF 与 VISION 的打通完整流程建议
   - 简单 CNN（5-loop + 固定 spacetime）支持路径与任务清单

---

## 🎯 使用场景

- **理解 AutoSA**: 阅读分析文档了解 AutoSA 的工作原理
- **参数配置**: 参考测试生成指南配置参数
- **功能对比**: 查看对比文档了解两个工具的差异
- **架构参考**: 参考 AutoSA 架构设计自己的实现

---

## 🔗 相关文档

- **Space-time 实现**: `../spacetime/` - 基于 AutoSA 的 spacetime 实现
- **测试验证**: `../testing/` - 测试结果和参考样本
- **问题分析**: `../issues/` - 与 AutoSA 对比发现的问题

---

## 📊 文档统计

| 指标 | 数值 |
|------|------|
| **文档数量** | 5 份 |
| **总字数** | ~40,000 字 |
| **主要用途** | 参考、分析、对比 |

---

**👉 推荐开始**: [AUTOSA_ANALYSIS.md](AUTOSA_ANALYSIS.md)

