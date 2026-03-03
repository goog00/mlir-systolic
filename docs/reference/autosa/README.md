# AutoSA 相关文档

> **目录**: `docs/reference/autosa/`  
> **目的**: 收集所有与 AutoSA 工具相关的分析、对比和参考文档

---

## 📚 文档列表

### 核心分析（推荐优先阅读）

| 文件 | 说明 |
|------|------|
| [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md) | AutoSA 源码与编译流水线、性能瓶颈、MLIR 优化机会（含 MTTKRP 访存） |
| [AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md) ⭐ | AutoSA 与 mlir-systolic 逐项对照、打通流程建议、CNN 支持路径 |

### 其他分析与参考

| 文件 | 说明 |
|------|------|
| [AUTOSA_ANALYSIS.md](AUTOSA_ANALYSIS.md) | AutoSA 架构、算法、参数影响 |
| [AUTOSA_ARCHITECTURE.md](AUTOSA_ARCHITECTURE.md) | AutoSA 架构与组件说明 |
| [AUTOSA_QUICK_REFERENCE.md](AUTOSA_QUICK_REFERENCE.md) | AutoSA 快速参考 |
| [AUTOSA_REFERENCE_STATUS.md](AUTOSA_REFERENCE_STATUS.md) | AutoSA 参考状态 |
| [AUTOSA_REFERENCE_TABLES.md](AUTOSA_REFERENCE_TABLES.md) | AutoSA 参考表 |
| [comparison_with_autosa.md](comparison_with_autosa.md) | 与 AutoSA 功能对比与差异 |

---

## 🎯 使用场景

- **理解 AutoSA**：阅读分析文档了解原理
- **与 mlir-systolic 对照**：见 AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS
- **架构/参数参考**：见 AUTOSA_ARCHITECTURE、AUTOSA_QUICK_REFERENCE

---

## 🔗 相关文档

- **设计/优化**：[../../design/](../../design/) — 脉动阵列优化与 L3/写时重排
- **测试与参考**： [../testing/](../testing/) — 测试结果与参考样本；根目录 `test/autosa_hls_refs/` 为 AutoSA 生成的 HLS 参考

