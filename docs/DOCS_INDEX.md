# mlir-systolic 文档全量索引

> 本文件列出**项目内所有 Markdown 文档**（不含 third_party），便于查找与整理。  
> 最后更新：2026-03

---

## 根目录

| 文件 | 说明 |
|------|------|
| [../README.md](../README.md) | 项目概述、构建、目录结构、与 AutoSA 对比 |
| [../PROJECT_STATUS_AND_ONBOARDING.md](../PROJECT_STATUS_AND_ONBOARDING.md) | **新环境/新 Agent 上手指南**：做了啥、如何验证、下一步、服务器环境 |
| [../RECENT_CHANGES_AND_NEXT_STEPS.md](../RECENT_CHANGES_AND_NEXT_STEPS.md) | 近期修改与下一步、文档索引、写时重排与 e2e 说明 |

---

## docs/（顶层）

| 文件 | 说明 |
|------|------|
| [README.md](README.md) | 文档导航入口，指向本索引与各分类 |
| [DOCS_INDEX.md](DOCS_INDEX.md) | 本文件：全量文档索引 |
| [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) | 系统架构总览 |
| [VISION_AND_DESIGN_GOALS.md](VISION_AND_DESIGN_GOALS.md) | 愿景与设计目标 |
| [BUILD_AND_SERVER_ENVIRONMENT.md](BUILD_AND_SERVER_ENVIRONMENT.md) | 构建依赖与 Ubuntu 18.04/老 LLVM 说明 |

---

## docs/design/（设计策略与优化）

| 文件 | 说明 |
|------|------|
| [SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md) | 单/多核与高性能策略（HLS≤2h、共用参数、高性能模板） |
| [SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md) | 脉动阵列优化改进计划（Phase 1–4、参数、MTTKRP/MM/CNN） |
| [EXISTING_OPTIMIZATIONS_IN_CODE.md](design/EXISTING_OPTIMIZATIONS_IN_CODE.md) | 代码中已有优化梳理（写时/读时重排、L2/L3/drain、FIFO） |
| [PARAMETER_SELECTION_AND_VALID_RANGE.md](design/PARAMETER_SELECTION_AND_VALID_RANGE.md) | 参数选择与合法范围（多面体分析、tilable_loops） |
| [FIFO_DEPTH_AND_PERFORMANCE_NEXT.md](design/FIFO_DEPTH_AND_PERFORMANCE_NEXT.md) | FIFO 深度策略与性能后续（RESOURCE、Pipeline 友好） |
| [SHIFT_VS_DIV_MOD_AND_HLS_CORRECTNESS.md](design/SHIFT_VS_DIV_MOD_AND_HLS_CORRECTNESS.md) | 移位/除取模与 HLS 正确性（强度削减、2 的幂） |
| [L3_COALESCE_AND_ACCESS_PATTERN.md](design/L3_COALESCE_AND_ACCESS_PATTERN.md) | L3 coalesce 与访问模式（tile 顺序、word_idx） |
| [L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md](design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md) | L3/写时重排与 host-serialize（暂不做的决策说明） |
| [CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md](design/CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md) | AutoSA 对比与通用化重构路线（支持 mttkrp/ttmc） |

---

## docs/status/（项目状态）

| 文件 | 说明 |
|------|------|
| [PROJECT_STATUS.md](status/PROJECT_STATUS.md) | 项目状态概览（亮点、组件、已知问题、测试现状） |
| [CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md) | 当前实现与下一步（构建、Pass、HLS 生成、e2e、建议行动） |

---

## docs/guide/（构建与开发指南）

| 文件 | 说明 |
|------|------|
| [BUILD_GUIDE.md](guide/BUILD_GUIDE.md) | 构建指南（统一构建、详细步骤、依赖、故障排除） |
| [CODE_STRUCTURE.md](guide/CODE_STRUCTURE.md) | 代码组织与结构 |
| [DEVELOPMENT_GUIDE.md](guide/DEVELOPMENT_GUIDE.md) | 开发指南 |

---

## docs/features/（特性与实现细节）

### Polymer

| 文件 | 说明 |
|------|------|
| [POLYMER_QUICK_START.md](features/polymer/POLYMER_QUICK_START.md) | Polymer 快速入门 |
| [POLYMER_INTEGRATION_COMPLETE.md](features/polymer/POLYMER_INTEGRATION_COMPLETE.md) | Polymer 集成完成报告 |

### Space-Time

| 文件 | 说明 |
|------|------|
| [README.md](features/spacetime/README.md) | Space-Time 特性概述 |
| [SPACETIME_IMPLEMENTATION_PLAN.md](features/spacetime/SPACETIME_IMPLEMENTATION_PLAN.md) | Space-Time 实现方案 |

### 写时重排

| 文件 | 说明 |
|------|------|
| [README.md](features/write-time-reordering/README.md) | 写时重排文档入口 |
| [WRITE_TIME_REORDERING_IMPLEMENTATION.md](features/write-time-reordering/WRITE_TIME_REORDERING_IMPLEMENTATION.md) | 写时重排实现细节 |
| [PHASE2_IMPLEMENTATION_SUMMARY.md](features/write-time-reordering/PHASE2_IMPLEMENTATION_SUMMARY.md) | Phase 2 实现总结 |
| [IMPLEMENTATION_IMPROVEMENTS.md](features/write-time-reordering/IMPLEMENTATION_IMPROVEMENTS.md) | 实现改进说明 |

---

## docs/reference/（参考资料）

| 文件 | 说明 |
|------|------|
| [PROJECT_STRUCTURE.md](reference/PROJECT_STRUCTURE.md) | 项目目录结构 |

### reference/autosa/

| 文件 | 说明 |
|------|------|
| [README.md](reference/autosa/README.md) | AutoSA 相关文档导航 |
| [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](reference/autosa/AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md) | AutoSA 源码与性能、MLIR 优化机会 |
| [AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md) | AutoSA 与 mlir-systolic 逐项对照 |
| [AUTOSA_ANALYSIS.md](reference/autosa/AUTOSA_ANALYSIS.md) | AutoSA 架构与算法分析 |
| [AUTOSA_ARCHITECTURE.md](reference/autosa/AUTOSA_ARCHITECTURE.md) | AutoSA 架构详细说明 |
| [AUTOSA_QUICK_REFERENCE.md](reference/autosa/AUTOSA_QUICK_REFERENCE.md) | AutoSA 快速参考 |
| [AUTOSA_REFERENCE_STATUS.md](reference/autosa/AUTOSA_REFERENCE_STATUS.md) | AutoSA 参考状态 |
| [AUTOSA_REFERENCE_TABLES.md](reference/autosa/AUTOSA_REFERENCE_TABLES.md) | AutoSA 参考表 |
| [comparison_with_autosa.md](reference/autosa/comparison_with_autosa.md) | 与 AutoSA 功能对比 |

### reference/allo/

| 文件 | 说明 |
|------|------|
| [README.md](reference/allo/README.md) | Allo 相关文档入口 |
| [ALLO_ANALYSIS_AND_PYTORCH_ROADMAP.md](reference/allo/ALLO_ANALYSIS_AND_PYTORCH_ROADMAP.md) | Allo 分析与 PyTorch 路线 |
| [ALLO_HLS_CODE_GENERATION_RULES.md](reference/allo/ALLO_HLS_CODE_GENERATION_RULES.md) | Allo HLS 代码生成规则 |
| [ALLO_INTEGRATION_ANALYSIS.md](reference/allo/ALLO_INTEGRATION_ANALYSIS.md) | Allo 集成分析 |

### reference/testing/

| 文件 | 说明 |
|------|------|
| [README.md](reference/testing/README.md) | 测试参考入口 |
| [REFERENCE_SAMPLES.md](reference/testing/REFERENCE_SAMPLES.md) | 参考样本 |
| [TEST_RESULTS.md](reference/testing/TEST_RESULTS.md) | 测试结果 |

---

## docs/issues/

| 文件 | 说明 |
|------|------|
| [README.md](issues/README.md) | 已知问题与注意事项 |

---

## docs/archive/（归档，历史/重复文档）

| 文件 | 说明 |
|------|------|
| [archive/README.md](archive/README.md) | 归档说明与列表 |
| [archive/DYNAMIC_ENUMERATION_ANALYSIS.md](archive/DYNAMIC_ENUMERATION_ANALYSIS.md) | 动态枚举 vs 固定模式分析（已归档） |
| [archive/DYNAMIC_ENUMERATION_VISUALIZATION.md](archive/DYNAMIC_ENUMERATION_VISUALIZATION.md) | 固定/动态枚举可视化对比（已归档） |
| [archive/QUICK_REFERENCE.md](archive/QUICK_REFERENCE.md) | 动态枚举恢复快速参考（已归档） |
| [archive/RESTORATION_REPORT.md](archive/RESTORATION_REPORT.md) | 动态枚举恢复报告（已归档） |
| [archive/IMPLEMENTATION_STATUS.md](archive/IMPLEMENTATION_STATUS.md) | 实现状态总结 2026-01（已归档，见 status/CURRENT_IMPLEMENTATION） |
| [archive/ROADMAP.md](archive/ROADMAP.md) | 实施路线图 2026-01（已归档） |
| [archive/NEXT_STEPS_ROADMAP.md](archive/NEXT_STEPS_ROADMAP.md) | 下一步路线图 2026-01（已归档） |

---

## 其他

| 文件 | 说明 |
|------|------|
| [../scripts/README.md](../scripts/README.md) | 脚本使用说明 |
| [../test/autosa_hls_refs/README.md](../test/autosa_hls_refs/README.md) | AutoSA 生成的 HLS 参考文件说明 |

---

## 使用建议

- **首次接手 / 换环境**：先看 [../PROJECT_STATUS_AND_ONBOARDING.md](../PROJECT_STATUS_AND_ONBOARDING.md)，再按需查本索引。
- **跟进近期修改**：看 [../RECENT_CHANGES_AND_NEXT_STEPS.md](../RECENT_CHANGES_AND_NEXT_STEPS.md)。
- **查某一类文档**：用本页表格按目录定位后打开对应文件。
