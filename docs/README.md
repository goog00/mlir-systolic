# mlir-systolic 文档导航

> **最后更新**: 2026-03  
> 根目录 **[../PROJECT_STATUS_AND_ONBOARDING.md](../PROJECT_STATUS_AND_ONBOARDING.md)** 为新环境/新 Agent 快速上手（做了啥、如何验证、下一步、服务器环境说明）。

---

## 📋 快速入口

| 文档 | 说明 |
|------|------|
| **[../PROJECT_STATUS_AND_ONBOARDING.md](../PROJECT_STATUS_AND_ONBOARDING.md)** | ⭐ 项目状态与上手指南（首选） |
| **[../RECENT_CHANGES_AND_NEXT_STEPS.md](../RECENT_CHANGES_AND_NEXT_STEPS.md)** | 近期修改与下一步、文档索引 |
| **[../README.md](../README.md)** | 项目概述、构建、目录结构 |
| **[BUILD_AND_SERVER_ENVIRONMENT.md](BUILD_AND_SERVER_ENVIRONMENT.md)** | 构建依赖与 Ubuntu 18.04/老 LLVM 说明 |

---

## 📚 核心文档

- **[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)** — 系统架构
- **[VISION_AND_DESIGN_GOALS.md](VISION_AND_DESIGN_GOALS.md)** — 愿景与设计目标
- **[guide/BUILD_GUIDE.md](guide/BUILD_GUIDE.md)** — 构建和安装
- **[guide/DEVELOPMENT_GUIDE.md](guide/DEVELOPMENT_GUIDE.md)** — 开发指南
- **[guide/CODE_STRUCTURE.md](guide/CODE_STRUCTURE.md)** — 代码组织
- **[reference/PROJECT_STRUCTURE.md](reference/PROJECT_STRUCTURE.md)** — 项目目录结构
- **[DOCS_INDEX.md](DOCS_INDEX.md)** — **全量文档索引**（所有 .md 列表与说明）

---

## 📊 项目状态

- **[status/PROJECT_STATUS.md](status/PROJECT_STATUS.md)** — 当前状态
- **[status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md)** — 实现状态与下一步
- 历史路线图与实现状态已归档至 **[archive/](archive/README.md)**

---

## 🎯 设计策略与优化

- **[design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md)** — 单/多核与高性能策略
- **[design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md)** — 脉动阵列优化改进计划
- **[design/EXISTING_OPTIMIZATIONS_IN_CODE.md](design/EXISTING_OPTIMIZATIONS_IN_CODE.md)** — 代码中已有优化梳理
- **[design/PARAMETER_SELECTION_AND_VALID_RANGE.md](design/PARAMETER_SELECTION_AND_VALID_RANGE.md)** — 参数选择与合法范围
- **[design/FIFO_DEPTH_AND_PERFORMANCE_NEXT.md](design/FIFO_DEPTH_AND_PERFORMANCE_NEXT.md)** — FIFO 深度与性能下一步
- **[design/SHIFT_VS_DIV_MOD_AND_HLS_CORRECTNESS.md](design/SHIFT_VS_DIV_MOD_AND_HLS_CORRECTNESS.md)** — 移位/除取模与 HLS 正确性
- **[design/L3_COALESCE_AND_ACCESS_PATTERN.md](design/L3_COALESCE_AND_ACCESS_PATTERN.md)** — L3 coalesce 与访问模式
- **[design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md](design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md)** — L3/写时重排与 host-serialize（暂不做）

---

## 🔍 参考资料

- **[reference/autosa/](reference/autosa/)** — AutoSA 分析与对照（含 AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md、AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md）
- **[reference/allo/](reference/allo/)** — Allo 分析与 PyTorch 路线
- **[reference/testing/](reference/testing/)** — 测试参考

---

## 🧩 特性与实现细节

- **features/polymer/** — Polymer 集成
- **features/spacetime/** — Space-Time 架构
- **features/write-time-reordering/** — 写时重排实现
- **issues/README.md** — 已知问题（若存在）

---

## 📖 使用建议

- **第一次接触 / 换环境**：先读 [../PROJECT_STATUS_AND_ONBOARDING.md](../PROJECT_STATUS_AND_ONBOARDING.md)，再按需查本页链接。
- **跟进近期修改**：读 [../RECENT_CHANGES_AND_NEXT_STEPS.md](../RECENT_CHANGES_AND_NEXT_STEPS.md)。
- **对比 AutoSA**：见 [reference/autosa/](reference/autosa/) 与根目录 `test/autosa_hls_refs/`。
