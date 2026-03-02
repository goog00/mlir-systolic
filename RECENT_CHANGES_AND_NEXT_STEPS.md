# 近期修改与下一步工作

> **更新日期**: 2026-03-02  
> 本文档记录本轮开发中的主要修改，并链到已有文档与后续计划。

---

## 工作区与文档整理说明

- **根目录**：新增 [RECENT_CHANGES_AND_NEXT_STEPS.md](RECENT_CHANGES_AND_NEXT_STEPS.md)（本文件），[README.md](README.md) 已加入指向本文件的链接。
- **测试**：`test/` 下保留 `minimal_matmul.mlir`、`minimal_mttkrp.mlir` 及 `run_mm_e2e.sh`、`run_mttkrp_e2e.sh`；`test/autosa_hls_refs/` 为 AutoSA 参考输出。
- **文档**：设计/策略类在 [docs/design/](docs/design/)，状态类在 [docs/status/](docs/status/)，AutoSA/Allo 分析在 [docs/reference/](docs/reference/)。完整导航见 [docs/README.md](docs/README.md)。

---

## 一、本轮完成的主要修改

### 1. 脉动阵列优化（读时/写时重排）

- **写时重排接入 drain**  
  当存在 2D 重排属性（`systolic.reorder.<array>.dims/perm`）时，`emitDrainSerialize` 按重排后的维度顺序写回 DRAM：先 unpack 入 buffer → 按重排顺序填入 buffer_linear → 再 pack 写出，实现连续 burst 写。
- **读时重排接入 L3**  
  当存在 2D 重排属性时，`emitIOL3InSerialize` 按重排维度顺序 (d0,d1) 生成读循环，假定输入已为重排布局，DRAM 读仍为顺序。
- **读/写重排与冲突说明**  
  读时重排作用于输入数组，写时重排作用于输出数组；约定一致（同一套 dims/perm），不冲突。详见 [docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md](docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md) 第 1.4 节。

**相关文档**：
- [docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md](docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md) — 代码中已有优化梳理（L2/L3/drain 接入点、写时/读时重排）
- [docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md) — 脉动阵列优化改进计划（Phase 1/2 任务清单）

### 2. 端到端测试与回归

- **MM 端到端**  
  [test/run_mm_e2e.sh](test/run_mm_e2e.sh)：opt → translate，检查生成 HLS 含 `kernel0`、`PIPELINE`、`DATAFLOW`、`PE_wrapper` 等。
- **MTTKRP 4-loop 端到端**  
  [test/minimal_mttkrp.mlir](test/minimal_mttkrp.mlir)（4-loop 8×8×8×8）+ [test/run_mttkrp_e2e.sh](test/run_mttkrp_e2e.sh)：验证 4-loop 可跑通 opt→translate。

### 3. Translate 参数化数组名（支持 MTTKRP 等）

- **从 kernel 参数推导数组名**  
  `deriveArrayNamesFromFunction`：遍历 kernel 的 memref 参数，取 `mlir.name` 或 `arg0/arg1/arg2`；前两个为输入名、最后一个为输出名（不足 3 个时退化为 A,B,C）。
- **类型/声明/kernel 全量使用推导名**  
  `emitTypeDefinitions`、`emitModuleDeclarations`、`emitTopKernel`、PE/drain/dummy 等均使用 `inputNames[0]`、`inputNames[1]`、`outputName`，支持 A/B/D 等任意命名。

### 4. 文档与清单更新

- **改进计划**  
  Phase 1 读时/写时重排、测试、文档项已勾选；Phase 2 增加 4-loop 已放行说明与 MTTKRP 测例/回归。
- **小规模配置表与写时重排接入清单**  
  见 [docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md) 第 6 节。
- **实现状态**  
  [docs/status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](docs/status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md) 中“建议的立即行动”与写时/读时重排、4-loop MTTKRP 已同步。

---

## 二、相关文档索引

| 类别     | 文档 |
|----------|------|
| 实现状态与下一步 | [docs/status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](docs/status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md) |
| 脉动阵列优化计划 | [docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md) |
| 已有优化梳理     | [docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md](docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md) |
| 单/多核与高性能策略 | [docs/design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](docs/design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md) |
| 愿景与设计目标   | [docs/VISION_AND_DESIGN_GOALS.md](docs/VISION_AND_DESIGN_GOALS.md) |
| AutoSA 对照分析  | [docs/reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](docs/reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md) |
| 文档导航         | [docs/README.md](docs/README.md) |

---

## 三、下一步工作（建议优先级）

1. **参数与多面体选择范围**  
   参数（array_part、latency、simd 等）由多面体分析给出选择范围后再选取；当前可文档化“在合法范围内手动取小”。参考 [third_party/AutoSA/docs/tutorials/getting_started.rst](third_party/AutoSA/docs/tutorials/getting_started.rst)。

2. **4-loop/MTTKRP 深化**  
   - 写时重排分析扩展至 4 维（当前 WriteTimeReorderingAnalyzer 仅 3 维）；  
   - 视需要为 MTTKRP 增加专用 PE/访存模板或 3D 写回逻辑。

3. **通用 loop body migration**  
   支持将任意 4-loop/5-loop 计算体正确迁移到 PE 与 IO 结构，为更多 kernel（CNN、TTMc 等）打基础。

4. **FIFO 深度与 HLS 友好化**  
   FIFO 深度由 dataflow 推导；pipeline 内减少 %、/ 等不利于 HLS 的形式（见改进计划 Phase 4）。

---

## 四、快速验证命令

```bash
# 构建（若已构建可跳过）
./scripts/build-systolic.sh

# MM 端到端回归
./test/run_mm_e2e.sh

# MTTKRP 4-loop 端到端回归
./test/run_mttkrp_e2e.sh
```
