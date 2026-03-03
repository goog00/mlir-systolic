# 脉动阵列优化改进计划

> **目的**：延续在 AutoSA 生成的 HLS 上对 MTTKRP 的手工优化成果，在 mlir-systolic 中系统化实现**脉动阵列优化**（多种手段，见下）；优先涉及随机读/写突出的 MTTKRP、TTMc，并兼顾多 kernel 场景下的 MM、CNN。**初期在小规模脉动阵列下进行**，便于代码审查、HLS 综合与上板验证。

---

## 1. 背景与目标

- **既有成果**：在 AutoSA 生成的 MTTKRP HLS 框架下手工修改（含**写时重排**等），已取得性能提升；该修改未回馈到 AutoSA 仓库。写时重排是**其中一种**已验证的优化方式，而非脉动阵列优化的全部。
- **脉动阵列优化**包含多种手段，例如：
  - **访存/布局**：写时重排（write-time reorder）、DRAM 读顺序与 stride-1 化、layout 变换与 host 重排、片上 transpose/gather。
  - **通信与缓冲**：FIFO 深度由 dataflow 推导、双缓冲显式化、backpressure 缓解。
  - **Reduction 与写回**：reduction 自动识别、drain/merge 结构化、写回 burst 化。
  - **HLS 友好化**：pipeline 内去除 %、/，strength reduction，pack/unpack 规范，分支外提。
  - **参数与调优**：space_time、array_part、latency、simd 等由**多面体分析给出选择范围**，再在范围内选取（见第 3 节）。
- **本计划目标**：
  - 将上述优化手段逐步沉淀为 **mlir-systolic 的可复用能力**（分析 + 代码生成）；写时重排作为其中之一优先接入。
  - 优先覆盖 **随机读/写突出的 kernel**：MTTKRP、TTMc。
  - 同时为 **多 kernel 共用** 打基础：MM、CNN 的小规模验证与优化。
  - **小规模优先**：在**多面体分析给出的参数选择范围内**选取较小配置，便于审查、综合与上板。

---

## 2. 优化范围与优先级

| 优先级 | Kernel | 主要问题 | 优化方向（多种手段） | 规模建议（初期） |
|--------|--------|----------|----------------------|------------------|
| **P0** | MTTKRP | 随机读/跨步访问；写回非连续 | 写时重排（已验证）；DRAM 读顺序/布局；FIFO/II | 在分析得到的 tilable 范围内取小 I/J/K/L、小 PE |
| **P0** | TTMc | 类似 MTTKRP 的访存模式 | 同上 | 同上 |
| **P1** | MM | 多 kernel 主形态；当前已有模板 | 写时重排接入；布局/coalescing；FIFO/II 微调 | 在分析范围内取小 N/M/K、小 PE |
| **P1** | CNN | 多 kernel；5-loop + reduction | reduction 与写回；小规模 5-loop 支持 | 在分析范围内取小 O,R,C,I,K |

---

## 3. 参数与“小规模验证”：依多面体分析的选择范围

**参数不能依赖单一全局预设**：array_part、latency、simd、space_time 等与**具体输入的循环界、依赖、调度**相关，同一组固定参数并不适合所有 kernel。在 AutoSA 中，参数是**通过多面体分析逐步得到选择范围**的（见 `third_party/AutoSA/docs/tutorials/getting_started.rst`）：

- **space_time**：先选 spacetime 模式，得到候选阵列形态。
- **array_part**：分析得到 **tilable_loops** 及各维上界（如 [64,64,64]）；tiling 因子需 ≤ 各维上界，且为循环界的子倍数。
- **latency**：在 array_part 之后，得到当前可做 latency hiding 的 **tilable_loops**（如 [16,16]），再在该范围内选因子。
- **simd**：分析得到可向量化循环及上界、是否需 layout 变换等，再在 **tilable_loops** 与 **legal** 约束下选取。

因此，**“小规模验证”** 应理解为：

- **在给定输入上**，先通过多面体分析（或等价地，由 SystolicTransform / 分析 pass）得到各步的**选择范围**（如 tilable_loops、合法 spacetime 列表）。
- 在该范围内**选取较小的参数**（如较小的 tiling 因子、较小的 PE 规模），以便快速综合与上板；而不是对任意输入强制使用同一组固定数值。
- 若当前 mlir-systolic 尚未输出“选择范围”接口，可先用手动指定参数，但应保证参数对当前输入合法（例如 tiling 能整除循环界）；后续再增加“分析 → 输出可选范围”的流程，与 AutoSA 的 tuning.json / 分步 manual 模式对齐。

---

## 4. 改进阶段划分

### Phase 1：写时重排与 MM 小规模闭环（当前周期）

**目标**：写时重排分析结果**完整接入代码生成**；MM 在小规模下可端到端跑通并可用于审查与综合。

1. **写时重排接入 codegen（MM 模板）**
   - 现状：`systolic.reorder.*.dims/perm` 已在 DataflowGeneration 中写入；translate 在 **L2 intra/inter trans** 中已使用 `getArrayDims`、`applyAccessPermutation`。
   - 缺口：**L3_in_serialize**（DRAM 读）、**drain_IO_L3_out_serialize**（DRAM 写）尚未按重排后的布局生成访问顺序。
   - 动作：
     - **L3_in_serialize**：当存在 reorder 属性时，按重排后的逻辑维度顺序生成读循环，使 DRAM 读顺序与重排布局一致（stride-1 或更连续）。
     - **drain_IO_L3_out_serialize**：当存在 reorder 属性时，按重排后的维度顺序生成写循环，使写回为连续 burst（写时重排的核心）。
   - 验收：对带 reorder 属性的 MM 生成 HLS，对比无 reorder 版本；写回循环应按 perm 后的维度顺序迭代。

2. **小规模验证与回归**
   - 参数由**多面体分析给出的选择范围**确定（或当前阶段由用户在合法范围内手动指定）；不对所有输入使用单一全局预设。
   - 提供 **小规模 MM 测例**（如 `test/minimal_matmul.mlir`），在**该输入对应的合法参数范围**内选取较小值（如 array_part/latency 与循环界 32 兼容），用于 L1（只生成 HLS）与 L2（C sim，可选 L3 综合）。

3. **文档与清单**
   - 在本文档中维护“小规模配置表”与“写时重排接入清单”；与 [SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md) 对齐。

### Phase 2：MTTKRP / TTMc 支持与写时重排

**目标**：4-loop MTTKRP（及同类 TTMc）能通过 mlir-systolic 生成 HLS，且写时重排/布局优化应用于读、写路径。

1. **4-loop 支持**
   - **SystolicTransform**：允许 band.size() ≥ 4（当前已有 ≥3；确认 4-loop 不早退）。
   - **SystolicDataflowGeneration**：对 4 维数组与 4 维循环的引用分组、IO/PE/Drain 分类、reduction 维标注；写时重排分析支持 4 维（若当前仅 3 维则扩展）。
   - **systolic-translate**：增加 **MTTKRP 模板分支**（或通用 4 数组/4-loop 分支）：识别 4-loop + 4 数组（如 A,B,C,D 或 D 为输出），生成 D 的 drain 与 A/B/C 的 IO；在 L3 serialize 与 drain serialize 中应用 reorder（与 Phase 1 逻辑一致）。

2. **MTTKRP 小规模测例**
   - 提供小规模 MTTKRP 的 Affine MLIR（I,J,K,L=16 或 32）；在分析得到的参数选择范围内选取较小值生成 HLS，用于审查与综合。

3. **TTMc**
   - 与 MTTKRP 同类访存特征；复用 4-loop 与写时重排逻辑，增加 TTMc 小规模测例。

### Phase 3：MM/CNN 多 kernel 与小规模验证

**目标**：MM 与 CNN 在小规模下均可生成、综合、上板；为多 kernel 共用参数打基础。

1. **MM**：已在 Phase 1 小规模闭环；本阶段做综合/上板验证与 II、FIFO 深度等微调。
2. **CNN**：5-loop 支持（允许 band.size()≥5）、一种 spacetime（如 ST0）、4D array_part/latency；translate 中 CNN 模板（cin/w/cout）；小规模 O,R,C,I,K。
3. **多 kernel**：仅预留接口或文档说明（共用参数求取见 [SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md)），不做本阶段实现重点。

### Phase 4：性能与鲁棒性

- FIFO 深度由 dataflow 推导（替代固定 2）。
- Pipeline 内去除 %、/ 等 HLS 不友好形式。
- 与 AutoSA 参考（`test/autosa_hls_refs`）做结构/指标对比。

---

## 5. 实现任务清单（可裁剪为 issue）

### Phase 1（当前）

- [x] **读时重排**：在 `emitIOL3InSerialize` 中，当存在 2D reorder 属性时，按重排后的维度顺序 (d0,d1) 生成 DRAM 读循环（假定输入为重排布局，读仍顺序）。
- [x] **写时重排**：在 `emitDrainSerialize` 中，当存在 reorder 属性时，按重排后的维度顺序生成 DRAM 写循环，实现写时重排。
- [x] **参数与选择范围（文档）**：已新增 [PARAMETER_SELECTION_AND_VALID_RANGE.md](PARAMETER_SELECTION_AND_VALID_RANGE.md)，说明“在合法范围内手动取小”及 AutoSA 参考；后续可增加分析输出可选范围的接口。
- [x] **测试**：小规模 MM 端到端（opt → translate，参数对应当前输入合法范围）；已提供 `test/run_mm_e2e.sh` 做回归（检查生成 cpp 含 kernel0、PIPELINE、DATAFLOW、PE_wrapper 等）；可选 C sim / 综合待做。
- [x] **文档**：将“参数来自分析范围”与“写时重排接入点”更新到本文档与 [CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](../status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md)（见下文小规模配置表与写时重排接入清单）。

### Phase 2

- [x] **4-loop（Transform 已放行）**：SystolicTransform 要求 band.size() ≥ 3（`applyMultiLevelTiling`），故 4-loop 已可通过；缺口在 DataflowGeneration 与 translate 对 4-loop/4 数组的模板与 reorder 支持。
- [x] **写时重排**：WriteTimeReorderingAnalyzer 已支持 2D/3D 数组；translate 端 drain 已增加 3D 重排路径（`hasReordering3D` + Phase 1/2/3）；若需 4 维数组可再扩展。
- [x] **MTTKRP 模板（参数化数组名）**：translate 从 kernel 函数参数推导数组名（2 输入 + 1 输出），支持 A/B/D 等任意命名；4-loop 已跑通 opt→translate。
- [x] **MTTKRP 小规模测例**：`test/minimal_mttkrp.mlir`（4-loop 8×8×8×8）+ `test/run_mttkrp_e2e.sh` 回归；L3/drain 仍用当前 2D 模板（3 数组共用），reorder 支持 2D。

### Phase 3

- [ ] **CNN 5-loop**：合法性、spacetime、4D 配置、translate CNN 模板。
- [ ] **MM/CNN 小规模**：综合与上板验证说明。

---

## 6. 小规模验证时参数选取原则

参数**不可**对所有输入使用同一组固定值；应**在对应输入的多面体分析给出的选择范围内**选取。参考 AutoSA：`third_party/AutoSA/docs/tutorials/getting_started.rst`。

| 步骤 | AutoSA 行为 | mlir-systolic 目标 |
|------|-------------|--------------------|
| space_time | 枚举候选阵列，用户或启发式选择 | 同：由分析得到候选，再选其一 |
| array_part | 输出 tilable_loops 及上界（如 [64,64,64]），tiling 因子 ≤ 上界且为循环界子倍数 | 同：分析得到范围，在范围内取小（小规模验证） |
| latency | 输出当前 tilable_loops（如 [16,16]），再选因子 | 同 |
| simd | 输出 tilable_loops、legal、scores 等，再选因子 | 同 |

小规模验证时：在**当前输入的**合法范围内，选取较小的 tiling 与 PE 规模（例如循环界 32 时可选 array_part=[8,8,8]、latency=[4,4] → PE 2×2），以利综合与上板。

### 小规模配置表示例（MM 32×32）

| 输入 | 循环界 | 示例合法参数（手动指定） | 回归命令 |
|------|--------|---------------------------|----------|
| test/minimal_matmul.mlir | 32,32,32 | --size=32 --array-part=8 --latency=4 --simd=1 | `./test/run_mm_e2e.sh` |

### 写时重排接入清单（2D）

| 模块 | 接入点 | 说明 |
|------|--------|------|
| L2 | getArrayDims / applyAccessPermutation | 用于 intra/inter trans 的 local 缓冲区维度与访问下标 |
| L3_in_serialize | hasReordering2D 时嵌套 (d0,d1) 读 | 假定输入为重排布局，DRAM 读仍顺序 |
| drain_serialize | hasReordering2D 时 buffer→buffer_linear→pack 写 | 按重排顺序写回 DRAM |

---

## 7. 相关文档

- **AutoSA 参数推导**：`third_party/AutoSA/docs/tutorials/getting_started.rst`（多面体分析逐步得到 space_time、array_part、latency、simd 的选择范围与 manual 模式）
- [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](../reference/autosa/AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md)（MTTKRP 随机读与多种优化建议）
- [SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md)（小规模验证、高性能模板）
- [AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](../reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md)（对照与打通）
- [CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](../status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md)（实现状态）
