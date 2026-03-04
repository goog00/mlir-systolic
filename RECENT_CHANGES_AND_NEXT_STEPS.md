# 近期修改与下一步工作

> **更新日期**: 2026-03-04  
> 本文档记录本轮开发中的主要修改，并链到已有文档与后续计划。

---

## 0. 2026-03-04 语义正确性结论（新增）

- **当前优先级已切换为语义正确性**：先让标准语义 `mttkrp/ttmc` 正确，再做性能对比。
- **标准 MTTKRP（2D 输出）已复现 csim 失败**：`hls_validation/mttkrp_std_mlirsystolic/CSIM_FINDINGS_2026-03-04.md` 记录了稳定可复现的 mismatch（典型 `hw=8, ref=64`），说明当前 PE 规约骨架尚未覆盖双规约语义。
- **TTMc（3D 输出）当前不支持**：`systolic-translate` 已加入保护性校验，输出 rank 非 2 时直接报错，避免继续生成语义错误代码。
- **新增设计结论文档**：见 [docs/design/CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md](docs/design/CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md)，包含 AutoSA 与当前生成逻辑差异、以及分阶段通用化路线（`mttkrp` 先行、`ttmc` 次之）。
- **回归基线状态**：`test/run_all_e2e.sh` 保持可回归；其中标准 `ttmc` / 3D reorder 相关脚本在当前阶段按“命中预期不支持错误即通过”判定。

### 双规约 PE 路径（num_time_loops ≥ 2）

- **语义属性**：`SystolicTransform` 在 kernel 上写入 `systolic.num_time_loops`；`systolic-translate` 读入后填入 `ContractionDesc.numReductions`，用于区分单规约（MM）与双规约（标准 MTTKRP）。
- **PE 双规约循环**：当 `contraction.numReductions >= 2` 时，在 PE 的 c2 与 c5 之间插入第二规约维循环 `r1`（0..size-1）；累加器在 `r1==0 && c2==0 && c5==0` 时清零，在 `r1==size-1 && c2==numTiles-1 && c5==c5Bound-1` 时写出，使每输出点完成 K×L 次累加（如 8×8=64）。
- **IO 与 FIFO 对齐**：为保持 DATAFLOW 下 FIFO 读写一致，在双规约时对以下模块同样增加 r1 循环（或 r1 次调用）：`emitDummyModules`、`emitIOL2In`、`emitIOL2InBoundary`、`emitIOL3In`、`emitIOL3InSerialize`（coalesced 分支）。
- **L3_serialize 按输入维度区分 3D/2D**：emitter 增加 `inputShapes`（由 kernel 的 memref 参数形状填充），并传入 `emitIOL3InSerialize`。双规约且该输入为 **3D**（如 MTTKRP 的 A[I,K,L]）时，coalesced 分支按 **r1 = 第三维切片** 读 DRAM：`word_idx = r1 * wordsPerPlane + (c0,c1,c3,c4g)`，其中 `wordsPerPlane = (dim0*dim1*4)/64`，使每个 r1 读到不同平面。2D 输入（B、C）仍按原公式（同一数据可被下游按 r1 重复使用）。
- **ContractionDesc 驱动 PE 条件**：`ContractionDesc` 增加 `hasExtraReductionLoop()`（即 `numReductions >= 2`）；emitter 中所有“双规约”分支统一用该接口。PE 的累加器清零与写出条件由 `emitPEInitCondition()` / `emitPEDrainCondition()` 根据 `contraction` 生成，便于后续扩展更多规约维。
- **ContractionDesc.Kind 与 rank-3 输出**：`ContractionDesc` 增加 `Kind`（MatmulLike / MttkrpLike / TtmcLike / Unsupported）。根据 `outputRank` 与 `num_time_loops` 设置；仅当 `Kind::Unsupported` 时报错（rank>3，或 rank-3 且 num_time_loops>2）。emitter 增加 `outputShape`，用于 drain serialize。**rank-3 且无写时重排**时，`emitDrainSerialize` 走 3D 顺序写回路径（fifo 顺序 = row-major i,j,k）。TTMc 当前为 5 循环、num_time_loops=3，仍视为 Unsupported，e2e 保持“预期失败”；后续支持 3 规约维后可改为 TtmcLike 并生成完整 3D 输出。
- **验证**：`run_all_e2e.sh` 全量通过（MM 单规约、标准 MTTKRP 双规约路径、写时重排 2D/3D、TTMc 预期失败）。
- **Transform 与 4 循环**：① **spaceTimeMode -1 视为 3**：调用 selectSpaceLoops 前将 -1 规范为 legacyMode=3（[i,j] 2D），避免 default 失败。② **4+ 循环 time = 剩余**：legacy selectSpaceLoops 中 time 改为“所有非 space 的循环”，故 4 循环得 space [0,1]、time [2,3]，num_time_loops=2。③ **num_time_loops 提前设置**：在 space-time 选择成功后、tiling 之前即写 `systolic.num_time_loops`，这样 tiling 失败（如 size=8 与默认 array_part=16）时 translate 仍能读到双规约。④ **4 循环不进入 tiling**：applyMultiLevelTiling 在 band.size()>3 时直接 failure，避免 4 维时 latency/arrayPart 越界；Level2 的 tileSizes2 对 i>=latency.size() 且 i>=arrayPart.size() 时用 1。
- **排错策略**：现阶段**先在本地开发**，通过**分析生成 HLS 文件内容**做初步排错；阶段性工作结束后再**统一在服务器上进行 HLS/csim 测试与具体调错**。本地可借助 [test/inspect_generated.sh](test/inspect_generated.sh) 与下方「生成文件本地检查清单」做快速自检。

### 生成文件本地检查清单

在未跑 HLS 的环境下，可通过以下方式对生成 `.cpp` 做初步自检（生成文件一般在 `/tmp/*_e2e_out.cpp`，由各 e2e 脚本写出）：

| 检查项 | 含义 | 建议 |
|--------|------|------|
| 出现 `r1` | 双规约路径是否生效（MTTKRP 应有，MM 通常无） | MTTKRP_std 生成中应有 r1；若为 0 可检查中间 MLIR 是否带 `systolic.num_time_loops` 及 kernel 选择 |
| `word_idx.*r1` | L3_serialize 对 3D 输入是否按 r1 切片读 | 仅 3D 输入（如 MTTKRP 的 A）对应模块应有 |
| 注释 `r1 = plane` | 3D 数组按平面读的注释 | 同上 |
| `buffer_linear` | 写时重排路径是否生效 | 写时重排测例生成中应有 |
| `PIPELINE II=1` / `DATAFLOW` | 基本 HLS pragma 是否存在 | 各 kernel 生成中应有 |

运行 `./test/inspect_generated.sh` 会先执行全量 e2e，再对上述模式做计数并打印，便于快速确认结构是否符合预期。HLS csim/综合留待阶段性工作结束后在服务器统一进行。

---

## 工作区与文档整理说明

- **根目录**：[RECENT_CHANGES_AND_NEXT_STEPS.md](RECENT_CHANGES_AND_NEXT_STEPS.md)（本文件）记录近期修改与下一步；**[PROJECT_STATUS_AND_ONBOARDING.md](PROJECT_STATUS_AND_ONBOARDING.md)** 为**新环境/新 Agent 上手指南**（做了啥、如何验证、下一步、服务器/Ubuntu18 说明）。[README.md](README.md) 已加入指向本文件的链接。
- **测试**：`test/` 下保留 `minimal_matmul.mlir`、`minimal_mttkrp.mlir` 及 `run_mm_e2e.sh`、`run_mttkrp_e2e.sh`、`run_all_e2e.sh`；`test/autosa_hls_refs/` 为 AutoSA 参考输出。
- **文档**：设计/策略类在 [docs/design/](docs/design/)，状态类在 [docs/status/](docs/status/)，AutoSA/Allo 分析在 [docs/reference/](docs/reference/)。**全量文档索引**见 [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md)；过时/重复文档已归档至 [docs/archive/](docs/archive/README.md)。导航入口 [docs/README.md](docs/README.md)。

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

- **全量 e2e**  
  [test/run_all_e2e.sh](test/run_all_e2e.sh)：依次执行 MM、MTTKRP、写时重排(2D)、写时重排(3D) 四个 e2e，汇总通过/失败，建议在改代码后跑一次。
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
| 参数选择与合法范围 | [docs/design/PARAMETER_SELECTION_AND_VALID_RANGE.md](docs/design/PARAMETER_SELECTION_AND_VALID_RANGE.md) |
| FIFO 深度与性能下一步 | [docs/design/FIFO_DEPTH_AND_PERFORMANCE_NEXT.md](docs/design/FIFO_DEPTH_AND_PERFORMANCE_NEXT.md) |
| 已有优化梳理     | [docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md](docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md) |
| 单/多核与高性能策略 | [docs/design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](docs/design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md) |
| 愿景与设计目标   | [docs/VISION_AND_DESIGN_GOALS.md](docs/VISION_AND_DESIGN_GOALS.md) |
| AutoSA 对照分析  | [docs/reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](docs/reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md) |
| 移位/除取模与正确性 | [docs/design/SHIFT_VS_DIV_MOD_AND_HLS_CORRECTNESS.md](docs/design/SHIFT_VS_DIV_MOD_AND_HLS_CORRECTNESS.md) |
| L3 coalesce 设计  | [docs/design/L3_COALESCE_AND_ACCESS_PATTERN.md](docs/design/L3_COALESCE_AND_ACCESS_PATTERN.md) |
| L3/写时重排与 host-serialize | [docs/design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md](docs/design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md) |
| 文档导航         | [docs/README.md](docs/README.md) |

---

## 三、本轮后续完成

### 写时重排分析 2D 支持

- **WriteTimeReorderingAnalyzer 支持 2D 数组**  
  `computeReordering` 现支持 2 维：非线性别在 dim0 时交换为 [dim1,dim0]，否则保持 [dim0,dim1]，便于 2D 输出（如 MM 的 C）从分析得到重排建议。3D 逻辑不变。
- **DataflowGeneration 调试输出**  
  存储 reorder 属性时的 LLVM_DEBUG 改为按 `reorderedDims`/`dimPermutation` 长度循环打印，兼容 2D/3D。

### Drain 写回 3D 重排（translate）

- **hasReordering3D / getLinearIndexFromReordered3D / getOriginalIndexExprs3D**  
  在 `systolic-translate` 中增加 3D 重排检测与索引换算，供 3D 输出数组写时重排使用。
- **emitDrainSerialize 的 3D 分支**  
  当输出数组存在 3D 重排属性时：Phase 1 从 FIFO 解包到 `buffer[s0][s1][s2]`（按计算顺序）；Phase 2 按重排后的 (d0,d1,d2) 顺序填入 `buffer_linear`；Phase 3 再 pack 写回 DRAM，实现 3D 写时重排。MTTKRP 等 3D 输出（如 D[I,J,L]）在分析给出 3D reorder 时可走该路径。

---

## 四、下一步工作（当前优先级）

- **全量 e2e 已通过**：`./test/run_all_e2e.sh`（MM、MTTKRP、写时重排 2D、写时重排 3D）已全部通过，可在此基础上推进 (3)(4)。
- **优先推进**  
  - **(3) 4-loop/MTTKRP 深化**：① **3D 写回验证**：新增 [test/minimal_reorder_write_3d.mlir](test/minimal_reorder_write_3d.mlir)（D[i*8+j,k,l]）与 [test/run_reorder_3d_e2e.sh](test/run_reorder_3d_e2e.sh)，用于验证 `emitDrainSerialize` 的 3D 重排路径。② **3 输入模板**：translate 已支持 3 输入 + 1 输出（`deriveArrayNamesFromFunction` 在 4 个 memref 时取前 3 个为 inputNames、最后 1 个为 outputName）；`emitTopKernel`、`emitPE`、`emitPEWrapper`、`emitDummyModules` 均按 `inputNames.size()` 循环生成 L3/L2/PE FIFO 与调用；PE 计算为所有输入的乘积（in0*in1 或 in0*in1*in2）。  
  - **(4) FIFO 深度与 HLS 友好化**：① **FIFO 深度可配置**：`systolic-translate` 新增 `--fifo-depth`（默认 2），与 AutoSA 一致。② **RESOURCE 系统化（已完成）**：所有 FIFO 声明补全 `#pragma HLS RESOURCE variable=... core=FIFO_SRL`（含输入 L3 序列化 FIFO）；PE/drain 局部数组已使用 `core=RAM_2P_BRAM`。③ **Pipeline 内 %、/ 强度削减（已完成）**：当 `array_part`、`latency` 或重排维度 s1/s2 为 2 的幂时，生成代码用位运算替代取模/除法（`x % N` → `x & (N-1)`，`x / N` → `x >> log2(N)`），有利于 HLS 达到 II=1、提高频率；涉及 L3 serialize 的 `split_idx`、drain inter_trans 的 `split_idx` 与 `c6/latency`、以及 2D/3D 写时重排中的 idx→(r,c) 或 (r0,r1,r2) 分解。④ **HLS 声明/定义一致**：L2 模块的 local 数组声明与定义、调用方（IO_L2_in / IO_L2_in_boundary）的 ping/pong 声明均改为使用 `getArrayDims()`，统一为 `[d0][d1][d2]`，避免声明与定义维度不一致。⑤ **L3 访问与 coalesce（进行中）**：L3_in_serialize 无重排分支改为按 **(c0,c1,c3,c4g)** 与显式 **word_idx** 顺序读 DRAM，与 L3_in 的 tile 顺序一致，便于 burst；设计见 [docs/design/L3_COALESCE_AND_ACCESS_PATTERN.md](docs/design/L3_COALESCE_AND_ACCESS_PATTERN.md)。下一步：与 autosa_hls_refs 逐项对比、在服务器上做 C sim/综合验证。
- **暂缓**  
  - **(2) 参数与多面体选择范围**：小规模测试与已知 kernel 下参数一般无问题，且已有 AutoSA 可参考；待需要再接入“分析给出选择范围”的流程。  
  - **HLS 综合/上板验证**：本机无 Xilinx HLS/XRT 环境，留待在服务器上做综合与功能验证。
- **Host-serialize**：**暂不支持**；在各级 IO 处理顺序与复用，避免 host 重排与传输成本。见 [L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md](docs/design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md)。

---

## 四（续）、不考虑 host-serialize 时的后续工作

在**不**做 host-serialize 的前提下，当前可推进的后续工作如下（按优先级）：

1. **与 autosa_hls_refs 逐项对比**  
   对同一 kernel（如 MM）、相近 size/array_part/latency，对比我们生成与 AutoSA 生成的：模块划分、FIFO 数量与位宽、PIPELINE/RESOURCE 分布、drain 与 L3 访问形式；列出差异并优先补对 II/频率/面积影响大的部分。

2. **服务器上 C sim + 综合验证**  
   在具备 Xilinx HLS/XRT 的环境做：生成 testbench 或复用 AutoSA 的 host 调用方式；C sim 验证数值正确性；综合看 II、频率、资源；必要时与 AutoSA 同参数结果对比。

3. **可选：L3/drain 的 BURST、BIND_STORAGE 等 pragma**  
   在 L3 读/写处加 `#pragma HLS BIND_STORAGE` 或 burst 相关提示，进一步对齐 AutoSA 的访存约束（在逐项对比后按需做）。

4. **可选：更多 kernel 与测例**  
   如简单 CNN、TTMc 等，扩展测例与 e2e 脚本，保证模板与写时重排的普适性。

5. **代码清理**（已做：systolic-translate）  
   已移除未使用接口：`getTypeName`、`getName`、`getLinearIndexFromReordered3D` 及成员 `valueNames`/`valueCounter`，消除编译 warning。清理后需重新构建并跑 `./test/run_all_e2e.sh` 确认通过。

### (3) 与 (4) 的先后顺序（基于当前代码）

- **(3) 涉及**：`systolic-translate` 中目前为**固定 2 输入 + 1 输出**（`inputNames[0/1]`、`outputName`），`emitTopKernel` / PE / drain 等全部写死 in0、in1、out。4-loop/MTTKRP 深化会扩展为支持 3 输入（或更多）、3D 输出写回路径等，即**扩展模板结构**（更多 FIFO、更多 L3/L2/PE 连接）。
- **(4) 涉及**：FIFO 深度当前在 `emitTopKernel` 中**全部硬编码为 depth=2**（约 10+ 处）；pipeline 内存在 `%`、`/`、`ceil(log2(...))` 等 HLS 不友好形式，需在 translate 或生成逻辑中做 strength reduction/外提。即**在现有结构上做优化**（深度由 dataflow 推导、生成代码更友好）。
- **顺序建议：先 (3) 再 (4)**  
  - 若先做 (4)：先为当前 2-in-1-out 做 FIFO 深度推导与 HLS 友好化；再做 (3) 时会出现**第三路输入**及一批新 FIFO，又要在新结构上再做一遍深度与友好化，**(4) 要做两轮**。  
  - 若先做 (3)：先把 3-in-1-out（及 3D 写回等）模板在 translate 里铺好，**(4) 只需在“最终”结构上做一次**：深度推导覆盖所有 FIFO，HLS 友好化针对完整生成代码即可。  
- **依赖与冲突**：两者改动的代码区域不同——(3) 主要是模板形态与数组/FIFO 数量，(4) 是深度数值与 pipeline 内运算形式；**无直接冲突**。先 (3) 可避免 (4) 的重复扩展，故推荐**先 (3) 后 (4)**。

---

## 五、写时重排验证（本轮新增）

- **分析器**：`WriteTimeReorderingAnalyzer` 现同时检查 **store 索引**（含 `affine.apply`），便于检测输出数组的非线性写。
- **前置 Pass**：新增 `--systolic-write-reorder-analysis`，需在 **transform 之前** 运行（transform outline 后 store 的 apply 会消失），用于在 kernel 上打 `systolic.reorder.*` 属性。
- **测例**：`test/minimal_reorder_write.mlir`（C[i*32+j][k] 写，触发 2D 重排）；`test/run_reorder_e2e.sh` 使用 `--systolic-write-reorder-analysis --systolic-transform --systolic-dataflow-generation` 并检查生成 HLS 含 `buffer_linear`。
- **Translate 2D**：drain 的 2D 重排从 reorder 信息读取 s0/s1，支持非方阵（如 1024×32）。

**Translate 修正**：若模块中第一个函数是私有函数（如 `@S0`），原先会错误地对该函数取 reorder 信息，导致 `arrayReordering` 为空、不生成 `buffer_linear`。已改为**优先选用带 `systolic.reorder.*.dims` 的 kernel 函数**，其次非 private，再退化为第一个函数；`extractReorderingInfo` / `deriveArrayNamesFromFunction` / `emitFunc` 均使用该 kernel。写时重排 2D e2e 已通过。**3D 路径**：当 MLIR 中为恒等置换（dims 与 original 相同、perm=[0,1,2]）时原先 `needsReordering()` 为 false 导致不进入 3D 分支；已改为 `hasReordering3D()` 仅判断是否存在 3D dims/perm 属性，确保 3D 输出仍走 buffer_linear 路径；全量 e2e（含写时重排 3D）已通过。

**本地验证**：先执行 `./scripts/build-systolic.sh` 再运行 `./test/run_reorder_e2e.sh`。若失败：  
- 看脚本提示是否在 `/tmp/reorder_e2e_out.mlir` 中找到 `systolic.reorder`。  
- 可单独验证分析 pass 是否打上属性：  
  `./build/bin/systolic-opt test/minimal_reorder_write.mlir --systolic-write-reorder-analysis -o - 2>/dev/null | grep systolic.reorder`  
  若无输出则问题在分析/ pass；若有属性但 e2e 仍无 buffer_linear 则检查 translate 是否选对 kernel（见上）。

---

## 六、头文件统一说明

- **WriteTimeReorderingAnalysis** 仅保留 **include/systolic/Analysis/WriteTimeReorderingAnalysis.h** 作为唯一头文件；**lib/Analysis/WriteTimeReorderingAnalysis.h** 已删除，避免与 include 版本重复（实现以 include 中的 `storeOps`/`loadOps` 等为准）。

---

## 七、快速验证命令

```bash
# 构建（若已构建可跳过）
./scripts/build-systolic.sh

# 全量 e2e 测试（推荐：一次跑完 MM、MTTKRP、写时重排）
./test/run_all_e2e.sh

# 或单独运行
./test/run_mm_e2e.sh
./test/run_mttkrp_e2e.sh
./test/run_reorder_e2e.sh
```
