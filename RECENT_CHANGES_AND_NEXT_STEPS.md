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
| 已有优化梳理     | [docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md](docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md) |
| 单/多核与高性能策略 | [docs/design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md](docs/design/SINGLE_MULTI_KERNEL_AND_HIGH_PERFORMANCE_STRATEGY.md) |
| 愿景与设计目标   | [docs/VISION_AND_DESIGN_GOALS.md](docs/VISION_AND_DESIGN_GOALS.md) |
| AutoSA 对照分析  | [docs/reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](docs/reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md) |
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
  - **(4) FIFO 深度与 HLS 友好化**：FIFO 深度由 dataflow 推导；pipeline 内减少 %、/ 等不利于 HLS 的形式（见改进计划 Phase 4）。
- **暂缓**  
  - **(2) 参数与多面体选择范围**：小规模测试与已知 kernel 下参数一般无问题，且已有 AutoSA 可参考；待需要再接入“分析给出选择范围”的流程。  
  - **HLS 综合/上板验证**：本机无 Xilinx HLS/XRT 环境，留待在服务器上做综合与功能验证。

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
