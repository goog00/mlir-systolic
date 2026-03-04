# 代码生成现状与下一步分析

> 基于最新一次提交（45a6dab，2026-03-04）的结论，明确当前代码生成流程的现状与准备修改的方向。

---

## 1. 最新提交与现状摘要

### 1.1 提交 45a6dab 内容

- **MM**：服务器 HLS 综合 + csim 通过；在 translate 中做了两处修复：
  - **PE**：`local_out` 在累加前于 `c2==0 && c5==0` 时置零。
  - **L2 intra_trans**：从 `local_*[c7][0][0]` 读 packed word，用 `split_idx=c5` 取 lane，避免读到未加载槽位。
- **MTTKRP 语义**：确认与 AutoSA 不一致；**AutoSA 的 MTTKRP 为标准计算流程与定义**。
- **保护性改动**：`systolic-translate` 对**输出 rank≠2** 直接报错并退出，避免为 rank-3 输出生成错误代码。
- **新增**：标准语义测例 `minimal_mttkrp_std.mlir`（D(i,j)+=A(i,k,l)*B(k,j)*C(l,j)）、`minimal_ttmc_std.mlir`；HLS 验证目录与对比/结论文档；设计文档 `CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md`。

### 1.2 当前代码生成的主要问题

| 问题 | 表现 | 根因（模板层） |
|------|------|----------------|
| 标准 MTTKRP csim 错误 | hw=8, ref=64（应 8×8=64 次累加） | PE 仅一层“规约”循环（c5），未覆盖双规约 k,l |
| rank-3 输出不支持 | 报错 "unsupported output rank 3" | drain/serialize 按 rank-2（矩阵）假设，未泛化 |
| 多 kernel 难以适配 | 不同 contraction 需不同 PE/IO 结构 | 无“语义驱动”的中间表示，模板与具体 kernel 绑定 |

---

## 2. 当前模板中“规约”与循环的对应关系

### 2.1 PE 循环骨架（`emitPE`，systolic-translate.cpp）

- **c0, c1, c2**：tile 循环，范围 `[0, numTiles-1]`，来自 `numTiles = size / tileSize`，对应**输出空间**的分块。
- **c5**：范围 `[0, c5Bound-1]`，`c5Bound = arrayPart/simd`，当前模板中**唯一**与“规约维”对应的循环。
- **c6, c7**：范围 `[0, latency-1]`，latency hiding，与累加器 `local_out[c7][c6]` 对应。

对 **MM**：单规约维 k，c5 对应 k 方向的分块/展开，语义正确。  
对 **标准 MTTKRP**：双规约维 k、l，应有两层规约循环（例如 8×8=64 次累加），但模板只有一层 c5（8 次），因此结果少了一个规约维 → csim 出现 8 vs 64。

### 2.2 上游来源

- `numTiles`、`arrayPart`、`latency`、`size` 等来自命令行或属性，**与具体 kernel 的“哪些维是规约”无关**。
- translate 当前**不**从 MLIR 中解析“规约维集合”或“输出 rank/形状”，仅用固定骨架 + 上述参数生成代码。

---

## 3. 设计文档中的方向（CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN）

- **根因**：抽象层级偏“矩阵 + 单规约”，缺少对一般张量 contraction（多规约、多输出 rank）的统一中间表示。
- **目标**：引入**语义驱动的生成**——用 Contraction 描述（TensorDesc、IteratorDesc、AccessMapDesc、ScheduleDesc、OutputDesc）驱动 PE/IO/drain。
- **阶段**：  
  - Phase 1：先支持**标准 MTTKRP**（2D 输出、双规约），PE 显式支持双规约循环；再支持 **TTMc**（3D 输出、双规约）。  
  - Phase 2：与 AutoSA 同语义/同规模对比性能。

---

## 4. 下一步分析建议（代码生成流程）

### 4.1 需要先搞清楚的点

1. **规约维从哪里来**
   - 在 MLIR 侧：**SystolicTransform** 已做 space/time 划分（ParametricSpaceTime）：space 循环 = PE 索引（如 MM 的 i,j），time 循环 = 执行顺序（如 MM 的 k）。对 MTTKRP_std，time 应为 [k,l]（双规约）。当前这些信息在 transform 里存在，但**未**以“规约维数/规约维索引”形式传到 dataflow 或 translate。
   - 在 translate 侧：目前**未**读取任何规约信息，仅用固定骨架 + 命令行参数生成代码。

2. **PE 骨架与“规约维数”的映射**
   - 当前：1 个规约维 → 一层 c5。
   - 目标：2 个规约维（如 MTTKRP 的 k,l）→ 两层规约循环（或等效的 c5/c5' 结构），使每输出点累加 64 次。

3. **IO/L2/L3 与规约维**
   - 输入 A/B/C 的 L2/L3 读顺序、FIFO 节奏是否依赖“规约维”的展开方式；双规约时是否要增加或调整循环/接口。

4. **drain 与输出 rank**
   - rank-2：当前 drain/serialize 已按矩阵来。
   - rank-3（TTMc）：需要在不破坏 rank-2 的前提下，扩展 drain/serialize 的维数与迭代次数。

### 4.2 建议的落地顺序

1. **规约语义提取（MLIR → 描述结构）**
   - 在 **SystolicDataflowGeneration** 或单独分析 pass 中：从 `affine.load/store` 与循环嵌套识别“规约维”（例如：某维只出现在 load 的索引、且对应 reduce 到同一输出元素）。
   - 写出最小 **ContractionDesc**（至少：输出 rank、规约维集合、每个张量的访问形式），并挂到函数属性或传给 translate。

2. **translate 读取规约信息**
   - 若当前没有属性，则先加“规约维数”或“规约维索引列表”的接口（属性或命令行），让 translate 能区分“单规约(MM)”与“双规约(MTTKRP_std)”。

3. **PE 双规约模板**
   - 在 `emitPE` 中：当规约维数为 2 时，增加内层规约循环（或把 c5 扩展为两维），使每输出点做 size_k*size_l 次乘加；保持与现有 MM（单规约）路径兼容。

4. **验证**
   - 用 `minimal_mttkrp_std.mlir` 生成 HLS，csim 期望 64（或与参考一致），再考虑与 AutoSA 的逐项对比。

---

## 5. 本仓库中可用的参考

- **设计**：[CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md](CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md)  
- **标准 MTTKRP 测例**：`test/minimal_mttkrp_std.mlir`（D(i,j)+=A(i,k,l)*B(k,j)*C(l,j)，双规约 k,l）  
- **标准 TTMc 测例**：`test/minimal_ttmc_std.mlir`（3D 输出，双规约 l,m）  
- **HLS 验证与 csim 结论**：`hls_validation/mttkrp_std_mlirsystolic/CSIM_FINDINGS_2026-03-04.md`，`hls_validation/mm_4pe_mlirsystolic/VALIDATION_REPORT_2026-03-04.md`  
- **PE/循环骨架实现**：`tools/systolic-translate/systolic-translate.cpp` 中 `emitPE()`（约 702–745 行）、`emitIOL2InIntraTrans`、`emitDrainIOL1` 等。

---

## 6. 延伸阅读

- **AutoSA 处理 MTTKRP 的完整路线**（调度驱动、PE 无固定模板）：[AUTOSA_MTTKRP_FLOW_AND_MLIR_CODEGEN.md](AUTOSA_MTTKRP_FLOW_AND_MLIR_CODEGEN.md)
- **通用化方案与 ContractionDesc**：[CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md](CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md)

本文档用于在“分析并修改代码生成流程”时统一现状与下一步，便于后续按 Phase 1 推进 MTTKRP_std 正确性，再扩展 TTMc 与通用化。
