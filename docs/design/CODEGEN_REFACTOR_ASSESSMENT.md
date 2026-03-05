# 代码生成器改进符合性评估

> 目标：评估当前 `systolic-translate` 与「由语义描述驱动、替代预设模板」的符合程度，并列出尚未满足的改进点，便于最终用新代码生成器完全替代旧模板。

---

## 0. 与 AutoSA 的参考关系（设计原则）

- **AutoSA** 是在 PPCG 上已验证的、基于多面体的脉动阵列 HLS 生成工具；尽管存在部分性能局限（如 MTTKRP 随机读、host-serialize 成本），其**整体流程、空间-时间划分、IO/PE/drain 层级、数据流与 FIFO 设计**等仍值得参考。
- **后续改进时**：在流程与结构上可**进一步参考 AutoSA 的处理方式**（如 sa-sizes、tilable_loops、IO 模块划分、drain 树等），以保持与成熟方案的可比性与正确性。
- **同时保留**：
  - **MLIR 优势**：单一 IR、Pass 管道、可扩展 Dialect、与 Polygeist/Polymer 等前端与多面体分析集成；
  - **既有改进**：ContractionDesc 驱动的多规约/多输出秩、写时/读时重排、L3 coalesced 读、FIFO 深度与 RESOURCE 可配置、强度削减等。
- 目标是在「参考 AutoSA 的成熟做法」与「发挥 MLIR 与既有优化」之间取得平衡，而非简单复刻或完全脱离 AutoSA。

---

## 1. 已符合改进要求的部分

### 1.1 语义描述驱动（ContractionDesc）

| 项目 | 实现位置 | 说明 |
|------|----------|------|
| **ContractionDesc** | `ContractionDesc` 结构体 | 集中描述 `outputRank`、`numReductions`、`Kind`（MatmulLike / MttkrpLike / TtmcLike / Unsupported） |
| **规约维驱动** | `hasExtraReductionLoop()` / `hasThirdReductionLoop()` | PE/IO/dummy 中 r1、r2 循环及条件统一由 `contraction` 决定，不再散落 `numReductions >= 2` 判断 |
| **PE 条件抽象** | `emitPEInitCondition()` / `emitPEDrainCondition()` | 累加器清零与写出条件由描述生成，单/双/三规约共用同一套接口 |
| **Kind 分类与报错** | `emit()` 中设置 `contraction.kind` | 仅对 `Unsupported` 报错；rank-3 + numReductions≤3 走 TtmcLike 并生成 3D drain |

### 1.2 从 MLIR 推导、非写死名称

| 项目 | 实现位置 | 说明 |
|------|----------|------|
| **数组名** | `deriveArrayNamesFromFunction()` | 从 kernel 的 memref 参数名（`mlir.name` 或 arg 下标）推导 `inputNames`、`outputName`，支持 A/B/C、A/B/D、A/B/C/D 等 |
| **输出形状** | `outputShape`、`inputShapes` | 从 kernel 最后一个 memref 与各参数 memref 的 shape 填充，用于 drain 3D 写回与 L3_serialize 的 3D 切片 |
| **规约维数** | `systolic.num_time_loops` | 从 Transform 写入的函数属性读取，填入 `contraction.numReductions`，驱动 r1/r2 与条件 |

### 1.3 多 kernel 类型支持

| 类型 | 支持情况 |
|------|----------|
| **MM（rank-2, 1 规约）** | 单规约路径，无 r1/r2 |
| **MTTKRP（rank-2, 2 规约）** | r1 循环 + 3D 输入按 r1 切片读 |
| **TTMc（rank-3, 3 规约）** | r2/r1 循环 + 3D drain 顺序写 + L3_serialize 三规约 3D 分支 |
| **写时重排 2D/3D** | 基于 `arrayReordering` 与 `hasReordering2D/3D` 分支，不依赖 kernel 名称 |

### 1.4 写回与读入路径

- **emitDrainSerialize**：按 `contraction.isRank3Output()`、`outShape.size()==3`、`hasReordering3D/2D` 分支；rank-3 无重排时用 `outputShape` 算 iterations 并顺序写。
- **emitIOL3InSerialize**：按 `inputShapes` 与 `contraction.hasExtraReductionLoop()/hasThirdReductionLoop()` 区分 2D/3D 及 r1/r2 循环与 word_idx。

---

## 2. 仍依赖预设/模板、待改进的部分

### 2.1 顶层入口与命名

| 问题 | 位置 | 现状 | 建议 |
|------|------|------|------|
| **入口函数名写死** | `emitTopKernel` | 固定生成 `kernel0` | 从 `funcOp.getName()` 生成，或由选项/属性指定；保留 `kernel0` 作默认/兼容 |
| **无 kernel 时默认名** | `emit()` 的 `else` 分支 | `inputNames.assign({"A","B"}); outputName="C"` | 可保留作 fallback，但应在文档标明“仅无 kernel 时使用” |

### 2.2 维度与规模来源

| 问题 | 位置 | 现状 | 建议 |
|------|------|------|------|
| **getArrayDims 回退** | `getArrayDims()` | 无 reorder 时返回 `{latency, 1, arrayPart}`，与具体 tensor shape 无关 | 有 `inputShapes`/output 时，按 arrayName 查对应 shape 作为 L2 缓冲区维度依据；仅无 shape 时再用 latency/arrayPart |
| **totalSize 单一标量** | `emit(ModuleOp)`、`emitDrainSerialize`、L3_serialize | 全用命令行 `size` 作为“问题规模”，2D 默认 `totalSize*totalSize` | 2D 输出时可用 `outputShape[0]`/`outputShape[1]`（若存在）替代单一 size，使矩形与方阵一致 |
| **iterations 默认 2D** | `emitDrainSerialize` | `iterations = (totalSize * totalSize * 4) / 64` | 已对 rank-3 用 `outputShape`；2D 也可改为用 `outputShape` 两维乘积（若存在） |

### 2.3 重排属性命名

| 问题 | 位置 | 现状 | 建议 |
|------|------|------|------|
| **kernel 选择与 reorder 属性** | `emit()`、`extractReorderingInfo` | 显式检查 `systolic.reorder.arg0.dims`、`...C.dims` 等；`extractReorderingInfo` 中用 `arrayName` 和 `arg` 下标两种形式 | 输出数组名已从 `deriveArrayNamesFromFunction` 得到，reorder 属性应统一为 `systolic.reorder.<outputName>.dims` 或 arg 下标，避免写死 "C"；kernel 选择可保留“有 reorder 的优先”逻辑 |

### 2.4 循环与位宽

| 问题 | 位置 | 现状 | 建议 |
|------|------|------|------|
| **循环变量位宽写死** | 多处 `ap_uint<3>`、`ap_uint<2>`、`ap_uint<4>` | c0/c1/c2 用 3 位，c3/c4 用 2 位，r1/r2 用 4 位 | 可根据 `numTiles`、`numPE`、`latency`、`size` 用 `ceil(log2(bound+1))` 生成位宽，避免大 size 时溢出或浪费 |
| **Drain L1/L2/L3 无 r1/r2** | `emitDrainIOL1`、`emitDrainIOL2`、`emitDrainIOL3` | 仅按 c0/c1/c3/c4/c5 迭代，未按规约维扩展 | 若未来 drain 侧也需要与 PE 一致的 r1/r2 节奏（例如多规约时 FIFO 深度或级数不同），可在此处按 `contraction` 增加相应循环；当前 FIFO 数据量由 PE 写出决定，若 csim 已通过可暂不改 |

### 2.5 类型与打包

| 问题 | 位置 | 现状 | 建议 |
|------|------|------|------|
| **输出打包宽度** | 类型定义、drain | 输出类型固定 `*_t{latency}`（如 4×float） | 与当前 PE 累加器 `local_out[latency][latency]` 一致，可保留；若将来支持更高 rank 或不同打包，再引入 OutputDesc |
| **512 位 / 16 float** | L3_serialize、drain | 多处写死 64 字节、16 float | 可提为常量（如 `kDramWordBytes`、`kFloatsPerDramWord`），便于以后扩展 |

---

## 3. 与「完全替代旧代码生成器」的差距总结

- **已具备**：规约维数、输出秩、输出/输入形状、数组名、Kind 分类、PE/IO 条件与 r1/r2 循环均由 **ContractionDesc + MLIR 属性/形状** 驱动，无按 kernel 名 if-else（MM/MTTKRP/TTMc 共用同一套分支逻辑）。
- **仍偏模板/预设**：
  1. 顶层名字固定 `kernel0`；
  2. `getArrayDims` 在无 reorder 时未使用 `inputShapes`/output shape；
  3. 2D 规模仍依赖单一 `size` 与 `totalSize*totalSize`，未统一用 `outputShape`；
  4. 重排属性仍含 "C.dims" 等写死名，与 `outputName` 未完全统一；
  5. 循环位宽、drain 树是否要 r1/r2 等为可选优化，不阻塞“语义正确、多 kernel 支持”目标。

---

## 4. 建议的后续改进顺序

1. **高优先级（语义与可维护性）**
   - 使 **getArrayDims** 在无 reorder 时优先使用 `inputShapes`/output 的 shape（按 arrayName 匹配），仅缺省时用 `latency/arrayPart`。
   - 2D 输出的 **iterations/totalSize** 在存在 `outputShape` 时用 `outputShape[0]*outputShape[1]` 推导。
   - 重排属性查找：在 `extractReorderingInfo` 与 kernel 选择中，对输出数组使用 `outputName`（或最后一参的 name），不再写死 "C"。

2. **中优先级（命名与可读性）**
   - 顶层入口名：从 `funcOp.getName()` 生成（或加 `--kernel-name`），默认仍可为 `kernel0`。

3. **低优先级（扩展与鲁棒性）**
   - ~~循环变量位宽按 bound 动态计算~~（**已做**：`bitsTiles`/`bitsPE`/`bitsSize`/`bitsLatency`/`bitsC5Bound` 在 `emit()` 中由 `requiredLoopBits` 计算，PE/IO/L2 等循环已使用）；
   - ~~将 512/16 等提为命名常量~~（**已做**：`kDramWordBytes`、`kFloatsPerDramWord`、`kDramWordBits`）；
   - 视需求决定 drain L1/L2/L3 是否引入 r1/r2。

完成上述高优先级项后，代码生成器即可视为**由语义描述驱动、基本不依赖 kernel 名与预设 2D 矩阵模板**，实现用新代码生成器完全替代原先“大量预设模板”的形态。
