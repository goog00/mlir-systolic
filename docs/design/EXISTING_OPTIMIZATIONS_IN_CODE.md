# 代码中已有的脉动阵列相关优化

> **目的**：梳理当前代码里已实现的优化方法与接入点，便于在此基础上继续实现脉动阵列优化。

---

## 1. 写时重排（Write-time reordering）

### 1.1 分析

| 位置 | 内容 |
|------|------|
| **lib/Analysis/WriteTimeReorderingAnalysis.cpp** | `WriteTimeReorderingAnalyzer`：遍历 load/store，提取访问模式；检测非线性和跨步访问；`computeReordering` / `computeReorderingWithISL` 计算 `reorderedDims`、`dimPermutation`。 |
| **lib/Analysis/LayoutOptimizer.cpp** | `applyPermutation`：按置换应用维度重排。 |
| **lib/Analysis/PolyhedralAccessAnalyzer** | stride、reuse distance 等访问分析（可为重排提供依据）。 |

### 1.2 属性写入

| 位置 | 内容 |
|------|------|
| **lib/Transforms/SystolicDataflowGeneration.cpp** | 在 DataflowGeneration 中调用 `WriteTimeReorderingAnalyzer::analyze()`，将结果写入函数属性 `systolic.reorder.<arrayName>.dims`、`systolic.reorder.<arrayName>.perm`（以及 `arg<N>` 形式）。 |

### 1.3 代码生成中的使用

| 位置 | 内容 |
|------|------|
| **tools/systolic-translate/systolic-translate.cpp** | `extractReorderingInfo()` 从函数属性读取重排信息到 `arrayReordering`。 |
| | `getArrayDims(arrayName)`：有重排时返回重排后维度，用于 L2 缓冲区声明。 |
| | `applyAccessPermutation(arrayName, indices)`：对给定下标做置换，用于 **L2** 的 `emitIOL2InIntraTrans`、`emitIOL2InInterTrans`、`emitIOL2InInterTransBoundary` 的 local 数组访问。 |
| | **已接入（2D）**：`emitDrainSerialize` 在存在 2D 重排属性时按重排顺序写回 DRAM（先 unpack 入 buffer → 按重排维度顺序填入 buffer_linear → 再 pack 写出）。 |
| | **已接入（2D）**：`emitIOL3InSerialize` 在存在 2D 重排属性时按重排维度顺序 (d0,d1) 生成读循环，假定输入已为重排布局，DRAM 读仍为顺序（word_idx 递增）。 |

### 1.4 读时重排与写时重排是否冲突？

**不冲突**，二者作用在不同数组、约定一致即可：

- **作用对象**：读时重排用于**输入**（如 A、B），写时重排用于**输出**（如 C）。MM 中无“同一数组既按重排读又按重排写”的情况。
- **布局约定一致**：都使用同一套 `systolic.reorder.<array>.dims/perm` 定义的“重排布局”（即按重排后的维度做 row-major 的物理顺序）。读时假定 host 已按该布局放好输入；写时 kernel 按该布局写回。因此 host 与 kernel 的约定一致即可。
- **Host 契约**：若某数组带 reorder 属性，则  
  - **输入**：host 须按重排布局写入 DRAM，kernel 才会按顺序读且逻辑正确；  
  - **输出**：kernel 按重排布局写回，host 读回时须按重排布局解释（或 host 事先按重排布局分配，kernel 只填该布局）。
- **同数组既读又写（in-place）**：若将来支持同一 buffer 既读又写，需保证读/写阶段在时间上分离，且读写使用同一重排约定，否则会冲突。当前 MM 模板不涉及。

---

## 2. 参数与配置

| 位置 | 内容 |
|------|------|
| **lib/Transforms/SystolicTransform.cpp** | space_time、array_part、latency 等由选项/属性注入；多级分块、循环置换。 |
| **lib/Analysis/ParametricSpaceTime.cpp** | ST0–ST5、reduction 维配置、数据流方向。 |
| **tools/systolic-translate** | `--size`、`--array-part`、`--latency`、`--simd` 控制生成；未与多面体分析给出的“选择范围”联动。 |

---

## 3. FIFO 与缓冲

| 位置 | 内容 |
|------|------|
| **lib/Transforms/DataflowGeneration.cpp** | 各层 depth 固定为 2（如 L3、L2、drain L1/L2）。 |
| **tools/systolic-translate** | 生成 `#pragma HLS STREAM variable=... depth=2`；深度未按 dataflow 推导。 |

---

## 4. Reduction

| 位置 | 内容 |
|------|------|
| **lib/Analysis/ParametricSpaceTime.cpp** | `ReductionDimConfig`、`hasReductionDim()`。 |
| **lib/Transforms/SystolicTransform.cpp** | 空间/时间划分时考虑 reduction 维（如置内层）。 |
| **lib/Transforms/SystolicDataflowGeneration.cpp** | PE 内循环体迁移时克隆最内层循环（含 reduction）；未单独做 reduction 识别与 merge 生成。 |

---

## 5. 小结：已用上 vs 待接入

- **已用上**：写时重排的分析 + 属性 + 在 **L2** 的 local 缓冲区维度和访问下标上的应用；**L3_in_serialize** 在存在 2D 重排时按重排维度顺序读；**drain_IO_L3_out_serialize** 在 2D/3D 重排时按重排顺序写回（buffer_linear 路径）。
- **待接入/可选**：  
  - FIFO 深度由简单 dataflow 推导（可选）。  
  - 参数与多面体分析给出的选择范围对接（可选）。
