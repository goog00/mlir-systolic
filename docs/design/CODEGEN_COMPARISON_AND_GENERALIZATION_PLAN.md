# 代码生成逻辑对比与通用化方案（面向 MTTKRP / TTMc）

> 日期：2026-03-04
> 目标：解释当前 mlir-systolic 代码生成为何在 `mttkrp/ttmc` 上出现语义问题，并给出可落地、可扩展的通用化重构路径。

---

## 1. 现状对比：AutoSA vs mlir-systolic

### 1.1 语义建模方式

- **AutoSA**
  - 从多面体调度和访问关系出发，生成的 PE/IO 结构与具体 kernel 语义（迭代空间、规约维、输出形状）绑定。
  - `mttkrp`/`ttmc` 示例本身是独立 kernel（见 `third_party/AutoSA/autosa_tests/large/*/kernel.c`），语义与测试黄金值一致。

- **mlir-systolic（当前）**
  - `systolic-translate` 采用“统一模板 + 参数替换”的方式：同一套 PE/IO/drain 模板覆盖多个 kernel。
  - 虽支持 `inputNames.size()` 变化（2/3 输入），但核心 PE 循环骨架仍是固定三层 tile + 一层规约计数，导致双规约/高阶输出语义表达不足。

### 1.2 当前模板的关键约束（导致“看起来奇怪”）

证据：`tools/systolic-translate/systolic-translate.cpp`。

1. **PE 累加结构固定为 2D tile**
   - `local_out[latency][latency]` 固定二维。
   - 这天然偏向 MM 类（二维输出）模板。

2. **规约循环骨架固定**
   - PE 中统一使用 `c0/c1/c2`（tile）+ `c5`（arrayPart/simd）+ `c6/c7`（latency）骨架。
   - 对 `mttkrp`（k/l 双规约）会出现“只覆盖一维规约”风险；csim 已出现 `8 vs 64`。

3. **输出序列化默认按 `size*size` 推导迭代数**
   - `emitDrainSerialize` 默认路径 `iterations = (totalSize * totalSize * 4) / 64`。
   - 对 rank-3 输出天然不匹配（除非走特定 3D reorder 分支且前置条件完整）。

4. **输入流向规则半固定**
   - 顶层连接中，`input[0]` 按行/列一种模式传播，其他输入按另一种模式传播。
   - 对复杂 contraction 不一定是最优或正确映射，缺少“由访问映射自动决定传播方向”的机制。

### 1.3 结果层面的表现

- `mttkrp`（标准语义）可生成 HLS C，但 `csim` 不通过（当前已复现）。
- `ttmc`（rank-3 输出）当前被保护性拦截（`unsupported output rank 3`），避免继续生成错误代码。

---

## 2. 根因总结

当前问题不是“某个 pragma 写错”，而是 **代码生成抽象层级不够通用**：

- 模板以“矩阵型输出 + 单规约骨架”为中心设计；
- 对一般张量 contraction（多规约 + 多输出 rank）缺少统一中间表示；
- 代码生成阶段没有把“语义信息（迭代角色、访问映射、规约集合）”作为一等输入。

---

## 3. 更优雅、更通用的方法（建议架构）

核心思想：在 `systolic-translate` 前半段建立 **Contraction Codegen IR**（轻量内部结构），后半段全部从该 IR 生成。

### 3.1 新的内部抽象（建议）

1. **TensorDesc**
   - 名称、rank、shape、打包宽度、存储顺序。

2. **IteratorDesc**
   - 迭代器集合，标注角色：`spatial_i/spatial_j/...`、`reduction_k/reduction_l/...`、`batch`。

3. **AccessMapDesc**
   - 每个 tensor 的访问函数：例如 `A(i,k,l)`、`B(k,j)`、`C(l,j)`、`D(i,j)`。

4. **ScheduleDesc**
   - tile/simd/latency 映射、PE 网格维度、每个输入的传播方向（由访问映射推导，而非硬编码）。

5. **OutputDesc**
   - 输出 rank、drain 树层级、serialize pack/unpack 规则。

### 3.2 生成器重构为“分层后端”

- **Layer A：语义提取/验证器**
  - 从 MLIR func + memref + affine.load/store 中提取 contraction IR。
  - 验证：规约维完整性、输出 rank、访存可映射性。

- **Layer B：结构规划器**
  - 根据 contraction IR 决定 PE 局部缓存维度、输入流拓扑、drain 树维度。

- **Layer C：代码模板库（参数化）**
  - `PE<NumInputs, NumReductions, OutRank>`
  - `IO<L3/L2>(TensorDesc, direction)`
  - `Drain<OutRank>`
  - `Serialize<OutRank>`（统一 N 维 pack/unpack，不再 `size*size` 特判）

- **Layer D：性能注入**
  - `PIPELINE/RESOURCE/STREAM`、strength reduction、coalescing 策略统一在该层注入。

---

## 4. 针对 mttkrp / ttmc 的最短落地顺序

### Phase 1（先打通语义正确）

1. **先支持标准 MTTKRP（2D 输出，双规约）**
   - 在 PE 生成中显式支持双规约循环集合。
   - 输出仍走 rank-2 drain/serialize。
   - 验收：`test/minimal_mttkrp_std.mlir` 对应 csim PASS。

2. **再支持标准 TTMc（3D 输出，双规约）**
   - 扩展 rank-3 输出 drain + serialize（不依赖 reorder 属性是否存在）。
   - 验收：`test/minimal_ttmc_std.mlir` 从“预期失败”改为 csim PASS。

### Phase 2（再做性能比较）

1. 与 AutoSA 在同语义/同规模下做 csynth 对比。
2. 重点比较：
   - 时钟、总延迟、BRAM/DSP/FF/LUT；
   - L3 访问式（是否规整、是否 burst 友好）；
   - FIFO 深度与背压热点。

---

## 5. 近期执行建议（具体）

1. 保持当前 rank-3 guard（防止错误代码扩散）。
2. 为 `mttkrp_std` 增加 translator 侧“语义单测”（检查是否识别双规约）。
3. 引入 `ContractionDesc` 最小版本（不一次性大改），先驱动 PE 双规约生成。
4. `mttkrp` csim 通过后，再迁移 drain serialize 到 rank-agnostic 实现。

---

## 6. 结论

- 当前“奇怪”的根源是：模板层抽象偏矩阵，尚未真正泛化到张量 contraction。
- 最优路线不是继续 patch 单点，而是引入“语义驱动的生成中间层”，再参数化生成 PE/IO/drain。
- 按上述分阶段推进，可以先快速恢复 `mttkrp` 语义正确，再扩展到 `ttmc` 与后续新 kernel。
