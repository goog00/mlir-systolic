# 单核快速验证、多核共用参数与高性能脉动阵列策略

> **目的**：回应三项建议——（1）单 kernel（MM、CNN）小规模下快速验证、HLS 综合控制在 2 小时内；（2）多 kernel 如何利用多面体信息求共用参数；（3）单 kernel 下相对 AutoSA 的高性能脉动阵列模板与改进点。

---

## 1. 单 kernel 小规模快速验证（HLS 综合 ≤ 2 小时）

### 1.1 目标

- 对 MM、CNN 等单 kernel，在**初步优化**后于**小规模**上做端到端验证。
- **HLS 综合时间**尽量控制在 **2 小时内**（便于迭代与 CI）。

### 1.2 约束来源

综合时间主要受以下影响：

- **问题规模**：循环界（N、M、K 或 O、R、C、I、K）越大，生成的循环与状态机越复杂。
- **PE 阵列规模**：空间维的并行度（如 P×Q）越大，实例化模块与布线越多。
- **array_part / latency / simd**：分块与向量化倍数越大，资源与调度越重。
- **FIFO 深度与 BRAM**：过深 FIFO 或大 buffer 会增加 BRAM/综合时间。

### 1.3 建议的“快速验证”配置

以下为**经验性上界**，可按具体器件与工具再收紧。

| 项目 | MM（3-loop） | CNN（5-loop） | 说明 |
|------|--------------|---------------|------|
| **问题规模** | N=M=K ≤ 64（建议 32） | O,R,C ≤ 16，I,K ≤ 8（或 4） | 减小循环界与迭代数 |
| **PE 阵列** | P×Q ≤ 4×4（如 2×2） | 1D 或 2×2 | PE 数少 → 综合快 |
| **array_part** | 每维 ≤ 16，如 [8,8,8] 或 [16,16,16] 配合小 N | [4,4,4,8] 或 [8,4,4,8] | 与问题规模匹配，避免过大 |
| **latency** | [4,4] 或 [8,8]（PE 数 = array_part/latency） | [2,2,4] 等 | 控制 PE 数 = array_part/latency |
| **simd** | 1 或 2 | 1 或 2 | 避免大 SIMD 带来的宽数据路径 |
| **目标** | 单 kernel 综合 < 2h | 单 kernel 综合 < 2h | 可在 CI 中跑“小规模 + 小 PE”配置 |

### 1.4 实现建议

- **参数来自多面体分析的选择范围**：不宜对任意输入使用单一全局预设（如固定 array_part/latency）；应像 AutoSA 那样，由多面体分析逐步得到 space_time、array_part、latency、simd 的**选择范围**（tilable_loops、循环界等），再在该范围内选取参数。小规模验证时，在**当前输入的**合法范围内选取较小值。参见 `third_party/AutoSA/docs/tutorials/getting_started.rst`。
- **与综合脚本联动**：文档或脚本中明确“快速验证”时参数须对应当前输入合法（如 tiling 整除循环界），避免用不适配的固定参数。
- **回归测试分层**：  
  - **L1**：仅 MLIR 变换 + 生成 HLS C++（不跑 HLS）；  
  - **L2**：在分析得到的参数范围内取小，跑 C sim；  
  - **L3**：同参数跑综合（预期 < 2h），用于周期性的深度验证。

---

## 2. 多 kernel 如何用多面体信息求共用参数

### 2.1 共用参数的含义（与愿景对齐）

“共用参数”指：**多个 kernel（如多个 Conv/GEMM）映射到同一套物理脉动阵列**时，需要确定的**统一架构参数**，包括：

- **PE 阵列维度与规模**：如 2D 阵列 (P, Q)，或 1D 阵列长度。
- **Dataflow 类型**：output-stationary / weight-stationary / input-stationary 等，决定数据流动方向与复用方式。
- **Space-time 模式**：哪些循环维作为空间、哪些作为时间（等价于选一种“共用”的 spacetime，如都按 ST3 或都按 ST0）。
- **Tile 与缓冲的“形状”**：array_part、latency 的**维度数**与**上界**（具体数值可为各 kernel 在该上界下的不同 tiling）。
- **IO 层级与 FIFO 拓扑**：L1/L2/L3 的层级数、是否双缓冲等（可先统一，再按 kernel 调深度）。

目标：用**多面体分析**为每个 kernel 得到“若单独实现”的需求，再在这些需求上**求交集或折中**，得到一套**所有 kernel 都能映射上去**的共用参数。

### 2.2 多面体能提供的信息

对每个 kernel（Affine 循环嵌套 + 数组访问），多面体/Polymer 可给出：

| 信息 | 用途 |
|------|------|
| **循环结构与维度** | 空间维数、时间维数、reduction 维；决定 spacetime 候选（如 3-loop→ST0–ST5，5-loop→CNN 的多种 ST）。 |
| **依赖距离** | 哪些维可做空间维（依赖距离 ≤1）；约束“合法”的 space 选择。 |
| **循环界（参数化）** | 各维上界 N、M、K 或 O、R、C、I、K；用于估计 tile 整除性、PE 利用率。 |
| **数组访问与 stride** | 读写的多面体/affine 映射；用于判断 burst 友好性、reduction 维、以及是否需 layout 重排。 |
| **调度树 / 合法变换空间** | 可枚举的 space/time 划分、tiling 可行性；用于生成“单核最优/候选”配置。 |

这些信息足以对**每个 kernel** 得到：  
- 合法 spacetime 集合；  
- 对每种 spacetime 的“合理”array_part/latency 范围（如 PE 数 = array_part/latency，且 tile 能整除问题规模）；  
- 对带宽/缓冲的粗略需求（读写的体积与 stride）。

### 2.3 从“单核需求”到“共用参数”的步骤

**步骤 1：为每个 kernel 生成单核候选**

- 对 kernel \(k\)，用 Polymer/多面体得到：循环数、依赖、访问。
- 枚举或启发式得到若干**合法 spacetime**（如 ST0、ST3、ST4）。
- 对每种 spacetime，结合循环界与资源启发式，得到一至多组 **(array_part, latency, simd)** 候选（或至少得到各维的**合理范围**）。
- 输出：每个 kernel 的**候选配置集合** \(C_k\)，以及每种配置下的（PE 维度、dataflow 类型、预估资源/带宽）。

**步骤 2：定义“可共用”的约束**

两个配置可共用，需要：

- **Dataflow / spacetime 类型一致**：例如都选 output-stationary（ST3 或 CNN 的某一种），这样 PE 连接与数据流方向一致。
- **PE 阵列形状一致**：都是 2D (P, Q) 或都是 1D；**规模取上界**：\(P = \max_k P_k\)，\(Q = \max_k Q_k\)，使得每个 kernel 都能“铺”在 P×Q 上（多出的 PE 可不用或做 padding）。
- **Tile 维度兼容**：array_part / latency 的**维数**一致（如都是 3 维或都是 4 维）；具体数值可以不同（每 kernel 用自己的 tile 大小，但不超过共用 PE 与 buffer 能力）。

**步骤 3：求共用参数（交集 / 资源约束下的折中）**

- **选项 A（严格交集）**：  
  只在“所有 kernel 都合法”的 spacetime 中选；再在每种 spacetime 下，取各 kernel 所需 PE 维度的**逐维最大值**，作为共用 (P, Q)；array_part/latency 取各 kernel 在该 (P,Q) 下可行范围的**交集**（若存在）。  
  适合 kernel 结构相似（如全是 GEMM 或全是 Conv）。

- **选项 B（资源约束下的折中）**：  
  设定片上资源上界（BRAM、DSP、LUT、FIFO 总深度等）；对每个 kernel 的候选配置，估计资源；在**不超资源**的前提下，选一组共用 (P, Q) 和 dataflow 类型，使得**所有 kernel 都能映射**（即每个 kernel 至少有一个候选在该共用架构下可行）。  
  可形式化为：在 (P, Q, dataflow) 空间搜索，使 \(\forall k,\, \exists \text{config}_k \in C_k\) 满足 (P,Q) ≥ 所需且资源 ≤ 上界。

- **选项 C（统一 spacetime + 最大 PE）**：  
  先固定**共用 spacetime**（如全部用 ST3），再对每个 kernel 在 ST3 下算“所需 PE 数”；取各 kernel 在各自空间维上的**最大**，得到共用 (P, Q)；array_part/latency 按 (P, Q) 反推一组默认值，各 kernel 的 tiling 在该默认值下再微调（如整除性、带宽）。

### 2.4 实现路径建议

- **阶段一（分析接口）**：  
  对每个 Affine kernel，用现有 Polymer + ParametricSpaceTime 输出：合法 spacetime 列表、各 spacetime 下的“推荐”或“可行”array_part/latency 范围、以及推导出的 PE 维度。可先做成**离线分析工具**或 **pass 输出 JSON/属性**。

- **阶段二（多核输入）**：  
  接受多个 kernel（多个 Affine 区域或多个函数），对每个跑阶段一；汇总为多组 \(C_k\)。

- **阶段三（共用参数求解）**：  
  实现上述选项 A/B/C 之一（建议先 C：统一 spacetime + 取各核 PE 上界），输出**一份**共用 (P, Q)、dataflow、array_part/latency 默认值；以及每个 kernel 的**在该共用架构下的 tiling 配置**（可覆盖到各 kernel 的循环界）。

- **阶段四（代码生成）**：  
  单份 SA 的 HLS 生成只依赖**共用参数**；每个 kernel 的调用/配置用各自的 tiling 与循环界生成（同一份 SA，多份“配置”或多次调用）。

### 2.5 小结

- **多面体信息**：循环结构、依赖、访问、调度树 → 单核的合法 spacetime 与 (array_part, latency, PE 形状) 候选。
- **共用参数**：dataflow/spacetime 统一；PE 规模取各核上界；tiling 维数一致、数值可 per-kernel。
- **实现**：先做“单核→候选配置”的稳定输出，再做“多核→取 max/交集/资源约束折中”的共用参数求解与映射。

---

## 3. 单 kernel 高性能脉动阵列模板（相对 AutoSA 的改进）

目标：在**单核**上，通过一套明确的**设计规则与代码形态**，形成“比 AutoSA 更优”的**基础高性能模板**，便于在 MLIR 生成中系统落地。

### 3.1 AutoSA 当前的主要不足（回顾）

结合 [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](../reference/autosa/AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md)：

- **布局/重排**：识别了 layout transform 但默认跳过，导致跨步/随机访问未从根本消除。
- **FIFO 深度**：多为固定小深度（如 2），易 backpressure，未从 dataflow 推导。
- **Reduction**：识别与 merge 不够自动，有时依赖交互。
- **HLS 友好性**：pipeline 内存在 %、/、复杂索引，影响 II/Fmax。
- **写回路径**：写时未系统做 reorder/merge，写回带宽易成瓶颈。

### 3.2 高性能模板的“设计契约”

下面列出我们建议的**基础高性能脉动阵列模板**应满足的规则，作为生成与优化的目标。

#### 3.2.1 访存与布局

| 规则 | 说明 | 相对 AutoSA |
|------|------|-------------|
| **DRAM 读：优先 stride-1** | 将 stride-1 维作为 burst 内连续维（最内层/向量化维）；否则做 host 重排或片上 transpose/gather。 | AutoSA 常保留原始访问顺序，跨步读多。 |
| **写时重排** | 写回地址非连续时，用局部 tile 缓存 + 排序/分桶 + 顺序 flush，或与 reduction merge 结合。 | AutoSA 未系统做，需手改。 |
| **Layout 与 SIMD 一致** | SIMD 维与物理连续维对齐；若需 layout transform，在 IR 层显式表达并生成一致 host/kernel 代码。 | AutoSA 识别但不执行 transform。 |

#### 3.2.2 FIFO 与缓冲

| 规则 | 说明 | 相对 AutoSA |
|------|------|-------------|
| **FIFO 深度可推导** | 基于简单 dataflow 模型：深度 ≥ (上游 burst 长度/下游消费率) + 启动延迟余量；或给出公式/属性由后端填。 | AutoSA 多固定 depth=2。 |
| **双缓冲显式化** | L2 等层双缓冲在 IR/生成中显式表达，避免隐式或不足。 | AutoSA 有双缓冲但深度偏小。 |

#### 3.2.3 Reduction 与写回

| 规则 | 说明 | 相对 AutoSA |
|------|------|-------------|
| **Reduction 自动识别** | 用依赖 + 访问模式区分 reduction 维，标注为 reduction；不依赖交互。 | AutoSA 部分场景需人工。 |
| **Drain/merge 结构化** | Reduction 结果经 tree merge 或有序 merge 再写回，减少写回次数与随机写。 | 可做得比 AutoSA 更统一。 |

#### 3.2.4 HLS 友好化

| 规则 | 说明 | 相对 AutoSA |
|------|------|-------------|
| **Pipeline 内无 %//** | 用计数器/状态机或预计算替代可静态推导的取模/除法。 | AutoSA 生成中常有 %/ 在 pipeline 内。 |
| **线性地址 strength reduction** | 地址表达式拆成 base + step，预计算基址与增量。 | 可做成 MLIR pass。 |
| **Pack/unpack 规范** | 固定宽度 ap_uint 的 slice 访问统一、可预测。 | 减少临时移位逻辑。 |
| **分支外提** | 尽量将条件移出最内层 pipeline，避免 II 波动。 | 与 AutoSA 同向，可更严格。 |

#### 3.2.5 参数与可调性

| 规则 | 说明 |
|------|------|
| **参数显式化** | space_time、array_part、latency、simd、FIFO 深度等作为属性或配置传入，便于多核共用与调优。 |
| **Cost 可估** | 对候选配置能估资源与带宽上界，便于“快速验证”与“多核共用”选参。 |

### 3.3 在 MLIR 中的落地顺序建议

1. **访存/布局**：实现 stride 分析 + burst 维决策；写时重排的 IR 表达与生成（可先支持 MM/CNN 的典型模式）。
2. **FIFO 深度**：在 dataflow 抽象或生成阶段，按简单公式/属性计算深度并写入 pragma。
3. **Reduction**：在现有 ParametricSpaceTime/reduction 维基础上，打通“识别 → 标注 → drain/merge 生成”。
4. **HLS 友好化**：增加一组 canonicalization/cleanup pass（strength reduction、去除 pipeline 内 %/、pack 规范、分支外提）。
5. **双缓冲**：在 IO 模块生成中显式双缓冲与深度，与 FIFO 深度一致。

### 3.4 小结

- **单核高性能模板** = 在保持与 AutoSA 相同的“脉动结构”（PE 阵列 + 多级 IO + drain）前提下，**系统化**解决：访存连续/写时重排、FIFO 深度、reduction 识别与 merge、HLS 友好化。
- 上述规则可作为**设计契约**与**回归指标**（如“生成代码中 pipeline 内无 %”“FIFO 深度 ≥ 某公式”），逐步在 mlir-systolic 中实现并对比 AutoSA 参考。

---

## 4. 文档与索引

- 愿景与多核共用： [VISION_AND_DESIGN_GOALS.md](../VISION_AND_DESIGN_GOALS.md)
- AutoSA 性能与 MLIR 机会： [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](../reference/autosa/AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md)
- AutoSA 与 mlir-systolic 对照： [AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md](../reference/autosa/AUTOSA_VS_MLIR_SYSTOLIC_COMPARATIVE_ANALYSIS.md)
- 当前实现与下一步： [CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](../status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md)
