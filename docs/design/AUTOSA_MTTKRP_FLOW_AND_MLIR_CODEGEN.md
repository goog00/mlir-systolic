# AutoSA 处理 MTTKRP 的路线与 MLIR 框架下代码生成器适配

> 目标：（1）分析 AutoSA 处理 MTTKRP 的完整路线与做法；（2）明确在 MLIR + Polygeist 框架下应如何做，以及代码生成器如何与新增 pass 配合、如何改得更优雅。

---

## 一、AutoSA 处理 MTTKRP 的路线

### 1.1 输入与语义

- **源程序**：`third_party/AutoSA/autosa_tests/large/mttkrp/kernel.c`
- **语义**：`D(i,j) += A(i,k,l) * B(k,j) * C(l,j)`（或等价的 `C[j][l]` 布局）
- **循环**：4 层 `i, j, k, l`；输出 D[I][J]，规约维为 **k** 和 **l**（双规约，每点累加 K×L 次）。
- **参数示例**：`space_time[3]`；`array_part[128,128,2]`；`latency[16,8]`；`simd[8,1]`。

### 1.2 整体流水线（从 C 到 HLS）

1. **SCoP 抽取**（PET/PPCG）  
   - 解析 C，提取 `#pragma scop` 内的循环与读写关系，得到多面体 SCoP（domain、access、schedule 初形）。

2. **Space-Time 变换**（`autosa_trans.cpp`）  
   - **sa_space_time_transform**：从**最外层可置换 band** 出发，按 `n_sa_dim`=1/2/3 枚举“空间维”组合，生成多组 `autosa_kernel` 候选（每个候选对应一个 space_time_id，如 space_time[3]）。  
   - **sa_space_time_loop_setup**：对每个候选，把 band 中一部分维度标成 **space**（对应 PE 网格维），其余为 **time**（对应执行顺序，包含所有规约维）。  
   - 对 4 循环 MTTKRP：若选 2 维空间（如 i,j），则 k,l 均为 **time**，即 PE 内会遍历 k 和 l，形成**两层规约循环**。

3. **Array partitioning / Latency / SIMD**（`autosa_trans.cpp`）  
   - 在含 space 的 band 上做 array partitioning（tile），再叠 latency、simd 等标注，得到带完整 tile/point 结构的 **调度树**。

4. **通信与模块生成**（`autosa_comm.cpp`、`autosa_codegen.cpp`）  
   - 按访问关系把数组分成 **IO 组 / PE 组 / drain 组**。  
   - **sa_pe_module_gen**：从当前 kernel 的 **schedule** 出发，移动到 “array” band → 再下到 “pe” 层，插入 copy-in/copy-out、pipeline/unroll 等 mark，得到 **PE 模块的 schedule**（`module->sched`）。  
   - 关键：PE 的 schedule 里**保留了所有调度维度**——包括 space（通过 pe_filter 约束到当前 PE）和 **time**（包括 k、l 两层），再叠加 array_part/latency 引入的 tile/point。因此 PE 的循环层数 = 该 schedule 的 band 维度数，**自然包含双规约**。

5. **从 Schedule 到 C 代码**（`autosa_codegen.cpp` + `autosa_print.cpp` / `autosa_xilinx_hls_c.cpp`）  
   - **sa_module_generate_code**：对每个 `autosa_hw_module`，用 **autosa_generate_ast_from_schedule(module->sched)** 从**调度树**生成 **ISL AST**。  
   - 打印时遍历 AST：遇到 “domain” 节点调用 **autosa_kernel_print_domain** 等，把语句体填进去。  
   - **结论**：AutoSA 的 PE 循环**不是手写模板**，而是 **由 schedule 维度数 + 边界约束 完全决定**；双规约 MTTKRP 的 k、l 都在 schedule 里，所以生成的 PE 必然有两层规约循环，语义正确。

### 1.3 AutoSA 路线的要点小结

- **单一事实来源**：调度树（含 space/time 标注、tile、latency、simd）驱动 IO/PE/drain 的**结构**与**循环形状**。  
- **PE 无“固定几层循环”**：循环层数 = 当前 PE schedule 的 band 维度，与 kernel 是 3 循环（MM）还是 4 循环（MTTKRP）一致。  
- **规约维**：未单独命名“reduction”，而是通过 **time 维** 体现——所有非 space 的维都在 PE 内执行，自然包含多规约。  
- **代码生成**：Schedule → AST → 打印；不依赖“单规约/双规约”分支，而是**通用**的“按 AST 打印”。

---

## 二、在 MLIR + Polygeist 框架下应如何做

### 2.1 已有能力与缺口

- **已有**：  
  - Polygeist 把 C 转成 MLIR（含 affine 循环、memref 访问）。  
  - SystolicTransform 用 Polymer/ISL 做依赖分析，并用 **ParametricSpaceTime** 做 space/time 划分（space 循环 = PE 索引，time 循环 = 执行顺序，即规约维）。  
  - 当前 **SystolicDataflowGeneration** 主要做数组分组、写时重排等，**未**把“time 维数 / 规约维集合”写成可被下游消费的**显式信息**。

- **缺口**：  
  - Translate 端**不读**“有多少个 time 维 / 规约维”，只按**固定骨架**（一层 c5 规约）生成 PE，导致 MTTKRP 只做一维规约（8 vs 64）。  
  - 没有“由调度/语义驱动循环形状”的中间表示，难以像 AutoSA 那样**一种代码生成逻辑**覆盖 MM、MTTKRP、TTMc。

### 2.2 建议方向：语义信息下传 + 由“描述”驱动生成

1. **在 Transform / Dataflow 侧显式标出“规约维”或“time 维数”**  
   - 利用现有 ParametricSpaceTime：**time 循环索引集合**（或其个数）即为规约维。  
   - 写入函数属性，例如 `systolic.time_loop_indices` 或 `systolic.n_time_loops`，供 translate 读取。  
   - 可选：更细的 **ContractionDesc**（输出 rank、每数组访问形式、规约维集合），便于后续扩展 drain/IO。

2. **Translate 以“描述”为输入，而非写死一层规约**  
   - 若 `n_time_loops == 1`：保持现有 PE 骨架（单层 c5 规约），兼容 MM。  
   - 若 `n_time_loops == 2`：PE 内生成**两层**规约循环（或等价的一层大循环 size_k*size_l），使每输出点累加次数 = K×L，与 MTTKRP_std 一致。  
   - 后续可再支持输出 rank=3（TTMc）、drain/serialize 按 rank 泛化。

3. **保持“一种生成逻辑、多种描述”**  
   - 目标：**不**为 MTTKRP 单独写一整套 PE 模板，而是**同一套**“根据 n_time_loops / 输出 rank 等生成循环与累加”的逻辑，通过**描述**区分 MM / MTTKRP / TTMc。这样新增 kernel 时主要是**前端/分析**补描述，代码生成器只做参数化扩展。

---

## 三、代码生成器如何适配新增 Pass、如何改得更优雅

### 3.1 当前“不优雅”的根源

- **Translate 与“语义”脱节**：不读规约维数、输出 rank、访问形式，只靠固定 c0/c1/c2/c5/c6/c7 和 “2 输入 × 1 输出” 的假设。  
- **模板与 kernel 强绑定**：PE、drain、L2/L3 的循环层数、是否双规约，都写在 if/switch 或字面常量里，难以扩展。  
- **缺少中间层**：从 MLIR 直接跳到“打印 C++”，没有一层**稳定的代码生成描述**（类似 AutoSA 的 schedule → AST，我们可要 “Schedule/ContractionDesc → 结构描述 → 打印”）。

### 3.2 适配新增 Pass 的接口约定

- **新增/扩展的 Pass**（如 “SystolicContractionAnalysis” 或扩展 DataflowGeneration）应产出**可被 translate 读取的显式信息**，例如：  
  - `systolic.n_time_loops`（或 `systolic.time_loop_indices`）  
  - `systolic.output_rank`（若与 memref 不一致可单独标）  
  - 可选：每数组的 IO 方向、是否参与规约等  

- **Translate 的契约**：  
  - **只根据这些属性 + 已有 array_part/latency/size 等** 决定：PE 有几层规约循环、drain 的迭代空间、serialize 的维度。  
  - **不**再根据“是 MM 还是 MTTKRP”做 if/else 分支，而是根据 **n_time_loops / output_rank** 等通用维度做分支或参数化。

### 3.3 更优雅的代码生成器结构（分阶段可落地）

- **阶段 A（最小改动）**  
  - 在 DataflowGeneration 或独立 pass 中：从 ParametricSpaceTime 得到 time 循环数（或规约维数），写入 `systolic.n_time_loops`。  
  - 在 translate 中：读取该属性；若为 2，则 PE 内生成双规约循环（或等价形式），使 MTTKRP_std csim 通过。  
  - 不改变整体架构，仅“拉一根线”把已有信息传到 translate，并扩展 PE 生成分支。

- **阶段 B（中间描述层）**  
  - 在 translate 前端：从 MLIR 函数 + 属性 构建 **ContractionCodegenDesc**（含 n_time_loops、output_rank、每数组的 shape/角色）。  
  - PE/IO/drain 的**循环结构**和**变量维度**全部从该描述推导（例如 PE 规约循环数 = n_time_loops，drain 维度 = output_rank），不再写死 c5 一层。  
  - 打印层仍可手写 C++ 字符串，但**控制流与维度**由描述驱动，便于后续支持 TTMc、更多 kernel。

- **阶段 C（可选，更接近 AutoSA）**  
  - 在 MLIR 内保留“带标注的调度/循环结构”（例如用 affine 或自定义 op 表示 tile/point 与 space/time），由**同一套** “从调度生成循环 + 填 body” 的逻辑生成 HLS，进一步统一 MM/MTTKRP/TTMc 的代码路径。

### 3.4 与新增 Pass 的配合方式

- **SystolicTransform**：继续负责 space/time 划分与分块；可**增加**写 `systolic.n_time_loops`（或 time 维索引列表）到 kernel 函数。  
- **SystolicDataflowGeneration**：可**增加**“规约/输出语义”的收集与写属性，或调用独立分析（如 ContractionAnalyzer）并写属性。  
- **systolic-translate**：  
  - **输入**：MLIR 模块 + 上述属性 + 现有 size/array_part/latency 等。  
  - **逻辑**：先构建内部描述（阶段 A 可仅为“n_time_loops + output_rank”），再根据描述生成 PE 的规约层数、drain 的 rank、serialize 的迭代；**避免**“看到 3 个输入就猜 MTTKRP”之类的隐式分支。  

这样，**新增 Pass 只负责把“语义/调度”变成属性或描述**，**代码生成器只依赖描述**，二者通过**稳定属性/描述格式**解耦，便于测试和扩展。

---

## 四、小结

| 维度 | AutoSA | 当前 mlir-systolic | 建议方向 |
|------|--------|---------------------|----------|
| 规约维 | 由 schedule 的 time 维自然包含，PE 循环=schedule 维度 | 固定单层 c5，无规约维信息输入 | 显式 n_time_loops，PE 按此生成规约层数 |
| 循环形状 | Schedule → AST → 打印，通用 | 固定 c0/c1/c2/c5/c6/c7 模板 | 由“描述”驱动循环层数与边界，同一套逻辑多 kernel |
| 输出 rank | 由访问与数组类型隐含 | 仅支持 rank 2，rank 3 报错 | 先 2 维双规约打通，再扩展 output_rank 与 drain |
| 与 Pass 关系 | 调度树贯穿 trans/codegen/print | Translate 几乎不读“语义”属性 | Pass 写属性/描述，translate 只读描述生成，接口清晰 |

实现上可先完成**阶段 A**：Dataflow 或新 pass 写 `systolic.n_time_loops`，translate 读之并实现双规约 PE 分支，用 `minimal_mttkrp_std.mlir` 验证 csim；再逐步引入 **ContractionCodegenDesc**（阶段 B）和更统一的调度驱动生成（阶段 C）。
