# AutoSA 源码逻辑、脉动阵列性能瓶颈与 MLIR 优化机会（面向 mlir-systolic）

## 背景与目标

AutoSA 是基于 PPCG/ISL/PET 的端到端脉动阵列（Systolic Array, SA）自动编译器：输入为带 `#pragma scop/endscop` 的 C 语言循环核，输出面向 FPGA 的 HLS C/C++（并可生成 host OpenCL 或 HLS testbench/host）。

`mlir-systolic` 的目标是在 MLIR 上复现并改进 AutoSA 的关键逻辑。目前项目已能初步将 `affine.for` 降为 HLS C（但尚未经过 HLS 工具验证）。本报告聚焦两件事：

- **AutoSA 源码层面**：它如何从 polyhedral schedule 走到 SA 结构与 HLS 代码？
- **性能层面**：AutoSA 生成 SA 的常见瓶颈与可改进空间是什么？在 **MLIR 框架** 下有哪些更系统、更可验证、可扩展的优化可以做？

本仓库中可复现/对照的材料：

- **AutoSA 源码**：`third_party/AutoSA/src/`（含 `autosa_*.cpp` 与 PPCG 相关的 `ppcg.c` 等）。
- **官方测试（需运行 autosa 生成 HLS）**：`third_party/AutoSA/autosa_tests/`。
- **已抽取的 HLS 参考代码**：`test/autosa_hls_refs/`（26 个配置变体，用于静态分析与对照）。

---

## 1. AutoSA 的源码结构与“编译流水线”

### 1.1 入口与 target 分发

AutoSA 的可执行程序入口在：

- `third_party/AutoSA/src/main.cpp`：`main()` 仅调用 `autosa_main_wrap(argc, argv)`
- `third_party/AutoSA/src/ppcg.c`：`autosa_main_wrap` 解析命令行参数并按 target 分发：
  - `generate_autosa_xilinx_hls_c(...)`
  - `generate_autosa_intel_opencl(...)`
  - `generate_autosa_catapult_hls_c(...)`
  - `generate_autosa_tapa_cpp(...)`

这表明 AutoSA 在 PPCG 框架之上“插入了新的后端 target”，并沿用 PPCG 的 SCoP 抽取与 schedule/AST 生成骨架。

### 1.2 核心数据结构（理解 AutoSA 最关键的一步）

AutoSA 的核心结构体集中在 `third_party/AutoSA/src/autosa_common.h`，其中最关键的对象包括：

- **`autosa_prog`**：全程序级别的 SCoP 表示（读写 union_map、array 信息、stmt 信息等）。
- **`autosa_kernel`**：AutoSA 将当前 SCoP 映射为“单 kernel + SA”的中心对象；包含：
  - SA 维度与 space-time 选择（`n_sa_dim`, `space_time_id` 等）
  - array partition、latency hiding、SIMD 的用户指定与实际使用 sizes（`sizes`, `used_sizes`）
  - `pe_filter` / `pe_ids`：将 domain 映射到 PE 坐标的约束与标识
  - `core` / `arrays`：核心计算与相关数组集合
- **`autosa_hw_module`**：最终要打印成 HLS 的硬件模块（`PE_MODULE / IO_MODULE / DRAIN_MODULE`）
  - I/O 层级（`level`）、是否双缓冲（`double_buffer`）、是否连接 DRAM（`to_mem`）、数据打包因子（`data_pack_*`）等
- **`autosa_array_ref_group`**：把同一数组的若干引用聚成组，并携带 IO 变换（`io_trans`）、tile/buffer（`io_buffers`）、mem port 映射（`n_mem_ports`）等信息

这组结构体直接体现了 AutoSA 的“架构建模方式”：不是把所有东西都揉在 schedule/AST 里，而是显式构造 **PE 组、IO 组、drain 组、tile、buffer、FIFO** 等硬件视角的实体，再把它们分别 codegen。

### 1.3 主要 pass/模块（文件级地图）

AutoSA 自研逻辑主要分布在：

- **`autosa_trans.cpp`**：关键的 schedule 变换与标注（space-time、array partition、latency hiding、SIMD 等）
  - 例如 SIMD 选择中用 `is_stride_coalesced(...)` 检查 stride-0/1，并记录 “需要 layout transform” 的信息，但默认会 **跳过需要 layout transform 的候选**（见后文改进点）。
- **`autosa_codegen.cpp` / `autosa_codegen.h`**：从 schedule/标注生成硬件模块列表、插入 host/device 相关节点、调用具体 codegen 打印。
- **`autosa_comm.cpp`**：通信与 IO 层级相关处理（构造 IO 模块、FIFO、边界处理、drain/merge 等）。
- **`autosa_xilinx_hls_c.cpp`**：Xilinx HLS C 打印器（生成 `#pragma HLS ...`、stream/FIFO、模块函数体与 top `DATAFLOW`）。
- **`autosa_tuning.cpp` / `autosa_tuning.h`**：参数/资源/延迟提取与调参相关基础设施（更像“离线估计+搜索”框架）。

### 1.4 典型输出结构（从 `autosa_hls_refs` 观察到的模板）

以 `test/autosa_hls_refs/mttkrp_default_st3_ap256x256x4_lat32x16_simd16x2.cpp` 为例，可观察到典型结构：

- 多级 IO 模块：`*_IO_L3_in`（从 DRAM 读）、`*_IO_L2_in`（跨 IO/PE 传播）、`*_IO_L2_in_boundary`（边界）
- PE 包装：`PE_wrapper(...)` 负责 PE 内核计算与与周边 FIFO 连接
- drain/merge：例如 `D_drain_*` 负责把 PE 侧结果收敛并写回 DRAM
- top kernel：`kernel0(...)` 中 `#pragma HLS DATAFLOW`，声明大量 `hls::stream<>` 并把 IO/PE/drain 模块串接

这说明 AutoSA 的核心思想确实是“把计算核心变成 PE 规则阵列 + 规则流动的 FIFO 通信”，但 IO/DRAM 的访问模式与 buffering 策略将强烈决定最终性能。

---

## 2. 性能瓶颈与可改进空间（以生成 SA 为中心）

从 FPGA SA 角度，性能常被以下因素夹住：

- **外存带宽利用率**：burst 读写、访问连续性、跨端口/通道映射、访存仲裁
- **片上存储组织**：BRAM/URAM/FF 的分配、banking、`ARRAY_PARTITION` 粒度、端口冲突
- **流水线能否稳定 II=1**：是否有除法/取模/复杂条件、是否出现对 stream 的 backpressure
- **通信与缓冲是否匹配**：FIFO 深度是否足以吸收 DRAM burst 与 PE 侧 steady-state 的速率差
- **边界/收敛逻辑开销**：drain/merge、reduction、boundary PE 的特殊处理
- **映射策略（space-time / array_part / latency / simd）是否与算子匹配**：同一套启发式在不同核上会出现“明显不合适”的决策

AutoSA 已经覆盖了很多“正确方向”的机制（多级 IO、latency hiding、SIMD、host serialize、资源估计/调参），但在源码与生成代码里仍能看到明显的提升空间，尤其是在 **布局/访存重排** 与 **自动化决策质量** 上。

### 2.1 参数如何决定 SA 结构与性能上限（面向调参与 MLIR 复现）

AutoSA 的核心参数（`--sa-sizes`）通常包含四类：

- **`space_time`**：选择空间-时间映射模式（决定数据在阵列中的流动方向与本地驻留对象）
- **`array_part[...]`**：第一级 tile（决定“阵列级”并行形态与 PE 组织）
- **`latency[...]`**：第二级 tile（latency hiding / point-loop 组织，决定 PE 数量与片上缓冲形态）
- **`simd[...]`**：向量化因子（决定单 PE 每拍处理的元素数与数据打包/布局需求）

在很多 ST 模式（尤其 ST3 类 2D 阵列）里，一个非常实用的结构估算关系是：

- **PE 阵列规模 \(\approx\) `array_part` / `latency`**（逐维相除后得到每个空间维的 PE 个数）

直观解释：

- `array_part` 更大 → 阵列并行度更高，但 IO/片上缓存/路由压力更大
- `latency` 更大 → 更强的延迟隐藏/更粗的 point-tile，但会改变 PE 数量与局部累加缓冲大小
- `simd` 更大 → 单拍吞吐更高，但更依赖 **布局/打包/连续 burst**（否则容易“算得快、喂不饱”）

因此，很多“看似随机读/带宽利用率差”的问题，本质不是 PE 计算映射错了，而是 **参数组合 + 布局/IO 决策**没有对齐到外存 burst 与片上 banking 的现实约束。

> 更完整的 AutoSA 参数—结构对应关系、ST0-ST5 的数据流语义与已验证的 PE 数量推导案例，可参考本仓库已有文档：`docs/reference/autosa/AUTOSA_ANALYSIS.md`。

---

## 3. MTTKRP 案例：典型“看起来像随机读”的外存访问

你提到在 MTTKRP 内核中，AutoSA 生成 SA 存在随机读取问题，并且你通过“写时重排（write-time reorder）”手工修改 HLS 代码验证能提升性能。

即使在本仓库抽取的 dense MTTKRP 参考代码里，也能看到外存访问“非连续/多流交错”的模式，容易导致：

- burst 长度变短、有效带宽下降
- 多路并行读取在同一 AXI 端口上被仲裁打散
- IO 模块/PE 侧被 backpressure，实际 II > 1 或出现间歇性空转

### 3.1 具体证据：`B` 的 DRAM 读是跨步访问（4 路交错，stride=21 words）

在 `mttkrp_default_st3_ap256x256x4_lat32x16_simd16x2.cpp` 的 `B_IO_L3_in` 中，外存读取形如：

- `B[16*c1 + 84*c2 + c3 + 21*c4]`，其中 `c4` 在最内层（0..3）

这意味着同一次 “access_coalesce” 内的连续读地址是：

- `... + c3 + 21*0`, `... + c3 + 21*1`, `... + c3 + 21*2`, `... + c3 + 21*3`

对 DRAM/AXI 来说这不是连续突发，而是 4 条相距 21 的读流交错。若把这些数据在 host 侧或片上做布局重排（让 `c4` 对应维度在物理内存上变成连续），就能把这类访问变成更理想的 burst。

### 3.2 结果写回端也存在“复杂索引/打包逻辑”可优化

同文件中 `D` 的写回包含：

- 运行时 `split_idx = (...) % 4`、再进行拼包后条件写回

在 HLS 中，取模/除法/复杂表达式出现在最内层 pipeline 中，常见后果是：

- 额外组合逻辑与时序压力
- 影响 II 或频率（Fmax）
- 更容易触发 HLS 的保守调度（尤其与 stream 读写混合时）

写回路径往往是“吞吐上限”的另一侧：即使 PE 计算很快，如果 drain/merge 或写回 IO 模块吞吐不足，也会反向施压整条数据流。

---

## 4. AutoSA 源码与生成策略层面的改进空间（可直接对标到代码文件/结构体）

### 4.1 关键缺口：**layout transform 被识别但默认“跳过”**

在 `autosa_trans.cpp` 的 SIMD 候选检测中，`is_stride_coalesced(...)` 会标记访问是否需要 layout transform，并把 `legal[...]` 设为 `!layout_transform`；随后在 `autosa_simd_tile_loop(...)` 中：

- **如果需要 layout transform，则直接跳过该 SIMD 候选循环**（即“看见了机会，但没做”）。

这会带来两个问题：

- 对很多本质上“可通过布局重排变得非常规整”的算子（包括 MTTKRP 类多维张量/矩阵访问），AutoSA 会错失最关键的带宽/向量化机会。
- 用户只能通过 `--host-serialize` 或手工改 HLS 代码来做重排，而编译器无法端到端保证一致性（host 数据布局与 kernel 访问布局的匹配）。

**改进建议（AutoSA 方向）**：

- 把 layout transform 从“诊断信息”升级为“可执行变换”：
  - **host 侧重排**：生成 `host_serialize/deserialize` 时引入 `memref.transpose/permutation` 的真实布局变换（AutoSA 已有 host serialize 机制，可扩展其变换种类）。
  - **片上重排**：在 IO 模块内增加一次性块搬运 + transpose（BRAM/URAM tile），把跨步流转换为连续 burst + 片上规整访问。
- 将 layout transform 纳入 cost model：只有当重排代价（额外 BRAM、搬运周期）低于带宽收益时才启用。

**在 MLIR 下更容易做**：见第 5 节（把布局变换作为 first-class 的 memref layout/packing pass，并自动生成 host 侧一致性代码）。

### 4.2 DRAM coalescing/多流聚合策略不足（以 MTTKRP 为代表）

在 `autosa_hls_refs` 中能看到 AutoSA 生成的 IO 模块常带 `// access_coalesce` 循环，但它未必真正在“物理连续维度”上做聚合。

**改进建议**：

- 为每个 DRAM 访问构建“地址表达式模型”（线性组合的 stride 向量），自动选择：
  - 哪一维作为 burst 内连续维（stride=1）
  - 哪些维度作为多路并行流（stride 常数但非 1）并进行 **gather-to-burst**：先将多流地址排序/分桶后成段读取，再片上散射
- 对 Xilinx 平台，进一步考虑：
  - AXI outstanding、burst 最大长度、对齐要求（512-bit 打包对齐）
  - 多 bundle/多端口映射（AutoSA 有 `n_mem_ports` / `mem_port_id`，但启发式可更精细）

### 4.3 FIFO 深度与 backpressure：当前明显偏保守

在 `kernel0` 的 top `DATAFLOW` 中，大量 FIFO 深度只有 `depth=2`（且多用 `FIFO_SRL`）。

**风险**：

- DRAM burst 与 PE steady-state 吞吐不匹配时，几乎没有缓冲吸收抖动
- 任一模块小幅变慢会导致全局 backpressure，吞吐下降

**改进建议**：

- 基于简单的 dataflow model 自动推导 FIFO depth：
  - 深度 ≥（上游 burst 长度 / 下游消费速率）+（模块间最大启动延迟）
- 把 FIFO depth 与 `latency hiding`、`double buffering` 联动：latency hiding 不应只体现在 PE 时间维，也应该反映到 IO buffering 上。

### 4.4 reduction/merge 自动识别能力不足（甚至依赖交互输入）

`autosa_trans.cpp` 中对 reduction loop 的识别在某些模式下需要人工回答（`Please input if the current loop is a reduction loop`）。这对自动化编译与调参极其不友好。

**改进建议**：

- 用依赖分析 + 访问模式识别自动区分：
  - true dependence vs reduction dependence
  - reduction 的结合律/交换律条件（至少支持常见 `+`, `max`, `min`）
- 生成更高效的 reduction 结构：
  - PE 内局部 tree reduction + drain merge（减少写回带宽）
  - 或者在 array partition 维度上做分段归约，最后再写时重排

MLIR 在这点上有天然优势：如果上游 IR 是 `linalg`/`tensor` 或者显式 reduction op，识别更可靠；就算在 `affine`，也可以通过模式匹配与 `affine.for` 归约形态检测得到。

### 4.5 生成的 HLS 代码“可综合但不够 HLS-friendly”

从参考代码可见一些典型点：

- 在最内层 pipeline 中出现 `%`、`/`、复杂索引表达式与条件写回
- 数据打包/解包用循环移位拆分（如 `ap_uint` shift），可能引入额外逻辑与影响 II

**改进建议**：

- 在 codegen 阶段做 HLS 友好化重写：
  - 将 `%`/`/` 替换为计数器/有限状态机（在可静态推导时）
  - 预计算线性地址基址与增量（strength reduction）
  - 用结构化 pack/unpack（固定宽度 `ap_uint` 的 slice 访问）替代循环移位
- 将这些“低层重写”从打印器逻辑抽离成统一 IR 级 pass（在 MLIR 更容易实现和验证）

### 4.6 调参/代价模型：从“可用”到“更稳定更泛化”

AutoSA 有 `autosa_tuning.*` 与资源/延迟提取接口，但行业经验表明仅靠启发式很难在不同算子上稳定拿到好结果。

**改进建议**：

- 更明确的目标函数：吞吐（elem/cycle）、带宽利用率、资源约束下的最小 latency
- 把“layout transform / coalescing / FIFO depth / banking”纳入搜索空间
- 采用分层搜索：
  - 先确定 memory/layout（决定上限带宽）
  - 再确定 array_part/latency hiding（决定并行度与启动延迟）
  - 最后在 SIMD 与细粒度 pragma 上做局部调优

---

## 5. 在 MLIR 框架下可落地的系统性优化（面向 mlir-systolic）

相比 AutoSA（C/PPCG/ISL 生态），“在 MLIR 上复现并改进”最大的收益是：**把关键决策做成显式 IR + 可组合 pass + 可验证分析**。下面按能力模块给出建议。

### 5.1 把 space-time / array_part / latency / simd 变成 MLIR 的一等公民

建议不要把这些信息只保存在命名或外部 JSON 中，而是：

- 用 attribute 或自定义 dialect 表达：
  - 循环维度的 `space/time` 归属
  - PE 坐标映射（space dims → PE ids）
  - latency hiding（时间维分段/软件流水）
  - SIMD lane 与 memref packing（向量化与数据布局的一致性）

这样后续的：

- IO 插入
- double buffer
- FIFO/stream 连接
- HLS pragma 发射

都可以在 IR 上系统地完成，而不需要像 AutoSA 那样把信息散落在 schedule node 的 mark 与打印器的上下文里。

### 5.2 访存分析与“布局/重排”作为核心优化 pass

针对你提到的 MTTKRP 随机读/重排收益，建议在 MLIR 中把它变成可复用的通用 pass：

- **Access stride 向量分析**（对每个 memref 访问，在给定 loop nest 上计算 stride）
- **burst/coalescing 决策**：
  - 若存在 stride=1 维，优先将其变成最内层/向量化维
  - 若存在多流交错（如 stride=const≠1），考虑生成：
    - host-side layout permutation（序列化重排）
    - 或片上 gather/transpose buffer
- **写时重排（write-time reorder）模式化**：
  - 对“写回地址非连续/跨步”的场景，用局部 tile 缓存 + 排序/分桶 + 顺序 flush
  - 与 drain/merge/reduction 联动，避免写回带宽成为瓶颈

MLIR 的优势在于：你可以把“重排”从最终 HLS C 的手工改动，上移到 IR 变换层，并自动生成 host 侧一致性代码（或生成新的输入布局约束）。

### 5.3 IO 层级与 dataflow 建模：从“模板化打印”走向“可计算的图”

建议将 SA 设计表示为显式 dataflow graph：

- 节点：IO 模块、PE 模块、drain/merge 模块
- 边：stream/FIFO（带宽、深度、生产/消费率）

然后基于图做：

- FIFO depth 自动推导
- backpressure 风险分析（找出可能形成瓶颈的边/节点）
- 与 HLS `DATAFLOW`、`STREAM depth`、`RESOURCE core` 的 pragma 自动对齐

### 5.4 reduction 自动识别与更高效的归约结构

在 MLIR 中优先考虑：

- 如果上游能保留 `linalg.reduce` / `arith.addf` 的 reduction 语义，直接继承语义信息
- 否则在 `affine.for` 上做 reduction pattern 识别，并将其“提升”为显式 reduction op/region

随后提供多种归约实现策略（由 cost model 选择）：

- PE 内局部归约（寄存器/BRAM）
- drain merge tree（减少写回次数）
- write-time reorder + 合并（匹配外存 burst）

### 5.5 HLS 友好化重写：把“硬件常识”固化成 canonicalization

推荐建立一组面向 HLS 的 canonicalization/cleanup pass，例如：

- strength reduction（线性地址表达式拆成 base + step）
- 去除 pipeline 内 `%`/`/`（可静态推导时）
- pack/unpack 规范化（统一位切片表达）
- 控制分支外提（尽量避免影响 II）

这些 pass 一旦稳定，就能显著提高“生成代码可综合且性能稳定”的概率。

### 5.6 验证与回归：把 `autosa_hls_refs` 变成回归基准集

`test/autosa_hls_refs` 非常适合作为：

- 结构回归（模块数量、FIFO 拓扑、pragma 模式）
- 静态性质回归（访存 stride、burst 连续性指标、预计吞吐上限）
- 后续接入 HLS 工具后的性能对照（latency/resource/utilization）

建议在 `mlir-systolic` 中为每个参考 kernel 建立：

- 一个“同等配置”的 MLIR 测例（对齐 ST/AP/LAT/SIMD）
- 自动抽取关键指标并生成对比表（至少包含：DRAM 访问连续性、FIFO depth、PE II 目标、写回模式）

---

## 6. 面向你当前痛点（MTTKRP 随机读 + 写时重排）的建议落地路径

结合你已经验证过“写时重排有效”，我建议把它在 MLIR 中升级为一个可自动触发的优化组合（而不仅是手工改 HLS）：

- **Step A：检测**  
  对 IO/写回路径的地址表达式做 stride/流数分析，识别“多流交错/跨步写回/难以 burst”的模式（如 `stride != 1` 且在最内层出现）。

- **Step B：选择策略（cost model）**  
  在以下策略中选一或组合：
  - host-side layout permutation（最省片上资源，但要求 host/输入可重排）
  - on-chip tile + transpose/gather（更通用，资源换带宽）
  - write-time reorder（对写回端尤其有效，可与 reduction 融合）

- **Step C：生成一致性代码**  
  若选择 host 重排，必须生成对应的 host 端重排/反重排（或者在接口上声明新布局）。

这样做的收益是：你已经手工验证的优化，会变成 **可复用的编译器能力**，并且可以用 `autosa_hls_refs` 做 A/B 对照回归。

---

## 7. 小结（建议优先级）

如果按“对性能贡献最大 + 最能体现 MLIR 优势”的优先级排序：

1. **布局变换/重排自动化**（解决 MTTKRP 随机读/跨步访问的根因；AutoSA 已识别但没执行）
2. **DRAM coalescing 与多流聚合**（burst 化 + 片上散射/重排）
3. **dataflow 图 + FIFO depth 推导**（避免 depth=2 带来的 backpressure 与吞吐不稳）
4. **reduction 自动识别与高效 merge**（从“交互/保守”变成“自动/可选结构”）
5. **HLS 友好化低层重写**（提升 II/Fmax 稳定性，减少手写“救火”）
6. **更强调参与 cost model**（把上述空间纳入统一搜索，避免仅靠启发式）

---

## 附：本仓库中关键路径索引

- AutoSA 入口与 target 分发：`third_party/AutoSA/src/main.cpp`、`third_party/AutoSA/src/ppcg.c`
- AutoSA 核心数据结构：`third_party/AutoSA/src/autosa_common.h`
- AutoSA 关键变换（space-time/latency/SIMD 等）：`third_party/AutoSA/src/autosa_trans.cpp`
- Xilinx HLS C 打印器：`third_party/AutoSA/src/autosa_xilinx_hls_c.cpp`
- HLS 参考代码（静态分析素材）：`test/autosa_hls_refs/*.cpp`

