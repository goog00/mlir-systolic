# AutoSA 与 mlir-systolic 对照分析：打通完整流程与简单 CNN 支持

> **目的**：综合 AutoSA 源码与 mlir-systolic 源码，结合 [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md) 与 [VISION_AND_DESIGN_GOALS.md](../../VISION_AND_DESIGN_GOALS.md)，做阶段/能力对照，并给出**快速打通完整流程**（可先固定单一 spacetime）与**简单 CNN 支持**的可行路径。

---

## 1. 文档与目标对齐

| 文档 | 要点 |
|------|------|
| **AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES** | AutoSA 流水线（schedule → space-time → compute/comm management → codegen）、性能瓶颈（随机读、FIFO、reduction）、MLIR 侧可做优化（layout、dataflow 图、reduction 识别、HLS 友好化）。 |
| **VISION_AND_DESIGN_GOALS** | 单核参考 AutoSA（单循环→单 SA）；扩展为共用 SA + 多算子映射；单核 pipeline 需与 AutoSA stage 对齐。 |

本对照分析在上述基础上，**逐阶段对齐 AutoSA 与 mlir-systolic**，并明确：为快速打通完整流程（可先支持某一固定 spacetime）并考虑简单 CNN，需要补齐/简化的具体项。

---

## 2. AutoSA 端到端流水线（源码级）

以下按执行顺序归纳 AutoSA 在 `generate_autosa_xilinx_hls_c` → `generate_sa` → `generate` 中的步骤（见 `ppcg.c`、`autosa_xilinx_hls_c.cpp`、`autosa_trans.cpp`）。

| 阶段 | AutoSA 实现位置 | 主要动作 |
|------|------------------|----------|
| **1. 调度** | `get_schedule(gen)` | 从 scop 得到 ISL schedule；`merge_outer_bands` 合并外层 band。 |
| **2. 合法性** | `sa_legality_check(schedule, scop)` | 检查是否可做 SA 映射。 |
| **3. 设备映射** | `sa_map_to_device(gen, schedule)` | 对 schedule tree 做 space-time 变换并生成/处理 kernel。 |
| 3a. Space-Time | `sa_space_time_transform` | 根据 `space_time_id` 在指定维度做 space/time 划分，得到 `autosa_kernel`（含 `n_sa_dim`、space/time 标注）。 |
| 3b. Kernel 元数据 | `process_kernel_meta_data` | 插入 local/array/kernel/PE mark，得到 host domain、arrays、contraction 等。 |
| 3c. 计算管理** | `compute_management(gen, kernel, ...)` | `sa_loop_init` → `sa_space_time_loop_setup` → `sa_io_update`；然后 **array partitioning**、**latency hiding**、**SIMD**（`sa_array_partitioning_optimize`、`sa_latency_hiding_optimize`、`sa_simd_vectorization_optimize`）。 |
| 3d. 通信管理** | `comm_management(kernel, gen)` | `sa_io_construct_optimize`：构建 IO 模块、FIFO、L1/L2/L3、drain；`localize_bounds`。 |
| **4. 代码生成** | `sa_generate_code` | 从 schedule 生成 AST。 |
| **5. 模块级 codegen** | `sa_module_generate_code` / `sa_filter_buffer_io_module_generate_code` | 按 `gen->hw_modules` 逐个生成 IO/PE/drain 等函数体。 |
| **6. Top + Drain** | `sa_top_module_generate_code`、`sa_drain_merge_generate_code` | 生成 kernel0、drain/merge 逻辑。 |
| **7. 可选** | `sa_host_serialize_generate_code` | Host 端序列化/重排。 |
| **8. 打印** | `print_hw`（如 `print_top_gen_host_code`） | 输出 HLS C + host。 |

**小结**：AutoSA 的“完整流程” = 调度 → 合法性 → **space-time 变换** → **compute_management（分块/延迟隐藏/SIMD）** → **comm_management（IO 构建）** → AST/模块/top/drain 生成 → 打印。其中 **compute_management** 与 **comm_management** 是“从 schedule 到具体硬件模块与 FIFO 拓扑”的核心。

---

## 3. mlir-systolic 当前流水线对照

| 阶段 | mlir-systolic 实现 | 与 AutoSA 对照 |
|------|--------------------|----------------|
| **调度** | Polymer 的 `getSchedule()`（在 SystolicTransform 内） | 对应 AutoSA 的 `get_schedule`；Polymer 输出 ISL schedule tree。 |
| **合法性** | `checkLegality(band)`（循环数、完美嵌套等） | 对应 `sa_legality_check`；当前偏 3-loop，5-loop 未显式支持。 |
| **Space-Time** | `selectSpaceLoopsParametric` + `ParametricSpaceTime::createFromMode` | 对应 `sa_space_time_transform`；ST0–ST5 已参数化。 |
| **计算管理** | `applyMultiLevelTiling`（array_part、latency）+ 属性注入 | 对应 `compute_management` 中的 array partitioning、latency hiding；SIMD 在 translate 侧用命令行参数，未与 IR 系统打通。 |
| **通信管理** | `SystolicDataflowGeneration`：数组分组、IO/PE/Drain 分类、写时重排分析、**属性**写入 | 对应 `comm_management` 的 IO 构建；当前**不**生成完整 SystolicDataflow 模块图，而是**属性 + 固定模板**驱动下游。 |
| **代码生成** | `systolic-translate` 内嵌 `SystolicHLSEmitter` | 对应 `sa_module_generate_code` + top；当前为**固定 A/B/C 三数组、MM 形态**的模板，未按 IR 的数组个数/名/维度泛化。 |
| **Top + Drain** | 同一 emitter 内 `emitTopKernel`、`emitDrain*` | 对应 AutoSA 的 top + drain；结构固定。 |
| **Host** | 未实现 | 对应 `sa_host_serialize_generate_code`；预留。 |

**结论**：

- **已对齐**：调度（Polymer）、合法性（部分）、space-time 参数化、分块与属性传递、数据流分析与属性。
- **部分对齐**：通信管理“逻辑”在 DataflowGeneration 中有（分组、层级），但**没有**像 AutoSA 那样输出“显式 hw_modules 列表 + 按模块 codegen”，而是由 **translate 的固定模板** 反推。
- **未对齐**：  
  - 任意数组名/维数的 **通用 IO/PE/Drain codegen**（当前仅 A/B/C + MM）；  
  - **5 循环 + 3 数组**（如简单 CNN）的 band 与配置（4D array_part/latency/simd）；  
  - **Reduction 识别** 与 local-reduce/drain merge 的显式支持；  
  - Host 端生成。

---

## 4. 逐项能力对照表

| 能力项 | AutoSA | mlir-systolic | 打通完整流程 / CNN 所需 |
|--------|--------|----------------|--------------------------|
| 输入 | C + `#pragma scop`（PET） | Affine MLIR（Polygeist 或手写） | 保持；CNN 需 5-loop Affine 或 linalg 降级。 |
| 调度 | ISL schedule | Polymer ISL | 已满足。 |
| Space-time 枚举 | space_time[0..9]（含 2D SA） | ST0–ST5（ParametricSpaceTime） | 先固定一种即可打通（如 MM 用 ST3，CNN 用 ST0 或 ST4）。 |
| 循环数 | 3（MM）、5/6（CNN） | 当前强调 ≥3，逻辑多针对 3-loop | **放宽或显式支持 5-loop**（简单 CNN）。 |
| array_part / latency / simd | 3D（MM）或 4D（CNN） | 3D 为主 | **CNN 需 4D 配置**（或先写死一组 4D）。 |
| Reduction | `--local-reduce`、`--reduce-op`、reduction 识别 | ParametricSpaceTime 有 reduction 维配置，未全面接上 | **识别 reduction 维**（如 CNN 的 i,p,q）并参与 PE/IO 决策。 |
| IO 构建 | `sa_io_construct_optimize` → 显式 hw_modules | 数组分组 + 属性，无显式模块图 | 快速路径：**保留“固定模板”**，用 **kernel 类型（MM vs CNN）** 选模板；中期再泛化为“按 IR 生成模块列表”。 |
| 模块 codegen | 按 `hw_modules` 逐模块打印 | 固定 A/B/C + L2/L3 + PE + C_drain | **增加“CNN 模板”**：cin/w/cout + 对应 IO/drain。 |
| FIFO/双缓冲 | 在 comm 与 codegen 中 | 模板中有 stream，深度等偏保守 | 后续优化；打通流程可沿用现状。 |
| 写时重排 / layout | 识别但多跳过 | 分析结果写属性，未参与生成 | 见 AUTOSA_SOURCE_PERF；打通流程可暂缓。 |

---

## 5. 快速打通“完整流程”的界定与建议

**“完整流程”** 此处指：**从 Affine 输入到可被 HLS 工具接受的单一 kernel 的 HLS C++ 输出**，且**不要求**当前就支持所有 spacetime 或所有 kernel。

建议采用**分目标**的快速路径：

### 5.1 目标 A：3-loop MM + 固定 ST3（当前最接近）

- **现状**：MM 的 3-loop + ST3 已能跑通（SystolicTransform → SystolicDataflowGeneration → systolic-translate），输出为固定 A/B/C 的 HLS。
- **建议**：
  - 明确将 **ST3 固定** 作为“默认/唯一”配置选项（命令行或属性），避免未指定时行为发散。
  - 补全**端到端测试**：从单一 Affine MM 到生成 `.cpp`，再与 `test/autosa_hls_refs` 中同配置（如 `mm_*_st3_*`）做**结构对比**（模块名、Pragma、大致循环结构），不要求逐行一致。
  - 文档化“当前仅保证 ST3 + MM”的边界，便于后续加 ST0/1/2/4/5 或 CNN 时对比。

### 5.2 目标 B：简单 CNN（5-loop，单层卷积）+ 固定一种 spacetime

- **AutoSA 参考**：`docs/examples/cnn.rst`、`test/autosa_hls_refs/cnn_default_st0_*.cpp` 等；CNN 为 5 维循环（o,r,c,i,p,q），3 数组（cin, w, cout），reduction 在 (i,p,q)。
- **最小可行**：
  1. **循环数**：允许 **5-loop band** 通过合法性检查（放宽 `band.size() >= 3` 的“仅 3-loop”假设，或对 5-loop 单独分支）。
  2. **Spacetime**：先只支持 **一种** CNN 用 spacetime（例如 **ST0**，对应 output-stationary [o]，与 `cnn_default_st0_*` 一致），避免一次实现 4/5/7/8。
  3. **配置**：为 5-loop 引入 **4D** 的 array_part/latency/simd（或先写死一组，如 `[8,4,4,8]`、`[4,2,4]`、`[1,1,1,2]`），并在 ParametricSpaceTime/属性中传递。
  4. **Reduction**：在 5-loop 上**识别 reduction 维**（最内几维为累加），并在数据流分析中标记为 reduction，供 PE 内累加与 drain 使用（可先做“识别+标注”，codegen 仍用固定 CNN 模板）。
  5. **Codegen**：在 **systolic-translate** 中增加 **“CNN 模板”** 分支：
     - 若检测到 3 数组且名为 cin/w/cout（或 5-loop + 某属性），则走 **CNN 模板**：生成 cin/w/cout 的 IO L1/L2/L3、PE、cout_drain，与 `cnn_default_st0_*` 结构类似；
     - 否则仍走现有 A/B/C MM 模板。

这样可在**不重写整套通用 codegen** 的前提下，先打通“MM（ST3）+ CNN（ST0）”两条固定路径，再逐步泛化。

---

## 6. 实现优先级建议（与愿景文档一致）

| 优先级 | 内容 | 说明 |
|--------|------|------|
| **P0** | 固定 ST3 的 MM 端到端固化与测试 | 明确默认配置、回归测试、与 AutoSA 参考结构对比。 |
| **P0** | 5-loop 合法性 + 一种 CNN spacetime（如 ST0） | 允许 5-loop band；ST0 + 4D 配置；reduction 维识别与标注。 |
| **P1** | systolic-translate 的 CNN 模板 | 基于 3 数组 cin/w/cout + 5-loop/属性，生成与 AutoSA cnn_st0 可比的 HLS 骨架。 |
| **P1** | Reduction 在 PE/drain 中的体现 | 从“仅标注”到在 CNN 模板中生成 PE 内累加与 drain merge 的合理形态。 |
| **P2** | 写时重排 / layout 接入生成 | 见 AUTOSA_SOURCE_PERF；在 MM/CNN 稳定后再接属性到 codegen。 |
| **P2** | 更多 spacetime（ST1/2/4/5）与 4-loop（MTTKRP） | 在 MM/CNN 两条线稳定后扩展。 |
| **P3** | 通用“按 IR 生成 hw_modules” | 替代固定模板，与愿景中的“单核→单 SA”完全对齐。 |

---

## 7. 小结

- **AutoSA**：调度 → 合法性 → space-time → **compute_management**（分块/latency/SIMD）→ **comm_management**（IO 构建）→ 按 hw_modules 做 codegen → top/drain/host。
- **mlir-systolic**：Polymer 调度 + SystolicTransform（space-time + 分块）+ SystolicDataflowGeneration（分组与属性）→ **固定模板** translate（当前仅 MM 的 A/B/C）。
- **快速打通**：  
  - **MM**：固定 ST3，固化流程并做与 AutoSA 参考的结构对比。  
  - **简单 CNN**：支持 5-loop、一种 spacetime（如 ST0）、4D 配置、reduction 识别，并在 translate 中增加“CNN 模板”分支（cin/w/cout），与现有 MM 模板并列。
- 在**不追求**“任意 spacetime + 任意 kernel”的前提下，上述两条线即可在较短时间内形成**可验证的完整流程**，并为后续泛化与写时重排、多 spacetime、共用 SA 等留出清晰接口。

---

## 8. 相关文档与源码索引

- AutoSA：`third_party/AutoSA/src/autosa_trans.cpp`（`generate`、`sa_map_to_device`、`compute_management`、`comm_management`）、`autosa_common.h`（`autosa_kernel`、`autosa_hw_module`）、`autosa_xilinx_hls_c.cpp`（打印）。
- mlir-systolic：`lib/Transforms/SystolicTransform.cpp`、`lib/Transforms/SystolicDataflowGeneration.cpp`、`tools/systolic-translate/systolic-translate.cpp`、`lib/Analysis/ParametricSpaceTime.cpp`。
- 文档： [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md)、[../../VISION_AND_DESIGN_GOALS.md](../../VISION_AND_DESIGN_GOALS.md)、[../../status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](../../status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md)。

---

## 附录：快速打通清单（可裁剪为 issue/任务）

### MM + ST3 固化
- [ ] 命令行/属性明确“默认 spacetime=3”且仅当指定时才用其他 mode。
- [ ] 建立端到端测试：`affine MM → systolic-opt → systolic-translate → .cpp`，并与 `test/autosa_hls_refs/mm_*_st3_*` 做结构对比（模块数量、Pragma、DATAFLOW 结构）。
- [ ] 在 README/status 中写明“当前仅保证 ST3 + 3-loop MM”。

### 简单 CNN（5-loop + ST0）
- [ ] **SystolicTransform**：允许 5-loop band（放宽或分支 `band.size() >= 3`）；对 5-loop 支持一种 spacetime（如 ST0）与 4D array_part/latency/simd（或写死一组）。
- [ ] **SystolicDataflowGeneration**：对 5-loop 做数组分组（cin/w/cout）；识别 reduction 维并写入属性。
- [ ] **systolic-translate**：增加“CNN 分支”：当检测到 5-loop + 3 数组（或显式属性）时，按 cin/w/cout 生成 CNN 模板（IO L1/L2/L3、PE、cout_drain），参考 `cnn_default_st0_ap8x4x4x8_*.cpp`。
- [ ] 提供一份最小 5-loop 卷积的 Affine MLIR 测例（或从 AutoSA kernel.c 转成 MLIR 的脚本），用于回归。
