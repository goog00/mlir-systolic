# Allo 分析与 PyTorch 兼容路线（面向 mlir-systolic）

> **最后更新**: 2025-03  
> **目的**: 分析 Allo 的贡献与可借鉴点，并给出 mlir-systolic 向上兼容 PyTorch 的可行路线  
> **与本目录其他文档的关系**: 本文档侧重“整体定位 + 借鉴 + PyTorch 路线”；[ALLO_INTEGRATION_ANALYSIS.md](ALLO_INTEGRATION_ANALYSIS.md) 侧重集成方案与阶段划分，[ALLO_HLS_CODE_GENERATION_RULES.md](ALLO_HLS_CODE_GENERATION_RULES.md) 侧重 Affine→HLS C++ 的逐项规则。

---

## 1. Allo 的定位与两大贡献

Allo 是基于 MLIR 的 HLS 生成器，**不做复杂多面体/脉动阵列自动变换**，其核心贡献可以概括为两点：

### 1.1 贡献一：Python 专用语言描述“计算 + 调度”

- **计算**：用 Python 函数 + 类型标注（如 `float32[M, N]`）描述算子逻辑，内部通过 `allo.dsl`（`grid`、`reduction`、`matmul`、`linear` 等）和 NumPy 风格运算在 **trace 时** 生成 Allo IR（MLIR，含 Allo Dialect 与后续的 Linalg/Affine）。
- **调度**：通过 `allo.customize(func)` 得到 `Schedule` 对象，用一套 **Schedule API** 显式描述硬件结构：
  - `s.partition(array, dim=..., factor=...)`：数组分区（对应 HLS `array_partition`）
  - `s.pipeline(loop)` / `s.unroll(loop)`：流水线、展开
  - `s.unfold(loop_name, [dims])`：把指定循环变为“空间维”（对应多实例/并行 PE）
  - `s.to(buffer, pe, axis=..., depth=...)`：在模块之间建立 FIFO/stream，并指定深度

这样做的效果是：**硬件形态（分区、流水、空间展开、FIFO）由用户在 Python 里用少量 API 声明，而不是在 C/MLIR 里手写**。对脉动阵列，Allo 提供 `allo.library.systolic` 等库函数，但 **空间循环、FIFO 连接、深度仍需用户通过 Schedule API 手写**，没有依赖距离分析与自动 space-time 映射。

### 1.2 贡献二：向上兼容 PyTorch（基于 MLIR 生态）

- **前端**：`allo.frontend.pytorch.from_pytorch(model, example_inputs)` 使用 **torch.fx** 追踪模型，将 PyTorch 算子映射到 Allo 的 DSL 调用（如 `dsl.linear`、`nn.conv2d`），再通过 `customize(code, global_vars)` 把生成的 Allo Python 代码字符串变成可调度的 `Schedule`。
- **后端**：Allo 的 lowering 把 Linalg 转为 Affine，**刻意不降低 Affine Dialect**，最终由 C++ 的 `EmitVivadoHLS` / `EmitTapaHLS` / `EmitIntelHLS` 等直接从 Affine MLIR **一对一翻译**成 HLS C++（for/load/store → C++，再根据属性插 pragma）。

因此，**“PyTorch 兼容”** 依赖的是：**torch.fx 追踪 → 生成 Allo Python 代码 → Allo 解析为 MLIR → Linalg→Affine → 直接生成 HLS**。没有 Polyhedral 自动变换，也没有我们期望的“自动脉动阵列 + 自动 IO 层次”。

---

## 2. Allo 源码结构速览（便于借鉴与对接）

本仓库 Allo 位于 `third_party/allo/`，关键目录与文件：

| 路径 | 作用 |
|------|------|
| `allo/customize.py` | `customize()` 入口、`Schedule` 类、partition/pipeline/unfold/to 等 API，以及每次调度后调用 `_mlir_lower_pipeline` |
| `allo/dsl.py` | 计算侧 DSL：`grid`、`reduction`、`matmul`、`add`、`relu` 等，trace 时用 NumPy 执行并记录到 IR |
| `allo/ir/builder.py` | 将 Python AST / trace 结果转为 Allo MLIR（Allo Dialect + 后续 Linalg 等） |
| `allo/ir/transform.py` | 循环查找、buffer 查找等，供 Schedule API 使用 |
| `allo/frontend/pytorch.py` | `from_pytorch()`、`AlloTracer`、`TorchBuilder.build()` 生成 Allo Python 代码字符串 |
| `allo/passes.py` | `_mlir_lower_pipeline`：empty-tensor-to-alloc-tensor、**convert-linalg-to-affine-loops**，且 **不 lower Affine** |
| `allo/backend/hls.py` | HLS 模块构建、调用 `_mlir_lower_pipeline`、通过 C API 调用 `emit_vhls`/`emit_thls`/`emit_ihls` |
| `allo/library/systolic.py` | 脉动阵列库函数；需用户手写 schedule（unfold、to 等） |
| `mlir/lib/Translation/EmitVivadoHLS.cpp` | 从 Affine MLIR 直接生成 Vivado HLS C++（visitor 模式，按 op 类型逐句翻译） |

要点：

- **计算与调度分离**：计算由 Python 函数 + DSL 描述，调度由 `Schedule` API 描述，二者在 MLIR 上通过 transform 和属性体现。
- **Affine 是“最后一层” IR**：HLS 代码生成直接读 Affine 的 for/load/store/if，不经过中间 HLS Dialect。
- **PyTorch 路径**：`from_pytorch` → 生成 Allo Python 代码字符串 → `customize(code, global_vars)` → 得到 `Schedule`，后续与手写 Allo 代码共用一个 lowering 与 codegen 路径。

---

## 3. 可借鉴之处（针对 mlir-systolic）

### 3.1 Schedule API 与“计算/调度分离”思想

- **借鉴点**：用少量高层 API（partition、pipeline、unfold、to）表达硬件决策，而不是在 C++/MLIR 里散落各种 pass 选项。mlir-systolic 若将来提供 Python 绑定，可以暴露“空间维/时间维/array_part/latency/FIFO 深度”等为类似 Schedule 的接口，便于调参与脚本化。
- **差异**：Allo 的 schedule 作用在“已展开的”循环/缓冲区上，不做依赖分析；我们的自动脉动阵列需要 **先依赖分析、再 space-time 映射、再生成 IO/PE/FIFO**，Schedule 更适合作“覆盖/微调”层，而不是唯一决策来源。

### 3.2 PyTorch 前端流程

- **借鉴点**：
  - **torch.fx 追踪** → 得到计算图，再映射到 DSL 调用（linear、conv2d、matmul 等）。
  - **生成 Python 代码字符串** → 再 `customize(code)` 进入同一套 Allo 流程，这样“模型侧”和“手写 kernel 侧”共用一个 lowering 与 codegen。
- **对 mlir-systolic 的启示**：若我们做 PyTorch 兼容，可以：
  - 方案 A：复用 Allo 的 `from_pytorch`，得到 Allo Python 代码或 Allo IR，再从中 **提取 Affine 子图** 喂给 mlir-systolic 的脉动阵列 pipeline；
  - 方案 B：自建 PyTorch→MLIR 前端（例如 Torch-MLIR、torch.fx → 我们的 DSL/IR），再与我们已有的 Affine→脉动阵列 pipeline 衔接。方案 A 见效快，方案 B 控制力与可维护性更好。

### 3.3 保留 Affine、不 lower 的 pipeline

- **借鉴点**：Allo 的 `_mlir_lower_pipeline` 只做到 Linalg→Affine，不再把 Affine 降到 SCF/Standard。这样 **Affine 循环/负载/存储** 就是 HLS 代码生成器的直接输入。mlir-systolic 当前正是从 Affine 开始做脉动变换与 dataflow 生成，因此“Allo 产出的 Affine”与“我们手写的 Affine”在格式上可以对齐，便于 **用 Allo 作为前端、我们作为中段** 的集成。

### 3.4 Dataflow 与多后端 harness

- **借鉴点**：Allo 的 `dataflow` 模块、`allo.harness`（vivado/vitis/tapa/catapult 等）负责把生成的 kernel 包成工程、脚本与 host。我们若只产 HLS C++，可以借鉴其 **工程模板与脚本组织**，而不是必须用 Allo 的 codegen。
- **HLS 生成方式**：Allo 是“Affine → 一对一 C++ 翻译 + 按属性插 pragma”；我们是“Affine → SystolicDataflow → HLS Dialect → 结构化 C++”。我们的路径更利于表达多级 IO、双缓冲、FIFO 拓扑，因此 **代码生成建议仍以 mlir-systolic 为主**，Allo 的 emit_vhls 可作为对比或备选。

---

## 4. 若后续要向上兼容 PyTorch，应如何做

### 4.1 目标

- **用户侧**：能够从 PyTorch 模型（或子图）得到可在 FPGA 上运行的 HLS 实现，其中“适合脉动阵列”的部分由 mlir-systolic 自动生成，其余部分可由简单 lowering 或调用 Allo/其他后端。
- **工程侧**：与现有 mlir-systolic 流程兼容，不破坏 Affine→脉动阵列→HLS 的主线。

### 4.2 路线一：复用 Allo 的 PyTorch 前端（推荐作为第一阶段）

1. **用 Allo 的 `from_pytorch`** 对给定 `model` 和 `example_inputs` 做追踪，得到 Allo Python 代码字符串（或直接拿到 Allo 的 `Schedule` / MLIR Module）。
2. **从 Allo 的 MLIR 中提取 Affine 区域**：找到顶层函数中“适合做脉动阵列”的循环嵌套（例如 GEMM、Conv2D 的稠密部分），将其转为 **标准 Affine MLIR**（func + memref + affine.for/load/store）。
3. **调用 mlir-systolic 的 pipeline**：对这些 Affine 区域做 SystolicTransform、SystolicDataflowGeneration、SystolicDataflowToHLS、EmitHLSCpp，得到结构化 HLS C++。
4. **与 Allo 的其余部分衔接**：
   - 若 Allo 支持“子图替换”，则用我们生成的 kernel 替换对应子图；
   - 否则，可把我们的 kernel 作为“黑盒子模块”由 Allo 或上层脚本在 DATAFLOW 里实例化（需要约定接口：memref 或 stream）。

**优点**：复用成熟 PyTorch→Allo 路径，快速打通“模型 → 可调度 IR”；我们只负责“Affine 子图 → 脉动阵列 HLS”。  
**缺点**：依赖 Allo 的 IR 格式与演化；提取 Affine 子图需要一定约定（循环命名、函数边界等）。

### 4.3 路线二：自建 PyTorch → MLIR 前端

1. **用 torch.fx 或 Torch-MLIR** 将 PyTorch 模型转为计算图或 MLIR（例如 linalg/tensor）。
2. **识别可脉动化的子图**：对 GEMM、Conv2D、MatMul 等做模式匹配，切出“稠密计算核”。
3. **将子图转为 Affine MLIR**：用现有或新建的 lowering（linalg→affine 或 tensor→memref+affine），使输出符合 mlir-systolic 对 Affine 的假设（循环结构、memref 访问形式）。
4. **与路线一相同**：Affine → mlir-systolic 全 pipeline → HLS C++；再与整体模型的其他部分（host、其他 kernel）组合。

**优点**：不依赖 Allo，控制力强，可与 Torch-MLIR、IREE 等生态对齐。  
**缺点**：工作量大，需维护 PyTorch 版本与 op 映射。

### 4.4 共同要点（无论路线一还是二）

- **接口约定**：mlir-systolic 的输入是“标准 Affine MLIR”（含 func、memref、affine.for/load/store），输出是 HLS C++（或 HLS Dialect）。PyTorch 前端或 Allo 只需产出/提取符合该约定的 Affine 子图即可。
- **子图识别**：优先支持 GEMM/MatMul、Conv2D 等稠密核；reduction、elementwise 可后续扩展或交给其他后端。
- **分阶段**：先实现“Allo 产出的 Affine → mlir-systolic → HLS”的对接与回归测试（可无 PyTorch）；再接入 `from_pytorch` 做端到端；最后再考虑自建前端。

### 4.5 跨层与共用脉动阵列（与愿景对齐）

对接 PyTorch 等网络时，**不应为每一层各生成一套脉动阵列**，而应寻求 **一套共用脉动阵列 + 多层映射**（见 [../../VISION_AND_DESIGN_GOALS.md](../../VISION_AND_DESIGN_GOALS.md)）。因此，从 PyTorch 得到多个 Affine 子图后，除“每子图单独跑单核 pipeline”的验证路径外，最终应走“多核 → 共用 SA 架构选择 → 每核映射到该 SA”的路径；自动调优也可在共用 SA 与每层映射参数上联合搜索。

---

## 5. 与现有参考文档的衔接

- **[ALLO_INTEGRATION_ANALYSIS.md](ALLO_INTEGRATION_ANALYSIS.md)**：详细写了 Allo 与 mlir-systolic 的对比、集成方案（含混合路径）、以及“先完成 mlir-systolic 核心、再集成 Allo 前端”的阶段划分。本文档的 PyTorch 路线与之一致，可作为“为什么这样分阶段”和“如何衔接 Allo IR”的补充说明。
- **[ALLO_HLS_CODE_GENERATION_RULES.md](ALLO_HLS_CODE_GENERATION_RULES.md)**：详细写了 Allo 如何从 Affine 操作生成 HLS C++（for/load/store、pragma、stream）。在做“用 Allo emit_vhls 作为备选输出”或“对比我们生成的 HLS 与 Allo 的差异”时，以该文档为准。

---

## 6. 小结

| 维度 | 要点 |
|------|------|
| **Allo 的贡献** | （1）Python DSL 描述计算+调度，Schedule API 表达分区/流水/空间展开/FIFO；（2）PyTorch 通过 torch.fx→Allo Python 代码→同一套 lowering，HLS 部分多为 Affine 的一对一翻译。 |
| **可借鉴** | Schedule 思想、PyTorch 前端流程、保留 Affine 不 lower、dataflow/harness 组织；不借鉴其“无自动脉动”的调度方式。 |
| **PyTorch 兼容** | 建议先复用 Allo 的 `from_pytorch`，提取 Affine 子图给 mlir-systolic；再视需要自建 PyTorch→MLIR 前端；接口统一为“Affine in, HLS out”。 |

本仓库 Allo 源码位置：`third_party/allo/`（`allo/` 为 Python，`mlir/` 为 C++/MLIR  dialect 与 translation）。
