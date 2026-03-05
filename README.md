# MLIR-Systolic

基于 MLIR 和 Polymer 的脉动阵列编译器。

---

## 📚 文档导航

**📖 [docs/README.md](docs/README.md) - 完整文档导航入口**

**📋 [PROJECT_STATUS_AND_ONBOARDING.md](PROJECT_STATUS_AND_ONBOARDING.md) - 项目状态与上手指南（新环境/新 Agent 首选）**

**📋 [RECENT_CHANGES_AND_NEXT_STEPS.md](RECENT_CHANGES_AND_NEXT_STEPS.md) - 近期修改与下一步工作（进度记录与文档索引）**

### 快速链接
- **[docs/ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)** — 系统架构概述
- **[docs/guide/CODE_STRUCTURE.md](docs/guide/CODE_STRUCTURE.md)** — 代码组织与结构
- **[docs/guide/BUILD_GUIDE.md](docs/guide/BUILD_GUIDE.md)** — 构建步骤与依赖说明
- **[docs/guide/DEVELOPMENT_GUIDE.md](docs/guide/DEVELOPMENT_GUIDE.md)** — 开发指南
- **[docs/reference/PROJECT_STRUCTURE.md](docs/reference/PROJECT_STRUCTURE.md)** — 项目目录结构
- **[docs/status/PROJECT_STATUS.md](docs/status/PROJECT_STATUS.md)** — 当前状态与进展
- **[docs/DOCS_INDEX.md](docs/DOCS_INDEX.md)** — 全量文档索引
- **[scripts/README.md](scripts/README.md)** — Scripts 使用指南

---

## 项目目标

将 Affine 循环嵌套自动转换为脉动阵列 HLS C++ 代码，结合：
- **Polymer** 的多面体分析能力 ⭐ **核心依赖** ✅ 已集成
  - 依赖距离分析（使用 ISL）
  - 空间循环自动选择
  - 调度树获取（用于 task 分解）
- **MLIR** 的变换和代码生成能力 ✅ 已实现
- **ParametricSpaceTime 框架** ✅ 已实现
  - 支持 ST0-ST5 全部 6 种 spacetime 配置
  - 参数化的空间/时间循环选择
  - 自动数据流方向推导

**为什么需要 Polymer**：
- AutoSA 基于多面体模型，使用 ISL Schedule Tree 进行依赖分析和循环变换
- **没有调度树，就无法正确分解 task 成多个独立的 module 函数**
- 这是 ScaleHLS 无法解决的问题，也是我们创建 mlir-systolic 的主要原因
- 详见：`docs/features/polymer/`（如 POLYMER_INTEGRATION_COMPLETE.md）与 [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md)

## 架构

**设计理念**：用 MLIR 的方式重新表达 AutoSA 的语义，而不是直接翻译。通过引入 `SystolicDataflow` Dialect，清晰地表达多层 IO 结构和双缓冲逻辑。

### AutoSA 的两部分架构

AutoSA 包含两个主要部分：

1. **FPGA Kernel 生成**（当前重点）：
   - 处理 SCoP 参数包裹的嵌套 for 循环
   - 通过多面体编译生成 FPGA 的 HLS C 文件
   - 包含 PE 阵列、IO 模块、双缓冲等硬件结构

2. **Host 端代码生成**（预留接口，暂不实现）：
   - 根据命令参数生成不同的 Host 端代码
   - HLS C Testbench：用于 Vivado HLS 仿真
   - OpenCL Host：用于 Xilinx OpenCL 运行时
   - 其他目标平台（TAPA、Catapult HLS 等）

**当前实现范围**：
- ✅ **FPGA Kernel 生成**：专注于 SCoP 区域的循环嵌套（对应 AutoSA 的 `#pragma scop` 区域）
  - ✅ Polymer 集成完成，支持多面体分析
  - ✅ ParametricSpaceTime 框架，支持 ST0-ST5 配置
  - ✅ HLS C++ 代码生成
- ⚠️ **Host 端代码生成**：接口已预留但暂不实现
- ⚠️ **SCoP 区域检测**：当前测试阶段只处理单个 `affine.for` 循环嵌套（后续使用 Polygeist 处理 C 文件导入）

### 编译流程

```
输入: Affine MLIR (来自 Polygeist 或手写)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Pass 1: SystolicTransform                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • 依赖分析 (Polymer/ISL)                          │  │
│  │ • 空间循环选择 (依赖距离 ≤ 1)                     │  │
│  │ • 循环置换 (空间循环外移)                         │  │
│  │ • 多级分块 (array_part + latency)                 │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
         ▼ 分块后的 Affine IR
┌─────────────────────────────────────────────────────────┐
│  Pass 2: SystolicDataflowGeneration ⭐ NEW              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • 数组引用分组 (IO/PE/Drain)                      │  │
│  │ • IO 层级分析 (L1/L2/L3)                         │  │
│  │ • 生成 SystolicDataflow Dialect:                  │  │
│  │   - systolic.io.module<L3> (Global Memory)        │  │
│  │   - systolic.io.module<L2> (双缓冲)               │  │
│  │   - systolic.io.module<L1> (PE 接口)              │  │
│  │   - systolic.pe.array (计算阵列)                  │  │
│  │   - systolic.drain.module (输出)                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
         ▼ SystolicDataflow Dialect IR
┌─────────────────────────────────────────────────────────┐
│  Pass 3: SystolicDataflowToHLS (可选优化) ⭐ NEW       │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • 缓冲区合并优化                                   │  │
│  │ • 流通道优化                                       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Pass 4: SystolicDataflowToHLS (降级) ⭐ NEW           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • systolic.io.module → hls.dataflow.task          │  │
│  │ • 双缓冲逻辑 → Affine loops + SCF if              │  │
│  │ • Stream 通道 → hls.dataflow.stream               │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
         ▼ HLS Dialect IR
┌─────────────────────────────────────────────────────────┐
│  Translation: EmitHLSCpp (FPGA Kernel)                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • HLS C++ 代码生成 (Kernel)                      │  │
│  │ • Pragma 插入 (pipeline, array_partition, etc.)   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
输出: HLS C++ Kernel (Vivado HLS / Vitis 兼容)

┌─────────────────────────────────────────────────────────┐
│  Translation: EmitHostCode (预留接口) ⚠️ 暂不实现      │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • HLS Testbench 生成                              │  │
│  │ • OpenCL Host 代码生成                            │  │
│  │ • 其他目标平台支持                                 │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
输出: Host 端代码 (根据目标平台)
```

**关键改进**：
- 引入 `SystolicDataflow` Dialect 作为中间抽象层，清晰表达硬件结构
- 支持多层（L1/L2/L3）双缓冲 IO 模块生成
- 利用 MLIR 的 dialect 系统和 pass 组合能力，便于后续扩展和优化

## 核心实现文件

本项目采用**分层设计**，充分利用 MLIR 的 dialect 系统：

| 文件 | 功能 | 行数估计 | 对应 AutoSA |
|------|------|---------|-------------|
| `SystolicTransform.cpp` | 分析 + 分块 + 置换 | ~600 | `sa_space_time_transform` + `compute_management` |
| `SystolicDataflowGeneration.cpp` ⭐ | 数据流抽象生成 | ~1200 | `comm_management` + `generate_hw_modules` |
| `SystolicDataflowToHLS.cpp` ⭐ | Dialect 降级 | ~800 | - |
| `EmitHLSCpp.cpp` | HLS C++ 输出 | ~1200 | `print_hw` |

**新增 Dialect 定义**：
- `include/systolic/Dialect/SystolicDataflow/` - Dialect 定义（TableGen）
- `lib/Dialect/SystolicDataflow/` - Dialect 实现

### 详细职责

**SystolicTransform.cpp** (Pass 1: `-systolic-transform`)
- 调用 Polymer/ISL 计算依赖距离
- 选择空间循环（依赖距离 ≤ 1）
- 执行循环置换（空间循环移到外层）
- 执行多级分块（array_part + latency）

**SystolicDataflowGeneration.cpp** ⭐ (Pass 2: `-systolic-dataflow-generation`)
- 数组引用分组分析（IO/PE/Drain）
- IO 层级分析（确定 L1/L2/L3）
- 生成 `SystolicDataflow` Dialect：
  - `systolic.io.module<L3>` - Global Memory 接口
  - `systolic.io.module<L2>` - 双缓冲中间层
  - `systolic.io.module<L1>` - PE 接口
  - `systolic.pe.array` - PE 阵列
  - `systolic.drain.module` - 输出模块
- 双缓冲逻辑生成（ping-pong）

**SystolicDataflowToHLS.cpp** ⭐ (Pass 3: `-systolic-dataflow-to-hls`)
- 将 `SystolicDataflow` Dialect 降级到 `HLS` Dialect
- 双缓冲逻辑转换为 Affine loops + SCF if
- Stream 操作映射到 HLS Stream

**EmitHLSCpp.cpp** (Translation: `-emit-hlscpp`)
- 遍历 HLS Dialect IR
- 生成 HLS C++ Kernel 代码
- 插入 Pragma（pipeline, array_partition 等）

**EmitHostCode.cpp** ⚠️ (Translation: `-emit-host-code`, 预留接口)
- 生成 HLS Testbench（用于 Vivado HLS 仿真）
- 生成 OpenCL Host 代码（用于 Xilinx OpenCL 运行时）
- 支持其他目标平台（TAPA、Catapult HLS 等）
- **当前状态**: 接口已预留，实现待后续开发

## 依赖

- LLVM/MLIR 18 (通过 Polygeist)
- Polygeist + Polymer (ISL 模式)
- ISL (Integer Set Library)

## 构建

本项目使用 **统一构建方式**（参考 [Polygeist README Option 2](third_party/Polygeist/README.md#option-2-using-unified-llvm-mlir-clang-and-polygeist-build)），一次性构建 LLVM/MLIR/Polly/Polygeist/Polymer 和 mlir-systolic。

### 快速开始

```bash
# 1. 初始化 submodule
git submodule update --init --recursive

# 2. 构建 Polygeist + Polymer
./scripts/build-polygeist.sh

# 3. 构建 mlir-systolic
./scripts/build-systolic.sh
```

### 快速回归测试（MM 端到端）

构建完成后，可用以下命令跑通 opt → translate 并检查生成的 HLS C++：

```bash
./test/run_mm_e2e.sh
# 成功时输出 PASS 及生成文件路径；失败时输出 FAIL 与缺失符号。
# 详见 docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md 与 docs/status/CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md
```

### 详细说明

#### 步骤 1: 构建 Polygeist + Polymer

使用统一构建方式（参考 Polygeist README Option 2）：

```bash
./scripts/build-polygeist.sh
```

这个脚本会：
- 使用 Polygeist 的 `llvm-project` submodule 统一构建 LLVM/MLIR/Polly/Polygeist/Polymer
- 自动检测系统内存并选择合适的并行度（防止 WSL 卡死）
- 构建 Polymer 库和 mlir-systolic 需要的 MLIR 库

**手动构建**（如果脚本不适用）：

```bash
cd third_party/Polygeist
mkdir -p build && cd build

cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;polly" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DPOLYGEIST_ENABLE_POLYMER=1 \
  -DPOLYGEIST_POLYMER_ENABLE_ISL=1

# 构建 Polymer 库和必要的 MLIR 库
ninja -j2 PolymerSupport PolymerTargetISL PolymerTransforms \
  MLIRArithTransforms MLIRArithValueBoundsOpInterfaceImpl MLIROptLib
```

#### 步骤 2: 构建 mlir-systolic

```bash
./scripts/build-systolic.sh
```

这个脚本会：
- 自动检测 Polygeist 构建目录
- 配置 CMake 并构建 mlir-systolic

**手动构建**（如果脚本不适用）：

```bash
mkdir -p build && cd build

cmake .. \
  -DMLIR_DIR=../third_party/Polygeist/build/lib/cmake/mlir \
  -DLLVM_DIR=../third_party/Polygeist/build/lib/cmake/llvm \
  -DPOLYGEIST_BUILD=../third_party/Polygeist/build

cmake --build . -j2
```

### 验证构建

构建完成后，检查工具是否可用：

```bash
./build/bin/systolic-opt --help
./build/bin/systolic-translate --help
```

### 故障排除

**问题 1: 找不到 MLIR_DIR**

确保 Polygeist 已构建完成，并且路径正确：
```bash
ls third_party/Polygeist/build/lib/cmake/mlir/MLIRConfig.cmake
```

**问题 2: 找不到 Polymer 库**

确保构建了 Polymer 库：
```bash
ls third_party/Polygeist/build/lib/libPolymer*.a
```

**问题 3: 构建时 WSL 卡死**

使用更少的并行任务：
```bash
ninja -j1  # 单线程，最安全
```

更多信息请参考 [docs/BUILD_AND_SERVER_ENVIRONMENT.md](docs/BUILD_AND_SERVER_ENVIRONMENT.md) 与 [docs/guide/BUILD_GUIDE.md](docs/guide/BUILD_GUIDE.md)。

## 使用示例

```bash
# 1. C 代码转换为 MLIR (使用 Polygeist)
cgeist matmul.c -function=matmul -S -memref-fullrank > matmul.mlir

# 2. 应用脉动阵列变换
systolic-opt matmul.mlir \
  -systolic-transform="array-part=16,16,16 latency=8,8" \
  -systolic-dataflow \
  -o matmul_systolic.mlir

# 3. 生成 HLS C++
systolic-translate matmul_systolic.mlir -o matmul_hls.cpp
```

## 与 AutoSA 的对比

| 功能 | AutoSA | mlir-systolic |
|------|--------|---------------|
| **FPGA Kernel 生成** | | |
| 前端 | PET (C解析) | Polygeist (C→MLIR) |
| 依赖分析 | ISL | Polymer (ISL) |
| 调度 | ISL schedule tree | MLIR Affine + Polymer |
| 变换 | ISL AST build | MLIR transforms |
| Kernel 代码生成 | 自定义 C printer | EmitHLSCpp |
| **Host 端代码生成** | | |
| HLS Testbench | ✅ 支持 | ⚠️ 预留接口（暂不实现） |
| OpenCL Host | ✅ 支持 | ⚠️ 预留接口（暂不实现） |
| 其他平台 | ✅ TAPA, Catapult | ⚠️ 预留接口（暂不实现） |

## 目录结构

```
mlir-systolic/
├── include/systolic/
│   ├── Dialect/
│   │   ├── HLS/                    # HLS Dialect 定义
│   │   └── SystolicDataflow/ ⭐    # SystolicDataflow Dialect 定义
│   ├── Analysis/                   # 分析接口
│   └── Transforms/                 # Pass 声明
├── lib/
│   ├── Dialect/
│   │   ├── HLS/                    # HLS.cpp (~300 行)
│   │   └── SystolicDataflow/ ⭐    # Dialect 实现
│   ├── Analysis/                   # SpaceTimeAnalysis.cpp (~400 行)
│   ├── Transforms/
│   │   ├── SystolicTransform.cpp        # (~600 行) ⭐核心
│   │   ├── SystolicDataflowGeneration.cpp ⭐ (~1200 行) ⭐核心
│   │   └── SystolicDataflowToHLS.cpp ⭐ (~800 行) ⭐核心
│   └── Translation/
│       └── EmitHLSCpp.cpp          # (~1200 行) ⭐核心
├── tools/systolic-opt/             # 主工具
├── test/
│   ├── matmul/                     # 矩阵乘测试用例
│   └── autosa_hls_refs/            # AutoSA 生成的参考 HLS C++（可选）
└── docs/
    └── DOCS_INDEX.md               # 全量文档索引；设计文档见 docs/design/
```

## 开发路线

### Phase 1: 基础设施 ✅
- [x] 项目结构
- [x] HLS Dialect 定义 (TableGen)
- [x] 分析接口设计
- [x] 构建系统重构 ✅ **最新完成**

### Phase 2: Polymer 集成 ✅ **已完成**
- [x] Polymer 集成框架 ✅
- [x] SCoP 提取（使用 Polymer）✅
- [x] 强制使用 Polymer（移除启发式方法）✅
- [x] 自动预处理（ExtractScopStmt）✅
- [x] 依赖距离分析 ✅
- [ ] 实现基于调度树的 task 分解（可选优化）
- [ ] 测试各种循环嵌套模式

### Phase 3: 核心实现 ✅ **主要完成**
- [x] SystolicDataflow Dialect 定义 (TableGen) ✅
- [x] SystolicDataflow Dialect 实现 ✅
- [x] SystolicTransform.cpp - 变换 Pass ✅（支持参数化 spacetime）
- [x] SystolicDataflowGeneration.cpp - 数据流抽象生成 ✅
- [x] SystolicDataflowToHLS.cpp - Dialect 降级 ✅
- [x] EmitHLSCpp.cpp - 代码生成 ✅

### Phase 4: 验证
- [x] MM/MTTKRP/TTMc 端到端测试（`./test/run_all_e2e.sh`）
- [ ] 服务器 C sim / 综合验证；与 AutoSA 输出对比（Kernel 部分）

### Phase 5: Host 端代码生成（预留，暂不实现）
- [ ] HLS Testbench 生成接口
- [ ] OpenCL Host 代码生成接口
- [ ] 其他目标平台支持接口

**最新进展**：详见 [项目进展总结](docs/status/PROJECT_STATUS.md) 与 [RECENT_CHANGES_AND_NEXT_STEPS.md](RECENT_CHANGES_AND_NEXT_STEPS.md)

## 文档

- **状态与下一步**：[PROJECT_STATUS_AND_ONBOARDING.md](PROJECT_STATUS_AND_ONBOARDING.md)、[RECENT_CHANGES_AND_NEXT_STEPS.md](RECENT_CHANGES_AND_NEXT_STEPS.md)
- **架构与开发**：[docs/ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)、[docs/guide/BUILD_GUIDE.md](docs/guide/BUILD_GUIDE.md)、[docs/guide/DEVELOPMENT_GUIDE.md](docs/guide/DEVELOPMENT_GUIDE.md)
- **AutoSA 参考**：[docs/reference/autosa/](docs/reference/autosa/)（含 AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES、对照分析等）
- **全量索引**：[docs/DOCS_INDEX.md](docs/DOCS_INDEX.md) — 所有 Markdown 文档列表与说明

**核心改进**：
1. 引入 `SystolicDataflow` Dialect 作为中间抽象层
2. 支持多层（L1/L2/L3）双缓冲 IO 模块
3. 充分利用 MLIR 的 dialect 系统和 pass 组合能力
4. 清晰的抽象层级，便于扩展和优化
5. ✅ **统一构建系统** - 使用 Polygeist 统一构建方式，确保版本兼容性

## 参考项目

- [ScaleHLS](https://github.com/UIUC-ChenLab/scalehls) - HLS Dialect 和代码生成
- [AutoSA](https://github.com/UCLA-VAST/AutoSA) - 脉动阵列生成算法
- [Polygeist/Polymer](https://github.com/llvm/Polygeist) - 多面体分析

## License

MIT License
