# mlir-systolic 架构概述

本文档提供 mlir-systolic 项目的高层架构视图，帮助理解系统的整体设计、组件交互和数据流。

## 目录

1. [系统概述](#系统概述)
2. [设计目标](#设计目标)
3. [核心组件](#核心组件)
4. [编译流程](#编译流程)
5. [数据流图](#数据流图)
6. [关键设计决策](#关键设计决策)
7. [扩展性考虑](#扩展性考虑)
8. [与 AutoSA 的对比](#与-autosa-的对比)

---

## 系统概述

**mlir-systolic** 是一个基于 MLIR 的脉动阵列代码生成器，用于将高层算法（如矩阵乘法、CNN、张量操作）自动转换为针对 FPGA 的高效 HLS C++ 代码。

### 核心功能
- **多态循环分析**: 基于 Polymer/ISL 的 Affine 分析
- **脉动配置生成**: 支持 6 种 spacetime 配置 (ST0-ST5)
- **数据流优化**: 空间循环展开、数据流插入、数组分区
- **HLS 代码生成**: 生成带优化指令的 HLS C++ 代码

### 技术栈
```
┌─────────────────────────────────────┐
│         用户算法 (C/MLIR)            │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│    MLIR Affine Dialect (IR)         │  ← 多态循环表示
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│   Polymer/ISL (依赖分析)            │  ← 多面体分析
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  mlir-systolic Transforms           │  ← 本项目核心
│  • SystolicTransform                │
│  • DataflowGeneration               │
│  • DataflowToHLS                    │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│      HLS C++ Code (.cpp)            │  ← Xilinx Vitis HLS
└─────────────────────────────────────┘
```

---

## 设计目标

> **愿景与双重使命**：mlir-systolic 以 **AutoSA 式单核脉动阵列生成为主线**（单嵌套循环 → 一套 SA，多面体驱动）；在此基础上，面向 PyTorch/网络时支持 **共用脉动阵列 + 多算子映射**（多卷积/多核映射到同一套 SA 上）与 **自动调优**。详见 [VISION_AND_DESIGN_GOALS.md](VISION_AND_DESIGN_GOALS.md)。

### 主要目标
1. **自动化**: 从算法到硬件代码的全自动生成
2. **可配置性**: 支持多种 spacetime 配置和优化策略
3. **通用性**: 支持多种计算模式（MM、CNN、张量操作）
4. **可扩展性**: 易于添加新的优化 pass 和代码生成策略
5. **跨层/网络**（扩展）: 多算子共用一套脉动阵列并映射，而非每层各生成一套 SA

### 当前限制 (待改进)
- ✅ **Spacetime 参数化**: 已支持 ST0-ST5 配置（通过 ParametricSpaceTime 框架）
- 🟡 **Kernel 泛化**: 主要针对 3-loop 矩阵乘法优化，其他 kernel 类型支持有限
- 🟡 **配置流**: 使用函数属性传递配置，可进一步优化为结构化属性
- 🟡 **Write-Time Reordering**: 分析结果未完全应用到代码生成

---

## 核心组件

### 1. Transform 层 (`lib/Transforms/`)

负责 Affine IR 的高层转换和优化。

#### SystolicTransform.cpp
- **职责**: 
  - 提取空间循环 (space loops)
  - 提取时间循环 (time loops)
  - 添加 systolic 属性到 IR 中
- **关键函数**:
  - `selectSpaceLoopsParametric()`: 使用 ParametricSpaceTime 框架选择空间循环（支持 ST0-ST5）
  - `selectSpaceLoops()`: 传统模式选择（向后兼容）
  - `annotateSystolicConfig()`: 添加属性标记
- **当前状态**: 
  - ✅ 已实现参数化 spacetime 配置（通过 ParametricSpaceTime 框架）
  - 🟡 主要针对 3-loop 结构优化，其他循环嵌套支持有限

#### ArrayPartitioning.cpp
- **职责**: 分析和标记数组分区策略
- **算法**: 根据访问模式决定 `complete`/`cyclic`/`block` 分区
- **输出**: 为每个数组添加分区属性

### 2. Translation 层 (`lib/Translation/`)

负责从 MLIR 转换到目标代码（HLS C++）。

#### SystolicDataflowGeneration.cpp
- **职责**:
  - 生成数据流 IR (中间表示)
  - 插入 PE (Processing Element) 函数
  - 处理数组分区和 FIFO 插入
- **关键数据结构**:
  - `DataflowNode`: 表示数据流节点（PE、load、store）
  - `FIFOConnection`: FIFO 连接信息
- **当前状态**:
  - ✅ 使用 ParametricSpaceTime 框架支持不同 spacetime 配置
  - ✅ 支持参数化的数据流方向分析（analyzeOperandFlowsParametric）
  - 🟡 FIFO 深度计算可进一步优化

#### SystolicDataflowToHLS.cpp
- **职责**:
  - 将数据流 IR 转换为 HLS C++ 代码
  - 生成 PE 函数体、dataflow 指令、数组声明
- **关键函数**:
  - `generatePEFunction()`: 生成 PE 函数
  - `generateDataflowPragmas()`: 生成 HLS 指令
  - `migrateLoopBody()`: 迁移循环体到 PE（⚠️ 未实现）
- **当前问题**:
  - 🔴 Loop body migration 是 TODO
  - 🟡 Write-time reordering 未应用

### 3. Analysis 层 (`lib/Analysis/`)

提供支持性分析功能。

#### WriteTimeReordering.cpp
- **职责**: 分析数组写时间，优化写顺序
- **算法**: 基于 ISL 依赖分析
- **当前状态**: ✅ 分析完成，🔴 结果未应用到代码生成

### 4. Dialect 层 (`lib/Dialect/Systolic/`)

定义 systolic-specific 的 MLIR dialect。

#### 当前状态
- 🟡 定义了基本的 systolic dialect
- 🟡 主要使用 Affine dialect + 属性（attributes）
- 未来可能扩展为完整的 systolic operations

### 5. Tools 层 (`tools/systolic-translate/`)

提供命令行工具。

#### systolic-translate.cpp
- **职责**: 整合所有 passes，提供 CLI
- **用法**:
  ```bash
  systolic-translate --emit-hls input.mlir -o output.cpp
  ```
- **当前状态**:
  - ✅ 支持从函数属性读取 spacetime 配置
  - ✅ 支持参数化的代码生成
  - 🟡 配置选项可进一步扩展

---

## 编译流程

### 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入: Affine MLIR                         │
│  Example: matmul.mlir (3 nested affine.for loops)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Polymer Analysis (Dependency Analysis)                │
│  • 构建多面体模型                                                │
│  • ISL 依赖分析                                                  │
│  • 生成访问关系 (access relations)                               │
│  Output: Scop 分析结果                                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Systolic Transform                                    │
│  • extractSpaceLoops() → 选择 i, j 作为空间循环                 │
│  • extractTimeLoops() → 选择 k 作为时间循环                     │
│  • annotateSystolicConfig() → 添加属性到 IR                     │
│  Output: Annotated Affine IR with systolic attributes           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Array Partitioning Analysis                           │
│  • 分析数组访问模式                                              │
│  • 决定分区策略 (complete/cyclic/block)                         │
│  • 计算分区维度和因子                                            │
│  Output: Array partition attributes                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Write-Time Reordering Analysis                        │
│  • 分析写操作时间依赖                                            │
│  • 计算优化的写顺序                                              │
│  ⚠️ Output: 分析结果（当前未使用）                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: Dataflow Generation                                   │
│  • 创建 DataflowNode 图                                          │
│  • 插入 PE 函数节点                                              │
│  • 生成 FIFO 连接                                                │
│  • 处理数组分区（声明 sub-arrays）                               │
│  Output: Dataflow IR (内部表示)                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 6: HLS Code Generation                                   │
│  • generateModuleHeader()                                        │
│  • generateArrayDeclarations() (with partition pragmas)          │
│  • generatePEFunctions() (⚠️ loop body migration TODO)           │
│  • generateDataflowPragmas() (#pragma HLS dataflow)             │
│  • generateTopFunction()                                         │
│  Output: HLS C++ (.cpp file)                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出: HLS C++ Code (kernel_*.cpp)                               │
│  Ready for Xilinx Vitis HLS synthesis                           │
└─────────────────────────────────────────────────────────────────┘
```

### 关键阶段详解

#### Phase 2: Space/Time Loop Extraction
```
使用 ParametricSpaceTime 框架进行参数化配置：

示例：spacetime=3 (MM kernel)
  i, j, k loops
  ↓
  ParametricSpaceTime::createFromMode(3)
  ↓
  Space loops: i, j (从配置获取)
  Time loops: k (剩余循环)
  ↓
  Generate: (t, i, j) systolic array

支持的配置 (通过 ParametricSpaceTime):
  - spacetime=0: space=[0] (1D row array)
  - spacetime=1: space=[1] (1D column array)
  - spacetime=2: space=[2] (1D reduction array)
  - spacetime=3: space=[0,1] (2D output-stationary) ✅ 主要测试
  - spacetime=4: space=[0,2] (2D weight-stationary)
  - spacetime=5: space=[1,2] (2D activation-stationary)
```

#### Phase 5: FIFO Depth Calculation
```cpp
// 当前实现（简化）
int fifoDepth = calculateFIFODepth(node1, node2) {
  // 基于循环边界和依赖距离
  int distance = analyzeDependency(node1, node2);
  return max(2, distance + 1);  // 至少深度 2
}
```

---

## 数据流图

### MM (Matrix Multiply) 数据流示例

```
输入: C[i][j] += A[i][k] * B[k][j]

Spacetime=3 配置:
  空间维度: i, j → 2D PE array
  时间维度: k → 迭代 K 次

数据流:
┌─────────────────────────────────────────────────────────────────┐
│                          Top Function                            │
└─────────┬───────────────────────────────────────────────────────┘
          │
          ├─► Load A[i][k] ──► FIFO_A[I][K] ──► PE Array
          │
          ├─► Load B[k][j] ──► FIFO_B[K][J] ──► PE Array
          │
          └─► PE Array [I][J] ──► FIFO_C[I][J] ──► Store C[i][j]

PE Array (2D):
    j=0     j=1     j=2    ...  j=J-1
  ┌─────┬─────┬─────┬───┬─────┐
i=0│ PE  │ PE  │ PE  │...│ PE  │  A[0][k] →
  ├─────┼─────┼─────┼───┼─────┤
i=1│ PE  │ PE  │ PE  │...│ PE  │  A[1][k] →
  ├─────┼─────┼─────┼───┼─────┤
i=2│ PE  │ PE  │ PE  │...│ PE  │  A[2][k] →
  ├─────┼─────┼─────┼───┼─────┤
 ...│ ... │ ... │ ... │...│ ... │
  ├─────┼─────┼─────┼───┼─────┤
i=I│ PE  │ PE  │ PE  │...│ PE  │  A[I-1][k] →
  └─────┴─────┴─────┴───┴─────┘
    ↑     ↑     ↑         ↑
  B[k][0] B[k][1] ...  B[k][J-1]

每个 PE:
  Input:  A element (from left), B element (from top), partial C
  Compute: C_partial += A * B
  Output: Updated C_partial
```

### FIFO 连接示例

```
Load_A ──FIFO(depth=2)──► PE_row_0
                          │
Load_B ──FIFO(depth=2)────┼──► PE[0][0] ──FIFO──► PE[0][1] ── ...
                          │        │
                          │        └──FIFO──► Store_C
                          │
                          └──► PE[1][0] ──FIFO──► PE[1][1] ── ...
```

---

## 关键设计决策

### 1. 为什么使用 Affine Dialect?

**优点**:
- 多面体分析友好（Polymer/ISL 直接支持）
- 循环结构清晰，易于提取 space/time loops
- MLIR 生态成熟

**缺点**:
- 限制了循环类型（必须是仿射循环）
- 动态边界支持较弱

### 2. 为什么不使用完整的 Systolic Dialect?

**当前设计**: Affine IR + 属性（attributes）
```mlir
affine.for %i ... attributes {systolic.space_loop} {
  affine.for %j ... attributes {systolic.space_loop} {
    affine.for %k ... attributes {systolic.time_loop} {
      ...
    }
  }
}
```

**理由**:
- 快速原型开发
- 利用现有 Affine 基础设施
- 降低学习曲线

**未来考虑**: 定义完整的 `systolic.pe`, `systolic.dataflow` operations

### 3. 配置流设计

**当前流程** (问题):
```
SystolicTransform: spacetime → string attribute
                      ↓
DataflowGeneration: parse string → enum
                      ↓
DataflowToHLS: enum → switch cases
```

**问题**:
- 多次序列化/反序列化
- 容易出错
- 扩展困难

**建议改进**:
```
定义 MLIR Attribute: SystolicConfigAttr
  - spacetime: IntegerAttr
  - spaceLoops: ArrayAttr<IntegerAttr>
  - timeLoops: ArrayAttr<IntegerAttr>
  - arrayPartitions: DictAttr

全流程传递结构化属性，避免字符串解析
```

### 4. 为什么 Loop Body Migration 是 TODO?

**原因**:
- 当前只支持简单的 `C[i][j] += A[i][k] * B[k][j]` 模式
- 硬编码了计算模式到 PE 函数
- 通用迁移需要处理：
  - 任意计算表达式
  - 多个语句
  - 嵌套条件

**影响**: 限制了支持的 kernel 类型

---

## 扩展性考虑

### 添加新的 Spacetime 配置

**当前硬编码位置**:
1. `SystolicTransform.cpp::extractSpaceLoops()` (~185-200)
2. `SystolicDataflowGeneration.cpp::selectSpaceLoops()` (~210-240)
3. `systolic-translate.cpp::parseConfig()` (~300-350)

**扩展步骤**:
1. 定义 `ParametricSpaceTime` 数据结构
2. 重构 `extractSpaceLoops()` 为参数化版本
3. 更新 dataflow generation 逻辑
4. 添加 CLI 选项 `--spacetime=<N>`

### 添加新的 Kernel 类型

**当前限制**: 假设 3-loop MM 结构

**扩展步骤**:
1. 定义 `KernelInfo` 结构（loop count, dependencies）
2. 实现通用的 loop selection 算法
3. 实现 loop body migration (替换 TODO)
4. 添加 kernel-specific 配置

**示例**: 支持 MTTKRP (4 loops)
```cpp
struct KernelInfo {
  int numLoops = 4;  // i, j, k, l
  std::vector<int> spaceLoops = {0, 1};  // i, j
  std::vector<int> timeLoops = {2, 3};   // k, l
};
```

### 添加新的优化 Pass

**示例**: Dataflow 优化

1. 创建文件 `lib/Transforms/DataflowOptimization.cpp`
2. 实现 `DataflowOptimizationPass`
3. 注册到 `systolic-translate.cpp`
4. 添加到 pass pipeline

```cpp
// 在 systolic-translate.cpp
pm.addPass(createDataflowOptimizationPass());
```

---

## 与 AutoSA 的对比

| 特性 | mlir-systolic | AutoSA | 说明 |
|-----|---------------|--------|------|
| **框架** | MLIR | PoCC/PPCG | mlir-systolic 基于现代 MLIR 基础设施 |
| **Spacetime 支持** | ✅ ST0-ST5 (参数化) | ✅ ST0-ST5 | 已实现 ParametricSpaceTime 框架支持全部 6 种配置 |
| **Kernel 支持** | 🟡 主要 MM (3-loop) | ✅ MM/CNN/MTTKRP 等 | AutoSA 更通用，mlir-systolic 主要针对 MM 优化 |
| **代码质量** | 🟡 中等 | ✅ 高 | AutoSA 经过大量优化 |
| **可扩展性** | ✅ 好 (MLIR) | 🟡 中等 (C++) | MLIR pass 系统更易扩展 |
| **文档** | 🟡 进行中 | ✅ 完善 | AutoSA 有完整论文和文档 |
| **社区** | 🔴 小 | 🟡 中等 | MLIR 社区活跃但 systolic 支持少 |

### 技术差异

**AutoSA 优势**:
- 成熟的 spacetime 映射算法
- 完整的 FPGA 优化（double buffering, latency hiding）
- 支持多种硬件后端

**mlir-systolic 潜力**:
- MLIR 生态系统（与 TensorFlow/PyTorch 集成）
- 模块化 pass 设计
- 未来可支持更多 dialect（GPU/TPU）

### 代码生成对比

**AutoSA 生成的代码**:
```cpp
// 完整的 double buffering
for (int t = 0; t < K; t++) {
  #pragma HLS pipeline II=1
  for (int i = 0; i < I; i++) {
    local_A[i][t%2] = A[i][t];  // Ping-pong buffer
  }
  PE_compute(local_A[0][(t-1)%2], ...);
}
```

**mlir-systolic 生成的代码** (当前):
```cpp
// 基础版本
for (int i = 0; i < I; i++) {
  for (int j = 0; j < J; j++) {
    #pragma HLS dataflow
    PE(A_fifo[i], B_fifo[j], C_fifo[i][j]);
  }
}
// ⚠️ 缺少 latency hiding, double buffering
```

---

## 下一步架构演进

### 短期目标 (1-2 个月)
1. ✅ **参数化 Spacetime**: 已完成 ParametricSpaceTime 框架，支持 ST0-ST5
2. 🟡 **通用化 Kernel**: 主要支持 3-loop MM，其他 kernel 类型支持有限
3. 🟡 **完善 Loop Body Migration**: 实现 TODO

### 中期目标 (3-6 个月)
1. 定义完整的 Systolic Dialect
2. 实现 double buffering 优化
3. 添加性能分析工具

### 长期目标 (6+ 个月)
1. 多后端支持 (Xilinx/Intel/GPU)
2. 与 ML 框架集成 (TensorFlow/PyTorch)
3. 自动调优系统

---

## 参考资源

### 代码导航
- 核心实现: [lib/Transforms/SystolicTransform.cpp](../lib/Transforms/SystolicTransform.cpp)
- 代码生成: [lib/Translation/SystolicDataflowToHLS.cpp](../lib/Translation/SystolicDataflowToHLS.cpp)
- 测试: [test/TESTING_GUIDE.md](../test/TESTING_GUIDE.md)
- 详细问题: [CODE_ISSUES_DETAILED_ANALYSIS.md](../CODE_ISSUES_DETAILED_ANALYSIS.md)

### 外部资源
- MLIR 文档: https://mlir.llvm.org/
- Polymer 项目: https://github.com/kumasento/polymer
- AutoSA 论文: [FPGA'21] AutoSA: A Polyhedral Compiler for High-Performance Systolic Arrays

---

## 附录: 术语表

| 术语 | 定义 | 示例 |
|-----|------|------|
| **Spacetime** | 空间-时间映射配置 | ST3 = 2D 空间 + 1D 时间 |
| **PE** | Processing Element (处理单元) | 脉动阵列中的计算节点 |
| **FIFO** | First-In-First-Out 队列 | PE 间的数据传输通道 |
| **Dataflow** | HLS 并行执行模式 | `#pragma HLS dataflow` |
| **Array Partition** | 数组分区优化 | 将数组拆分到多个 BRAM |
| **Affine Loop** | 仿射循环 | 边界和索引是仿射表达式的循环 |
| **Scop** | Static Control Part | 多面体模型可分析的代码区域 |
| **ISL** | Integer Set Library | 多面体分析库 |

---

**文档维护**: 此文档应在架构重大变更时更新。
**最后更新**: 2026-01 (更新：反映 ParametricSpaceTime 框架实现)
