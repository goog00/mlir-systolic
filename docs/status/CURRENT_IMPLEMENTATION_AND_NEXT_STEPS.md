# 当前实现状态与下一步工作

> **更新日期**: 2026-03  
> **目的**: 结合文档与源码梳理已实现与下一步。**最新下一步与阶段清单**见根目录 [RECENT_CHANGES_AND_NEXT_STEPS.md](../../RECENT_CHANGES_AND_NEXT_STEPS.md) 与 [PHASE_CODEGEN_AND_HLS_TEST.md](PHASE_CODEGEN_AND_HLS_TEST.md)。

---

## 一、已验证的当前实现

### 1. 构建与工具链 ✅

- 构建脚本：`build-polygeist.sh` + `build-systolic.sh`，可完成编译。
- 工具可用：
  - `systolic-opt`：支持 `--systolic-transform`、`--systolic-dataflow-generation`。
  - `systolic-translate`：接受 MLIR 输入，生成 HLS C++（无单独 `-emit-hlscpp` 选项，默认为 HLS 输出）。

### 2. Pass 1: SystolicTransform ✅

- **Polymer 集成**：可用；无 scop.stmt 时自动跑 Reg2Mem + ExtractScopStmt。
- **依赖分析**：通过 Polymer/ISL 得到依赖距离（如 3 个依赖）。
- **空间/时间循环**：支持参数化选择（ParametricSpaceTime），ST0–ST5。
- **多级分块**：array_part、latency 等通过选项/属性注入。
- **输出**：Affine IR + 函数属性（如 `systolic.space_time_mode`、`systolic.array_part`、`systolic.latency`、`systolic.pe_array_size` 等）。

**运行示例**（3 循环 matmul）：

```bash
./build/bin/systolic-opt test/minimal_matmul.mlir --systolic-transform --systolic-dataflow-generation -o out.mlir
# 输出: Preprocessing done, Dependence analysis OK, 空间循环选择与配置打印
```

### 3. Pass 2: SystolicDataflowGeneration ✅（以属性形式）

- **数组引用分组**：analyzeArrayReferences → IO/PE/Drain 分类。
- **写时重排分析**：WriteTimeReorderingAnalyzer 结果写入函数属性（`systolic.reorder.<array>.dims/perm`）。
- **配置读取**：从 SystolicTransform 写入的属性读取 pe_array_size、latency、array_part。
- **参数化数据流**：ParametricSpaceTime 下的数据流方向分析。
- **说明**：当前并未把 IR 替换为 SystolicDataflow Dialect 的 op，而是**在原有 Affine/Func 上附加属性**，供下游 translate 使用。

### 4. HLS C++ 生成 ✅

- **systolic-translate**：内置 `SystolicHLSEmitter`，根据模块中的 `func.func` 及上述属性，生成固定模板的 HLS C++。
- **内容**：A/B 的 IO L3/L2（serialize、intra/inter trans、boundary）、PE、C drain（L1/L2/L3、serialize）及顶层 kernel 调用。
- **Pragma**：含 `#pragma HLS PIPELINE II=1`、`ARRAY_PARTITION`、stream 等。
- **参数**：`--array-part`、`--latency`、`--simd`、`--size` 等可调。

**运行示例**：

```bash
./build/bin/systolic-translate /tmp/out.mlir -o /tmp/out.cpp
# 成功生成完整 HLS C++ 文件（含类型定义、各模块声明与实现）
```

### 5. 其他已存在组件

- **SystolicDataflowToHLS**：框架在，将 SystolicDataflow 降到 HLS Dialect；当前主流程是「Affine + 属性 → translate 内嵌发射器」，未强制经过该降级。
- **EmitHLSCpp（lib/Translation）**：若与 translate 内嵌逻辑重复，可视为备用或待统一。
- **ParametricSpaceTime / SpaceTimeAnalysis**：ST0–ST5 与数据流方向已接入。
- **WriteTimeReorderingAnalysis**：分析已做并写属性；**2D 重排已接入代码生成**：L2 用 getArrayDims/applyAccessPermutation，L3_in_serialize 与 drain_serialize 在存在 reorder 属性时按重排顺序读/写（见 docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md）。

---

## 二、与设想/文档的对照

| 设想/文档目标 | 当前状态 |
|---------------|----------|
| 单核：Affine → 依赖分析 → space-time → 分块 → IO/PE/FIFO → HLS | ✅ 主路径打通（属性驱动 + 固定模板） |
| ST0–ST5 参数化 | ✅ 已支持 |
| 与 AutoSA 行为对齐（单 kernel） | 🟡 主流程有，未做系统对比与回归测试 |
| 多 kernel（MM/MTTKRP/TTMc）| ✅ translate 支持 3 输入 + 1 输出、双/三规约 r1/r2、2D/3D 输出 |
| 写时重排/读时重排应用到生成 | ✅ 2D 已接入 L2、L3_in_serialize、drain_serialize（见 EXISTING_OPTIMIZATIONS_IN_CODE.md） |
| SystolicDataflow 作为显式 IR 层 | 🟡 有 Dialect 与降级框架，主流程仍用属性 |
| Host 端（Testbench/OpenCL） | ⚠️ 预留，未实现 |
| 跨层/共用脉动阵列 + 多算子映射 | 📋 愿景文档有，未实现 |
| 自动调优 | 📋 路线图有，未实现 |

---

## 三、下一步工作

**当前优先级与具体条目**见根目录 [RECENT_CHANGES_AND_NEXT_STEPS.md](../../RECENT_CHANGES_AND_NEXT_STEPS.md) 与 [PHASE_CODEGEN_AND_HLS_TEST.md](PHASE_CODEGEN_AND_HLS_TEST.md)（本阶段清单与后续改进）。主要包括：服务器 C sim/综合验证、写时重排分析扩展、FIFO 深度推导、更多 kernel 等。

---

## 四、验证与脚本

- **全量 e2e**：`./test/run_all_e2e.sh`（MM、MTTKRP、TTMc、写时重排 2D/3D，共 5 项）
- **单测**：`./test/run_mm_e2e.sh`、`./test/run_mttkrp_e2e.sh`、`./test/run_ttmc_std_e2e.sh`、`./test/run_reorder_e2e.sh`、`./test/run_reorder_3d_e2e.sh`
- **生成供服务器 HLS**：`./test/generate_hls_for_server.sh` → `build/hls_for_server/`
- **写时/读时重排**：已接入 L2、L3_in_serialize、drain_serialize，见 [EXISTING_OPTIMIZATIONS_IN_CODE.md](../design/EXISTING_OPTIMIZATIONS_IN_CODE.md)

---

## 五、参考文档

- 状态与阶段： [PROJECT_STATUS.md](PROJECT_STATUS.md)、[PHASE_CODEGEN_AND_HLS_TEST.md](PHASE_CODEGEN_AND_HLS_TEST.md)
- 设计： [SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](../design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md)、[EXISTING_OPTIMIZATIONS_IN_CODE.md](../design/EXISTING_OPTIMIZATIONS_IN_CODE.md)、[HLS_SEMANTIC_AUDIT.md](../design/HLS_SEMANTIC_AUDIT.md)
- 愿景与架构： [VISION_AND_DESIGN_GOALS.md](../VISION_AND_DESIGN_GOALS.md)、[ARCHITECTURE_OVERVIEW.md](../ARCHITECTURE_OVERVIEW.md)、[guide/CODE_STRUCTURE.md](../guide/CODE_STRUCTURE.md)
- 全量索引： [DOCS_INDEX.md](../DOCS_INDEX.md)
