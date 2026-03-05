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
| 通用 loop body migration | 🟡 未完成，偏 3-loop MM |
| 写时重排/读时重排应用到生成 | ✅ 2D 已接入 L2、L3_in_serialize、drain_serialize（见 EXISTING_OPTIMIZATIONS_IN_CODE.md） |
| SystolicDataflow 作为显式 IR 层 | 🟡 有 Dialect 与降级框架，主流程仍用属性 |
| Host 端（Testbench/OpenCL） | ⚠️ 预留，未实现 |
| 跨层/共用脉动阵列 + 多算子映射 | 📋 愿景文档有，未实现 |
| 自动调优 | 📋 路线图有，未实现 |

---

## 三、下一步应完成的工作（按设想优先级）

### 短期（1–2 周，与 NEXT_STEPS_ROADMAP / PROJECT_STATUS 一致）

1. **测试与验证**
   - 为 MM 跑齐 ST0–ST5 的用例（当前多为 ST3）。
   - 用 `test/minimal_matmul.mlir` 或与 AutoSA 对齐的 C→MLIR 用例，做端到端测试并对比 AutoSA 输出结构。
   - 可选：引入简单 lit 或脚本，自动化「opt → translate → 检查生成文件存在/关键符号」。

2. **循环体迁移补齐**
   - 实现**通用 loop body migration**，使非 3-loop MM（如 4 循环 MTTKRP、5 循环 CNN）也能正确迁移到 PE/IO 结构。
   - 文档与代码中的 FIXME/TODO：SystolicDataflowToHLS 的「通用循环体迁移」、DataflowGeneration 的「迁移到 PE 的循环体」。

3. **写时重排接到代码生成**
   - 将 `systolic.reorder.*` 属性在 systolic-translate（或统一后的 EmitHLSCpp）中真正用于生成逻辑（访问顺序/缓冲布局），而不是只做分析。

### 中期（1–2 月）

4. **配置流重构（可选）**
   - 定义 `SystolicConfigAttr`（或等价结构化属性），将 space_time、array_part、latency、pe_array_size 等从零散属性改为单一结构化配置，全流程传递。

5. **Kernel 与测试扩展**
   - MTTKRP、TTMc、CNN、LU 等：在通用 loop body migration 基础上，为每种 kernel 增加测试与（若可能）与 AutoSA 的结构对比。
   - 文档中「待补充：ST0/1/2/4/5、CNN、MTTKRP、TTMc、TTM、LU」逐步勾选。

6. **双缓冲与 HLS 质量**
   - 在 SystolicDataflowToHLS 或生成侧完善双缓冲逻辑；减少生成冗余、统一代码风格，便于与 AutoSA 对比。

### 远期（愿景文档）

7. **共用脉动阵列与多算子映射**
   - 多核描述（多 Affine 核/多 linalg op）→ 共用 SA 架构选择 → 每核在固定 SA 上的 tiling/调度生成。
8. **Host 端**
   - HLS Testbench、OpenCL Host 等。
9. **自动调优**
   - 单核与多核的搜索空间、代价模型或启发式，与脚本/框架集成。

---

## 四、建议的立即行动

1. **运行并固化一条端到端命令**（便于后续回归）✅  
   已提供 **`test/run_mm_e2e.sh`**：运行 opt → translate，并检查生成 cpp 含 kernel0、PIPELINE、DATAFLOW、PE_wrapper 等，输出 PASS/FAIL。示例：
   ```bash
   ./test/run_mm_e2e.sh
   # 或指定输入/输出: ./test/run_mm_e2e.sh test/minimal_matmul.mlir /tmp/out.mlir /tmp/out.cpp
   ```

2. **补充/整理测试**  
   将 `test/minimal_matmul.mlir` 纳入正式测试目录（如 `test/matmul/`），并增加 README 或 TESTING_GUIDE 中对该流程的说明。

3. **写时/读时重排应用到生成** ✅  
   2D 已接入：L2（getArrayDims/applyAccessPermutation）、L3_in_serialize、drain_serialize。见 [EXISTING_OPTIMIZATIONS_IN_CODE.md](../design/EXISTING_OPTIMIZATIONS_IN_CODE.md)。

4. **通用 loop body migration**  
   这是支持 MTTKRP、CNN 等更多 kernel 的前提。当前 4-loop MTTKRP 已可 opt→translate 生成 HLS（数组名从参数推导，见 `test/minimal_mttkrp.mlir`、`test/run_mttkrp_e2e.sh`）。

---

## 五、参考文档

- **脉动阵列优化改进计划**：`docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md`（MTTKRP/TTMc 写时重排、MM/CNN 小规模、分阶段实现与任务清单）
- 愿景与阶段：`docs/VISION_AND_DESIGN_GOALS.md`
- 状态概览：`docs/status/PROJECT_STATUS.md`
- 下一步路线：`docs/status/NEXT_STEPS_ROADMAP.md`
- 代码结构与 FIXME：`docs/guide/CODE_STRUCTURE.md`
- 架构与目标：`docs/ARCHITECTURE_OVERVIEW.md`
