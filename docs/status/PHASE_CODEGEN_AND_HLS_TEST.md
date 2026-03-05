# 阶段：代码生成器重构与服务器 HLS 测试

> 本阶段目标：完成代码生成器重构、生成 MM/MTTKRP/TTMc 的 HLS 代码，在服务器上进行 HLS 测试；MTTKRP 与 TTMc 需能体现写时重排等优化。

---

## 一、本阶段目标与产出

| 目标 | 说明 |
|------|------|
| 重构代码生成器 | 由 ContractionDesc 与 MLIR 属性/形状驱动，替代大量预设模板；支持单/双/三规约、rank-2/3 输出。 |
| 生成 MM / MTTKRP / TTMc 代码 | 供服务器 HLS 综合与 C sim 的 .cpp（及可选写时重排版）。 |
| MTTKRP、TTMc 体现写时重排 | 通过「先跑写时重排分析再 transform」的流水线生成 _reorder 版本；若分析器命中则 drain 走 buffer_linear 路径。 |

**产出**：
- 统一输出目录 `build/hls_for_server/`：`mm.cpp`、`mttkrp_std.cpp`、`mttkrp_std_reorder.cpp`、`ttmc_std.cpp`、`ttmc_std_reorder.cpp`（及对应 .mlir 中间文件）。
- 脚本：`./test/generate_hls_for_server.sh [输出目录]`。

---

## 二、本阶段仍需完成的工作

| 序号 | 工作项 | 状态 | 说明 |
|------|--------|------|------|
| 1 | 代码生成器重构（ContractionDesc、Kind、r1/r2、常量、位宽等） | ✅ 已完成 | 见 [CODEGEN_REFACTOR_ASSESSMENT.md](../design/CODEGEN_REFACTOR_ASSESSMENT.md) |
| 2 | 生成脚本与统一输出目录 | ✅ 已完成 | `test/generate_hls_for_server.sh` → `build/hls_for_server/` |
| 3 | 服务器 HLS 测试 | 待用户执行 | 将 `build/hls_for_server/` 拷到服务器，跑 C sim + 综合；验证 MM/MTTKRP/TTMc 正确性与性能 |
| 4 | 写时重排在 MTTKRP/TTMc 上的体现 | 部分 | 当前写时重排分析对 MTTKRP/TTMc 的 store 模式可能未命中，故 _reorder 与基础版可能相同；可在 .cpp 中查 `buffer_linear` 确认。若需强制体现，可扩展 WriteTimeReorderingAnalyzer 或为 MTTKRP/TTMc 增加专用 reorder 测例（类似 minimal_reorder_write.mlir） |

**建议**：
- 在服务器上先跑 **MM** 的 C sim/综合，确认基线正确。
- 再跑 **MTTKRP**、**TTMc** 的 C sim，对照参考结果或黄金值。
- 若需对比「带写时重排」的性能，可先用现有写时重排 2D/3D 测例（`run_reorder_e2e.sh` / `run_reorder_3d_e2e.sh`）生成的 .cpp 做参考；MTTKRP/TTMc 的 _reorder 版本待分析器扩展后再对比。

---

## 三、本阶段结束后的可改进项

以下可在本阶段（服务器 HLS 测试通过或结论明确）之后推进，不阻塞当前交付。

### 3.1 正确性与语义

- **MTTKRP csim 与 AutoSA 对齐**：若当前仍有 mismatch，根据 [CSIM_FINDINGS_2026-03-04.md](../../hls_validation/mttkrp_std_mlirsystolic/CSIM_FINDINGS_2026-03-04.md) 继续修 PE/IO 语义或数据顺序。
- **TTMc csim**：在服务器上跑 TTMc 的 C sim，确认 rank-3 输出与三规约维 r2 的语义正确。

### 3.2 写时重排与优化

- **扩展写时重排分析**：使 MTTKRP（2D 输出）、TTMc（3D 输出）在分析阶段获得 `systolic.reorder.*`，从而 _reorder 版本真正走 buffer_linear。
- **与 autosa_hls_refs 逐项对比**：同一 kernel、相近参数下，对比模块划分、FIFO、PIPELINE/RESOURCE、drain/L3 访问，补对 II/频率/面积影响大的差异。
- **BURST、BIND_STORAGE 等 pragma**：按需在 L3/drain 增加，提升访存效率。

### 3.3 代码生成器与流程

- **getArrayDims 与 inputShapes**：无 reorder 时优先用 shape 推导 L2 缓冲维度（见评估文档高优先级）。
- **顶层 kernel 名**：支持从 `funcOp.getName()` 或 `--kernel-name` 生成，默认保留 kernel0。
- **Drain L1/L2/L3 与 r1/r2**：若多规约时 FIFO 节奏需与 PE 完全一致，再考虑在 drain 树中加规约维循环。
- **参考 AutoSA 的 IO/drain 结构**：在流程与结构上进一步对齐 AutoSA 的成熟实现，保留 MLIR 与既有优化（见 [CODEGEN_REFACTOR_ASSESSMENT.md](../design/CODEGEN_REFACTOR_ASSESSMENT.md) 第 0 节）。

### 3.4 扩展与生态

- **更多 kernel**：如 CNN、其他张量 contraction；参数与多面体选择范围（当前小规模可暂缓）。
- **Host 端与驱动**：若需在服务器上做完整端到端，可补 host 端调用与 testbench（当前以 kernel .cpp 为主）。

---

## 四、快速命令

```bash
# 本地：生成全部 HLS 代码到 build/hls_for_server/
./test/generate_hls_for_server.sh

# 本地：全量 e2e 回归
./test/run_all_e2e.sh

# 服务器：将生成目录拷出后，在 HLS 环境中综合与 C sim（具体命令依 Vivado/Vitis 环境而定）
# 例如：将 build/hls_for_server/*.cpp 拷到服务器，用 Vitis 或 Vivado HLS 打开并运行 C simulation / 综合
```

---

## 五、相关文档

- [RECENT_CHANGES_AND_NEXT_STEPS.md](../../RECENT_CHANGES_AND_NEXT_STEPS.md) — 近期修改与 e2e 状态  
- [CODEGEN_REFACTOR_ASSESSMENT.md](../design/CODEGEN_REFACTOR_ASSESSMENT.md) — 代码生成器符合性评估与 AutoSA 参考原则  
- [PROJECT_STATUS_AND_ONBOARDING.md](../../PROJECT_STATUS_AND_ONBOARDING.md) — 项目状态与上手指南  
