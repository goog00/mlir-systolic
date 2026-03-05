# 近期修改与下一步工作

> **更新日期**: 2026-03  
> 供**服务器或新环境上的 Agent** 快速了解当前状态与下一步。详细历史与设计见 [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md)。

---

## 给 Agent 的快速入口

### 当前状态

- **语义修复（已完成）**：L2 原先每 (c0,c1) 只加载一次数据却对全部 c2 复用，已修复为**每个 c2 先 inter_trans 再 intra_trans**；L3_serialize（MatmulLike）按 c2 重复输出、L3_in 增加 c2 循环，与 L2 消费一致。见 [docs/design/HLS_SEMANTIC_AUDIT.md](docs/design/HLS_SEMANTIC_AUDIT.md)。
- **e2e**：`./test/run_all_e2e.sh` 共 5 项（MM、MTTKRP、标准 TTMc、写时重排 2D/3D），应全部通过。
- **支持 kernel**：MM、MTTKRP（4 循环）、TTMc（3D 输出、三规约 r2）、写时重排 2D/3D；最多 3 输入 + 1 输出。

### 验证命令（本地）

```bash
./scripts/build-systolic.sh
./test/run_all_e2e.sh
./test/generate_hls_for_server.sh   # 输出 → build/hls_for_server/
```

### 下一步（建议顺序）

| 谁做 | 做什么 |
|------|--------|
| **服务器** | 将 `build/hls_for_server/` 拷到服务器，对 mm.cpp、mttkrp_std.cpp、ttmc_std.cpp 做 **C sim + 综合**，确认数值正确性与 II/资源。 |
| **本地/服务器** | 若 csim 通过：与 AutoSA 同 kernel/参数对比性能；若失败：根据 [docs/design/HLS_SEMANTIC_AUDIT.md](docs/design/HLS_SEMANTIC_AUDIT.md) 与 [docs/status/PHASE_CODEGEN_AND_HLS_TEST.md](docs/status/PHASE_CODEGEN_AND_HLS_TEST.md) 的待验证项排查。 |
| **可选** | 写时重排分析扩展（使 MTTKRP/TTMc 的 _reorder 版本真正带上 buffer_linear）；FIFO 深度由 dataflow 推导；更多 kernel（CNN 等）。 |

**关键文档**：[PROJECT_STATUS_AND_ONBOARDING.md](PROJECT_STATUS_AND_ONBOARDING.md)（上手）、[docs/status/PHASE_CODEGEN_AND_HLS_TEST.md](docs/status/PHASE_CODEGEN_AND_HLS_TEST.md)（本阶段清单与后续改进）、[docs/design/HLS_SEMANTIC_AUDIT.md](docs/design/HLS_SEMANTIC_AUDIT.md)（语义审计）。

---

## 最新修改：L2 与 c2 语义修复（2026-03）

- **问题**：L2 仅在 (c0,c1) 调用一次 `inter_trans`，再对 c2 循环多次 `intra_trans`，导致所有 c2 共用同一批 A/B 数据，与 MM 规约语义不符。
- **修复**：  
  - **L2**：`emitIOL2In` / `emitIOL2InBoundary` 将 `inter_trans` 移入 c2 循环内，每个 (c0,c1,c2) 先加载再消费。  
  - **L3_serialize**：MatmulLike 时增加外层 c2 循环，同一批 DRAM 数据按 c2 重复输出 numTiles 次（128×4 words）。  
  - **L3_in**：MatmulLike 时增加 c2 循环，读/写次数与 L3_serialize 一致。
- **审计文档**：[docs/design/HLS_SEMANTIC_AUDIT.md](docs/design/HLS_SEMANTIC_AUDIT.md)。

---

## 主要修改摘要（历史）

- **ContractionDesc**：输出秩、规约维数（num_time_loops）、Kind（MatmulLike/MttkrpLike/TtmcLike）；PE/IO 初值/写出条件、r1/r2 循环由 ContractionDesc 驱动。
- **写时/读时重排**：2D/3D 接入 L2、L3_in_serialize、drain_serialize；kernel 选择优先带 `systolic.reorder.*.dims` 的函数。
- **HLS 友好化**：RESOURCE 系统化、2 的幂时 %/ 用位运算、DRAM 常量命名、循环位宽按 bound 计算。
- **e2e**：5 项全量（MM、MTTKRP、TTMc、写时重排 2D/3D）；`generate_hls_for_server.sh` 生成供服务器测试的 .cpp。

---

## 文档索引

| 类别 | 文档 |
|------|------|
| **上手/状态** | [PROJECT_STATUS_AND_ONBOARDING.md](PROJECT_STATUS_AND_ONBOARDING.md)、[docs/status/PHASE_CODEGEN_AND_HLS_TEST.md](docs/status/PHASE_CODEGEN_AND_HLS_TEST.md) |
| **语义/正确性** | [docs/design/HLS_SEMANTIC_AUDIT.md](docs/design/HLS_SEMANTIC_AUDIT.md) |
| **代码生成器** | [docs/design/CODEGEN_REFACTOR_ASSESSMENT.md](docs/design/CODEGEN_REFACTOR_ASSESSMENT.md)、[docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md](docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md) |
| **设计/策略** | [docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md)、[docs/VISION_AND_DESIGN_GOALS.md](docs/VISION_AND_DESIGN_GOALS.md) |
| **全量索引** | [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md) |
