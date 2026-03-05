# 项目状态（概览）

> **当前状态与下一步**以根目录 [PROJECT_STATUS_AND_ONBOARDING.md](../../PROJECT_STATUS_AND_ONBOARDING.md)、[RECENT_CHANGES_AND_NEXT_STEPS.md](../../RECENT_CHANGES_AND_NEXT_STEPS.md) 及 [PHASE_CODEGEN_AND_HLS_TEST.md](PHASE_CODEGEN_AND_HLS_TEST.md) 为准。

**最后更新**: 2026-03

---

## 当前亮点

- 构建：`build-polygeist.sh` + `build-systolic.sh`，稳定可用
- Polymer 集成：SystolicTransform 使用 Polymer/ISL 分析
- HLS 生成：systolic-translate 生成完整 kernel（L3/L2/PE/drain），支持 3 输入 + 1 输出
- 支持 kernel：MM、MTTKRP（4 循环）、TTMc（3D 输出、三规约 r2）、写时重排 2D/3D
- 语义修复：L2 每 c2 先 inter_trans 再 intra_trans；L3 按 c2 重复输出（MatmulLike）
- 写时/读时重排：2D/3D 已接入 L2、L3_in_serialize、drain_serialize
- e2e：5 项全量通过（`./test/run_all_e2e.sh`）

## 关键组件

- **SystolicTransform**：依赖分析、空间/时间循环选择（含 num_time_loops）、属性注入、多级分块
- **SystolicDataflowGeneration**：数组分组、写时重排属性、数据流/PE/FIFO 框架（属性形式）
- **systolic-translate**：ContractionDesc 驱动 PE/IO/drain 生成，支持单/双/三规约、2D/3D 输出

## 测试现状

- **e2e**：MM、MTTKRP、标准 TTMc、写时重排 2D、写时重排 3D（5 项）
- **生成脚本**：`./test/generate_hls_for_server.sh` → `build/hls_for_server/`（供服务器 HLS/csim）
- 历史 AutoSA 对照与 HLS 验证见 `hls_validation/`

## 待办/可选

- 服务器 C sim + 综合验证（MM/MTTKRP/TTMc）
- 写时重排分析扩展（使 MTTKRP/TTMc 的 _reorder 版本命中 buffer_linear）
- FIFO 深度由 dataflow 推导；更多 kernel（如 CNN）

---

详见 [CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md)（实现细节）、[DOCS_INDEX.md](../DOCS_INDEX.md)（全量文档索引）。
