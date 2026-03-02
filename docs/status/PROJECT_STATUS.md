# 项目状态（概览版）

**最后更新**: 2026-01-06 (更新：反映 ParametricSpaceTime 框架实现状态)  
**状态**: 核心功能已可用，正推进多配置/多 kernel 支持

---

## 当前亮点
- ✅ 构建系统稳定：统一脚本 `build-polygeist.sh` + `build-systolic.sh`
- ✅ Polymer 集成完成：`SystolicTransform` 强制使用 Polymer/ISL 分析
- ✅ Space-time=3 主路径可用：PIPELINE 数量与 AutoSA 对齐（24）
- ✅ HLS 生成可用：IO_L2_in、C_drain 等模块生成完成
- ✅ AutoSA 参考用例：11 个 ST3 配置全部通过

## 关键组件状态
- **SystolicTransform**: 依赖分析、空间/时间循环选择、属性注入完成
- **DataflowGeneration**: 数据流图/PE/FIFO 生成框架完成，循环体迁移部分完成
- **DataflowToHLS**: HLS 代码生成可用，缺少通用 loop-body migration
- **PolymerAnalysis**: SCoP 提取与调度树获取可用

## 已知问题（优先级）
- ✅ Spacetime 参数化已完成（通过 ParametricSpaceTime 框架支持 ST0-ST5）
- 🟡 Kernel 主要支持 3-loop MM，循环体迁移 TODO
- 🟡 配置流使用函数属性传递，可进一步优化为结构化属性
- 🟡 Write-time reordering 分析结果未应用到生成

## 测试现状
- **通过**：11 个 AutoSA ST3 配置（MM）
- **待补充**：ST0/1/2/4/5、CNN、MTTKRP、TTMc、TTM、LU
- 参考文档：`test/TESTING_GUIDE.md`

## 近期工作摘要（2024-12 ~ 2026-01）
- 修复构建缺少 MLIR 库问题，脚本化并行度控制
- 强制使用 Polymer，自动运行 ExtractScopStmt
- 完成 IO/L2/L3/PE 数据流生成框架和 pragma 插入（24 条）

---

## 下一步短期计划（1-2 周）
1) ✅ **Spacetime 参数化**：已完成 ParametricSpaceTime 框架，支持 ST0-5
2) **循环体迁移补齐**：通用 loop-body migration，支持非 MM kernel
3) **配置流重构**：定义 `SystolicConfigAttr`，全流程结构化传递（可选优化）
4) **测试扩充**：新增 ST0/1/2/4/5 + CNN/MTTKRP/TTMc/TTM/LU 的测试验证

## 中期计划（1-2 月）
- 写时重排：将分析结果应用到 HLS 生成
- 性能优化：double buffering、更多 PIPELINE、资源利用优化
- 代码质量：减少生成冗余，统一代码风格

## 远期计划（>3 月）
- Host 端代码生成（testbench/OpenCL）
- 多后端支持（Xilinx/Intel/GPU）
- 自动调优/性能分析工具链

---

**参考**: 详尽路线图见 `docs/status/ROADMAP.md`，代码结构见 `docs/guide/CODE_STRUCTURE.md`。  
**实现与下一步梳理**: 见 [CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md](CURRENT_IMPLEMENTATION_AND_NEXT_STEPS.md)（结合文档与运行结果的实现状态与下一步建议）。