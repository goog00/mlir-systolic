# MTTKRP csim 问题定位（2026-03-04）

## 结论摘要
- 已完成 `csim`：`hls_validation/mttkrp_mlirsystolic/run_hls_csim.sh`
- 结果：**FAIL**（`448` 处 mismatch）
- 关键现象：`Non-zero outputs: 64 / 512`

## 复现信息
- 输入：`test/minimal_mttkrp.mlir`
- translate 参数：`--size=8 --array-part=8 --latency=4 --simd=1`
- testbench：`hls_validation/mttkrp_mlirsystolic/tb_kernel.cpp`
- csim 日志：`hls_validation/mttkrp_mlirsystolic/hls_csim.log`

## 关键证据
1. `tb_kernel.cpp` 对 8x8x8 输出做全量校验，参考值为全 8（A/B 全 1，沿 k 规约）。
2. `csim` 显示仅 64 个非零输出，其余 448 个为 0。
3. 生成的 `kernel.cpp` 中，`arg2_drain_IO_L3_out_serialize` 仅写 `arg2[0..3]`（4 个 512-bit words = 64 个 float）。

## 根因判断
- 当前 `systolic-translate` 的代码模板是 **rank-2 输出（矩阵型）** 假设。
- 对 `minimal_mttkrp` 这种 **rank-3 输出**（`memref<8x8x8xf32>`）会静默生成不完整代码：
  - 只覆盖部分输出；
  - 无法表达完整 4-loop 语义。

## 已做修复（保护性）
- 在 `tools/systolic-translate/systolic-translate.cpp` 增加输出 rank 校验：
  - 若输出 rank != 2，直接报错并停止生成。
- 验证结果：对当前 `mttkrp_dataflow.mlir` 会报
  - `unsupported output rank 3`
  - 并返回非零退出码。

## 后续建议
- 若要“正确支持该 mttkrp 变体”，需要扩展 translator 的模板能力（rank-3 输出与对应 drain/serialize/PE 调度）。
- 在该能力落地前，建议先使用 rank-2 输出问题做对比验证，避免误判性能与正确性。
