# MM 4PE 小规模 HLS 验证报告（2026-03-04）

## 1) 验证目标
- 在独立目录验证 mlir-systolic 生成的 2x2 PE（约 4PE）MM kernel。
- 先检查 HLS 综合是否可通过，再用 csim 做功能正确性验证。

## 2) 验证对象与配置
- 输入 MLIR: `test/minimal_matmul.mlir`
- 生成参数: `--size=32 --array-part=8 --latency=4 --simd=1 --fifo-depth=2`
- 生成文件: `hls_validation/mm_4pe_mlirsystolic/kernel.cpp`
- 目标器件: `xcu200-fsgd2104-2-e`
- 时钟: `5ns`

## 3) 执行命令
### 3.1 生成 kernel
```bash
cd /data/mlir-workspace/mlir-systolic
./build/bin/systolic-opt test/minimal_matmul.mlir \
  --systolic-transform --systolic-dataflow-generation \
  -o hls_validation/mm_4pe_mlirsystolic/mm_dataflow.mlir

./build/bin/systolic-translate hls_validation/mm_4pe_mlirsystolic/mm_dataflow.mlir \
  --size=32 --array-part=8 --latency=4 --simd=1 --fifo-depth=2 \
  -o hls_validation/mm_4pe_mlirsystolic/kernel.cpp
```

### 3.2 跑综合
```bash
enable_vitis
cd hls_validation/mm_4pe_mlirsystolic
./run_hls_csynth.sh
```

### 3.3 跑 csim
```bash
enable_vitis
cd hls_validation/mm_4pe_mlirsystolic
./run_hls_csim.sh
```

## 4) 综合结果（csynth）
- 结果：通过（生成 RTL 成功）
- 报告：`hls_mm_4pe_prj/solution1/syn/report/kernel0_csynth.rpt`
- 关键指标：
  - Estimated clock: 4.375 ns（目标 5.00 ns）
  - Estimated Fmax: 228.57 MHz
  - Latency: 8755 cycles
  - Resource: BRAM=174, DSP=20, FF=11901, LUT=18372
- 备注：存在 `Loop constraints NOT satisfied`（主要在 `arg2_drain_IO_L3_out_serialize`）。

## 5) csim 结果
- 结果：通过（`CSIM PASS`）
- testbench：`tb_kernel.cpp`（A/B 全 1，期望 C 全 32）
- 最终结果：所有输出与参考一致（0 errors）。

## 6) 根因定位进展
### 6.1 已确认并修复的问题
- 位置：`tools/systolic-translate/systolic-translate.cpp` 的 `emitPE()`
- 问题：PE 局部累加器 `local_out`（即 `local_arg2`）在累加前未初始化。
- 修复：在 `c2 == 0 && c5 == 0` 时先置零，再做累加。

### 6.2 本轮新增修复
- 位置：`tools/systolic-translate/systolic-translate.cpp` 的 `emitIOL2InIntraTrans()`
- 问题：`intra_trans` 从 `local_*[c7][0][c5]` 读取，`c5>0` 时会读到未加载槽位。
- 修复：改为从 `local_*[c7][0][0]` 读取 packed word，再由 `split_idx=c5` 取对应 lane。

## 7) 结论
- **综合链路可用**：当前 4PE 配置可以在 Vitis/Vivado HLS 2019.2 下通过综合。
- **功能正确性通过**：在当前 MM 4PE 配置下，csim 已通过。

## 8) 下一步建议
1. 基于当前修复后代码，回跑一次 `run_hls_csynth.sh` 并记录与前次报告的指标差异。
2. 增加一个非全 1 输入的 MM testbench 用例（随机或结构化），提升 csim 覆盖度。
3. 开始与 AutoSA 同参数配置做模块/pragma/资源逐项对比。
