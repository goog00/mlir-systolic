# mlir-systolic 小规模 HLS 验证（MM, 4PE）

## 配置
- 输入: `test/minimal_matmul.mlir`
- translate 参数: `--size=32 --array-part=8 --latency=4 --simd=1 --fifo-depth=2`
- 预期阵列规模: `PE Array: 2 x 2`（约 4 PE）
- 目标器件: `xcu200-fsgd2104-2-e`
- 时钟: `5ns` (200MHz)

## 目录内容
- `mm_dataflow.mlir`: 经过 transform + dataflow-generation 的 MLIR
- `kernel.cpp`: 由 systolic-translate 生成的 HLS C++
- `tb_kernel.cpp`: C testbench（用于 Vivado HLS csim）
- `hls_csim.tcl`: Vivado HLS csim 脚本
- `run_hls_csim.sh`: 一键运行 csim
- `hls_csynth.tcl`: Vivado HLS 综合脚本
- `run_hls_csynth.sh`: 一键运行脚本

## 运行步骤
1. 进入仓库根目录并确保已构建 `systolic-opt/systolic-translate`
2. 配置 Xilinx 环境（2019.2）
3. 运行：

```bash
cd hls_validation/mm_4pe_mlirsystolic
chmod +x run_hls_csynth.sh
./run_hls_csynth.sh
```

## 运行 csim

```bash
cd hls_validation/mm_4pe_mlirsystolic
chmod +x run_hls_csim.sh
./run_hls_csim.sh
```

## 报告路径
- `hls_mm_4pe_prj/solution1/syn/report/kernel0_csynth.rpt`

## 当前状态（2026-03-04）
- csynth：通过（可综合，见 `kernel0_csynth.rpt`）
- csim：通过（`tb_kernel.cpp` 全 1 输入用例与参考一致）
- 已修复两处生成器问题：
	- `PE` 累加器 `local_arg2` 未初始化；
	- `IO_L2_in_intra_trans` 读取使用未加载槽位索引。
- 详细过程与结论见 `VALIDATION_REPORT_2026-03-04.md`
