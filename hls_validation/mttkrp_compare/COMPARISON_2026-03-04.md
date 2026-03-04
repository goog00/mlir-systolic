# MTTKRP 对比（mlir-systolic vs AutoSA，同参数，无 host-serialize）

## 参数与流程
- mlir-systolic：`test/minimal_mttkrp.mlir` -> `systolic-opt` -> `systolic-translate`
- translate 参数：`--size=8 --array-part=8 --latency=4 --simd=1`
- AutoSA：在 `third_party/AutoSA` 下运行，`--output-dir=./autosa.tmp/output`（默认路径）
- AutoSA 参数：`space_time[3]; array_part[8,8,2]; latency[4,4]; simd[1,1]`
- 目标器件/时钟：`xcu200-fsgd2104-2-e`，5ns

## HLS C 规模对比
| 指标 | mlir-systolic | AutoSA |
|---|---:|---:|
| 代码行数 | 1195 | 1846 |
| `hls::stream` 声明次数 | 138 | 107 |
| `#pragma HLS PIPELINE` 次数 | 24 | 28 |
| `#pragma HLS RESOURCE` 次数 | 45 | 55 |
| `#pragma HLS DATAFLOW` 次数 | 1 | 1 |

## csynth 对比（Vivado HLS 2019.2）
| 指标 | mlir-systolic | AutoSA |
|---|---:|---:|
| 目标时钟 (ns) | 5.00 | 5.00 |
| 估计时钟 (ns) | 4.375 | 4.375 |
| 数据流总延迟 (cycles) | 159 | 1409329165 |
| BRAM_18K | 142 | 188 |
| DSP48E | 20 | 38 |
| FF | 12167 | 10976 |
| LUT | 16811 | 23910 |

## 脉动阵列结构确认（基于 HLS C）

### 共同点
- 两边 `kernel0` 都是 `#pragma HLS DATAFLOW` 组织。
- 两边都实例化了 4 个 `PE_wrapper`，坐标分别是 `(0,0) (0,1) (1,0) (1,1)`，即 **2x2 PE 网格**。
- 两边都采用了“边界 dummy + drain 树”的收尾模式：右边界/下边界由 `*_PE_dummy_in` 吸收，输出走 `*_drain_IO_L1/L2/L3_out`。

### mlir-systolic 结构（`mlirsystolic_kernel.cpp`）
- 头部显式标注：`PE Array: 2 x 2`。
- PE 端口：`arg0_in/out`、`arg1_in/out`、`arg2_drain_out`（2 输入乘加，1 输出）。
- 网格连线：
	- `arg0` 沿列方向传播（如 `fifo_arg0_PE_0_0 -> fifo_arg0_PE_0_1 -> fifo_arg0_PE_0_2`）。
	- `arg1` 沿行方向传播（如 `fifo_arg1_PE_0_0 -> fifo_arg1_PE_1_0 -> fifo_arg1_PE_2_0`）。
- PE 内核循环规模较小（`c5<=7`, `c6<=3`, `c7<=3`），与本次小规模验证目标一致。

### AutoSA 结构（`autosa_kernel.cpp`）
- PE 端口：`A_in/out`、`B_in/out`、`C_in/out`、`D_drain_out`（3 输入乘加，1 输出）。
- 网格连线：
	- `A` 沿列方向传播（`fifo_A_PE_*_0/1/2`）。
	- `B`、`C` 沿行方向传播（`fifo_B_PE_0/1/2_*`、`fifo_C_PE_0/1/2_*`）。
- 同样是 2x2 PE 网格，但 PE 内部循环边界明显更大（`c0<=31`, `c1<=41`, `c2<=127`, `c6<=255`），直接导致秒级综合估计延迟。

### 结构差异结论
- **拓扑层面**：两者同为 2x2 脉动阵列，dataflow + stream 连接模式一致。
- **计算语义层面**：AutoSA PE 为三输入乘加（A*B*C），mlir 版本当前为两输入乘加（arg0*arg1）。
- **规模层面**：AutoSA 当前生成保留了更大的迭代空间，导致延迟与资源表现不在同一数量级；后续公平对比需先对齐问题规模。

## 初步结论
- 本组参数下，两者估计时钟一致（4.375ns）。
- mlir-systolic 延迟更低（159 cycles）而 AutoSA 延迟更高（1409329165 cycles）。
- 资源上，AutoSA 的 FF 低于 mlir-systolic（10976 vs 12167），但 BRAM/DSP/LUT 更高（188/38/23910 vs 142/20/16811）。
- 说明：本轮 AutoSA `mttkrp` 结果呈现秒级延迟（报告中约 7s 量级），与 mlir-systolic 的微秒级结果差距极大，提示两边当前问题规模/循环边界可能尚未完全对齐；后续需先对齐同一数据规模再做严格性能对比。
