# MM 4PE 对比（mlir-systolic vs AutoSA，同参数，无 host-serialize）

## 参数与流程
- 参数：`array_part=8`, `latency=4`, `simd=1`, 目标时钟 5ns，器件 `xcu200-fsgd2104-2-e`
- AutoSA 在 `third_party/AutoSA` 根目录执行，输出保留并覆盖在 `autosa.tmp/output`
- 本次 AutoSA 命令不带 `--host-serialize`

## HLS C 文件对比
- mlir-systolic: `hls_validation/mm_4pe_compare/mlirsystolic_kernel.cpp`
- AutoSA: `hls_validation/mm_4pe_compare/autosa_kernel.cpp`

| 指标 | mlir-systolic | AutoSA |
|---|---:|---:|
| 代码行数 | 1195 | 1372 |
| `hls::stream` 声明次数 | 138 | 83 |
| `#pragma HLS PIPELINE` 次数 | 24 | 21 |
| `#pragma HLS RESOURCE` 次数 | 45 | 42 |
| `#pragma HLS DATAFLOW` 次数 | 1 | 1 |

## csynth 对比（Vivado HLS 2019.2）
- mlir-systolic 报告：`hls_validation/mm_4pe_compare/mlirsystolic_kernel0_csynth.rpt`
- AutoSA 报告：`hls_validation/mm_4pe_compare/autosa_kernel0_csynth.rpt`

| 指标 | mlir-systolic | AutoSA |
|---|---:|---:|
| 目标时钟 (ns) | 5.00 | 5.00 |
| 估计时钟 (ns) | 4.375 | 4.375 |
| 数据流总延迟 (cycles) | 8755 | 65560 |
| BRAM_18K | 174 | 108 |
| DSP48E | 20 | 20 |
| FF | 12681 | 7367 |
| LUT | 19072 | 15859 |

## 模块级归因（基于 csynth Instance 表）

### 关键延迟上界（cycles，取该分组实例中的 `max`）

| 分组 | mlir-systolic | AutoSA |
|---|---:|---:|
| PE | 8203 | 65547 |
| Input(A/B) | 8745 | 67860 |
| Drain(C) | 546 | 2178 |

说明：这里是“分组内关键实例的上界”，不是可直接相加的总时延；总时延仍以 dataflow 汇总行为准。

### 模块资源归因（按 Instance 表分组求和）

| 分组 | BRAM_18K (mlir/AutoSA) | DSP (mlir/AutoSA) | FF (mlir/AutoSA) | LUT (mlir/AutoSA) |
|---|---:|---:|---:|---:|
| PE | 4 / 4 | 20 / 20 | 4008 / 3256 | 4628 / 3980 |
| Input(A/B) | 64 / 64 | 0 / 0 | 3254 / 1362 | 3652 / 3874 |
| Drain(C) | 16 / 16 | 0 / 0 | 850 / 602 | 2242 / 2526 |
| AXI 接口 | 90 / 24 | 0 / 0 | 4245 / 1839 | 4755 / 2361 |
| 其它控制 | 0 / 0 | 0 / 0 | 153 / 153 | 279 / 279 |

备注：该分组求和对应的是 Utilization 的 Instance 子表（总和闭合到 `Instance Total`），与顶层 `Total` 的差值主要来自 FIFO/表达式/多路器等非实例项。

## 结果解读
- 两边估计时钟一致（4.375ns，约 228.57MHz）。
- 在“无 host-serialize”的同参数下，AutoSA 端到端总延迟仍明显高于 mlir-systolic（65560 vs 8755 cycles）。
- 从模块上界看，差异主要集中在 `PE` 与 `Input(A/B)` 链路（AutoSA 约 6.8 万级，mlir-systolic 约 0.8 万级）。
- 资源上两者 DSP 相同；AutoSA 的 AXI 与 FF 总量更低，而 mlir-systolic 在 BRAM/FF/LUT 更高。
