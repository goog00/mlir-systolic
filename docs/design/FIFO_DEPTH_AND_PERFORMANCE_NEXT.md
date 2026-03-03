# FIFO 深度策略与脉动阵列性能下一步

> 结合 `test/autosa_hls_refs` 与 AutoSA 源码，说明：是否应由 dataflow 推导 FIFO 深度、可能副作用，以及面向性能的改进顺序。

---

## 1. AutoSA 的 FIFO 深度做法

- AutoSA 使用**单一全局** `fifo_depth`（选项 `--fifo-depth`，默认 2），**不**按 dataflow 为每个 FIFO 单独推导深度。
- 源码：`ppcg_options.c` 中 `fifo_depth` 为整型选项；`autosa_xilinx_hls_c.cpp` 等用 `prog->scop->options->autosa->fifo_depth` 为**所有** FIFO 生成 `depth=N`。
- 参考 HLS：`test/autosa_hls_refs` 中所有 MM/MTTKRP/TTMc 生成代码均为 `#pragma HLS STREAM variable=... depth=2`，并配合 `#pragma HLS RESOURCE variable=... core=FIFO_SRL`。

结论：AutoSA 采用**可配置的单一深度**，与当前 mlir-systolic 的 `--fifo-depth` 一致；**没有**“由 dataflow 推导每 FIFO 深度”的实现。

---

## 2. 由 dataflow 推导 FIFO 深度是否会有副作用？

**会。** 推导错误或过于激进时，容易带来性能或正确性问题：

| 情况 | 后果 |
|------|------|
| **估小** | FIFO 过浅 → 生产者/消费者速率不匹配时易 **backpressure、流水线 stall**，甚至 **死锁** → **性能下降**。 |
| **估大** | 面积增大（深度×位宽大时 HLS 可能用 BRAM 而非 SRL）；行为一般仍正确。 |
| **按 FIFO 分别推导** | 需对**每条** FIFO 的 dataflow 分析都正确；一条估错就可能导致整条 DATAFLOW 链 stall。 |

因此：

- **不建议**在未经验证的分析与综合/仿真前，用“由 dataflow 推导”的深度**替代**当前单一可配置深度。
- **建议**保持**单一可配置深度**（`--fifo-depth`，默认 2），与 AutoSA 对齐；若将来做推导，应作为**可选**增强，且需在典型 kernel 上做综合与 C sim 验证，避免盲目替换。

---

## 3. 面向脉动阵列性能的下一步（对照 autosa_hls_refs）

以 `test/autosa_hls_refs` 与 [AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md](../reference/autosa/AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md) 为参照，建议按**性能收益/风险比**排序：

### 3.1 优先：RESOURCE 与 pipeline 友好（无深度推导）

1. **RESOURCE pragma 对齐**（已完成）  
   AutoSA 对 FIFO 统一使用 `core=FIFO_SRL`，对 PE/drain 的 local buffer 使用 `core=RAM_2P_BRAM`。mlir-systolic 已系统化：**所有** kernel0 内声明的 FIFO（含输入 L3 序列化、L2、PE、drain 各层）均带 `#pragma HLS RESOURCE variable=... core=FIFO_SRL`；IO_L2 的 ping/pong 与 PE 的 local_out、drain 的 local_* 均带 `core=RAM_2P_BRAM`。与 AutoSA 参考一致。
2. **Pipeline 内减少 %、/**（已完成）  
   当 `array_part`、`latency` 或写时重排维度 s1/s2 为 2 的幂时，systolic-translate 生成**位运算**替代取模/除法：`x % N` → `x & (N-1)`，`x / N` → `x >> log2(N)`。应用于：L3 serialize 的 `split_idx`、drain inter_trans 的 `split_idx` 与 `c6/latency`、以及 2D/3D drain 中 idx→(r,c) 或 (r0,r1,r2) 的分解，有利于**稳定 II=1** 和频率。

### 3.2 其次：访存与结构

3. **L3 访问与 coalesce**（已实现无重排分支）  
   L3_in_serialize 无重排时改为按 (c0,c1,c3,c4g) 循环与显式 word_idx 读 DRAM，word_idx 单调递增、与 L3_in 的 tile 顺序一致，利于连续 burst。详见 [L3_COALESCE_AND_ACCESS_PATTERN.md](L3_COALESCE_AND_ACCESS_PATTERN.md)。可进一步对照 AutoSA 索引形式做逐项对比。
4. **与 autosa_hls_refs 的逐项对比**  
   对同一 kernel（如 MM 512）、相近参数，对比我们生成与 AutoSA 生成的：模块划分、FIFO 数量与位宽、PIPELINE/RESOURCE 分布、drain 结构；列出差异并优先补我们缺失的、对 II/频率/面积影响大的部分。

### 3.3 暂不优先：FIFO 深度由 dataflow 推导

- 在**没有**可靠 dataflow 模型与综合/仿真验证前，**不**将“由 dataflow 推导每 FIFO 深度”作为默认或唯一方案。
- 保持 `--fifo-depth` 作为主入口；若后续引入推导，建议：  
  - 作为可选模式（例如 `--fifo-depth=auto`）；  
  - 输出每个 FIFO 的推导深度便于检查；  
  - 在 MM/MTTKRP 等典型 kernel 上做 C sim + 综合，确认无 stall/死锁且性能不劣于固定深度。

---

## 4. 小结

| 问题 | 结论 |
|------|------|
| 由 dataflow 推导 FIFO 深度会不会有副作用？ | **会**：估小易导致 backpressure/stall/死锁，估大增加面积；按 FIFO 分别推导时单点错误影响整链。 |
| 当前建议 | 继续使用**单一可配置** `--fifo-depth`（默认 2），与 AutoSA 一致；不急于用推导深度替代。 |
| 性能下一步 | 优先做 **RESOURCE 系统化** 与 **pipeline 内减少 %、/**；其次 L3 coalesce 与和 autosa_hls_refs 的对比；FIFO 深度推导留待有验证能力后再考虑。 |
