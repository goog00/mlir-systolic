# 移位替代除/取模的适用范围、HLS 优化与生成代码正确性

> 回答：（1）索引中除法是否都能用移位处理；（2）HLS 工具是否已做该优化；（3）生成 HLS C 的逻辑/语法与计算正确性检查。

---

## 1. 是否所有索引中的除法都能用移位处理？

**不能；只能处理除数为 2 的幂的情况。**

- **能用移位/掩码等价替换的**（当前已实现）：
  - `x % N` → `x & (N-1)` **仅当 N 为 2 的幂**（否则 `& (N-1)` 与 `% N` 不等价）。
  - `x / N` → `x >> log2(N)` **仅当 N 为 2 的幂**（否则整数除法与右移不等价，例如 7/3 ≠ 7>>1）。
- **不能这样替换的**：
  - 除数为非 2 的幂（如 3、5、10、12）时，必须保留 `%` 和 `/`，或改用其它方式（查表、乘逆元等），不能简单用移位/掩码。

因此，本项目的“强度削减”**只针对 2 的幂**：在 `systolic-translate` 中用 `isPowerOf2(N)` 判断，仅当为真时才生成 `& (N-1)` 和 `>> log2(N)`，否则仍生成 `% N` 和 `/ N`。典型配置（如 `array_part=8/16`、`latency=4/8`）以及写时重排的常见维度（如 8、32）多为 2 的幂，故多数热点路径已用移位/掩码。

---

## 2. HLS 工具是否已经会做“移位代替除/取模”？

**部分会，但不保证；在生成阶段做更稳妥。**

- **Vivado HLS / Vitis HLS** 会对常量除数做一定优化，有时会把 `x % 8` 优化成 `x & 7`、`x / 8` 优化成 `x >> 3`，但在 **pipeline 区域** 内行为依赖版本和上下文，**不保证**一定优化，且有时会劣化 II。
- 在**源码生成阶段**就写出移位/掩码，可以：
  - 不依赖 HLS 的优化能力，**稳定**地在 pipeline 内避免 %、/；
  - 有利于**达到并保持 II=1**、提高频率；
  - 与“在 C 里写清楚意图、减少对后端优化猜测”的实践一致。

结论：**不能依赖 HLS 自动做全；在 translate 里对 2 的幂做强度削减是必要且有益的。**

---

## 3. 生成 HLS C 的检查：逻辑、语法与计算正确性

### 3.1 已修复：声明与定义维度一致

- **问题**：模块**声明**中 L2 的 local 数组为 `[latency][1]`（如 `[4][1]`），而**定义**中 `getArrayDims()` 返回三维（如 `[4][1][8]`），导致声明与定义不一致，存在编译/链接风险。
- **修复**：
  - `emitModuleDeclarations()` 中，对每个输入数组用 `getArrayDims(name)` 得到 `d0,d1,d2`，声明为 `local_name[d0][d1][d2]`。
  - `emitIOL2In()` 与 `emitIOL2InBoundary()` 中，local_*_ping / local_*_pong 的声明同样改为使用 `getArrayDims(arrayName)` 的 `[d0][d1][d2]`。
- 这样**声明、定义、调用方**的数组维度一致，避免未定义行为。

### 3.2 强度削减与等价性（计算正确性）

- **L3 serialize 的 `split_idx`**：`c5` 范围为 `[0, c5Bound-1]`，`c5Bound = arrayPart/simd`；当 `arrayPart` 为 2 的幂时，`c5 % arrayPart` 与 `c5 & (arrayPart-1)` 在 `c5 < arrayPart` 时等价（当前循环边界保证这一点）。
- **Drain inter_trans 的 `split_idx` 与 `c6/latency`**：`c6` 范围为 `[0, latency-1]`，故 `c6 % latency == c6`，`c6 & (latency-1) == c6`；`c6 / latency == 0`，`c6 >> log2(latency) == 0`，等价。
- **2D/3D 写时重排的 idx 分解**：`idx` 为行主序线性下标，`r = idx/s1`、`c = idx%s1`（2D）及 `r0,r1,r2`（3D）与行主序语义一致；当 `s1`、`s2` 为 2 的幂时，用 `>>` 与 `&` 与用 `/`、`%` 数学上等价。

因此，**在 2 的幂前提下，当前强度削减与原始除/取模语义等价，不改变计算结果。**

### 3.3 MM / MTTKRP / 写时重排 kernel 的语义

- **MM**：PE 内 `local_arg2[c7][c6] += local_arg0[0][0] * local_arg1[0][0]`，累加后写 drain；数据流与空间循环结构未改，仅索引计算从 `%`/`/` 改为位运算，等价，故 **MM 仍为矩阵乘**。
- **MTTKRP / 其它 kernel**：同样仅替换了索引计算方式，未改数据依赖与读写顺序，**计算语义保持不变**。
- **写时重排 2D/3D**：Phase 1 解包到 buffer、Phase 2 按重排顺序写入 buffer_linear、Phase 3 写回 DRAM；idx→(r,c) 或 (r0,r1,r2) 的分解与行主序一致，位运算仅在 2 的幂时替代除/取模，**重排语义与正确性不变**。

### 3.4 ap_uint 拼接语法

- 生成代码中的 `out_data = (data_split[3], data_split[2], data_split[1], data_split[0]);` 在标准 C++ 中是逗号运算符（只取最后一项），但在 **Vivado/Vitis HLS** 中，对 `ap_uint` 类型该写法被扩展为**位拼接**（高位到低位），与 AutoSA 参考一致，无需修改。

### 3.5 全量 e2e 生成文件检查结论（MM / MTTKRP / 写时重排 2D·3D）

对 `run_all_e2e.sh` 生成的四个 HLS C 文件（`/tmp/mm_e2e_out.cpp` 等）做抽查结论：

- **声明与定义一致**：L2 输入模块的声明、定义与调用方一致——`local_arg0[4][1][8]`、`local_arg1[4][1][8]`，调用方 `local_arg0_ping/pong[4][1][8]`，与 `getArrayDims()` 一致；drain 的 `local_arg2[4][1]` 为 2D，符合设计。
- **写时重排路径**：2D 重排含 `buffer[1024][32]`、`buffer_linear[32768]` 及 Phase 2 `buffer_linear[linear] = buffer[d1][d0]`；3D 重排含 `buffer[64][8][8]`、`buffer_linear[4096]` 及 Phase 2 按重排顺序写 `buffer_linear`，结构正确。
- **RESOURCE / 位运算**：FIFO 带 `core=FIFO_SRL`，local 带 `core=RAM_2P_BRAM`；L3/drain 的 `split_idx` 及 2D/3D 的 idx 分解在 2 的幂下使用 `&`、`>>`。未发现明显语法或维度不一致问题；数值正确性建议在具备 HLS 环境时做 C sim 验证。

### 3.6 建议的持续检查

- **语法**：用脚本或 CI 对生成 C 做一次简单解析/编译（如 `g++ -fsyntax-only` 或 HLS 的 parser），确保无语法错误。
- **逻辑**：对 MM/MTTKRP 等做小规模 **C sim**（或 RTL sim），对比参考实现或黄金输出，确认数值一致。
- **回归**：每次改 emit 逻辑后跑全量 e2e（`run_all_e2e.sh`），确认仍生成 `buffer_linear`、位运算等预期模式。

---

## 4. 小结

| 问题 | 结论 |
|------|------|
| （1）是否所有除/取模都能用移位？ | **否**；仅当除数为 2 的幂时可等价替换，当前实现已按此条件生成。 |
| （2）HLS 是否会自动做？ | **不一定**；在生成阶段做强度削减更稳妥，有利于 II 与频率。 |
| （3）生成 HLS C 正确性？ | **声明与定义已统一为 getArrayDims**；强度削减在 2 的幂下与除/取模等价；MM/MTTKRP/重排语义未改。建议加语法检查与 C sim 回归。 |
