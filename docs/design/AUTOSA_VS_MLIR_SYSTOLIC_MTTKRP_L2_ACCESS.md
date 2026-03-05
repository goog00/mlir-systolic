# AutoSA 与 mlir-systolic MTTKRP HLS：L2 阶段随机读取对比

> 对比参考：AutoSA 生成的 `third_party/AutoSA/autosa.tmp/output/src/kernel_kernel.cpp`（MTTKRP）与 mlir-systolic 生成的 MTTKRP HLS（如 `build/hls_for_server/mttkrp_std.cpp`）。  
> 项目内随机读取问题说明见：`docs/reference/autosa/AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md` §3、§4。

---

## 1. 随机读取发生位置（文档共识）

- **主要发生在 IO 的 L2 阶段**：从 L2 的 local buffer（ping/pong）向 PE 侧提供数据时，若访问下标是“多维 + 循环变量表达式”，容易导致：
  - 同一 word 被多次非连续地读（同一 bank 被反复访问、顺序不规整）；
  - 最内层 pipeline 中出现 `%`、`/`、复杂下标，增加组合逻辑、影响 II/Fmax。
- L3（DRAM）也可能存在跨步/多流交错（如文档中 B 的 `stride=21` 案例），但当前对比重点在 **L2 intra_trans** 的 local 读模式。

---

## 2. AutoSA MTTKRP：L2 intra_trans 的访问模式

### 2.1 A 的 L2 读

```cpp
// A_IO_L2_in_intra_trans: 从 local_A 读到 fifo
in_data = local_A[c8][c5][4 * c6 / 16];
// ...
int split_idx = (c6) % 4;
out_data = data_split[split_idx];
```

- **下标**：`[c8][c5][4*c6/16]`，第三维随 `c6` 每 4 步变一次（0,0,0,0, 1,1,1,1, …），即阶梯式、非单维顺序扫描。
- **循环顺序**：外层 c5 → c6 → c7 → c8（内层），对 local_A 的访问是 (c8, c5, c6/4)，同一 (c5,c6/4) 下 c8 连续，但整体上存在“同一 word 被多轮 c6 重复读 + 用 c6%4 取不同 lane”的模式。
- **取模**：最内层 pipeline 中 `split_idx = (c6) % 4`。

### 2.2 B 的 L2 读

```cpp
// B_IO_L2_in_intra_trans
in_data = local_B[c5][c7 / 8];
// ...
int split_idx = (c7) % 8;
```

- **下标**：`[c5][c7/8]`，第二维随 c7 每 8 步变一次；内层是 c8 循环，但读的只是 `(c5, c7/8)`，即**同一 word 被多次读**（c8 从 0 到 15 都读同一位置），仅通过 `c7%8` 取不同 lane。
- **取模**：`split_idx = (c7) % 8`。

### 2.3 C 的 L2 读

```cpp
// C_IO_L2_in_intra_trans
in_data = local_C[c7][4 * c6 / 16];
// ...
int split_idx = (c6) % 4;
```

- 与 A 类似：第二维 `4*c6/16`（即 c6/4）阶梯变化，访问模式非单维顺序；最内层有 `(c6) % 4`。

### 2.4 AutoSA L3（本文件）

- A：`A[8*c2+4096*c3+256*c4+4*c5+c6]`，最内层 c6，stride=1，**连续**。
- B：`B[16*c2+c3+8*c4]`，c4 最内，步长 8，本配置下相对规整。
- C：`C[32*c3+4*c4+c5]`，最内层 c5，stride=1，**连续**。  
（文档中提到的“B  stride=21”等随机读来自其他 sa-size/问题规模配置，此处不展开。）

**小结（AutoSA L2）**：L2 intra_trans 从 local buffer 读时，使用 **多维下标 + 除法/取模**（如 `[c8][c5][4*c6/16]`、`[c5][c7/8]`、`[c7][4*c6/16]`），导致访问非单维顺序、同一 word 多次读、最内层含 `%`/`/`，属于文档所指的“L2 阶段随机/非连续访问”问题。

---

## 3. mlir-systolic MTTKRP：L2 intra_trans 的改进

### 3.1 L2 读：单维顺序 + 固定其余维

```cpp
// arg0_IO_L2_in_intra_trans（对应 A 类输入）
in_data = local_arg0[c7][0][0];
// ...
int split_idx = (c5) & 7;   // 2 的幂用位与，无 %
```

- **下标**：仅 **c7** 变化，第二、三维固定为 **0**，即对 local buffer 的访问是 **沿第一维的顺序扫描**（c7=0,1,2,3…），无阶梯、无重复读同一 word 的复杂模式。
- **取 lane**：`split_idx = (c5) & 7`，用位与代替取模（strength reduction，且无 `%`）。

arg1、arg2 的 L2 intra_trans 同理：`local_arg1[c7][0][0]`、`local_arg2[c7][0][0]`。

### 3.2 L2 inter_trans 写

- 写 local buffer 时同样简化：`local_arg0[c4][0][0]`（只变一维），与读侧一致，保证 L2 内访问模式简单、可预测。

### 3.3 L3 读

- 使用显式线性 `word_idx = ...`（如 `r1*4 + c0*32 + c1*32 + c3*16 + c4g`）做 coalesced 读，按 tile 顺序生成顺序 burst；与 AutoSA 本文件中 A/C 的 L3 类似，但公式统一、便于扩展写时重排等。

---

## 4. 对比结论（随机读取上是否应用了改进）

| 项目           | AutoSA MTTKRP（本文件）        | mlir-systolic MTTKRP        |
|----------------|--------------------------------|-----------------------------|
| **L2 读下标**  | 多维+表达式：`[c8][c5][4*c6/16]`、`[c5][c7/8]`、`[c7][4*c6/16]` | 单维顺序：`local_arg*[c7][0][0]` |
| **同一 word 多次读** | 是（如 B 的 `[c5][c7/8]` 下 c8 全读同一位置） | 否（按 c7 顺序扫）         |
| **最内层 % / /** | 有：`(c6)%4`、`(c7)%8`         | 用 `& 7` 等替代（2 的幂）   |
| **L2 访问性质** | 非单维顺序、阶梯/重复读        | 单维顺序、顺序读           |

因此，**在 L2 阶段，mlir-systolic 相对 AutoSA 已应用了针对“随机/非连续读”的改进**：

1. **L2 读**：从“多维+表达式”改为 **单维顺序**（仅 c7 变化，其余维固定 0），避免阶梯式、重复读同一 word 的模式。
2. **强度约简**：在最内层用 `& (simd-1)` 替代 `% simd`，减少组合逻辑与时序压力。

注意：两边的 sa-size、问题规模（以及数组命名 A/B/C vs arg0/arg1/arg2）可能不同，此处只对比 **L2 访问模式与表达式形式**，结论不受参数差异影响。  
若需在服务器上验证性能差异，可在相同/相近问题规模下对比 II、Fmax 与综合资源。

---

## 5. 相关文档

- 随机读与优化建议：`docs/reference/autosa/AUTOSA_SOURCE_PERF_AND_MLIR_OPPORTUNITIES.md` §3、§4  
- 写时重排与 L2 接入：`docs/design/EXISTING_OPTIMIZATIONS_IN_CODE.md`  
- L3 coalesce 与写时重排关系：`docs/design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md`
