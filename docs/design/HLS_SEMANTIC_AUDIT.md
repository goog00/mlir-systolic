# 生成 HLS C 的语义正确性检查

> 本文档对 `systolic-translate` 生成的 HLS C 做语义层面的审计，指出已发现的问题与待验证点。  
> 涉及：L3 读、L2 读写、PE 计算、drain 写回、FIFO 数量一致性。

---

## 1. 审计范围与方法

- **对照语义**：MM（C[i,j]+=sum_k A[i,k]*B[k,j]）、MTTKRP（D[i,j]+=sum_kl A[i,k,l]*B[k,j]*C[l,j]）、TTMc（3D 输出、三规约）。
- **检查点**：L3_in_serialize 产出量、L2 inter_trans/intra_trans 与 c2 的对应、PE 累加/初值/写出条件、drain 写出顺序与迭代次数。

---

## 2. 已发现的语义问题

### 2.1 【高】L2 仅在 (c0,c1) 加载一次，却对全部 c2 复用（MM/单规约）

**现象**：

- 在 `emitIOL2In()` 中，对每个 `(c0, c1)` 只调用**一次** `inter_trans`，再在**内层**对 `c2 = 0 .. numTiles-1` 调用多次 `intra_trans`。
- `inter_trans` 从 FIFO 读入 `latency` 个 word（如 4 个），写入 `local_*[c4][0][0]`；`intra_trans` 则按 (c5, c6, c7) 从同一 buffer 读出并送给 PE。
- 因此**同一批 4 个 word（32 个 float）被用于全部 c2 迭代**，PE 在 4 个 c2 上重复使用同一批 A/B 数据。

**应有语义（以 MM 为例）**：

- `c2` 对应规约维 k 的 tile 索引；不同 c2 应对应不同的 k 段，即不同的 A、B 数据。
- 每个 (c0, c1, **c2**) 应使用**新的一批** A/B 数据，即 L2 应在**每个 c2** 从 FIFO 再加载一次。

**结论**：当前“每 (c0,c1) 只加载一次、多 c2 复用”与 MM/单规约语义不符，存在**语义错误**。

**修复（已实现）**：

- **L2**：已将 `inter_trans` / `inter_trans_boundary` 移入 c2 循环内（`emitIOL2In`、`emitIOL2InBoundary`）。对每个 (c0, c1, c2) 先执行一次 `inter_trans` 从 FIFO 读入本 c2 对应的 words，再执行 `intra_trans`。
- **L3_serialize**：对 MatmulLike（单规约、非 3D）增加外层 c2 循环，将同一批 DRAM 数据按 c2 重复输出 numTiles 次，使 L2 每个 c2 能读到完整一批 words。
- **L3_in**：对 MatmulLike 增加 c2 循环，使读/写 FIFO 的迭代次数与 L3_serialize 产出一致（128×numTiles words）。

---

## 3. 已核对无问题的部分

### 3.1 L3_in_serialize 产出数量（MM，2D）

- `totalDramWords = (32*32*4)/64 = 64`（以 32×32、512-bit 为单位）。
- `wordsPerDram = 16/8 = 2`，每 DRAM word 产出 2 个 `arg0_t8` word。
- 循环 (c0, c1, c3, c4g) 且内层 `slot` 2 次 → 4×4×2×2×2 = **128** 个 `arg0_t8` word，与 32×32 元素 / 8 = 128 word 一致。
- 若按 2.1 修复（L2 按 c2 加载），则 L3 需按 (c0, c1, **c2**, c3, c4g) 顺序产出更多 word，数量需与 L2 消费一致。

### 3.2 PE 初值/写出条件（MM）

- 初值：`c2==0 && c5==0` 时 `local_out[c7][c6]=0`，符合“每个输出点在一个规约段的开始清零”。
- 写出：`c2==numTiles-1 && c5==c5Bound-1` 时写出，符合“规约结束写回”。

### 3.3 Drain 写出量（2D 输出）

- `emitDrainSerialize` 中 2D 用 `outputShape[0]*outputShape[1]` 算迭代/字数，与 2D 输出规模一致。

### 3.4 L2 intra_trans 读下标

- 使用 `local_*[c7][0][0]` 且 `split_idx = c5 & (arrayPart-1)`，避免未初始化槽位；与当前“单 buffer 维”设计一致（参见 CODEGEN_STATUS_AND_NEXT_ANALYSIS）。

---

## 4. 待进一步验证的点

| 项目 | 说明 |
|------|------|
| **L3 word_idx 与 (i,j,k) 映射** | word_idx 的线性公式是否与 host 端矩阵布局（行主序等）一致，需结合具体 kernel 与 host 约定核对。 |
| **MTTKRP / TTMc 的 r1、r2 与 L3** | 双规约/三规约时，L3 的 r1/r2 循环与 word_idx 是否与 MTTKRP/TTMc 的 A/B/C 访问顺序一致。 |
| **Drain 顺序与输出布局** | 无写时重排时，drain 写出顺序是否与 host 期望的 C/D 布局一致。 |
| **FIFO 深度与 backpressure** | 深度为 2 时，L3/L2/PE/drain 之间是否存在 backpressure 导致死锁或结果错；建议在 C sim 或综合后仿真中观察。 |

---

## 5. 建议的修复优先级

1. **P0**：修复 L2 与 c2 的对应关系（2.1），使每个 c2 使用新加载的 A/B 数据；并同步调整 L3_in_serialize 的循环与产出，使 L3 产出与 L2 消费一致。
2. **P1**：在修复后对 MM 做 C sim 或小规模数值对比，确认 C 与参考实现一致。
3. **P2**：系统性核对 MTTKRP/TTMc 的 L3 顺序与 drain 顺序，并补充到本文档。

---

## 6. 相关文档

- [CODEGEN_STATUS_AND_NEXT_ANALYSIS.md](CODEGEN_STATUS_AND_NEXT_ANALYSIS.md)：规约维、PE 骨架、L2 读下标。
- [EXISTING_OPTIMIZATIONS_IN_CODE.md](EXISTING_OPTIMIZATIONS_IN_CODE.md)：写时重排、L2 维度与置换。
- [CODEGEN_REFACTOR_ASSESSMENT.md](CODEGEN_REFACTOR_ASSESSMENT.md)：代码生成器重构与语义驱动目标。
