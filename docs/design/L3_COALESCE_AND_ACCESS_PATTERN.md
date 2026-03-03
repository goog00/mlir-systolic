# L3 访问与 Coalesce 设计

> 目标：使 L3 serialize/deserialize 的 DRAM 访问与 AutoSA 一致，**连续 burst、tile 顺序**，便于在服务器上验证性能。

---

## 1. AutoSA 的 L3/coalesce 做法

- AutoSA 将 DRAM 读写在 **L2 层** 与 tile 循环 **(c0, c1, c2, c3, c4, c5)** 绑定，用**显式线性索引**访问全局数组，例如：
  - MM 输入 A：`A[((32*c0 + 8*c3 + c4)*64 + (32*c2 + 4*c5)) / 16]`
  - MM 输入 B：`B[128*c1 + 2*c2 + 4*c3 + c4]`
- 设计原则：**内层循环对应最快变化的维度**，使连续迭代访问**连续或接近连续的 DRAM 地址**，利于 HLS 推断 burst、提高带宽利用率。
- 索引由多面体 schedule 与 array partition 推导，与 tile/array_part/latency 一致。

---

## 2. mlir-systolic 当前行为

- **L3_in_serialize**：对 2D 无重排输入，用单层循环 `i = 0 .. (totalSize²×4/64)-1`，读 `array[i]`（512-bit 字），再按 `arrayPart` 拆成多段写 FIFO。  
  - **优点**：DRAM 已是顺序访问（i 递增），本身适合 burst。  
  - **可改进**：循环结构与下游 **L3_in** 的 (c0, c1, c3, c4) 不一致，且未显式体现 tile 顺序，不利于与 AutoSA 对标和后续扩展（如按 tile 的 BIND_STORAGE/partition 提示）。
- **L3_in**：按 (c0, c1, c3, c4) 从 FIFO 读、再写 FIFO，不直接访问 DRAM。
- **Drain L3_out_serialize**：写时重排路径已按 buffer_linear 顺序写回，等价于连续写。

---

## 3. 目标：L3 显式 tile 顺序 + 保持 coalesced 读

- **语义不变**：总读入量仍为 `totalSize²` 个 float，按 512-bit 字读，再按 `arrayPart` 写入 FIFO；顺序与当前单循环 `i` 等价（同一线性顺序）。
- **结构改进**：
  - 用与 **L3_in** 一致的 **(c0, c1, c3)** 加一层“半字”内层循环，显式计算 **word_idx**，使：
    - `word_idx` 随 (c0, c1, c3, half) 单调递增，**DRAM 仍为顺序读**；
    - 循环顺序与下游 (c0, c1, c3, c4) 一致，便于调度与后续加 pragma（如 BURST、BIND_STORAGE）。
  - 约定：2D、行主序、每 512-bit 字 16 float；每字拆成 `16/arrayPart` 个 FIFO 字（如 arrayPart=8 时为 2）。
- **索引公式（2D 行主序，无重排）**  
  设 `numTiles = size / tileSize`，`tileSize = latency * numPE`，每 512-bit 字 16 float，则总字数为 `totalWords = (size*size*4)/64`。  
  将 (c0, c1, c3, half) 映射到 [0, totalWords-1]：
  - `wordsPerTile = (numPE * latency) / (16/arrayPart)`，即每个 (c0,c1) tile 对应的 512-bit 字数（当前 16/arrayPart=2，故每 tile 读 `numPE*latency/2` 个字）。
  - 线性索引：  
    `word_idx = c0 * (numTiles * numPE * latency / 2) + c1 * (numPE * latency / 2) + c3 * (latency / 2) + half`  
    其中 `half in {0,1}` 表示一个 512-bit 字内的前/后半段（各 8 float 当 arrayPart=8）。  
  这样 **word_idx 连续递增**，DRAM 访问保持 coalesced。

---

## 4. 实现要点（emitIOL3InSerialize）

- **无重排分支**（已实现）：用 `for (c0) for (c1) for (c3) for (c4g)` 与内层 `for (slot)`；`wordsPerDram = 16/arrayPart`，`c4GroupBound = totalDramWords/(numTiles²×numPE)`；`word_idx = c0×strideC0 + c1×strideC1 + c3×c4GroupBound + c4g` 保证 word_idx 单调递增、DRAM 顺序读；每 (c0,c1,c3,c4g) 读一次 `array[word_idx]`，内层按 slot 写出 `wordsPerDram` 个 arrayPart 宽 FIFO 字。
- **重排分支**：保持现有 (d0, d1) 重排顺序读逻辑。
- **PIPELINE**：在内层 slot 循环加 `#pragma HLS PIPELINE II=1`。注释已标明 “Coalesced L3 read: tile order (c0,c1,c3,c4_group), word_idx sequential for burst”。

---

## 5. 与 autosa_hls_refs 的对照

- 对同一 kernel（如 MM）、相近 size/array_part/latency，对比：
  - 我们生成的 **word_idx** 与 AutoSA 的 **A[...]/B[...]** 索引是否同序（或等价线性顺序）；
  - 循环嵌套 (c0, c1, c3, …) 是否与 AutoSA 的 L2 读循环一致。
- 若一致，则 L3 访问模式与 AutoSA 对齐，便于在服务器上做正确性与性能对比。

---

## 6. 小结

| 项       | 内容 |
|----------|------|
| 目的     | L3 读与 AutoSA 一致：连续 burst、tile 顺序，便于性能验证。 |
| 做法     | L3_in_serialize 无重排分支改为 (c0,c1,c3,c4g) + 显式 word_idx，保持 DRAM 顺序读。 |
| 语义     | 总数据量、顺序与现有一致；仅循环结构与索引显式化。 |
| 与写时重排 | 无冲突：L3 coalesce 管**输入** L3 读，写时重排管**输出** drain 写；见 [L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md](L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md)。 |
| 后续     | 可与 autosa_hls_refs 逐项对比；必要时加 BURST/BIND_STORAGE 等 pragma。 |
