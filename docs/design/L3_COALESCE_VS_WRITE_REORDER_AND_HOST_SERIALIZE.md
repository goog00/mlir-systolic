# L3 Coalesce 与写时重排的关系、Host-Serialize 策略

> 回答：（1）L3 coalesce 与写时重排是否冲突；（2）是否要支持 AutoSA 的 --host-serialize，还是在各级 IO 处理数据复用。

---

## 1. L3 coalesce 与写时重排的关系与冲突

**结论：两者无冲突，作用在不同端、不同数组。**

| 优化 | 作用位置 | 作用对象 | 说明 |
|------|----------|----------|------|
| **L3 coalesce** | **输入**：L3_in_serialize（DRAM→FIFO） | 输入数组 A、B 等 | 用 (c0,c1,c3,c4g) + word_idx 按 tile 顺序、顺序读 DRAM，便于 burst；**不改变**数据语义，仅循环结构与索引显式化。 |
| **写时重排** | **输出**：drain_IO_L3_out_serialize（FIFO→buffer→DRAM） | 输出数组 C、D 等 | 当存在非线性写（如 C[i*32+j,k]）时，在 FPGA 上先 unpack→按重排顺序填 buffer_linear→再 pack 写回，使写 DRAM 连续。 |

- **数据流**：输入 L3 → L2 → PE → drain L1/L2 → drain L3。L3 coalesce 只影响**输入**的 L3_in_serialize；写时重排只影响**输出**的 drain serialize。二者不共享同一数组、同一段流水线。
- **L3_in_serialize 内部**：有两个分支——**有重排**（hasReordering2D）时走 (d0,d1) 读时重排；**无重排**时走 (c0,c1,c3,c4g) coalesced 读。写时重排不参与 L3 输入分支选择。
- **总结**：L3 coalesce 与写时重排可以同时启用，无逻辑冲突；一个优化输入读顺序与 burst，一个优化输出写顺序与连续写。

---

## 2. AutoSA 的 --host-serialize 在做什么

- **含义**：在 **Host 端**（CPU）对输入/输出数据做**重排与可能的数据重复**，使传到 FPGA 的数据流**直接**符合各级 IO 期望的顺序与复用需求；FPGA 侧只需按顺序读/写流，无需复杂索引或大 buffer 重排。
- **好处**：FPGA kernel 更简单（顺序访问）、易达到更好 II/频率；部分复杂 layout 可在 host 一次性算完。
- **代价**：  
  - Host 计算与 **数据传输量** 可能增大（重复数据要传多份）；  
  - Host 需生成/维护 serialize/deserialize 代码，与 kernel 的 tile/schedule 强绑定。

---

## 3. 我们是否要支持 host-serialize？

**当前策略：不依赖 host-serialize，在 FPGA 各级 IO 完成“顺序化/复用”。**

- **现状**：mlir-systolic 生成的 kernel 假定 DRAM 上的输入是**自然布局**（如行主序）；由 **L3_in_serialize** 通过 word_idx 等按 tile 顺序读入，**L2/L1** 通过 FIFO 与 local buffer 在 FPGA 内完成数据分发与复用；**drain** 侧通过写时重排在 FPGA 上完成非线性写→连续写。即：重排与复用都在 **FPGA 内** 完成，**不增加** host 到 device 的传输量。
- **与 host-serialize 的对比**：
  - **不支持 host-serialize（当前）**：传输量小（无重复）、host 简单；FPGA 侧 L3/L2 稍复杂（索引公式、FIFO/buffer），但已用 coalesced 读与写时重排优化。
  - **支持 host-serialize**：需生成 host 端 serialize/deserialize 代码，并约定“传进 FPGA 的流顺序”；可减少 FPGA 索引复杂度，但增加传输与 host 逻辑。

**决策：暂不支持 host-serialize。**

- Host 端做两次数据重排代价大，且增加传输成本；AutoSA 基于 PPCG，受当时代码结构所限，在 IO 层次做进一步优化较难，我们则在 FPGA 各级 IO 完成顺序化与复用，不依赖 host-serialize。
- **预期输入布局**：自然顺序（如行主序）；重排与复用均在 FPGA 内（L3/L2 + drain 写时重排）完成。

---

## 4. 小结

| 问题 | 结论 |
|------|------|
| L3 coalesce 与写时重排是否冲突？ | **不冲突**：前者管输入 L3 读顺序，后者管输出 drain 写顺序；可同时启用。 |
| 是否要支持 host-serialize？ | **暂不支持**：在各级 IO 处理顺序与复用，避免 host 重排与传输成本；不计划实现 host-serialize。 |
