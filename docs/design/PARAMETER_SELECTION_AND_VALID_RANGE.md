# 参数选择与合法范围

> 参数（array_part、latency、simd、space_time 等）**不应依赖单一全局预设**，而应由多面体分析给出选择范围后再选取。本文档说明思路及当前用法，并指向 AutoSA 参考。

---

## 1. 原则

- 同一组固定参数并不适合所有 kernel；参数与**具体输入的循环界、依赖、调度**相关。
- 在 AutoSA 中，参数是**通过多面体分析逐步得到选择范围**的，再在该范围内选取（如为快速综合选取较小值）。
- **“小规模验证”**：在分析得到的合法范围内，选取较小参数以便快速综合与上板，而不是对任意输入强制同一组数值。

---

## 2. AutoSA 中的参数与选择范围（参考）

详见 [third_party/AutoSA/docs/tutorials/getting_started.rst](../../third_party/AutoSA/docs/tutorials/getting_started.rst)：

| 参数 | 含义 | 选择范围来源 |
|------|------|----------------|
| **space_time** | 时空映射模式 | 合法性检查后得到的可选 spacetime 列表 |
| **array_part** | 数组划分（tiling 因子） | **tilable_loops**：各维上界（如 [64,64,64]）；因子需 ≤ 各维上界且为循环界的子倍数 |
| **latency** | 延迟隐藏 | 在 array_part 之后，**tilable_loops**（如 [16,16]）内的并行循环；在该范围内选因子 |
| **simd** | 向量化宽度 | 可向量化循环及上界、**tilable_loops** 与 **legal** 约束 |

配置文件（如 `autosa_config.json`）中会给出各步的 `tilable_loops`、`legal` 等，用于指导手工或自动选取参数。

---

## 3. 当前 mlir-systolic 用法

- 若尚未实现“分析 → 输出可选范围”的接口，可**手动指定参数**，但需保证对当前输入合法，例如：
  - tiling 因子能整除循环界；
  - array_part / latency 与循环维数、band 结构一致。
- **建议**：对给定测试 kernel（如 MM 32×32、MTTKRP 8×8×8×8），在合法范围内取较小值（如 array_part=16、latency=8、小 PE 规模），用于 L1（只生成 HLS）与 L2（C sim / 综合）。
- 后续可增加“分析 → 输出可选范围”的流程，与 AutoSA 的 tuning 分步模式对齐。

---

## 4. 相关文档

- [SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md](SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md) 第 3 节：参数与“小规模验证”
- [RECENT_CHANGES_AND_NEXT_STEPS.md](../../RECENT_CHANGES_AND_NEXT_STEPS.md)：下一步工作中的“参数与多面体选择范围”
