# 实施路线图（精简版）

**最后更新**: 2026-01-06  
**范围**: Space-time 全配置 + 多 kernel 支持

---

## 阶段 1（进行中）: Space-time=3 巩固
- ✅ PIPELINE 数量与 AutoSA 对齐（24）
- 📌 继续：减少生成冗余行数，完善 loop-body migration

## 阶段 2（计划）：1D 脉动阵列（ST0, ST1）
- PE/IO 方向：ST0 (B horizontal), ST1 (A vertical)
- 简化 IO 层级（无 L3），支持 direct 数据流
- 目标：PIPELINE≈17，代码行数≈1080

## 阶段 3（计划）：Reduction 支持（ST2）
- 自动检测 reduction 循环 (load→add→store)
- FIFO_C in/out，PE 内累加 + 逐 PE 传递
- 目标：PIPELINE≈18，代码行数≈1105

## 阶段 4（计划）：2D Reduction（ST4, ST5）
- 支持 2D 阵列的 reduction；C 水平或垂直累加
- IO/PE 数据流方向随 spacetime 变化
- 目标：PIPELINE≈24，代码行数接近 AutoSA 参考

## 阶段 5（计划）：Kernel 泛化
- 通用 loop-body migration：支持 CNN / MTTKRP / TTMc / TTM / LU
- KernelInfo 结构：loop 计数、space/time 选择、依赖/访存模式
- HLS 生成模板化：PE 签名、FIFO 拓扑、pragma 策略解耦

## 阶段 6（计划）：配置流重构
- 定义 `SystolicConfigAttr`，结构化携带 spacetime/loops/partitions
- 贯穿 Transform → Dataflow → HLS，无需字符串序列化

## 阶段 7（计划）：性能与可维护性
- 写时重排：将分析结果应用到 HLS 生成
- Double buffering / latency hiding / 资源优化
- 统一代码风格，减少冗余生成

---

## 里程碑与测试
- **每个 spacetime 完成标准**：
  - 生成 HLS 代码编译通过
  - PIPELINE 数量与参考对齐
  - 代码行数接近参考（±5%）
  - AutoSA 对比通过（结构与功能）

- **测试矩阵**：
  - ST0-5 × {MM, CNN, MTTKRP, TTMc, TTM, LU}
  - 覆盖不同数组维度/latency/simd 组合

---

**阅读顺序建议**
1) `docs/status/PROJECT_STATUS.md` - 当前状态
2) `docs/CODE_STRUCTURE.md` - 代码组织与问题列表
3) 本文档 - 路线图与阶段目标
4) `docs/guide/BUILD_GUIDE.md` - 构建
5) `test/TESTING_GUIDE.md` - 测试
