# 下一步工作路线图

> **最后更新**: 2026-01-06  
> **目的**: 明确下一步工作重点和优先级

---

## 当前状态总结

### ✅ 已完成

1. **Polymer 集成**: 完成并集成到编译流程
2. **ParametricSpaceTime 框架**: 支持参数化 spacetime 配置
3. **Spacetime 动态枚举**: ✅ **最新完成** - 支持任意循环数量的动态枚举
4. **HLS 代码生成**: 基本功能已实现
5. **文档更新**: 已更新以反映实际实现状态

### 🟡 部分完成

1. **Spacetime 测试**: 3 循环测试通过，4+ 循环待测试
2. **Kernel 支持**: 主要支持 MM，其他 kernel 支持有限
3. **命令行选项**: 可通过代码设置，命令行选项待完善

---

## 下一步工作（按优先级）

### 优先级 1: 测试和验证 🔴

#### 1.1 创建 AutoSA 参考代码生成脚本 ✅

**状态**: 已完成脚本 (`scripts/generate_autosa_reference.sh`)

**任务**:
- [x] 创建生成脚本
- [ ] 测试脚本功能
- [ ] 生成完整的参考代码集
- [ ] 验证参数与代码对应关系

**产出**:
- 可重现的 AutoSA 参考代码
- 每个文件的元数据（参数信息）

#### 1.2 扩展 Spacetime 测试覆盖

**状态**: 🟡 部分完成（3 循环已测试，4+ 循环待测试）

**已完成**:
- [x] 实现动态枚举功能
- [x] 3 循环 kernel (MM) 测试通过

**待完成**:
- [ ] 为 MM 测试所有 ST0-ST5 配置（当前只测试了默认 ST3）
- [ ] 为 MTTKRP (4 循环) 枚举并测试所有 spacetime 配置
- [ ] 为 TTMC (4 循环) 枚举并测试所有 spacetime 配置
- [ ] 为 CNN (5 循环) 枚举并测试 spacetime 配置

**验证**:
- 与 AutoSA 输出对比
- 确保配置对应关系正确

**参考**: [SPACETIME_TEST_RESULTS.md](SPACETIME_TEST_RESULTS.md)

#### 1.3 端到端测试

**任务**:
- [ ] 为每个 kernel 类型创建测试用例
- [ ] 验证生成的 HLS 代码可以编译
- [ ] 对比功能正确性（如果可能）

---

### 优先级 2: 功能完善 🟡

#### 2.1 动态 Spacetime 枚举 ✅ **已完成**

**状态**: ✅ 已实现并测试通过

**实现内容**:
1. ✅ 实现动态枚举所有可能的 spacetime 配置
2. ✅ 根据循环数量自动生成配置列表
3. ✅ 自动选择默认配置（优先 ST3）

**实现位置**: `lib/Transforms/SystolicTransform.cpp::enumerateSpaceTimeConfigs()`

**测试状态**: 3 循环 kernel 测试通过，4+ 循环待测试

**参考**: [SPACETIME_TEST_RESULTS.md](SPACETIME_TEST_RESULTS.md)

#### 2.2 Kernel 泛化

**任务**:
- [ ] 实现通用的 loop body migration
- [ ] 支持 4+ 循环嵌套
- [ ] 支持不同的计算模式（不仅仅是累加）

**当前限制**: 主要针对 3-loop MM 优化

#### 2.3 Write-Time Reordering 集成

**任务**:
- [ ] 将分析结果应用到 HLS 代码生成
- [ ] 支持更多数组维度（不仅仅是 3D）

---

### 优先级 3: 工具和体验 🟢

#### 3.1 简化版交互模式（可选）

**任务**:
- [ ] 实现 `--list-spacetime-configs` 命令
- [ ] 增强 `--verbose` 输出
- [ ] 支持配置文件（JSON/YAML）

**参考**: `docs/INTERACTIVE_MODE_ANALYSIS.md`

#### 3.2 测试脚本和工具

**任务**:
- [ ] 创建自动化测试脚本
- [ ] 创建对比工具（与 AutoSA 输出对比）
- [ ] 创建性能分析工具

---

### 优先级 4: Host 端代码生成 🔵

#### 4.1 HLS Testbench 生成

**任务**:
- [ ] 实现 HLS Testbench 生成
- [ ] 支持数据序列化/反序列化
- [ ] 支持不同的测试场景

#### 4.2 OpenCL Host 代码生成

**任务**:
- [ ] 实现 OpenCL Host 代码生成
- [ ] 支持 Xilinx Vitis 平台
- [ ] 支持多 DDR/HBM 端口映射

---

## 工作流程建议

### 短期（1-2 周）

1. **测试脚本**: 使用 `generate_autosa_reference.sh` 生成参考代码
2. ✅ **Spacetime 枚举**: 动态枚举功能已实现
3. **测试覆盖**: 为 MM 测试所有 ST0-ST5 配置，扩展 4+ 循环测试

### 中期（1-2 月）

1. **多 Kernel 测试**: 测试 MTTKRP, TTMC, CNN
2. **功能完善**: Loop body migration, Write-time reordering
3. **工具完善**: 测试脚本、对比工具

### 长期（3+ 月）

1. **Host 端代码**: 实现 Testbench 和 OpenCL Host 生成
2. **性能优化**: Double buffering, 资源优化
3. **自动调优**: 实现启发式自动选择

---

## 关键决策点

### 1. 是否需要完整交互模式？

**建议**: 先实现简化版（枚举 + verbose），根据需求决定是否实现完整交互模式

**参考**: `docs/INTERACTIVE_MODE_ANALYSIS.md`

### 2. 测试策略

**建议**: 
- 先与 AutoSA 输出进行结构对比
- 再验证功能正确性（如果可能）
- 最后进行性能对比

### 3. Kernel 支持优先级

**建议**:
1. MM (3 循环) - 最常用，优先完善
2. MTTKRP/TTMC (4 循环) - 测试 spacetime 枚举
3. CNN (5 循环) - 测试复杂场景

---

## 总结

**核心工作**: 
1. ✅ **测试和验证** - 确保功能正确性
2. 🟡 **功能完善** - 支持更多 kernel 和配置
3. 🟢 **工具完善** - 提升开发体验
4. 🔵 **Host 端代码** - 完整流程支持

**关键原则**:
- 先测试，再优化
- 先覆盖，再深入
- 先自动化，再智能化

---

**下一步行动**:
1. 运行 `generate_autosa_reference.sh` 生成参考代码
2. ✅ 实现 spacetime 动态枚举（已完成）
3. 扩展测试覆盖（4+ 循环 kernel）
4. 完善命令行选项支持
