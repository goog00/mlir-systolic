# mlir-systolic 动态枚举 vs 固定模式分析报告

> **日期**: 2026-01-14  
> **目标**: 分析项目是否真的需要动态枚举、固定模式的局限性、以及推荐方案

---

## 执行摘要

### 关键发现

| 维度 | 现状 | 发现 |
|------|------|------|
| **MM Kernel** | 固定 ST3 | ✅ 可行 (3 个循环) |
| **4+ 循环 Kernel** | 需要动态枚举 | ❌ 固定模式**严重失效** |
| **代码现状** | 已有两套代码 | ⚠️ 删除了动态枚举，退回固定模式 |
| **AutoSA 对标** | SpaceTime 0-5 | ✅ 仅对 3 循环工作 |

### 结论

**✅ 必须需要动态枚举** —— 特别是为了支持 4+ 循环的 kernel（MTTKRP、CNN）

---

## 第一部分：AutoSA 空间-时间映射机制

### 1. AutoSA 的 Space-Time 设计

在 AutoSA 中，空间-时间映射由 `struct autosa_kernel` 中的以下字段定义：

```c
struct autosa_kernel {
  int n_sa_dim;          // PE 阵列维度数：1 或 2（很少用 3）
  int sa_dim[3];         // 空间维度的循环索引（最多 3 个）
  int space_parallel[3]; // 各空间维度的并行度
  int space_time_id;     // 空间-时间模式 ID（0-5 for 3-loop）
};
```

**关键点**：
- `space_time_id` 是一个**预定义的 ID**（ST0-ST5），**只对 3 循环 kernel 工作**
- 对于 4+ 循环，AutoSA 通过**动态枚举**所有可能的空间维度组合

### 2. AutoSA 的 6 种 Space-Time 模式（仅限 3 循环）

对于矩阵乘法 `for i, j, k`：

| 模式 | 空间维度 | 时间维度 | PE 阵列 | 数据流特征 |
|------|---------|---------|---------|-----------|
| **ST0** | [0] = i | [1, 2] = j, k | 1D 行 | 行脉动 |
| **ST1** | [1] = j | [0, 2] = i, k | 1D 列 | 列脉动 |
| **ST2** | [2] = k | [0, 1] = i, j | 1D 归约 | 针对特定模式 |
| **ST3** | [0, 1] = i, j | [2] = k | 2D 输出驻留 | **默认、最常用** |
| **ST4** | [0, 2] = i, k | [1] = j | 2D 权重驻留 | 权重沿 k 方向流 |
| **ST5** | [1, 2] = j, k | [0] = i | 2D 激活驻留 | 激活沿 j,k 方向流 |

**注意**：这 6 种模式**是硬编码的**，循环索引固定为 0, 1, 2。

---

## 第二部分：mlir-systolic 当前实现

### 1. 固定模式实现（selectSpaceLoops）

[lib/Transforms/SystolicTransform.cpp L300-370]

```cpp
static LogicalResult selectSpaceLoops(
    const SmallVectorImpl<LoopDepInfo> &depInfos,
    unsigned spaceTimeMode,  // 0-5
    SmallVectorImpl<unsigned> &spaceLoopIndices,
    SmallVectorImpl<unsigned> &timeLoopIndices) {
  
  unsigned numLoops = depInfos.size();
  if (numLoops < 3) {
    return failure();  // 需要至少 3 个循环！
  }
  
  switch (spaceTimeMode) {
    case 0:  // [i] - 1D row
      spaceLoopIndices.push_back(0);      // 硬编码循环 0
      timeLoopIndices.push_back(1);       // 硬编码循环 1
      timeLoopIndices.push_back(2);       // 硬编码循环 2
      break;
    case 3:  // [i,j] - 2D output-stationary
      spaceLoopIndices.push_back(0);      // 硬编码循环 0
      spaceLoopIndices.push_back(1);      // 硬编码循环 1
      timeLoopIndices.push_back(2);       // 硬编码循环 2
      break;
    // ... ST1, ST2, ST4, ST5 类似
  }
  return success();
}
```

**问题所在**：
- ❌ 假设总是有**恰好 3 个循环**
- ❌ 循环索引**硬编码为 0, 1, 2**
- ❌ **不支持 4+ 循环的 kernel**

### 2. 参数化框架（ParametricSpaceTime）

[include/systolic/Analysis/ParametricSpaceTime.h L103-160]

```cpp
class ParametricSpaceTime {
  unsigned configId;                          // 配置 ID
  SmallVector<SpaceDimConfig, 2> spaceDimConfigs;
  TimeDimConfig timeDimConfig;
  ReductionDimConfig reductionDimConfig;
  
  // 支持任意循环索引和维度数
};
```

**优势**：
- ✅ **支持任意循环数**（3, 4, 5+）
- ✅ **支持任意循环索引组合**
- ✅ 每个配置有唯一 ID（对应 AutoSA 的 space_time_id）
- ✅ 配置**参数化和可组合**

### 3. 已删除的动态枚举（enumerateSpaceTimeConfigs）

[git diff 显示已删除 161 行代码]

```cpp
// 已删除！
static LogicalResult enumerateSpaceTimeConfigs(
    const SmallVector<AffineForOp> &loops,
    const SmallVectorImpl<LoopDepInfo> &depInfos,
    unsigned maxSADim,  // 最大 PE 维度（1, 2, 或 3）
    SmallVector<ParametricSpaceTime, 8> &configs) {
  
  // 1. 枚举所有 1D 配置：每个循环一个
  // 2. 枚举所有 2D 配置：每对循环组合一个
  // 3. 枚举所有 3D 配置（可选）：每三个循环组合一个
}
```

**被删除的功能**：
- ❌ 为任意循环数**自动生成所有可能的 spacetime 配置**
- ❌ 支持 `maxSADim` 参数限制 PE 维度
- ❌ 返回配置列表供用户选择或自动选择

---

## 第三部分：为什么固定模式对 4+ 循环失效？

### 问题分析

#### 情况 1：MTTKRP (4 循环)

**原始循环**：
```
for i, j, k, l:
  D[i,j] += A[i,k,l] * B[k,l,j]
```

**可能的 Space-Time 映射**（部分示例）：

| 配置 | 空间维度 | 时间维度 | 说明 |
|------|---------|---------|------|
| [0] | i | j, k, l | 1D 行脉动 |
| [1] | j | i, k, l | 1D 列脉动 |
| [0,1] | i, j | k, l | 2D 输出驻留 |
| [0,2] | i, k | j, l | 权重驻留（沿 k） |
| [0,3] | i, l | j, k | 新模式（ST 中无） |
| [1,2] | j, k | i, l | 新模式 |
| [1,3] | j, l | i, k | 新模式 |
| [2,3] | k, l | i, j | 新模式 |
| **共计** | **10 种 1D + 2D** | - | - |

**固定模式的失效**：
```cpp
// 错误！循环索引硬编码
case 3:
  spaceLoopIndices.push_back(0);  // 假设循环 0 是 i
  spaceLoopIndices.push_back(1);  // 假设循环 1 是 j
  timeLoopIndices.push_back(2);   // 假设循环 2 是 k
  // 但如果输入是其他循环顺序，就错了！
  // 而且完全无法表达 [0,3]、[1,3] 等新配置
```

#### 情况 2：CNN (5 循环)

**原始循环**（简化）：
```
for h, w, c_in, c_out, n:
  out[h,w,c_out,n] += filter[h,w,c_in,c_out] * input[h,w,c_in,n]
```

**可能的组合**：
- 1D：5 种（每个循环一个）
- 2D：C(5,2) = 10 种（任意两个循环）
- 3D：C(5,3) = 10 种（任意三个循环）
- **共计：25+ 种**

**固定模式彻底失效**：固定 ST0-ST5 只能表达 6 种，无法覆盖 25 种配置。

### 理论基础：AutoSA 的动态枚举

AutoSA 对 4+ 循环的处理：

```
对于 N 个循环：
  1. 列举所有 1D 配置：N 种
  2. 列举所有 2D 配置：C(N, 2) 种
  3. 列举所有 3D 配置（可选）：C(N, 3) 种
  总数 = N + C(N,2) + C(N,3) + ...
```

**示例**：
- N=3: 3 + 3 + 1 = **7 种**（ST0-ST5 只有 6 种，因为 ST2 有重复）
- N=4: 4 + 6 + 4 = **14 种**
- N=5: 5 + 10 + 10 + 5 = **30 种**

---

## 第四部分：项目文档中的规划

### 1. ParametricSpaceTime 框架说明

[include/systolic/Analysis/ParametricSpaceTime.h L1-10]

```cpp
// 文件注释
// 此模块定义了脉动阵列生成的参数化空间-时间配置表示，
// 支持 ST0–ST5 **及其他配置**。
// 与原始实现中的硬编码 "spacetime=3" 不同，
// 这提供了一个统一的、可扩展的框架。
```

**关键词**：**"及其他配置"** —— 明确指出需要超越 ST0-ST5

### 2. 动态枚举实现计划

[docs/features/spacetime/SPACETIME_IMPLEMENTATION_PLAN.md]

```markdown
## 实现步骤

### 步骤 2: 实现动态枚举函数

```cpp
static LogicalResult enumerateSpaceTimeConfigs(
    const SmallVector<AffineForOp> &loops,
    const SmallVector<LoopDepInfo> &depInfos,
    unsigned maxSADim,  // 最大 PE 维度（默认：2）
    SmallVector<ParametricSpaceTime> &configs) {
    
    // 枚举所有可能的 spacetime 配置
}
```

### 测试计划

**测试用例 1: 3 循环 (MM)**
  预期输出：6 种配置

**测试用例 2: 4 循环 (MTTKRP)**
  预期输出：10+ 种配置

**测试用例 3: 5 循环 (CNN)**
  预期输出：25+ 种配置
```

**明确的需求**：系统规划了对 4+ 循环的支持。

### 3. 实现状态文档

[docs/status/IMPLEMENTATION_STATUS.md]

```markdown
## 测试状态

### 已测试
- ✅ ST3 配置的矩阵乘法（11 个 AutoSA 参考用例通过）
- ✅ **Spacetime 动态枚举**: 3 循环 kernel 测试通过（2026-01-06）

### 待测试
- 🟡 ST0, ST1, ST2, ST4, ST5 配置（枚举功能已实现，待验证）
- 🟡 4 循环 kernel（MTTKRP, TTMC）- 枚举功能已支持，待测试
- 🟡 5+ 循环 kernel（CNN）- 枚举功能已支持，待测试
```

**注意**：文档说 "枚举功能已实现"，但实际上已被删除！

### 4. 生成脚本中的参考

[scripts/generate_autosa_reference.py L56-58]

```python
'mttkrp': {
    'spacetime_configs': {
        i: {'name': f'spacetime_{i}', 'flags': []}
        for i in range(6)  # MTTKRP 有 ~6 种 spacetime 配置
    }
},
'ttmc': {
    'spacetime_configs': {
        i: {'name': f'spacetime_{i}', 'flags': []}
        for i in range(10)  # TTMC 有 ~10 种 spacetime 配置
    }
}
```

**说明**：明确记录了 MTTKRP 和 TTMC 的多个 spacetime 配置。

---

## 第五部分：代码现状分析

### 对比表

| 功能 | MM (3-loop) | MTTKRP (4-loop) | TTMC (4-loop) | CNN (5-loop) |
|------|-------------|-----------------|---------------|--------------|
| **ST0-ST5 固定模式** | ✅ 工作 | ❌ 失效 | ❌ 失效 | ❌ 失效 |
| **ParametricSpaceTime 框架** | ✅ 可用 | ✅ 支持 | ✅ 支持 | ✅ 支持 |
| **动态枚举函数** | ❌ 已删除 | ❌ 已删除 | ❌ 已删除 | ❌ 已删除 |
| **项目规划** | ✅ 已完成 | 🟡 计划中 | 🟡 计划中 | 🟡 计划中 |

### 关键代码位置

1. **固定模式（已使用）**：
   - [lib/Transforms/SystolicTransform.cpp L300-370]：`selectSpaceLoops()`
   - 限制：仅支持 3 循环，硬编码索引

2. **参数化框架（已实现但未充分使用）**：
   - [include/systolic/Analysis/ParametricSpaceTime.h]：框架定义
   - [lib/Analysis/ParametricSpaceTime.cpp L137-214]：ST0-ST5 预设
   - [lib/Transforms/SystolicTransform.cpp L220-280]：参数化循环选择

3. **动态枚举（已删除）**：
   - 之前 161 行的 `enumerateSpaceTimeConfigs()` 被删除
   - 相关文档和测试也被删除

---

## 第六部分：是否需要动态枚举？

### 综合评估

| 需求 | 是否需要 | 理由 |
|------|---------|------|
| **支持 MM (ST0-ST5)** | ❌ 否 | 固定模式足够 |
| **支持 MTTKRP (4-loop)** | ✅ **是** | 10+ 种配置，超出 ST0-ST5 |
| **支持 TTMC (4-loop)** | ✅ **是** | 10+ 种配置 |
| **支持 CNN (5-loop)** | ✅ **是** | 25+ 种配置 |
| **与 AutoSA 对标** | ✅ **是** | AutoSA 对 4+ 循环使用动态枚举 |
| **项目完整性** | ✅ **是** | 项目规划明确要求支持 4+ 循环 |

### 推荐方案

#### ✅ 推荐方案 A：恢复并完善动态枚举

**步骤**：
1. 恢复删除的 `enumerateSpaceTimeConfigs()` 函数
2. 实现参数化循环选择 (`selectSpaceLoopsParametric`)
3. 添加 4+ 循环的测试用例
4. 更新文档和相关说明

**优势**：
- ✅ 完全支持 AutoSA 的所有 kernel
- ✅ 符合项目规划和文档承诺
- ✅ 与 AutoSA 完全对标

**工作量**：中等（恢复代码 + 测试）

#### ⚠️ 备选方案 B：保留固定模式，限制范围

**步骤**：
1. 保持当前固定模式 (ST0-ST5)
2. 明确文档说明 "仅支持 3 循环 kernel"
3. 删除 MTTKRP、TTMC、CNN 的相关参考
4. 专注于 MM kernel 的优化

**劣势**：
- ❌ 背离项目初心和设计目标
- ❌ 无法支持真实世界的多种 kernel
- ❌ 与 AutoSA 功能不对标

#### ❌ 不推荐方案 C：混合使用两种方式

**说明**：
- 有时用固定模式，有时用参数化模式
- 导致代码复杂、维护困难

---

## 第七部分：技术实现概要

### 如何实现动态枚举？

#### 1. 枚举算法（伪代码）

```cpp
enumerateSpaceTimeConfigs(
    numLoops: int,
    maxSADim: int,  // 1, 2, 或 3
    configs: List<ParametricSpaceTime>
) {
  configId = 0
  
  // 枚举 1D 配置
  for i in [0..numLoops-1]:
    if depInfos[i].canBeSpaceLoop:
      config = createFromSpaceLoopIndices([i])
      config.setId(configId++)
      configs.add(config)
  
  // 枚举 2D 配置
  if maxSADim >= 2:
    for i in [0..numLoops-1]:
      for j in [i+1..numLoops-1]:
        if depInfos[i].canBeSpaceLoop and depInfos[j].canBeSpaceLoop:
          config = createFromSpaceLoopIndices([i, j])
          config.setId(configId++)
          configs.add(config)
  
  // 枚举 3D 配置（如果需要）
  if maxSADim >= 3:
    for i, j, k in all triples:
      ...
}
```

#### 2. 参数化配置创建

```cpp
ParametricSpaceTime createFromSpaceLoopIndices(
    spaceLoopIndices: List<int>,
    timeLoopIndices: List<int>,
    loopNames: List<String>
) {
  config = new ParametricSpaceTime()
  
  for idx in spaceLoopIndices:
    config.addSpaceDim(idx, loopNames[idx])
  
  for idx in timeLoopIndices:
    config.addTimeDim(idx, loopNames[idx])
  
  return config
}
```

#### 3. 集成到 Pass

```cpp
struct SystolicTransformPass {
  runOnOperation() {
    // 提取循环
    getLoopNest(outerLoop, loops)
    
    // 分析依赖
    analyzeLoopDependencies(loops, depInfos)
    
    // 动态枚举配置
    SmallVector<ParametricSpaceTime> configs
    enumerateSpaceTimeConfigs(loops.size(), maxSADim, configs)
    
    // 选择配置（默认 ST3 或用户指定）
    config = selectConfig(configs, options.spaceTimeMode)
    
    // 使用参数化配置进行变换
    applySpaceTimeTransform(config)
  }
}
```

---

## 建议

### 短期（立即）

**✅ 建议恢复动态枚举功能**

理由：
1. 项目规划明确要求
2. 文档已承诺
3. 代码框架已完整
4. 只需恢复已删除的 161 行 + 适当测试

步骤：
```bash
# 恢复被删除的代码
git log --all --oneline -- lib/Transforms/SystolicTransform.cpp  # 查找历史
git show <old-commit>:lib/Transforms/SystolicTransform.cpp | grep -A 160 "enumerateSpaceTimeConfigs"

# 重新实现
# 添加测试：4 循环 MTTKRP 和 5 循环 CNN
```

### 中期（下一个版本）

1. 完整测试 4+ 循环 kernel
2. 与 AutoSA 生成的参考代码对齐
3. 性能评估和优化

### 长期

1. 支持自定义 spacetime 模式
2. 自动化 kernel 参数探索
3. 与 AutoSA tuning 框架集成

---

## 总结

### 核心结论

| 问题 | 答案 |
|------|------|
| **是否需要动态枚举？** | ✅ **是的，必须需要** |
| **为什么？** | 固定 ST0-ST5 仅适用于 3 循环；4+ 循环需要 10-30+ 种配置 |
| **项目规划？** | ✅ 已明确规划支持 4+ 循环 |
| **技术可行性？** | ✅ 框架完整，只需恢复已删除的实现 |
| **AutoSA 对标？** | ✅ AutoSA 本身就用动态枚举处理 4+ 循环 |
| **建议？** | 恢复动态枚举，完成项目承诺 |

### 删除动态枚举的隐患

当前状态（固定模式 + 已删除的枚举代码）：
- ❌ **无法支持 MTTKRP、TTMC、CNN** 等多循环 kernel
- ❌ **背离项目设计目标**
- ❌ **与已有文档和规划不符**
- ❌ **存在误导性**：代码和文档说支持，但实际不支持

### 最终推荐

**立即恢复动态枚举功能**，理由充分：

1. ✅ 项目规划明确
2. ✅ 文档已承诺
3. ✅ 代码框架完整
4. ✅ 工作量不大
5. ✅ 与 AutoSA 对标
6. ✅ 支持真实世界的多样化 kernel

---

## 附录：参考链接

### 代码位置

- **固定模式**：[lib/Transforms/SystolicTransform.cpp#L300-L370](lib/Transforms/SystolicTransform.cpp#L300-L370)
- **参数化框架**：[include/systolic/Analysis/ParametricSpaceTime.h](include/systolic/Analysis/ParametricSpaceTime.h)
- **参数化实现**：[lib/Transforms/SystolicTransform.cpp#L220-L280](lib/Transforms/SystolicTransform.cpp#L220-L280)

### 文档位置

- **AutoSA 分析**：[docs/reference/autosa/AUTOSA_ANALYSIS.md](docs/reference/autosa/AUTOSA_ANALYSIS.md)
- **实现计划**：[docs/features/spacetime/SPACETIME_IMPLEMENTATION_PLAN.md](docs/features/spacetime/SPACETIME_IMPLEMENTATION_PLAN.md)
- **实现状态**：[docs/status/IMPLEMENTATION_STATUS.md](docs/status/IMPLEMENTATION_STATUS.md)
- **代码结构**：[docs/guide/CODE_STRUCTURE.md#L354](docs/guide/CODE_STRUCTURE.md#L354)

### AutoSA 源码

- **autosa_kernel 定义**：[third_party/AutoSA/src/autosa_common.h#L183](third_party/AutoSA/src/autosa_common.h#L183)

