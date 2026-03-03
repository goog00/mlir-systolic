# 固定模式 vs 动态枚举：可视化对比

## 固定模式的局限（当前实现）

### 代码结构

```
selectSpaceLoops(spaceTimeMode, depInfos) 
│
├─ case 0: space=[0], time=[1,2]    ← 硬编码循环索引
├─ case 1: space=[1], time=[0,2]    ← 假设总是 3 个循环
├─ case 2: space=[2], time=[0,1]    ← 索引 0,1,2 固定
├─ case 3: space=[0,1], time=[2]    ← 无法表达其他组合
├─ case 4: space=[0,2], time=[1]
└─ case 5: space=[1,2], time=[0]
   
总共：6 种（ST0-ST5）
```

### 处理 3 循环时（工作正常）

```
for i, j, k:          空间-时间映射
  ...         ───────────────────→
             
case 3:
  space = [0, 1] = [i, j]  ✅ 正确
  time = [2] = [k]         ✅ 正确
```

### 处理 4 循环时（失效）

```
for i, j, k, l:       空间-时间映射
  ...          ───────────────────→
              
case 3:
  space = [0, 1] = [i, j]  ✅ 索引对
  time = [2] = [k]         ❌ 只处理 k，丢失 l！
  
需要的是：
  space = [0, 1] = [i, j]
  time = [2, 3] = [k, l]   ← 代码无法表达
```

### 处理 5 循环时（彻底失效）

```
for h, w, c_in, c_out, n:
  ...

固定模式无法表达任何配置：
  [0,1], [0,2], [0,3], [0,4]
  [1,2], [1,3], [1,4]
  [2,3], [2,4]
  [3,4]
  ... 共 10 种 2D + 10 种 3D = 20 种

需要的：[h,w]、[c_in,c_out]、[n,h] 等任意组合
```

---

## 动态枚举的能力（已删除）

### 枚举算法流程

```
enumerateSpaceTimeConfigs(numLoops, maxSADim)
│
├─ 循环 1：
│  ├─ 1D 配置：每个循环一个
│  │  loop[0] → space=[0], time=[1,2,3]
│  │  loop[1] → space=[1], time=[0,2,3]
│  │  loop[2] → space=[2], time=[0,1,3]
│  │  loop[3] → space=[3], time=[0,1,2]
│  │
│  └─ 总数：N 种（4 种）
│
├─ 循环 2：
│  ├─ 2D 配置：所有循环对
│  │  [0,1] → space=[0,1], time=[2,3]
│  │  [0,2] → space=[0,2], time=[1,3]
│  │  [0,3] → space=[0,3], time=[1,2]
│  │  [1,2] → space=[1,2], time=[0,3]
│  │  [1,3] → space=[1,3], time=[0,2]
│  │  [2,3] → space=[2,3], time=[0,1]
│  │
│  └─ 总数：C(4,2) = 6 种
│
└─ 总配置数：4 + 6 = 10 种（+ 可选 3D）
   配置 ID：0-9（每个唯一）
```

### 处理 3 循环（完全兼容）

```
for i, j, k:

自动枚举得到：
  ID=0: space=[0], time=[1,2]     (ST0)
  ID=1: space=[1], time=[0,2]     (ST1)
  ID=2: space=[2], time=[0,1]     (ST2)
  ID=3: space=[0,1], time=[2]     (ST3 ✅ 默认)
  ID=4: space=[0,2], time=[1]     (ST4)
  ID=5: space=[1,2], time=[0]     (ST5)

✅ 与 ST0-ST5 完全对应
```

### 处理 4 循环（自动支持）

```
for i, j, k, l:

自动枚举得到：
  ID=0: space=[0], time=[1,2,3]
  ID=1: space=[1], time=[0,2,3]
  ID=2: space=[2], time=[0,1,3]
  ID=3: space=[3], time=[0,1,2]
  ID=4: space=[0,1], time=[2,3]     ← MTTKRP 常用
  ID=5: space=[0,2], time=[1,3]
  ID=6: space=[0,3], time=[1,2]
  ID=7: space=[1,2], time=[0,3]
  ID=8: space=[1,3], time=[0,2]
  ID=9: space=[2,3], time=[0,1]

✅ 10 种配置，用户可选或自动选择最佳
```

### 处理 5 循环（自动扩展）

```
for h, w, c_in, c_out, n:

自动枚举得到：
  1D: 5 种 (space=[i] for i in [0..4])
  2D: 10 种 (space=[i,j] for all pairs)
  3D: 10 种 (space=[i,j,k] for all triples, if maxSADim >= 3)
  
总计：25 种

✅ CNN 自动支持，无需修改代码
```

---

## 实际效果对比

### 矩阵乘法 (MM, 3 循环)

| 方案 | 支持 | 配置数 | 质量 |
|------|------|--------|------|
| **固定模式** | ✅ 是 | 6 | ✅ 好 |
| **动态枚举** | ✅ 是 | 6 | ✅ 好 |
| **推荐** | — | — | 两者都可，动态枚举更灵活 |

### MTTKRP (4 循环)

| 方案 | 支持 | 配置数 | 质量 |
|------|------|--------|------|
| **固定模式** | ❌ 否 | 0 | ❌ 失效 |
| **动态枚举** | ✅ 是 | 10+ | ✅ 优 |
| **推荐** | — | — | **必须用动态枚举** |

### TTMC (4 循环)

| 方案 | 支持 | 配置数 | 质量 |
|------|------|--------|------|
| **固定模式** | ❌ 否 | 0 | ❌ 失效 |
| **动态枚举** | ✅ 是 | 10+ | ✅ 优 |
| **推荐** | — | — | **必须用动态枚举** |

### CNN (5 循环)

| 方案 | 支持 | 配置数 | 质量 |
|------|------|--------|------|
| **固定模式** | ❌ 否 | 0 | ❌ 完全失效 |
| **动态枚举** | ✅ 是 | 25+ | ✅ 优 |
| **推荐** | — | — | **必须用动态枚举** |

---

## 代码示例对比

### 固定模式的问题

```cpp
// 当前代码
static LogicalResult selectSpaceLoops(
    const SmallVectorImpl<LoopDepInfo> &depInfos,
    unsigned spaceTimeMode) {
  
  unsigned numLoops = depInfos.size();
  
  if (numLoops < 3) {
    return failure();  // ❌ 需要至少 3 个循环！
  }
  
  switch (spaceTimeMode) {
    case 0:
      spaceLoopIndices.push_back(0);      // ❌ 硬编码
      spaceLoopIndices.push_back(1);      // ❌ 假设 3 个循环
      timeLoopIndices.push_back(2);
      break;
    // ... 其他 case
  }
}

// 使用问题
for i, j, k, l:  // 4 个循环
  selectSpaceLoops(depInfos, 3);
  // 会将循环 0,1 作为空间循环，循环 2 作为时间循环
  // 但循环 3 会被忽略！❌
```

### 动态枚举的优雅

```cpp
// 新代码
static LogicalResult enumerateSpaceTimeConfigs(
    const SmallVector<AffineForOp> &loops,
    unsigned maxSADim,
    SmallVector<ParametricSpaceTime, 8> &configs) {
  
  unsigned numLoops = loops.size();
  unsigned configId = 0;
  
  // 自动枚举 1D
  for (unsigned i = 0; i < numLoops; ++i) {
    auto config = ParametricSpaceTime::createFromLoopIndices(
        {i}, getOtherLoops(i), loopNames);
    config.setConfigId(configId++);
    configs.push_back(config);
  }
  
  // 自动枚举 2D（当 maxSADim >= 2）
  if (maxSADim >= 2) {
    for (unsigned i = 0; i < numLoops; ++i) {
      for (unsigned j = i + 1; j < numLoops; ++j) {
        auto config = ParametricSpaceTime::createFromLoopIndices(
            {i, j}, getOtherLoops(i, j), loopNames);
        config.setConfigId(configId++);
        configs.push_back(config);
      }
    }
  }
  
  return success();
}

// 使用效果
for i, j, k, l:  // 4 个循环
  enumerateSpaceTimeConfigs(loops, 2);
  // 自动生成 4 + 6 = 10 种配置
  // 支持 [0,1], [0,2], [0,3], [1,2], [1,3], [2,3] 等 ✅
```

---

## 性能和复杂度分析

### 枚举复杂度

| 循环数 | 1D | 2D | 3D | 总计 |
|--------|----|----|----|----|
| 3 | 3 | 3 | 1 | **7** |
| 4 | 4 | 6 | 4 | **14** |
| 5 | 5 | 10 | 10 | **25** |
| 6 | 6 | 15 | 20 | **41** |

**复杂度**：$O(N^k)$ 其中 k 是 PE 维度（通常 k=2）

**实际**：对于 N≤5，完全在可接受范围内（<100 个配置）

### 运行时开销

- **枚举**：一次性，编译期
- **存储**：每个配置 ~100 字节，25 个配置 ~2.5KB（可忽略）
- **选择**：$O(1)$ 根据 ID 选择

**结论**：性能开销几乎为零

---

## 总结表

```
┌──────────────┬──────────────┬──────────────┐
│   方案       │   固定模式   │   动态枚举   │
├──────────────┼──────────────┼──────────────┤
│ MM (3-loop)  │     ✅       │      ✅      │
│ 4-loop       │     ❌       │      ✅      │
│ 5-loop       │     ❌       │      ✅      │
│ 扩展性       │     差       │      优      │
│ 代码复杂度   │     低       │      中      │
│ 维护成本     │     低       │      中      │
│ AutoSA 对标  │     差       │      优      │
│ 项目规划     │     不符     │      符      │
└──────────────┴──────────────┴──────────────┘
```

---

## 推荐行动

### 立即行动（高优先级）

恢复 `enumerateSpaceTimeConfigs()` 函数：

```bash
# 1. 查找历史版本（大约 1-2 周前有完整实现）
git log --all --oneline --grep="spacetime\|parametric" | head

# 2. 查看该版本的实现
git show <commit-hash>:lib/Transforms/SystolicTransform.cpp | head -300

# 3. 恢复代码
git checkout <commit-hash> -- lib/Transforms/SystolicTransform.cpp
# 或手动重新实现（161 行）

# 4. 添加测试
# test/4-loop-mttkrp.mlir
# test/5-loop-cnn.mlir
```

### 验证步骤

```bash
# 编译
cd build && ninja

# 测试 3 循环（应该与之前相同）
./bin/systolic-opt -systolic-transform ../test_matmul.mlir

# 测试 4 循环（动态枚举新支持）
./bin/systolic-opt -systolic-transform ../test_mttkrp.mlir

# 测试 5 循环
./bin/systolic-opt -systolic-transform ../test_cnn.mlir
```

