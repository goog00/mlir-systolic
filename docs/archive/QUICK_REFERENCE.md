# 恢复快速参考指南

## 恢复了什么?

恢复了 mlir-systolic 项目的**动态空间-时间配置枚举**功能，该功能对4环及以上的循环嵌套支持至关重要。

## 修改了哪些文件?

```
✏️  include/systolic/Analysis/ParametricSpaceTime.h
    ├─ 新增: createFromLoopIndices() 工厂方法
    ├─ 新增: setConfigId() / getConfigId() 方法
    └─ 新增: unsigned configId 成员变量

✏️  lib/Analysis/ParametricSpaceTime.cpp
    └─ 新增: createFromLoopIndices() 函数实现

✏️  lib/Transforms/SystolicTransform.cpp
    ├─ 新增: #include "systolic/Analysis/ParametricSpaceTime.h"
    ├─ 新增: enumerateSpaceTimeConfigs() 函数 (~160行)
    └─ 修改: SystolicTransformOptions 结构
         ├─ int spaceTimeMode = -1
         ├─ unsigned maxSADim = 2
         └─ bool listConfigs = false

📄 RESTORATION_REPORT.md (新创建)
   └─ 详细的恢复报告
```

## 何时需要这个功能?

| 场景 | 环数 | 需要动态枚举 |
|------|-----|-----------|
| MatMul (MM) | 3 | ❌ (ST0-ST5足够) |
| MTTKRP | 4 | ✅ (14种配置) |
| TTMC | 4 | ✅ (14种配置) |
| CNN | 5 | ✅ (15种配置) |

## 编译验证

```bash
cd /workspaces/mlir-systolic/build
ninja  # 应该成功编译

# 检查库文件
[ -f lib/libSystolicAnalysis.a ] && echo "✓ Analysis library OK"
[ -f lib/libSystolicTransforms.a ] && echo "✓ Transforms library OK"
```

## 保留了什么?

所有新增文件都被保留:
- ✅ `scripts/generate_autosa_reference.py` - AutoSA参考生成器
- ✅ `DYNAMIC_ENUMERATION_ANALYSIS.md` - 详细分析
- ✅ `DYNAMIC_ENUMERATION_VISUALIZATION.md` - 可视化对比
- ✅ `ANALYSIS_SUMMARY.txt` - 执行摘要

## 核心功能

### enumerateSpaceTimeConfigs() 函数

这个函数枚举所有可能的空间-时间配置:

```cpp
// 输入: 循环列表 + 依赖信息 + 最大PE阵列维度
// 输出: 所有有效的ParametricSpaceTime配置

static LogicalResult enumerateSpaceTimeConfigs(
    const SmallVector<AffineForOp> &loops,
    const SmallVectorImpl<LoopDepInfo> &depInfos,
    unsigned maxSADim = 2,
    SmallVector<ParametricSpaceTime, 8> &configs);
```

### 配置创建工厂

```cpp
// 从循环索引动态创建配置
ParametricSpaceTime config = 
    ParametricSpaceTime::createFromLoopIndices(
        spaceLoopIndices,  // 例如: [0, 1]
        timeLoopIndices,   // 例如: [2]
        loopNames);        // 例如: ["i", "j", "k"]
```

## 下一步

### 1. 测试基础功能
```bash
cd /workspaces/mlir-systolic/build
./bin/systolic-opt ../test/matmul.mlir \
  --systolic-transform \
  -debug-only=systolic-parametric-spacetime
```

### 2. 测试4环支持
需要编写或查找4环测试用例(例如MTTKRP)

### 3. 测试5环支持
需要编写或查找5环测试用例(例如CNN)

## 关键代码改进

### 配置ID管理
```cpp
// 配置ID对应AutoSA的space_time_id字段
config.setConfigId(0);  // 第一个配置
unsigned id = config.getConfigId();  // 获取ID
```

### 动态模式选择
```cpp
// SystolicTransformOptions现在支持动态模式:
// - spaceTimeMode < 0: 使用默认/启发式选择
// - spaceTimeMode >= 0: 使用该索引的配置
```

## 恢复方案说明

### 为什么不完全回滚?

❌ **不能做**: `git revert 13c18ae`
- 会删除 generate_autosa_reference.py
- 会删除分析文档和报告
- 会损失后续改进

✅ **已做**: 选择性手动应用
- 提取 enumerateSpaceTimeConfigs() 函数
- 更新必要的结构和方法
- 保留所有新增工作

## 验证清单

- [x] ParametricSpaceTime.h 已更新
- [x] ParametricSpaceTime.cpp 已实现
- [x] SystolicTransform.cpp 已恢复
- [x] 项目编译成功
- [x] 库文件已生成
- [x] 新增文件已保留
- [x] 文档已更新

## 相关文档

- [RESTORATION_REPORT.md](RESTORATION_REPORT.md) - 完整恢复报告
- [DYNAMIC_ENUMERATION_ANALYSIS.md](DYNAMIC_ENUMERATION_ANALYSIS.md) - 详细技术分析
- [DYNAMIC_ENUMERATION_VISUALIZATION.md](DYNAMIC_ENUMERATION_VISUALIZATION.md) - 可视化对比
- [ANALYSIS_SUMMARY.txt](ANALYSIS_SUMMARY.txt) - 执行摘要

---

最后更新: 本次恢复
状态: ✅ 完成
