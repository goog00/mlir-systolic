# Allo 相关参考文档

> **目录**: `docs/reference/allo/`  
> **目的**: 收集与 Allo 工具相关的分析、集成方案与 HLS 代码生成规则

---

## 文档列表

1. **[ALLO_ANALYSIS_AND_PYTORCH_ROADMAP.md](ALLO_ANALYSIS_AND_PYTORCH_ROADMAP.md)**  
   - Allo 的定位与两大贡献（Python 计算+调度 DSL、PyTorch 兼容）
   - 可借鉴点与 PyTorch 兼容路线（复用前端 vs 自建、接口约定、分阶段）

2. **[ALLO_INTEGRATION_ANALYSIS.md](ALLO_INTEGRATION_ANALYSIS.md)**  
   - Allo 与 mlir-systolic 的对比与集成方案
   - 混合路径与实施阶段（先核心再前端）

3. **[ALLO_HLS_CODE_GENERATION_RULES.md](ALLO_HLS_CODE_GENERATION_RULES.md)**  
   - Allo 如何将 Affine MLIR 转为 HLS C++（操作映射、pragma、stream 等）

---

## 推荐阅读顺序

- 了解 Allo 与 PyTorch 路线：ALLO_ANALYSIS_AND_PYTORCH_ROADMAP.md  
- 做集成设计：ALLO_INTEGRATION_ANALYSIS.md  
- 对照 HLS 生成细节：ALLO_HLS_CODE_GENERATION_RULES.md  
