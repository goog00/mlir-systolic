# 写时重排（Write-time Reordering）文档

> **目录**: `docs/features/write-time-reordering/`  
> **目的**: 写时重排功能相关实现与改进说明。当前 e2e 与 Pass 顺序见根目录 [RECENT_CHANGES_AND_NEXT_STEPS.md](../../../RECENT_CHANGES_AND_NEXT_STEPS.md)。

---

## 📚 本目录文档列表

| 文件 | 说明 |
|------|------|
| [WRITE_TIME_REORDERING_IMPLEMENTATION.md](WRITE_TIME_REORDERING_IMPLEMENTATION.md) | 写时重排实现细节与步骤 |
| [PHASE2_IMPLEMENTATION_SUMMARY.md](PHASE2_IMPLEMENTATION_SUMMARY.md) | Phase 2 实现总结 |
| [IMPLEMENTATION_IMPROVEMENTS.md](IMPLEMENTATION_IMPROVEMENTS.md) | 实现改进说明 |

---

## 🔗 相关文档

- **设计/优化**：写时重排与 L3、host-serialize 关系见 [../../design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md](../../design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md)；已有优化梳理见 [../../design/EXISTING_OPTIMIZATIONS_IN_CODE.md](../../design/EXISTING_OPTIMIZATIONS_IN_CODE.md)。
- **验证**：`test/run_reorder_e2e.sh`、`test/run_reorder_3d_e2e.sh`，需先跑 `--systolic-write-reorder-analysis` 再 transform/dataflow。
