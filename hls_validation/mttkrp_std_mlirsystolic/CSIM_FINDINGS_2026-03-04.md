# Standard MTTKRP csim 结果（2026-03-04）

## 结论
- `hls_validation/mttkrp_std_mlirsystolic/tb_kernel.cpp` + `run_hls_csim.sh` 运行结果：**FAIL**。
- 典型错误：`hw=8, ref=64`（全 1 输入时）。
- 这说明当前生成的 HLS C 只覆盖了一个规约维（8 次累加），未覆盖标准 MTTKRP 的双规约（8x8=64 次累加）。

## 复现命令
```bash
cd hls_validation/mttkrp_std_mlirsystolic
enable_vitis
./run_hls_csim.sh
```

## 关键证据
1. 标准语义用例：`test/minimal_mttkrp_std.mlir`
   - 语义：`D(i,j) += A(i,k,l) * B(k,j) * C(l,j)`，应有 k/l 双规约。
2. 生成内核 PE 结构（`kernel.cpp`）中仅有一层“规约型”循环计数（`c5`），
   对应单规约模板。
3. csim 结果每个输出是 8 而非 64，符合“只做一维规约”的失真模式。

## 影响
- 当前 `mttkrp` 虽然可生成 HLS C，但语义不正确，不能作为性能对比基线。
- `ttmc` 更复杂（同样涉及双规约且输出 rank-3），当前后端同样尚不具备正确支持能力。

## 建议
- 先在 translator 中显式区分“单规约模板”与“双规约模板”，
  在双规约模板未完成前，不应把 mttkrp/ttmc 标记为可比性能对象。
