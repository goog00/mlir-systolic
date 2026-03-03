#!/usr/bin/env bash
# 写时重排端到端：使用 minimal_reorder_write.mlir（输出 C 第一维为 i*32+j，触发 reorder），
# 检查生成的 HLS 中 drain_IO_L3_out_serialize 包含 Write-time reordering 注释或 buffer_linear。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
INPUT="${1:-$REPO_ROOT/test/minimal_reorder_write.mlir}"
OUT_MLIR="${2:-/tmp/reorder_e2e_out.mlir}"
OUT_CPP="${3:-/tmp/reorder_e2e_out.cpp}"

OPT="$BUILD_DIR/bin/systolic-opt"
TRANS="$BUILD_DIR/bin/systolic-translate"

if [[ ! -x "$OPT" ]] || [[ ! -x "$TRANS" ]]; then
  echo "FAIL: build first (e.g. ./scripts/build-systolic.sh)"
  exit 1
fi
if [[ ! -r "$INPUT" ]]; then
  echo "FAIL: input not found: $INPUT"
  exit 1
fi

SIZE=32
ARRAY_PART=8
LATENCY=4
SIMD=1

# 写时重排分析必须在 transform 之前运行（transform 会 outline 循环，store 的 affine.apply 会消失）
"$OPT" "$INPUT" --systolic-write-reorder-analysis --systolic-transform --systolic-dataflow-generation -o "$OUT_MLIR"
"$TRANS" "$OUT_MLIR" --size=$SIZE --array-part=$ARRAY_PART --latency=$LATENCY --simd=$SIMD -o "$OUT_CPP"

if [[ ! -r "$OUT_CPP" ]]; then
  echo "FAIL: generated file missing: $OUT_CPP"
  exit 1
fi

# 验证写时重排路径：应出现 buffer_linear 或 Write-time reordering 注释
if grep -q "buffer_linear" "$OUT_CPP"; then
  echo "PASS: Write-time reordering path (buffer_linear) present in $OUT_CPP"
  exit 0
fi

# 失败时提示：若 MLIR 中无 systolic.reorder 则问题在 pass；若有则问题可能在 translate
if grep -q "systolic.reorder" "$OUT_MLIR" 2>/dev/null; then
  echo "FAIL: buffer_linear missing in cpp; reorder attrs found in $OUT_MLIR (check translate hasReordering2D/outputName)"
else
  echo "FAIL: no buffer_linear in cpp and no systolic.reorder in $OUT_MLIR (run --systolic-write-reorder-analysis before transform)"
fi
exit 1
