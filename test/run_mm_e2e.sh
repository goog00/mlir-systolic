#!/usr/bin/env bash
# MM 端到端回归：opt -> translate，并检查生成的 HLS C++ 包含关键符号。
# 参数应对应当前输入合法范围（见 docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md）。
# 用法: ./test/run_mm_e2e.sh [input.mlir] [out.mlir] [out.cpp]
# 若无参数则使用 test/minimal_matmul.mlir 与 /tmp 下输出路径。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
INPUT="${1:-$REPO_ROOT/test/minimal_matmul.mlir}"
OUT_MLIR="${2:-/tmp/mm_e2e_out.mlir}"
OUT_CPP="${3:-/tmp/mm_e2e_out.cpp}"

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

# 示例参数：对 32x32 matmul 合法（循环界 32 的因子）
SIZE=32
ARRAY_PART=8
LATENCY=4
SIMD=1

"$OPT" "$INPUT" --systolic-transform --systolic-dataflow-generation -o "$OUT_MLIR"
"$TRANS" "$OUT_MLIR" --size=$SIZE --array-part=$ARRAY_PART --latency=$LATENCY --simd=$SIMD -o "$OUT_CPP"

if [[ ! -r "$OUT_CPP" ]]; then
  echo "FAIL: generated file missing: $OUT_CPP"
  exit 1
fi

# 检查关键符号
MISSING=""
grep -q "kernel0" "$OUT_CPP" || MISSING="${MISSING} kernel0"
grep -q "PIPELINE" "$OUT_CPP" || MISSING="${MISSING} PIPELINE"
grep -q "DATAFLOW" "$OUT_CPP" || MISSING="${MISSING} DATAFLOW"
grep -q "hls::stream" "$OUT_CPP" || MISSING="${MISSING} hls::stream"
grep -q "PE_wrapper" "$OUT_CPP" || MISSING="${MISSING} PE_wrapper"

if [[ -n "$MISSING" ]]; then
  echo "FAIL: missing in $OUT_CPP:$MISSING"
  exit 1
fi

echo "PASS: MM e2e opt->translate; output: $OUT_CPP"
