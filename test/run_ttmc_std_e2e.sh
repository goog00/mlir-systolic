#!/usr/bin/env bash
# 标准语义 TTMc e2e: opt -> translate（期望成功，rank-3 + num_time_loops=3 已支持）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
INPUT="${1:-$REPO_ROOT/test/minimal_ttmc_std.mlir}"
OUT_MLIR="${2:-/tmp/ttmc_std_e2e_out.mlir}"
OUT_CPP="${3:-/tmp/ttmc_std_e2e_out.cpp}"
ERR_LOG="${4:-/tmp/ttmc_std_translate.err}"

OPT="$BUILD_DIR/bin/systolic-opt"
TRANS="$BUILD_DIR/bin/systolic-translate"

[[ -x "$OPT" && -x "$TRANS" ]] || { echo "FAIL: build first (e.g. ./scripts/build-systolic.sh)"; exit 1; }
[[ -r "$INPUT" ]] || { echo "FAIL: input not found: $INPUT"; exit 1; }

SIZE=8
ARRAY_PART=8
LATENCY=4
SIMD=1

"$OPT" "$INPUT" --systolic-transform --systolic-dataflow-generation -o "$OUT_MLIR"

"$TRANS" "$OUT_MLIR" --size=$SIZE --array-part=$ARRAY_PART --latency=$LATENCY --simd=$SIMD -o "$OUT_CPP" 2>"$ERR_LOG" || { echo "FAIL: TTMc translate failed."; cat "$ERR_LOG"; exit 1; }

[[ -f "$OUT_CPP" ]] || { echo "FAIL: no output cpp: $OUT_CPP"; exit 1; }
# 三规约维应生成 r2 循环
grep -q "r2" "$OUT_CPP" || { echo "FAIL: expected r2 loop in TTMc output."; exit 1; }

echo "PASS: Standard TTMc e2e (translate success, r2 present)."
