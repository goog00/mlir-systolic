#!/usr/bin/env bash
# 标准语义 MTTKRP e2e: opt -> translate（期望成功）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
INPUT="${1:-$REPO_ROOT/test/minimal_mttkrp_std.mlir}"
OUT_MLIR="${2:-/tmp/mttkrp_std_e2e_out.mlir}"
OUT_CPP="${3:-/tmp/mttkrp_std_e2e_out.cpp}"

OPT="$BUILD_DIR/bin/systolic-opt"
TRANS="$BUILD_DIR/bin/systolic-translate"

[[ -x "$OPT" && -x "$TRANS" ]] || { echo "FAIL: build first (e.g. ./scripts/build-systolic.sh)"; exit 1; }
[[ -r "$INPUT" ]] || { echo "FAIL: input not found: $INPUT"; exit 1; }

SIZE=8
ARRAY_PART=8
LATENCY=4
SIMD=1

"$OPT" "$INPUT" --systolic-transform --systolic-dataflow-generation -o "$OUT_MLIR"
"$TRANS" "$OUT_MLIR" --size=$SIZE --array-part=$ARRAY_PART --latency=$LATENCY --simd=$SIMD -o "$OUT_CPP"

[[ -r "$OUT_CPP" ]] || { echo "FAIL: generated file missing: $OUT_CPP"; exit 1; }

MISSING=""
grep -q "kernel0" "$OUT_CPP" || MISSING="$MISSING kernel0"
grep -q "DATAFLOW" "$OUT_CPP" || MISSING="$MISSING DATAFLOW"
grep -q "PE_wrapper" "$OUT_CPP" || MISSING="$MISSING PE_wrapper"
grep -q "arg2_IO_L3_in" "$OUT_CPP" || MISSING="$MISSING third_input_path"

if [[ -n "$MISSING" ]]; then
  echo "FAIL: missing in $OUT_CPP:$MISSING"
  exit 1
fi

echo "PASS: Standard MTTKRP e2e opt->translate; output: $OUT_CPP"
