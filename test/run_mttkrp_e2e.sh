#!/usr/bin/env bash
# 4-loop MTTKRP 端到端：opt -> translate，检查生成 HLS 存在且含关键符号。
# 参数应对应当前输入合法范围（8x8x8 等小规模）。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
INPUT="${1:-$REPO_ROOT/test/minimal_mttkrp.mlir}"
OUT_MLIR="${2:-/tmp/mttkrp_e2e_out.mlir}"
OUT_CPP="${3:-/tmp/mttkrp_e2e_out.cpp}"

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

# 8x8x8 小规模，合法参数示例
SIZE=8
ARRAY_PART=8
LATENCY=4
SIMD=1

"$OPT" "$INPUT" --systolic-transform --systolic-dataflow-generation -o "$OUT_MLIR"
"$TRANS" "$OUT_MLIR" --size=$SIZE --array-part=$ARRAY_PART --latency=$LATENCY --simd=$SIMD -o "$OUT_CPP"

if [[ ! -r "$OUT_CPP" ]]; then
  echo "FAIL: generated file missing: $OUT_CPP"
  exit 1
fi

MISSING=""
grep -q "kernel0" "$OUT_CPP" || MISSING="${MISSING} kernel0"
grep -q "PIPELINE" "$OUT_CPP" || MISSING="${MISSING} PIPELINE"
grep -q "DATAFLOW" "$OUT_CPP" || MISSING="${MISSING} DATAFLOW"
grep -q "PE_wrapper" "$OUT_CPP" || MISSING="${MISSING} PE_wrapper"
grep -q "drain_IO_L3_out_serialize" "$OUT_CPP" || MISSING="${MISSING} drain_IO_L3_out_serialize"

if [[ -n "$MISSING" ]]; then
  echo "FAIL: missing in $OUT_CPP:$MISSING"
  exit 1
fi

echo "PASS: MTTKRP 4-loop e2e opt->translate; output: $OUT_CPP"
