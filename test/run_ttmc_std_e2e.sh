#!/usr/bin/env bash
# 标准语义 TTMc e2e: opt -> translate（当前期望失败，因 rank-3 输出尚不支持）
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

set +e
"$TRANS" "$OUT_MLIR" --size=$SIZE --array-part=$ARRAY_PART --latency=$LATENCY --simd=$SIMD -o "$OUT_CPP" 2>"$ERR_LOG"
RC=$?
set -e

if [[ $RC -eq 0 ]]; then
  echo "FAIL: TTMc translate unexpectedly succeeded; rank-3 output support check may be bypassed."
  exit 1
fi

if ! grep -q "unsupported output rank 3" "$ERR_LOG"; then
  echo "FAIL: translate failed, but not with expected rank-3 unsupported message."
  echo "---- stderr ----"
  cat "$ERR_LOG"
  echo "--------------"
  exit 1
fi

echo "PASS: Standard TTMc e2e got expected failure (rank-3 unsupported)."
