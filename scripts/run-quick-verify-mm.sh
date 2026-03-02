#!/usr/bin/env bash
# Run MM pipeline (opt -> translate) for the given input.
# Parameters (size, array-part, latency, simd) must be valid for the input;
# they should come from polyhedral analysis selection range, not a global preset.
# For minimal_matmul.mlir (32x32), example valid options: --size=32 --array-part=8 --latency=4 --simd=1.
# See docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md and
# third_party/AutoSA/docs/tutorials/getting_started.rst.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
INPUT="${1:-$REPO_ROOT/test/minimal_matmul.mlir}"
OUT_MLIR="${2:-}"
OUT_CPP="${3:-}"

if [[ -z "$OUT_MLIR" ]]; then
  OUT_MLIR="/tmp/mm_out.mlir"
fi
if [[ -z "$OUT_CPP" ]]; then
  OUT_CPP="/tmp/mm_out.cpp"
fi

OPT="$BUILD_DIR/bin/systolic-opt"
TRANS="$BUILD_DIR/bin/systolic-translate"

if [[ ! -x "$OPT" ]] || [[ ! -x "$TRANS" ]]; then
  echo "Build mlir-systolic first (e.g. ./scripts/build-systolic.sh)"
  exit 1
fi

echo "Input: $INPUT"
echo "  opt -> $OUT_MLIR"
echo "  translate -> $OUT_CPP (pass size/array-part/latency/simd as needed for this input)"

"$OPT" "$INPUT" --systolic-transform --systolic-dataflow-generation -o "$OUT_MLIR"
# Example for 32x32 matmul: parameters must be valid for the input (e.g. divisors of loop bounds)
"$TRANS" "$OUT_MLIR" --size=32 --array-part=8 --latency=4 --simd=1 -o "$OUT_CPP"

echo "Done. Generated: $OUT_CPP"
