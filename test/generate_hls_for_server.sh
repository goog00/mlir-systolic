#!/usr/bin/env bash
# 生成供服务器 HLS 测试用的全部 .cpp：MM、MTTKRP、TTMc（基础版 + 写时重排版）。
# 用法: ./test/generate_hls_for_server.sh [输出目录]
# 默认输出目录: build/hls_for_server/
# 生成文件: mm.cpp, mttkrp_std.cpp, mttkrp_std_reorder.cpp, ttmc_std.cpp, ttmc_std_reorder.cpp

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
OUT_DIR="${1:-$BUILD_DIR/hls_for_server}"

OPT="$BUILD_DIR/bin/systolic-opt"
TRANS="$BUILD_DIR/bin/systolic-translate"

if [[ ! -x "$OPT" ]] || [[ ! -x "$TRANS" ]]; then
  echo "FAIL: build first (e.g. ./scripts/build-systolic.sh)"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "========== 生成 HLS 代码 → $OUT_DIR =========="

# 公共参数（与 e2e 一致）
MM_SIZE=32
MM_AP=8
MM_LAT=4
MM_SIMD=1

KERNEL_SIZE=8
KERNEL_AP=8
KERNEL_LAT=4
KERNEL_SIMD=1

# ---- MM（基础） ----
echo "[1/5] MM..."
"$OPT" "$REPO_ROOT/test/minimal_matmul.mlir" --systolic-transform --systolic-dataflow-generation -o "$OUT_DIR/mm.mlir"
"$TRANS" "$OUT_DIR/mm.mlir" --size=$MM_SIZE --array-part=$MM_AP --latency=$MM_LAT --simd=$MM_SIMD -o "$OUT_DIR/mm.cpp"
echo "      → $OUT_DIR/mm.cpp"

# ---- MTTKRP 标准（基础） ----
echo "[2/5] MTTKRP 标准..."
"$OPT" "$REPO_ROOT/test/minimal_mttkrp_std.mlir" --systolic-transform --systolic-dataflow-generation -o "$OUT_DIR/mttkrp_std.mlir"
"$TRANS" "$OUT_DIR/mttkrp_std.mlir" --size=$KERNEL_SIZE --array-part=$KERNEL_AP --latency=$KERNEL_LAT --simd=$KERNEL_SIMD -o "$OUT_DIR/mttkrp_std.cpp"
echo "      → $OUT_DIR/mttkrp_std.cpp"

# ---- MTTKRP 标准（写时重排：先分析再 transform） ----
echo "[3/5] MTTKRP 标准 + 写时重排..."
"$OPT" "$REPO_ROOT/test/minimal_mttkrp_std.mlir" --systolic-write-reorder-analysis --systolic-transform --systolic-dataflow-generation -o "$OUT_DIR/mttkrp_std_reorder.mlir"
"$TRANS" "$OUT_DIR/mttkrp_std_reorder.mlir" --size=$KERNEL_SIZE --array-part=$KERNEL_AP --latency=$KERNEL_LAT --simd=$KERNEL_SIMD -o "$OUT_DIR/mttkrp_std_reorder.cpp"
echo "      → $OUT_DIR/mttkrp_std_reorder.cpp"

# ---- TTMc 标准（基础） ----
echo "[4/5] TTMc 标准..."
"$OPT" "$REPO_ROOT/test/minimal_ttmc_std.mlir" --systolic-transform --systolic-dataflow-generation -o "$OUT_DIR/ttmc_std.mlir"
"$TRANS" "$OUT_DIR/ttmc_std.mlir" --size=$KERNEL_SIZE --array-part=$KERNEL_AP --latency=$KERNEL_LAT --simd=$KERNEL_SIMD -o "$OUT_DIR/ttmc_std.cpp"
echo "      → $OUT_DIR/ttmc_std.cpp"

# ---- TTMc 标准（写时重排） ----
echo "[5/5] TTMc 标准 + 写时重排..."
"$OPT" "$REPO_ROOT/test/minimal_ttmc_std.mlir" --systolic-write-reorder-analysis --systolic-transform --systolic-dataflow-generation -o "$OUT_DIR/ttmc_std_reorder.mlir"
"$TRANS" "$OUT_DIR/ttmc_std_reorder.mlir" --size=$KERNEL_SIZE --array-part=$KERNEL_AP --latency=$KERNEL_LAT --simd=$KERNEL_SIMD -o "$OUT_DIR/ttmc_std_reorder.cpp"
echo "      → $OUT_DIR/ttmc_std_reorder.cpp"

echo ""
echo "========== 完成 =========="
echo "输出目录: $OUT_DIR"
echo "  mm.cpp, mttkrp_std.cpp, mttkrp_std_reorder.cpp, ttmc_std.cpp, ttmc_std_reorder.cpp"
echo "可将上述目录打包到服务器进行 HLS 综合与 C sim。"
echo "写时重排版若分析未命中则与基础版相同；可检查 .cpp 中是否含 buffer_linear 确认。"
