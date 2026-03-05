#!/usr/bin/env bash
# 本地生成文件检查：先跑全量 e2e 生成 HLS，再对输出 .cpp 做关键模式检查，便于本地排错。
# 用法: ./test/inspect_generated.sh
# 依赖: 先构建并跑 e2e（本脚本会先执行 run_all_e2e.sh）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== 1. 运行全量 e2e 生成 HLS =========="
"$SCRIPT_DIR/run_all_e2e.sh" || true

echo ""
echo "========== 2. 生成文件关键模式检查 =========="

count_in() { local c; c=$(grep -c "$1" "$2" 2>/dev/null); echo "${c:-0}"; }

check_file() {
  local path="$1"
  local name="$2"
  local skip_reason="$3"
  if [[ ! -f "$path" ]]; then
    if [[ -n "$skip_reason" ]]; then
      echo "--- $name ---"
      echo "  (未生成 cpp: $skip_reason)"
    else
      echo "[$name] 文件不存在: $path"
    fi
    return
  fi
  echo "--- $name ($path) ---"
  echo "  PE/IO 内 r1 循环: $(count_in "r1" "$path")"
  echo "  PE/IO 内 r2 循环: $(count_in "r2" "$path")"
  echo "  word_idx 含 r1: $(count_in "word_idx.*r1" "$path")"
  echo "  注释 3D r1=plane: $(count_in "r1 = plane" "$path")"
  echo "  buffer_linear: $(count_in "buffer_linear" "$path")"
  echo "  PIPELINE II=1: $(count_in "PIPELINE II=1" "$path")"
  echo "  DATAFLOW: $(count_in "DATAFLOW" "$path")"
}

check_file "/tmp/mm_e2e_out.cpp" "MM"
check_file "/tmp/mttkrp_e2e_out.cpp" "MTTKRP_std"
check_file "/tmp/ttmc_std_e2e_out.cpp" "标准TTMc"
check_file "/tmp/reorder_e2e_out.cpp" "写时重排(2D)"
check_file "/tmp/reorder_3d_e2e_out.cpp" "写时重排(3D)"

echo ""
echo "说明: 本地仅做生成与模式检查；HLS csim/综合留待阶段性工作结束后在服务器统一进行。"
