#!/usr/bin/env bash
# 全量端到端测试：依次执行 MM、MTTKRP、标准语义 MTTKRP/TTMc、写时重排 e2e 脚本，汇总结果。
# 用法: ./test/run_all_e2e.sh
# 依赖: 先构建 build (e.g. ./scripts/build-systolic.sh)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FAILED=()
PASSED=()

run_one() {
  local name="$1"
  local script="$2"
  if [[ ! -x "$script" ]]; then
    echo "SKIP: $script not executable"
    return 1
  fi
  if "$script"; then
    echo "--- $name: PASS"
    PASSED+=("$name")
    return 0
  else
    echo "--- $name: FAIL"
    FAILED+=("$name")
    return 1
  fi
}

echo "========== mlir-systolic 全量 e2e 测试 =========="
run_one "MM"           "$SCRIPT_DIR/run_mm_e2e.sh"            || true
run_one "MTTKRP"       "$SCRIPT_DIR/run_mttkrp_e2e.sh"        || true
run_one "标准语义TTMc"   "$SCRIPT_DIR/run_ttmc_std_e2e.sh"      || true
run_one "写时重排(2D)" "$SCRIPT_DIR/run_reorder_e2e.sh"       || true
run_one "写时重排(3D)" "$SCRIPT_DIR/run_reorder_3d_e2e.sh"    || true

echo ""
echo "========== 汇总 =========="
echo "通过: ${#PASSED[@]} — ${PASSED[*]:-无}"
echo "失败: ${#FAILED[@]} — ${FAILED[*]:-无}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "结果: 存在失败用例"
  exit 1
fi
echo "结果: 全部通过"
exit 0
