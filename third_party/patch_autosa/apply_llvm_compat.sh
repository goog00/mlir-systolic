#!/bin/sh
# 在 mlir-systolic 下：先对本目录执行此脚本完成 ISL ppcg 补丁 + LLVM 18 / Ubuntu 22.04 兼容性补丁，
# 并改写 install.sh（去掉子模块拉取与 ISL 打补丁步骤）。子模块应由 mlir-systolic 递归拉取，无需在 install.sh 中再拉取。
# 用法（在 AutoSA 仓库根目录执行）:
#   ../patch_autosa/apply_llvm_compat.sh
# 或从 third_party 执行:
#   ./patch_autosa/apply_llvm_compat.sh

set -e
PATCHES_DIR="$(cd "$(dirname "$0")" && pwd)"
AUTOSA_ROOT="$(cd "$PATCHES_DIR/../AutoSA" && pwd)"
if [ ! -d "$AUTOSA_ROOT/src" ]; then
	echo "Error: AutoSA not found at $AUTOSA_ROOT (expected patch_autosa and AutoSA to be siblings)."
	exit 1
fi

# 1) 先执行 ISL ppcg 补丁（将 autosa_scripts/ppcg_changes/isl 覆盖到 src/isl），与 install.sh 原流程一致
echo "=== Applying ISL ppcg patch (isl_patch.sh) ==="
if [ -f "$AUTOSA_ROOT/autosa_scripts/ppcg_changes/isl/isl_patch.sh" ]; then
	(cd "$AUTOSA_ROOT/autosa_scripts/ppcg_changes/isl" && ./isl_patch.sh)
	echo "  OK."
else
	echo "  Skip: isl_patch.sh not found (optional in some layouts)."
fi

apply_patch() {
	local dir="$1"
	local patch_file="$PATCHES_DIR/$2"
	local p="${3:-1}"
	if [ ! -f "$patch_file" ]; then
		echo "Skip (file not found): $patch_file"
		return 0
	fi
	if [ ! -d "$dir" ]; then
		echo "Skip (dir not found): $dir"
		return 0
	fi
	echo "Applying $(basename "$patch_file") in $dir (-p$p) ..."
	if patch -p"$p" -d "$dir" --forward --dry-run < "$patch_file" 2>/dev/null; then
		patch -p"$p" -d "$dir" --forward < "$patch_file"
		echo "  OK."
	else
		echo "  Failed (maybe already applied or conflict)."
		return 1
	fi
}

echo "=== Applying LLVM 18 / Ubuntu 22.04 compatibility patches ==="
# pet/isl 补丁路径无顶层目录前缀，用 -p0
apply_patch "$AUTOSA_ROOT/src/pet" "pet-llvm18-ubuntu22.patch" 0 || true
apply_patch "$AUTOSA_ROOT/src/pet" "pet-ldadd-clang-cpp.patch" 0 || true
apply_patch "$AUTOSA_ROOT/src/isl" "isl-interface-llvm18.patch" 0 || true
if [ -d "$AUTOSA_ROOT/src/pet/isl" ]; then
	apply_patch "$AUTOSA_ROOT/src/pet/isl" "isl-interface-llvm18.patch" 0 || true
fi
apply_src_patch() {
	local patch_file="$PATCHES_DIR/src-llvm18.patch"
	if [ ! -f "$patch_file" ]; then return 0; fi
	echo "Applying $(basename "$patch_file") (top-level src) ..."
	if patch -p0 -d "$AUTOSA_ROOT" --forward --dry-run < "$patch_file" 2>/dev/null; then
		patch -p0 -d "$AUTOSA_ROOT" --forward < "$patch_file"
		echo "  OK."
	else
		echo "  Failed (maybe already applied or conflict)."
		return 1
	fi
}
apply_src_patch || true

# 4) 修改 install.sh：去掉子模块 init/update 与「Patch ISL」步骤，适配 mlir-systolic 先打补丁再 install 的流程
if [ -f "$PATCHES_DIR/install-sh-mlir-systolic.patch" ]; then
	echo "=== Patching install.sh for mlir-systolic workflow ==="
	if patch -p1 -d "$AUTOSA_ROOT" --forward --dry-run < "$PATCHES_DIR/install-sh-mlir-systolic.patch" 2>/dev/null; then
		patch -p1 -d "$AUTOSA_ROOT" --forward < "$PATCHES_DIR/install-sh-mlir-systolic.patch"
		echo "  OK."
	else
		echo "  Skip or already applied."
	fi
fi

# 5) requirements.txt：sklearn 已改名为 scikit-learn
if [ -f "$PATCHES_DIR/requirements-txt-scikit-learn.patch" ]; then
	echo "=== Patching requirements.txt (sklearn -> scikit-learn) ==="
	if patch -p1 -d "$AUTOSA_ROOT" --forward --dry-run < "$PATCHES_DIR/requirements-txt-scikit-learn.patch" 2>/dev/null; then
		patch -p1 -d "$AUTOSA_ROOT" --forward < "$PATCHES_DIR/requirements-txt-scikit-learn.patch"
		echo "  OK."
	else
		echo "  Skip or already applied."
	fi
fi

# 6) src/autogen.sh：首次 autoreconf 后把 build/ltmain.sh 复制到 src/，供 barvinok 等子目录的 libtoolize 使用（避免 required file './ltmain.sh' not found）
if [ -f "$PATCHES_DIR/autogen-sh-ltmain.patch" ]; then
	echo "=== Patching src/autogen.sh (ltmain for subdirs) ==="
	if patch -p0 -d "$AUTOSA_ROOT/src" --forward --dry-run < "$PATCHES_DIR/autogen-sh-ltmain.patch" 2>/dev/null; then
		patch -p0 -d "$AUTOSA_ROOT/src" --forward < "$PATCHES_DIR/autogen-sh-ltmain.patch"
		echo "  OK."
	else
		echo "  Skip or already applied."
	fi
fi

# 7) isl/barvinok autogen.sh：在 autoreconf 前先运行 libtoolize，确保 ./ltmain.sh 存在，避免第一次 autogen 失败需跑两遍
if [ -f "$PATCHES_DIR/isl-autogen-libtoolize.patch" ] && [ -d "$AUTOSA_ROOT/src/isl" ]; then
	echo "=== Patching src/isl/autogen.sh (libtoolize first) ==="
	if patch -p0 -d "$AUTOSA_ROOT/src/isl" --forward --dry-run < "$PATCHES_DIR/isl-autogen-libtoolize.patch" 2>/dev/null; then
		patch -p0 -d "$AUTOSA_ROOT/src/isl" --forward < "$PATCHES_DIR/isl-autogen-libtoolize.patch"
		echo "  OK."
	else
		echo "  Skip or already applied."
	fi
fi
if [ -f "$PATCHES_DIR/barvinok-autogen.patch" ] && [ -d "$AUTOSA_ROOT/src/barvinok" ]; then
	echo "=== Patching src/barvinok/autogen.sh (ltmain + libtoolize for one-pass autogen) ==="
	if patch -p0 -d "$AUTOSA_ROOT/src/barvinok" --forward --dry-run < "$PATCHES_DIR/barvinok-autogen.patch" 2>/dev/null; then
		patch -p0 -d "$AUTOSA_ROOT/src/barvinok" --forward < "$PATCHES_DIR/barvinok-autogen.patch"
		echo "  OK."
	else
		echo "  Skip or already applied."
	fi
fi

echo "=== Done. You can now run install.sh or: (cd $AUTOSA_ROOT/src && ./autogen.sh && ./configure && make -j4) ==="
