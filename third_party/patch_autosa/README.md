# AutoSA LLVM 18 / Ubuntu 22.04 兼容性补丁

本目录包含使 AutoSA 在 **Ubuntu 22.04**（及更新系统）上使用 **LLVM/Clang 18**（或 14–18）编译和运行的补丁。  
官方 Docker 使用 Ubuntu + LLVM 9，若在较新系统上直接编译会因 API 变更而失败。

**放置方式**：请将整个 `patch_autosa` 文件夹放到 **AutoSA 的上级目录**（与 `AutoSA` 并列）。例如：

- `third_party/AutoSA/`  ← AutoSA 仓库（从 GitHub 拉取）
- `third_party/patch_autosa/`  ← 本目录（补丁与脚本）

这样 AutoSA 保持为纯净的 GitHub 仓库，补丁独立放在外面。

## 补丁说明

| 文件 | 作用 |
|------|------|
| `pet-llvm18-ubuntu22.patch` | 对 **pet** 子模块的修改：Host.h、LangStandard、setLangDefaults、createFileID(FileEntryRef)、ArgStringList、getBeginLoc/getEndLoc、ArraySizeModifier、comparator const、**CLANG_CPP_LIB 检测与链接**、CREATETARGETINFO/ADDPATH/CREATEPREPROCESSOR/CREATEDIAGNOSTICS 等 configure 检测，以及 pet_codegen/pet_check_code 的 `-lclang-cpp` 与 `-Wl,--no-as-needed` |
| `pet-ldadd-clang-cpp.patch` | 对 **pet** 的补充：为可执行程序 `pet`、`pet_scop_cmp` 的 LDADD 增加 `$(CLANG_CPP_LIB)` 及 `-Wl,--no-as-needed`，解决链接时 `clang::RISCV::RVVIntrinsic`、`clang::SourceMgrAdapter` 等未定义引用（LLVM 18） |
| `isl-interface-llvm18.patch` | 对 **isl** 的 `interface/extract_interface.cc` 的修改：使用 `llvm/TargetParser/Host.h`（LLVM 15+）。脚本会同时应用到 `src/isl` 与 `src/pet/isl`（若存在） |
| `src-llvm18.patch` | 对 **顶层 src** 的修改：`configure.ac` 中在 bundled pet 时检测并导出 `CLANG_CPP_LIB`；`Makefile.am` 中为 autosa 增加 `@CLANG_CPP_LIB@` 与 `-Wl,--no-as-needed`；`autosa_common.cpp` 中 `index < 0` 改为 `!index`（指针判空） |
| `install-sh-mlir-systolic.patch` | 修改 **install.sh**：去掉子模块 init/update 与「Patch ISL」步骤；编译时**自动执行两次 autogen**（第一次 `|| true`，第二次必过），避免 barvinok/parker 等缺 `ltmain.sh` 导致需手动再跑一次 |
| `requirements-txt-scikit-learn.patch` | 修改 **requirements.txt**：`sklearn>=0.0` 改为 `scikit-learn>=0.0`（PyPI 包名已变更） |
| `autogen-sh-ltmain.patch` | 修改 **src/autogen.sh**：**开头**若存在 `../ltmain.sh`（官方仓库在根目录提交的 ltmain.sh），则复制到 `src/` 与 `src/build/`，再执行 `autoreconf -i`；并在调用 barvinok 前把 `ltmain.sh` 预复制到 `barvinok/` 与 `barvinok/parker/`，实现与官方一致的**单次 autogen** |
| `isl-autogen-libtoolize.patch` | 修改 **src/isl/autogen.sh**：在 `autoreconf` 前先运行 `libtoolize --install --copy`，确保一次 autogen 即可成功 |
| `barvinok-autogen.patch` | 修改 **src/barvinok/autogen.sh**：先准备 `ltmain.sh`（从 `../` 或 `../build/` 复制 + `libtoolize`），复制到 **parker**，再运行 `autoreconf`；若首次失败则再次准备并重跑一次，避免 `required file './ltmain.sh' not found`，保证单次执行 autogen 即可成功 |

以上补丁均为必要：pet/isl/src 为 LLVM 18 兼容，install/requirements 为流程与依赖，autogen 相关三份解决 ltmain.sh 与一次生成 configure 的问题。

### 根目录 ltmain.sh 与官方「一次 autogen」说明

- **根目录 `ltmain.sh`**：AutoSA 官方仓库在**仓库根目录**提交了 `ltmain.sh`（GNU libtool 2.4.6），`git ls-files` 可看到。该文件**未被官方脚本显式引用**；官方能一次 autogen 通过，是因为在**旧版 autotools 环境**（如官方 Docker 使用的 Ubuntu + 旧 libtool）下，`autoreconf` 调用的 `libtoolize` 会在每个需要 libtool 的子目录（如 `barvinok/`）里**自动生成或复制** `ltmain.sh` 到当前目录。
- **与 autogen 版本的关系**：在新系统（如 Ubuntu 22.04）上，`libtoolize`/`autoreconf` 的行为可能变化（例如 aux 文件只放到 `AC_CONFIG_AUX_DIR`、不往子目录当前目录写），导致 `barvinok/` 或 `barvinok/parker/` 在首次 autogen 时缺少 `./ltmain.sh`，从而报错 `required file './ltmain.sh' not found`。因此**与 autotools/libtool 版本有关**，官方一次过是在其固定环境下成立。
- **本补丁的做法**：利用仓库已有的根目录 `ltmain.sh`，在 `src/autogen.sh` 中**先**把它复制到 `src/` 和 `src/build/`，并**在运行 `autoreconf -i` 之前**就预复制到 `barvinok/` 和 `barvinok/parker/`（因为 `src` 的 automake 会递归处理 SUBDIRS 里的 barvinok）；在调用 `barvinok/autogen.sh` 前再次用 `src/`、`src/build/` 或 repo 根目录的 ltmain 确保 barvinok/parker 有文件。`barvinok-autogen.patch` 在 barvinok 的 autogen 里增加 `ensure_ltmain`（从 `../`、`../build/` 取 ltmain），与 `src/autogen.sh` 的预复制一起尽量保证一次通过；若仍失败，`install.sh` 的「执行两次 autogen」作为兜底。

### 补丁应用顺序与前置状态

- **顺序**：脚本先执行 **ISL ppcg 补丁**（`autosa_scripts/ppcg_changes/isl/isl_patch.sh`），再按下列顺序应用本目录补丁：pet → isl interface → src → install.sh → requirements.txt → src/autogen.sh → src/isl/autogen.sh → src/barvinok/autogen.sh。
- **前置状态**：`isl-interface-llvm18.patch`、`isl-autogen-libtoolize.patch` 应用对象是 **已执行 isl_patch.sh 之后** 的 `src/isl`。此时 `src/isl/autogen.sh` 仅有两行（`#!/bin/sh`、`autoreconf -i`），`isl-autogen-libtoolize.patch` 已按此 2 行上下文编写，可无 fuzz 应用。
- **barvinok-autogen.patch**：补丁已包含完整上下文（polylib/isl/pet 三个 if 块），避免应用后丢失 isl/pet 的 autogen 调用；hunk 为 `@@ -1,12 +1,22 @@`（整文件 12 行替换为 22 行）。

## 使用方式

### 1. 在 mlir-systolic 下的推荐流程（子模块已递归拉取）

在 **mlir-systolic** 中拉取子模块后，**先** 在 **third_party** 执行本脚本（会完成 ISL ppcg 补丁、LLVM 兼容补丁，并改写 `install.sh`），**再** 在 AutoSA 中执行 `install.sh` 或手动编译：

```bash
# 在 third_party 下执行（AutoSA 与 patch_autosa 并列）
./patch_autosa/apply_llvm_compat.sh

# 然后进入 AutoSA 安装/编译（install.sh 已去掉子模块拉取与「Patch ISL」步骤）
cd AutoSA && ./install.sh
```

或从 **AutoSA 根目录** 执行补丁脚本：

```bash
# 在 third_party/AutoSA 下执行
../patch_autosa/apply_llvm_compat.sh
```

`apply_llvm_compat.sh` 会依次：执行 `isl_patch.sh`、应用 pet/isl/src 的 LLVM 兼容补丁、对 `install.sh` 打 `install-sh-mlir-systolic.patch`（去掉子模块 init/update 与「Patch ISL」步骤，避免与“先打补丁再 install”的流程冲突）。

### 2. 仅打兼容性补丁后手动编译

若已手动完成 ISL ppcg 补丁或不需要改 `install.sh`，可在应用本脚本后自行编译：

```bash
(cd third_party/AutoSA && ../patch_autosa/apply_llvm_compat.sh)
(cd third_party/AutoSA/src && ./autogen.sh && ./configure && CCACHE_DISABLE=1 make -j4)
```

### 3. 重新拉取 AutoSA 后从头打补丁

```bash
cd third_party/AutoSA
git pull
git submodule update --init --recursive   # 或在 mlir-systolic 根目录递归拉取子模块

# 应用本脚本（内含 isl_patch.sh + 兼容补丁 + install.sh 修改）
../patch_autosa/apply_llvm_compat.sh

# 再执行 install.sh 或编译
./install.sh
# 或 (cd src && ./autogen.sh && ./configure && make -j4)
```

### 4. 若遇 ccache 权限错误

编译时若出现 `ccache: error: failed to create temporary file ... Permission denied`，可禁用 ccache 再编译：

```bash
CCACHE_DISABLE=1 make -j4
```

## 涉及的 LLVM/Clang 变更摘要

- **LLVM 15+**：`getDefaultTargetTriple` 从 `llvm/Support/Host.h` 移至 `llvm/TargetParser/Host.h`。
- **LLVM 18+**：`CompilerInvocation::setLangDefaults` 改为 `LangOptions::setLangDefaults`；`createFileID` 接受 `FileEntryRef`；`setInvocation` 接受 `shared_ptr`；部分头文件移至 `Basic/`（如 DiagnosticOptions）。
- **Clang 9+**：`getLocStart`/`getLocEnd` 改为 `getBeginLoc`/`getEndLoc`。
- **C++/STL**：用于 `std::set` 的 comparator 的 `operator()` 需为 `const`。
- **Clang API**：`VariableArrayType::Static` 改为 `ArraySizeModifier::Static`；`ext_implicit_function_decl` 改为 `warn_implicit_function_decl`；`ArgStringList` 使用 `llvm::opt::ArgStringList` 等。

## 生成/更新补丁（维护用）

在修改了 pet、isl 或顶层 src 的兼容代码后，可在 AutoSA 仓库内重新生成补丁，再复制到本目录：

```bash
# 在 third_party/AutoSA 下
cd src/pet && git diff --no-prefix -- . ':(exclude)isl' > ../../../patch_autosa/pet-llvm18-ubuntu22.patch
cd ../isl && git diff --no-prefix -- interface/extract_interface.cc > ../../../patch_autosa/isl-interface-llvm18.patch
cd ../.. && git diff --no-prefix -- src/configure.ac src/Makefile.am src/autosa_common.cpp > ../patch_autosa/src-llvm18.patch
```

应用前请确保子模块与工作区处于预期状态，再执行 `apply_llvm_compat.sh` 做一次验证。
