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
| `install-sh-mlir-systolic.patch` | 修改 **install.sh**：去掉子模块 init/update 与「Patch ISL」步骤，适配先打补丁再 install 的流程 |
| `requirements-txt-scikit-learn.patch` | 修改 **requirements.txt**：`sklearn>=0.0` 改为 `scikit-learn>=0.0`（PyPI 包名已变更） |

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
