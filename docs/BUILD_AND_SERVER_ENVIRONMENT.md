# 构建与服务器环境说明

> 简要说明 mlir-systolic 的构建依赖，以及在 **Ubuntu 18.04 / 老版本 LLVM** 环境下的注意事项。  
> 详细上手指南见根目录 [../PROJECT_STATUS_AND_ONBOARDING.md](../PROJECT_STATUS_AND_ONBOARDING.md)。

---

## 构建依赖（与系统 LLVM 无关）

- **LLVM/MLIR 来源**：来自 **Polygeist 子项目** 的 `third_party/Polygeist/llvm-project` 构建结果，**不**使用系统已安装的 LLVM（例如你用来编译 AutoSA 的那套）。
- **构建顺序**：先 `./scripts/build-polygeist.sh`（会构建 LLVM/MLIR/Polly/Polygeist/Polymer），再 `./scripts/build-systolic.sh`。
- **所需工具**：
  - **CMake ≥ 3.20**（见根目录 `CMakeLists.txt`）
  - **C++17**（GCC 7+ 或 Clang 5+）
  - **Ninja**
  - 可选：GMP（Polymer/ISL 依赖，多数系统已装）

---

## Ubuntu 18.04 与“老 LLVM”说明

- **AutoSA** 使用你本机/服务器上已有的旧版 LLVM 编译，与 mlir-systolic **无关**。
- **mlir-systolic** 的 LLVM/MLIR 完全由 Polygeist 的源码在 `third_party/Polygeist` 内构建，不会链接系统 LLVM。
- **Ubuntu 18.04**：
  - 默认 CMake 可能低于 3.20，需自行安装：如 [Kitware APT 仓库](https://apt.kitware.com/) 或 `snap install cmake --classic`。
  - 默认 GCC 通常为 7.x，支持 C++17，可满足构建。
- 若 Polygeist 构建失败，请根据报错排查（常见：GMP、Ninja、内存不足时减少 `-j`）。

---

## 仅在服务器上跑 HLS 综合时

若只需在服务器上做 **Vivado/Vitis 综合与 C sim**，可在本机完成：

```bash
systolic-opt ... -o out.mlir
systolic-translate out.mlir -o kernel.cpp
```

再将生成的 `kernel.cpp` 拷到服务器即可，**不强制**在服务器上完整编译 mlir-systolic 和 Polygeist。
