# mlir-systolic 项目状态与上手指南

> **用途**：在新环境（例如服务器）或新 Agent 接手时，快速了解当前做了什么、如何验证、下一步该做什么。  
> **最后更新**：2026-03-04

---

## 一、项目在做什么

- **目标**：在 MLIR 上复现并改进 AutoSA 的脉动阵列生成流程，将 Affine 循环（来自 Polygeist 或手写）变为 **FPGA HLS C++**（Vivado/Vitis 兼容），**不依赖 host-serialize**，在各级 IO 完成数据顺序化与复用。
- **核心依赖**：**Polymer**（多面体分析，ISL） + **MLIR**（变换与代码生成）。LLVM/MLIR 来自 **Polygeist 子项目** 的构建，**不使用系统自带的 LLVM**。
- **当前能力**：
  - **Pass 链**：`systolic-write-reorder-analysis`（可选）→ `systolic-transform` → `systolic-dataflow-generation` → 生成 SystolicDataflow IR；再由 **systolic-translate** 生成 HLS C++。
  - **支持的 kernel**：MM（矩阵乘）、MTTKRP（4 循环）、写时重排 2D/3D 测例；模板支持最多 3 输入 + 1 输出。
  - **已做优化**：写时重排（2D/3D）、读时重排（2D）、L3 coalesced 读（tile 顺序 + word_idx）、FIFO 深度可配置、RESOURCE 系统化（FIFO_SRL/RAM_2P_BRAM）、Pipeline 内 %/ 强度削减（2 的幂时用位运算）、L2 声明/定义维度一致。

### 当前阶段结论（2026-03-04）

- **目标收敛**：从“先比性能”切换为“先语义正确（csim）”。
- **标准 MTTKRP**：已建立标准语义用例并复现 csim mismatch，详见 `hls_validation/mttkrp_std_mlirsystolic/CSIM_FINDINGS_2026-03-04.md`。
- **标准 TTMc**：当前仍受 rank-3 输出模板能力限制；translate 中对非 2D 输出已做保护性拦截。
- **设计路线**：已形成对比与通用化方案，详见 [docs/design/CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md](docs/design/CODEGEN_COMPARISON_AND_GENERALIZATION_PLAN.md)。

---

## 二、如何验证（本地 / 服务器）

### 开发与测试策略

- **本地**：以「构建 + 全量 e2e + 分析生成文件内容」为主，不做 HLS 综合/csim。通过查看生成的 `.cpp`（结构、循环、word_idx、r1、注释等）做初步排错。可用 `./test/inspect_generated.sh` 做一次生成并输出关键模式检查结果。
- **服务器**：阶段性工作告一段落后，再在具备 Xilinx HLS/XRT 的环境统一做 csim、综合与具体调错。生成好的 `.cpp` 可拷贝到服务器使用，不必在服务器上完整编译本仓库。

### 2.1 构建

```bash
# 1. 子模块
git submodule update --init --recursive

# 2. 先构建 Polygeist（内含 LLVM/MLIR，耗时较长）
./scripts/build-polygeist.sh

# 3. 再构建 mlir-systolic
./scripts/build-systolic.sh
```

**重要**：mlir-systolic **不**使用系统已安装的“老版本 LLVM”（例如你用来跑 AutoSA 的那套）。它使用 **third_party/Polygeist** 里自带的 llvm-project 构建出的 MLIR/LLVM。更多说明见 [docs/BUILD_AND_SERVER_ENVIRONMENT.md](docs/BUILD_AND_SERVER_ENVIRONMENT.md)。因此：

- 服务器上是 **Ubuntu 18.04 + 老 LLVM** 也不影响本仓库的编译，只要能在该环境下成功构建 Polygeist 即可。
- 需要满足：**CMake ≥ 3.20**、**C++17**（GCC 7+）、**Ninja**；若系统 CMake 过旧，需自行安装 3.20+（如 Kitware 或 snap）。

### 2.2 全量端到端测试

```bash
./test/run_all_e2e.sh
```

- 会跑：MM、MTTKRP、写时重排(2D)、写时重排(3D) 四个 e2e。
- 成功时输出「通过: 4 … 结果: 全部通过」。
- 生成文件在 `/tmp/`：`mm_e2e_out.cpp`、`mttkrp_e2e_out.cpp`、`reorder_e2e_out.cpp`、`reorder_3d_e2e_out.cpp`。

### 2.3 单独测 MM 或写时重排

```bash
./test/run_mm_e2e.sh
./test/run_reorder_e2e.sh
./test/run_reorder_3d_e2e.sh
```

### 2.4 写时重排测例的 Pass 顺序

写时重排测例需要**先**跑写重排分析，再跑 transform 与 dataflow，例如：

```bash
systolic-opt test/minimal_reorder_write.mlir \
  --systolic-write-reorder-analysis \
  --systolic-transform --systolic-dataflow-generation -o /tmp/out.mlir
systolic-translate /tmp/out.mlir --size=32 -o /tmp/out.cpp
# 检查 /tmp/out.cpp 中是否出现 buffer_linear
```

---

## 三、下一步工作（不考虑 host-serialize）

以下为当前规划的后续工作（**不做** host-serialize）：

1. **与 autosa_hls_refs 逐项对比**  
   同一 kernel（如 MM）、相近参数下，对比我们与 AutoSA 的：模块划分、FIFO、PIPELINE/RESOURCE、drain 与 L3 访问；补对 II/频率/面积影响大的差异。

2. **服务器上 C sim + 综合**  
   在具备 Xilinx HLS/XRT 的环境做 C sim（正确性）、综合（II、频率、资源），并与 AutoSA 同参数对比。

3. **可选**：L3/drain 的 BURST、BIND_STORAGE 等 pragma；更多 kernel（如 CNN、TTMc）与测例；参数与多面体选择范围（当前小规模可暂缓）。

更细的条目见 **[RECENT_CHANGES_AND_NEXT_STEPS.md](RECENT_CHANGES_AND_NEXT_STEPS.md)** 中「四（续）、不考虑 host-serialize 时的后续工作」。

---

## 四、关键文档索引

| 文档 | 说明 |
|------|------|
| **[RECENT_CHANGES_AND_NEXT_STEPS.md](RECENT_CHANGES_AND_NEXT_STEPS.md)** | 近期修改、e2e 状态、下一步与文档链接（首选入口） |
| **[README.md](README.md)** | 项目概述、构建、目录结构 |
| **[docs/README.md](docs/README.md)** | 文档导航与结构 |
| **[docs/DOCS_INDEX.md](docs/DOCS_INDEX.md)** | **全量 Markdown 文档索引**（所有 .md 列表与说明） |
| **[docs/design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md](docs/design/L3_COALESCE_VS_WRITE_REORDER_AND_HOST_SERIALIZE.md)** | L3 coalesce 与写时重排关系、host-serialize 暂不做的说明 |
| **[docs/design/FIFO_DEPTH_AND_PERFORMANCE_NEXT.md](docs/design/FIFO_DEPTH_AND_PERFORMANCE_NEXT.md)** | FIFO 深度策略与性能后续 |
| **test/autosa_hls_refs/** | AutoSA 生成的 HLS 参考（用于对比） |

---

## 五、服务器环境（Ubuntu 18.04 / 老 LLVM）说明

- **AutoSA** 使用你本机/服务器上已有的旧版 LLVM 编译，与 mlir-systolic **无关**。
- **mlir-systolic** 的 LLVM/MLIR 来自 **Polygeist 的源码构建**（`third_party/Polygeist/llvm-project`），不链接系统 LLVM。
- 在 **Ubuntu 18.04** 上建议：
  - 安装 **CMake 3.20+**（若 `cmake --version` 不足则用 Kitware 或 snap 安装）。
  - 使用 **GCC 7+**（18.04 默认通常满足 C++17）。
  - 先 `./scripts/build-polygeist.sh`，再 `./scripts/build-systolic.sh`；若 Polygeist 构建失败，再根据报错排查（如 GMP、Ninja、内存等）。
- 若仅需在服务器上跑 **HLS 综合与 C sim**，可在本机完成 `systolic-opt` + `systolic-translate`，把生成的 `.cpp` 拷到服务器用 Vivado/Vitis 综合即可；不强制在服务器上完整编译 mlir-systolic。
