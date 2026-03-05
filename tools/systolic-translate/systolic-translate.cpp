//===----------------------------------------------------------------------===//
//
// systolic-translate - Systolic Array Translation Tool
//
// This tool translates MLIR to HLS C++ code for systolic arrays.
//
//===----------------------------------------------------------------------===//

#include "systolic/Dialect/HLS/HLS.h"
#include "systolic/Dialect/SystolicDataflow/SystolicDataflow.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <optional>
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

#define DEBUG_TYPE "systolic-translate"

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<unsigned> arrayPartSize(
    "array-part", llvm::cl::desc("Array partition size (default: 8)"),
    llvm::cl::init(8));

static llvm::cl::opt<unsigned> latencySize(
    "latency", llvm::cl::desc("Latency hiding factor (default: 4)"),
    llvm::cl::init(4));

static llvm::cl::opt<unsigned> simdFactor(
    "simd", llvm::cl::desc("SIMD factor (default: 1)"),
    llvm::cl::init(1));

static llvm::cl::opt<unsigned> problemSize(
    "size", llvm::cl::desc("Problem size N for NxN matrix (default: 32)"),
    llvm::cl::init(32));

static llvm::cl::opt<unsigned> fifoDepth(
    "fifo-depth", llvm::cl::desc("FIFO depth for hls::stream (default: 2)"),
    llvm::cl::init(2));

//===----------------------------------------------------------------------===//
// HLS C++ Emitter
//===----------------------------------------------------------------------===//

namespace {

/// 表示数组的布局重排信息
struct ArrayReorderingInfo {
  std::string arrayName;
  
  // 原始维度和重排后维度
  SmallVector<int64_t, 3> originalDims;
  SmallVector<int64_t, 3> reorderedDims;
  
  // 维度置换：new_dim[i] = originalDims[permutation[i]]
  SmallVector<unsigned, 3> dimPermutation;
  
  // 是否需要重排
  bool needsReordering() const {
    return originalDims != reorderedDims;
  }
  
  // 应用置换到索引
  // 输入索引数组：[idx0, idx1, idx2]
  // 输出索引数组：[idx_perm[0], idx_perm[1], idx_perm[2]]
  SmallVector<std::string, 3> applyPermutation(
      const SmallVector<std::string, 3> &indices) const {
    SmallVector<std::string, 3> result(3);
    for (unsigned i = 0; i < 3 && i < dimPermutation.size(); i++) {
      unsigned origIdx = dimPermutation[i];
      if (origIdx < indices.size()) {
        result[i] = indices[origIdx];
      } else {
        result[i] = "0";  // 默认值
      }
    }
    return result;
  }
};

/// 轻量级的 contraction 描述，用于逐步把硬编码模板
/// 重构为“由描述驱动”的生成逻辑。
struct ContractionDesc {
  /// 输出张量的秩（2 或 3）
  unsigned outputRank = 2;
  /// 规约维数量（如 MM=1，标准 MTTKRP=2），对应 time-loop 数量。
  unsigned numReductions = 1;

  enum class Kind {
    MatmulLike,   // rank-2, 1 reduction (MM)
    MttkrpLike,   // rank-2, 2 reductions (MTTKRP)
    TtmcLike,     // rank-3 output (TTMc 等)，支持 numReductions<=3 时生成 3D drain
    Unsupported,  // rank>3 或 rank-3 且 numReductions>3
  };
  Kind kind = Kind::MatmulLike;

  /// 是否需要在 PE/IO 中插入“第二规约维”循环（r1, 0..size-1）。
  bool hasExtraReductionLoop() const { return numReductions >= 2; }
  /// 是否需要在 PE/IO 中插入“第三规约维”循环（r2, 0..size-1），用于 TTMc 等 3 规约维。
  bool hasThirdReductionLoop() const { return numReductions >= 3; }
  /// 是否生成 3D 输出的 drain 写回路径。
  bool isRank3Output() const { return outputRank == 3; }
};

// DRAM word / pack 常量（与 AutoSA 等 512-bit 总线约定一致，便于扩展与对照）
static const unsigned kDramWordBytes = 64;   // 512 bits
static const unsigned kFloatsPerDramWord = 16;
static const unsigned kDramWordBits = 512;

// Helpers for pipeline-friendly strength reduction: use bitwise when divisor is power-of-2
static bool isPowerOf2(unsigned n) { return n && !(n & (n - 1)); }
static unsigned log2Po2(unsigned n) { return (unsigned)llvm::Log2_32(n); }
/// 循环变量所需位宽（bound 为 inclusive 上界），避免大 size 溢出，参考 AutoSA 风格按界计算
static unsigned requiredLoopBits(uint64_t bound) {
  if (bound == 0) return 1;
  uint64_t v = bound + 1;
  unsigned bits = 0;
  while (v) { bits++; v >>= 1; }
  return std::min(32u, std::max(1u, bits));
}

class SystolicHLSEmitter {
public:
  SystolicHLSEmitter(raw_ostream &os, unsigned arrayPart, unsigned latency, 
                     unsigned simd, unsigned size, unsigned fifoDepth = 2)
      : os(os), arrayPart(arrayPart), latency(latency), simd(simd), size(size),
        fifoDepth(fifoDepth),
        numPE(arrayPart / latency),
        tileSize(latency * numPE),  // Each tile is latency * numPE
        numTiles(size / tileSize)   // Number of tile iterations per dimension
        {}

  LogicalResult emit(ModuleOp module);

private:
  raw_ostream &os;
  unsigned arrayPart;
  unsigned latency;
  unsigned simd;
  unsigned size;
  unsigned fifoDepth;
  unsigned numPE;
  unsigned tileSize;  // = latency * numPE
  unsigned numTiles;  // = size / tileSize
  unsigned indentLevel = 0;

  /// 循环变量位宽（按 bound 计算，避免大 size 溢出，参考 AutoSA 等按界生成）
  unsigned bitsTiles = 3;   // c0,c1,c2: bound numTiles-1
  unsigned bitsPE = 2;      // c3,c4: bound numPE-1
  unsigned bitsSize = 4;    // r1,r2: bound size-1
  unsigned bitsLatency = 3; // c6,c7: bound latency-1
  unsigned bitsC5Bound = 3; // c5: bound c5Bound-1

  // 由上游 Pass 传入的 contraction 级别信息（目前仅使用 numReductions/outputRank）
  ContractionDesc contraction;

  // 写时重排信息：存储所有数组的重排信息
  std::unordered_map<std::string, ArrayReorderingInfo> arrayReordering;
  
  // 从 kernel 函数参数推导的数组名：2 或 3 个输入 + 1 个输出（如 MM: A,B,C；MTTKRP: A,B,D；3-input: A,B,C,D）
  SmallVector<std::string, 4> inputNames;
  std::string outputName;
  /// 每个输入的 memref 维度（与 inputNames 一一对应），用于 L3_serialize 双规约时 3D 按 r1 切片读
  SmallVector<SmallVector<int64_t, 3>, 4> inputShapes;
  /// 输出 memref 的形状，用于 rank-3 时 drain serialize 写回
  SmallVector<int64_t, 3> outputShape;
  void deriveArrayNamesFromFunction(func::FuncOp funcOp);
  
  // Indentation helpers
  raw_ostream &indent() { indentLevel++; return os; }
  raw_ostream &dedent() { if (indentLevel > 0) indentLevel--; return os; }
  raw_ostream &emitIndent() {
    for (unsigned i = 0; i < indentLevel; i++) os << "  ";
    return os;
  }
  
  // Write-time reordering methods
  void extractReorderingInfo(func::FuncOp funcOp);
  SmallVector<int64_t, 3> getArrayDims(StringRef arrayName) const;
  SmallVector<std::string, 3> applyAccessPermutation(
      StringRef arrayName,
      const SmallVector<std::string, 3> &originalIndices) const;
  /// True if array has 2D reordering (for drain serialize write-time reorder).
  bool hasReordering2D(StringRef arrayName) const;
  /// True if array has 3D reordering (for drain serialize write-time reorder).
  bool hasReordering3D(StringRef arrayName) const;
  /// Emit PE accumulator init condition (e.g. "c2 == 0 && c5 == 0" or "r1 == 0 && c2 == 0 && c5 == 0").
  void emitPEInitCondition();
  /// Emit PE drain write condition (last iteration of reduction loops).
  void emitPEDrainCondition(unsigned c5Bound);
  /// Linear index in original layout from reordered (d0,d1). sizeLiteral used in expr (0 => "size").
  std::string getLinearIndexFromReordered2D(StringRef arrayName,
                                           StringRef d0Var, StringRef d1Var,
                                           unsigned sizeLiteral) const;
  /// Original index exprs [orig0, orig1] from (d0,d1) for buffer[orig0][orig1].
  SmallVector<std::string, 2> getOriginalIndexExprs2D(StringRef arrayName,
                                                      StringRef d0Var, StringRef d1Var) const;
  /// Original index exprs [orig0, orig1, orig2] from (d0,d1,d2) for buffer[orig0][orig1][orig2].
  SmallVector<std::string, 3> getOriginalIndexExprs3D(StringRef arrayName,
                                                      StringRef d0Var, StringRef d1Var, StringRef d2Var) const;
  
  // Emission methods
  void emitFileHeader();
  void emitTypeDefinitions();
  void emitModuleDeclarations();
  void emitIOL3InSerialize(StringRef arrayName, StringRef typeName,
                           unsigned totalSize,
                           llvm::ArrayRef<int64_t> arrayShape = {});
  void emitIOL3In(StringRef arrayName, StringRef typeName);
  void emitIOL2InIntraTrans(StringRef arrayName);
  void emitIOL2InInterTrans(StringRef arrayName);
  void emitIOL2InInterTransBoundary(StringRef arrayName);
  void emitIOL2In(StringRef arrayName);
  void emitIOL2InBoundary(StringRef arrayName);
  void emitPE();
  void emitPEWrapper();
  void emitDummyModules();
  void emitDrainIOL1(StringRef arrayName);
  void emitDrainIOL2(StringRef arrayName);
  void emitDrainIOL3(StringRef arrayName);
  void emitDrainSerialize(StringRef arrayName, unsigned totalSize,
                          llvm::ArrayRef<int64_t> outShape = {});
  void emitTopKernel(func::FuncOp funcOp);
  
  LogicalResult emitFunc(func::FuncOp funcOp);
};

} // namespace

void SystolicHLSEmitter::emitFileHeader() {
  os << "//===----------------------------------------------------------------------===//\n";
  os << "// Generated by mlir-systolic (systolic-translate)\n";
  os << "// Configuration: array_part=" << arrayPart << ", latency=" << latency 
     << ", simd=" << simd << ", fifo_depth=" << fifoDepth << "\n";
  os << "// PE Array: " << numPE << " x " << numPE << "\n";
  os << "//===----------------------------------------------------------------------===//\n\n";
  
  os << "#include <ap_int.h>\n";
  os << "#include <hls_stream.h>\n\n";
  
  os << "#define min(x,y) ((x < y) ? x : y)\n";
  os << "#define max(x,y) ((x > y) ? x : y)\n\n";
}

void SystolicHLSEmitter::deriveArrayNamesFromFunction(func::FuncOp funcOp) {
  inputNames.clear();
  outputName = "C";
  SmallVector<std::string, 4> names;
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    if (!funcOp.getArgument(i).getType().isa<MemRefType>()) continue;
    if (auto attr = funcOp.getArgAttrOfType<StringAttr>(i, "mlir.name"))
      names.push_back(attr.getValue().str());
    else
      names.push_back("arg" + std::to_string(i));
  }
  if (names.size() >= 4) {
    inputNames.push_back(names[0]);
    inputNames.push_back(names[1]);
    inputNames.push_back(names[2]);
    outputName = names[3];
  } else if (names.size() >= 3) {
    inputNames.push_back(names[0]);
    inputNames.push_back(names[1]);
    outputName = names[2];
  } else {
    inputNames.push_back("A");
    inputNames.push_back("B");
  }
}

void SystolicHLSEmitter::emitTypeDefinitions() {
  os << "/* Data Type */\n";
  for (const auto &n : inputNames) {
    os << "typedef float " << n << "_t1;\n";
    os << "typedef ap_uint<" << kDramWordBits << "> " << n << "_t16;\n";
    os << "typedef ap_uint<" << (arrayPart * 32) << "> " << n << "_t" << arrayPart << ";\n";
  }
  os << "typedef float " << outputName << "_t1;\n";
  os << "typedef ap_uint<" << kDramWordBits << "> " << outputName << "_t16;\n";
  os << "typedef ap_uint<" << (latency * 32) << "> " << outputName << "_t" << latency << ";\n";
  os << "/* Data Type */\n\n";
}

void SystolicHLSEmitter::emitModuleDeclarations() {
  os << "/* Module Declarations */\n";
  for (const auto &name : inputNames) {
    auto dims = getArrayDims(name);
    int64_t d0 = dims.size() > 0 ? dims[0] : (int64_t)latency;
    int64_t d1 = dims.size() > 1 ? dims[1] : 1;
    int64_t d2 = dims.size() > 2 ? dims[2] : (int64_t)arrayPart;
    os << "void " << name << "_IO_L3_in(hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_in, "
       << "hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_local_out);\n";
    os << "void " << name << "_IO_L3_in_serialize(" << name << "_t16 *" << name << ", hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_local_out);\n";
    os << "void " << name << "_IO_L2_in_intra_trans(int idx, int c0, int c1, int c2, " << name << "_t" << arrayPart
       << " local_" << name << "[" << d0 << "][" << d1 << "][" << d2 << "], hls::stream<float> &fifo_" << name << "_local_out, bool intra_trans_en);\n";
    os << "void " << name << "_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, " << name << "_t" << arrayPart
       << " local_" << name << "[" << d0 << "][" << d1 << "][" << d2 << "], hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_in, hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_out, bool inter_trans_en);\n";
    os << "void " << name << "_IO_L2_in_inter_trans_boundary(int idx, int c0, int c1, int c2, " << name << "_t" << arrayPart
       << " local_" << name << "[" << d0 << "][" << d1 << "][" << d2 << "], hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_in, bool inter_trans_en);\n";
    os << "void " << name << "_IO_L2_in(int idx, hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_in, "
       << "hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_out, "
       << "hls::stream<float> &fifo_" << name << "_local_out);\n";
    os << "void " << name << "_IO_L2_in_boundary(int idx, hls::stream<" << name << "_t" << arrayPart << "> &fifo_" << name << "_in, hls::stream<float> &fifo_" << name << "_local_out);\n";
  }
  os << "void PE_wrapper(int idx, int idy, ";
  for (size_t i = 0; i < inputNames.size(); i++)
    os << (i ? ", " : "") << "hls::stream<float> &fifo_" << inputNames[i] << "_in, hls::stream<float> &fifo_" << inputNames[i] << "_out";
  os << ", hls::stream<float> &fifo_" << outputName << "_drain_out);\n";
  os << "void " << outputName << "_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, " << outputName << "_t" << latency
     << " local_" << outputName << "[" << latency << "][1], hls::stream<float> &fifo_" << outputName << "_drain_local_in);\n";
  os << "void " << outputName << "_drain_IO_L1_out_inter_trans(int idx, int idy, int c0, int c1, " << outputName << "_t" << latency
     << " local_" << outputName << "[" << latency << "][1], hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_in, hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_out);\n";
  os << "void " << outputName << "_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int c0, int c1, " << outputName << "_t" << latency
     << " local_" << outputName << "[" << latency << "][1], hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_out);\n";
  os << "void " << outputName << "_drain_IO_L1_out_wrapper(int idx, int idy, "
     << "hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_in, "
     << "hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_out, "
     << "hls::stream<float> &fifo_" << outputName << "_drain_local_in);\n";
  os << "void " << outputName << "_drain_IO_L1_out_boundary_wrapper(int idx, int idy, "
     << "hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_out, "
     << "hls::stream<float> &fifo_" << outputName << "_drain_local_in);\n";
  os << "void " << outputName << "_drain_IO_L2_out(int idx, hls::stream<" << outputName << "_t" << latency
     << "> &fifo_" << outputName << "_drain_in, hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_out, hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_local_in);\n";
  os << "void " << outputName << "_drain_IO_L2_out_boundary(int idx, hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_out, hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_local_in);\n";
  os << "void " << outputName << "_drain_IO_L3_out(hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_out, hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_local_in);\n";
  os << "void " << outputName << "_drain_IO_L3_out_serialize(" << outputName << "_t16 *" << outputName << ", hls::stream<" << outputName << "_t" << latency << "> &fifo_" << outputName << "_drain_local_in);\n";
  os << "/* Module Declarations */\n\n";
}

void SystolicHLSEmitter::emitIOL3InSerialize(StringRef arrayName,
                                              StringRef typeName,
                                              unsigned totalSize,
                                              llvm::ArrayRef<int64_t> arrayShape) {
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_IO_L3_in_serialize(" << arrayName 
     << "_t16 *" << arrayName << ", hls::stream<" << arrayName << "_t" 
     << arrayPart << "> &fifo_" << arrayName << "_local_out) {\n";
  os << "#pragma HLS INLINE OFF\n";

  if (hasReordering2D(arrayName)) {
    // Read-time reordering: iterate (d0,d1) in reordered layout order; DRAM read is still
    // sequential (word_idx = reordered_linear / kFloatsPerDramWord). Assumes input is in reordered layout.
    unsigned loopBits = (unsigned)ceil(log2(totalSize + 1));
    unsigned wordShift = log2Po2(kFloatsPerDramWord);
    unsigned wordMask = kFloatsPerDramWord - 1;
    os << "  /* Read in reordered dimension order (input assumed reordered layout) */\n";
    os << "  " << arrayName << "_t16 current_word;\n";
    os << "  union { float f; uint32_t u; } tmp;\n";
    os << "  float buf[" << arrayPart << "];\n";
    os << "  #pragma HLS ARRAY_PARTITION variable=buf complete\n";
    os << "  unsigned reordered_linear, word_idx, offset;\n";
    os << "  int count = 0;\n";
    os << "  for (ap_uint<" << loopBits << "> d0 = 0; d0 < " << totalSize << "; d0++)\n";
    os << "    for (ap_uint<" << loopBits << "> d1 = 0; d1 < " << totalSize << "; d1++) {\n";
    os << "      reordered_linear = d0 * " << totalSize << " + d1;\n";
    os << "      word_idx = reordered_linear >> " << wordShift << ";\n";
    os << "      offset = reordered_linear & " << wordMask << ";\n";
    os << "      if (offset == 0) current_word = " << arrayName << "[word_idx];\n";
    os << "      tmp.u = current_word.range(32*(int)(offset+1)-1, 32*(int)offset).to_uint();\n";
    os << "      buf[count++] = tmp.f;\n";
    os << "      if (count == " << arrayPart << ") {\n";
    os << "        " << arrayName << "_t" << arrayPart << " fifo_data;\n";
    os << "        for (int i = 0; i < " << arrayPart << "; i++) {\n";
    os << "          tmp.f = buf[i];\n";
    os << "          fifo_data.range(32*(i+1)-1, 32*i) = tmp.u;\n";
    os << "        }\n";
    os << "        fifo_" << arrayName << "_local_out.write(fifo_data);\n";
    os << "        count = 0;\n";
    os << "      }\n";
    os << "    }\n";
  } else {
    // Coalesced L3 read: same (c0,c1,c3) tile order as L3_in, explicit word_idx for sequential DRAM burst
    unsigned wordsPerDram = kFloatsPerDramWord / arrayPart;
    bool doubleReduction = contraction.hasExtraReductionLoop();
    bool thirdReduction = contraction.hasThirdReductionLoop();
    bool use3DSlice = (doubleReduction || thirdReduction) && arrayShape.size() == 3;
    unsigned totalDramWords;
    unsigned wordsPerPlane = 0;
    if (use3DSlice) {
      wordsPerPlane = (unsigned)((arrayShape[0] * arrayShape[1] * 4) / kDramWordBytes);
      if (wordsPerPlane == 0) wordsPerPlane = 1;
      // 3D array total words = wordsPerPlane * dim2; for triple reduction we repeat read (r2,r1) so same total
      totalDramWords = wordsPerPlane * (thirdReduction ? (unsigned)arrayShape[2] : size);
      if (totalDramWords == 0) totalDramWords = wordsPerPlane * size;
    } else {
      totalDramWords = (totalSize * totalSize * 4) / kDramWordBytes;
    }
    unsigned c4GroupBound = totalDramWords / (numTiles * numTiles * numPE);
    if (c4GroupBound == 0) c4GroupBound = 1;
    unsigned strideC0 = numTiles * numPE * c4GroupBound;
    unsigned strideC1 = numPE * c4GroupBound;
    bool triple3D = thirdReduction && use3DSlice;  // repeat same 3D data for each (r2,r1)
    bool matmulLike = !doubleReduction && !triple3D && !use3DSlice;  // MM: repeat same DRAM data per c2 (reduction tile)
    os << "  /* Coalesced L3 read: tile order (c0,c1,c3,c4_group), word_idx sequential for burst";
    if (use3DSlice)
      os << "; 3D array, r1" << (triple3D ? ",r2 repeat" : " = plane index");
    if (matmulLike)
      os << "; MatmulLike: repeat per c2 (reduction tile)";
    os << " */\n";
    os << "  " << arrayName << "_t" << arrayPart << " fifo_data;\n";
    os << "  " << arrayName << "_t16 mem_data;\n";
    os << "  unsigned word_idx;\n";
    if (matmulLike) {
      // L2 loads once per (c0,c1,c2): output same 128 words numTiles times so each c2 gets a full batch
      os << "  for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1)\n";
    }
    os << "  for (ap_uint<" << bitsTiles << "> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
    os << "    for (ap_uint<" << bitsTiles << "> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1)\n";
    if (triple3D)
      os << "      for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
    if (doubleReduction)
      os << "      for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
    os << "      for (ap_uint<" << bitsPE << "> c3 = 0; c3 <= " << (numPE - 1) << "; c3 += 1)\n";
    os << "        for (ap_uint<" << (unsigned)ceil(log2(c4GroupBound + 1)) << "> c4g = 0; c4g <= " << (c4GroupBound - 1) << "; c4g += 1) {\n";
    if (use3DSlice && !triple3D)
      os << "          word_idx = r1 * " << wordsPerPlane << "U + c0 * " << strideC0 << "U + c1 * " << strideC1 << "U + c3 * " << c4GroupBound << "U + c4g;\n";
    else if (triple3D)
      os << "          word_idx = c0 * " << strideC0 << "U + c1 * " << strideC1 << "U + c3 * " << c4GroupBound << "U + c4g;\n";
    else
      os << "          word_idx = c0 * " << strideC0 << "U + c1 * " << strideC1 << "U + c3 * " << c4GroupBound << "U + c4g;\n";
    os << "          mem_data = " << arrayName << "[word_idx];\n";
    os << "          for (ap_uint<2> slot = 0; slot <= " << (wordsPerDram - 1) << "; slot += 1) {\n";
    os << "          #pragma HLS PIPELINE II=1\n";
    os << "            fifo_data = mem_data(" << (arrayPart * 32 - 1) << ", 0);\n";
    os << "            mem_data = mem_data >> " << (arrayPart * 32) << ";\n";
    os << "            fifo_" << arrayName << "_local_out.write(fifo_data);\n";
    os << "          }\n";
    os << "        }\n";
    if (doubleReduction)
      os << "      }\n";
    if (triple3D)
      os << "      }\n";
  }
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitIOL3In(StringRef arrayName, StringRef typeName) {
  bool doubleReduction = contraction.hasExtraReductionLoop();
  bool thirdReduction = contraction.hasThirdReductionLoop();
  bool matmulLike = !doubleReduction && !thirdReduction;
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_IO_L3_in(hls::stream<" << arrayName << "_t" 
     << arrayPart << "> &fifo_" << arrayName << "_in, hls::stream<" 
     << arrayName << "_t" << arrayPart << "> &fifo_" << arrayName 
     << "_local_out) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  for (ap_uint<" << bitsTiles << "> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "    for (ap_uint<" << bitsTiles << "> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  if (matmulLike)
    os << "      for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1) {\n";
  if (thirdReduction)
    os << "      for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
  if (doubleReduction)
    os << "      for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
  os << "      // io_L3\n";
  os << "      for (ap_uint<" << bitsPE << "> c3 = 0; c3 <= " << (numPE - 1) << "; c3 += 1) {\n";
  os << "        // io_L2\n";
  os << "        for (ap_uint<" << bitsLatency << "> c4 = 0; c4 <= " << (latency - 1) << "; c4 += 1) {\n";
  os << "        #pragma HLS PIPELINE II=1\n";
  os << "          // access_coalesce\n";
  os << "          // access_serialize\n";
  os << "          {\n";
  os << "            " << arrayName << "_t" << arrayPart << " in_data;\n";
  os << "            " << arrayName << "_t" << arrayPart << " out_data;\n";
  os << "            in_data = fifo_" << arrayName << "_in.read();\n";
  os << "            out_data = in_data;\n";
  os << "            fifo_" << arrayName << "_local_out.write(out_data);\n";
  os << "          }\n";
  os << "        }\n";
  os << "      }\n";
  if (doubleReduction)
    os << "      }\n";
  if (thirdReduction)
    os << "      }\n";
  if (matmulLike)
    os << "      }\n";
  os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitIOL2InIntraTrans(StringRef arrayName) {
  unsigned c5Bound = arrayPart / simd;
  
  // 获取声明维度（考虑重排）
  auto dims = getArrayDims(arrayName);
  
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_IO_L2_in_intra_trans(int idx, int c0, int c1, int c2, "
     << arrayName << "_t" << arrayPart << " local_" << arrayName 
     << "[" << dims[0] << "][" << dims[1] << "][" << dims[2] << "], "
     << "hls::stream<float> &fifo_" << arrayName << "_local_out, bool intra_trans_en) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx; // module id\n";
  os << "  ap_uint<32> data_split[" << arrayPart << "];\n";
  os << "  #pragma HLS ARRAY_PARTITION variable=data_split complete\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  if (!intra_trans_en) return;\n\n";
  os << "  // io_L2\n";
  os << "  // io_L1\n";
  os << "  // pe\n";
  os << "  for (ap_uint<4> c5 = 0; c5 <= " << (c5Bound - 1) << "; c5 += 1) {\n";
  os << "    // latency\n";
  os << "    for (ap_uint<3> c6 = 0; c6 <= " << (latency - 1) << "; c6 += 1) {\n";
  os << "      // latency\n";
  os << "      for (ap_uint<3> c7 = 0; c7 <= " << (latency - 1) << "; c7 += 1) {\n";
  os << "      #pragma HLS PIPELINE II=1\n";
  os << "        {\n";
  os << "          " << arrayName << "_t" << arrayPart << " in_data;\n";
  os << "          " << arrayName << "_t1 out_data;\n";
  
  // 应用维度置换到数组访问
  SmallVector<std::string, 3> originalIdx = {"c7", "0", "0"};
  SmallVector<std::string, 3> permutedIdx = applyAccessPermutation(arrayName, originalIdx);
  os << "          in_data = local_" << arrayName << "[" << permutedIdx[0] << "]["
     << permutedIdx[1] << "][" << permutedIdx[2] << "];\n";
  os << "          for (ap_uint<4> n = 0; n < " << arrayPart << "; n++) {\n";
  os << "          #pragma HLS UNROLL\n";
  os << "            data_split[n] = in_data(31, 0);\n";
  os << "            in_data = in_data >> 32;\n";
  os << "          }\n";
  if (isPowerOf2(arrayPart))
    os << "          int split_idx = (c5) & " << (arrayPart - 1) << ";\n";
  else
    os << "          int split_idx = (c5) % " << arrayPart << ";\n";
  os << "          union {unsigned int ui; float ut;} u;\n";
  os << "          u.ui = (unsigned int)data_split[split_idx];\n";
  os << "          out_data = u.ut;\n";
  os << "          fifo_" << arrayName << "_local_out.write(out_data);\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitIOL2InInterTrans(StringRef arrayName) {
  // 获取声明维度（考虑重排）
  auto dims = getArrayDims(arrayName);
  
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, "
     << arrayName << "_t" << arrayPart << " local_" << arrayName 
     << "[" << dims[0] << "][" << dims[1] << "][" << dims[2] << "], "
     << "hls::stream<" << arrayName << "_t" << arrayPart << "> &fifo_" << arrayName << "_in, "
     << "hls::stream<" << arrayName << "_t" << arrayPart << "> &fifo_" << arrayName << "_out, "
     << "bool inter_trans_en) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx; // module id\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  if (!inter_trans_en) return;\n\n";
  os << "  for (ap_uint<2> c3 = p0; c3 <= " << (numPE - 1) << "; c3 += 1) {\n";
  os << "    // io_L2\n";
  os << "    if (c3 == p0) {\n";
  os << "      for (ap_uint<3> c4 = 0; c4 <= " << (latency - 1) << "; c4 += 1) {\n";
  os << "      #pragma HLS PIPELINE II=1\n";
  os << "        // access_coalesce\n";
  os << "        {\n";
        os << "          " << arrayName << "_t" << arrayPart << " in_data;\n";
        os << "          " << arrayName << "_t" << arrayPart << " out_data;\n";
        os << "          in_data = fifo_" << arrayName << "_in.read();\n";
        os << "          out_data = in_data;\n";
        
        // 应用维度置换到写入索引
        SmallVector<std::string, 3> writeIdx = {"c4", "0", "0"};
        SmallVector<std::string, 3> permutedWriteIdx = applyAccessPermutation(arrayName, writeIdx);
        os << "          local_" << arrayName << "[" << permutedWriteIdx[0] << "]["
           << permutedWriteIdx[1] << "][" << permutedWriteIdx[2] << "] = out_data;\n";
  os << "        }\n";
  os << "      }\n";
  os << "    } else {\n";
  os << "      for (ap_uint<3> c4 = 0; c4 <= " << (latency - 1) << "; c4 += 1) {\n";
  os << "      #pragma HLS PIPELINE II=1\n";
  os << "        // access_coalesce\n";
  os << "        {\n";
  os << "          " << arrayName << "_t" << arrayPart << " in_data;\n";
  os << "          " << arrayName << "_t" << arrayPart << " out_data;\n";
  os << "          in_data = fifo_" << arrayName << "_in.read();\n";
  os << "          out_data = in_data;\n";
  os << "          fifo_" << arrayName << "_out.write(out_data);\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitIOL2InInterTransBoundary(StringRef arrayName) {
  // 获取声明维度（考虑重排）
  auto dims = getArrayDims(arrayName);
  
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_IO_L2_in_inter_trans_boundary(int idx, int c0, int c1, int c2, "
     << arrayName << "_t" << arrayPart << " local_" << arrayName 
     << "[" << dims[0] << "][" << dims[1] << "][" << dims[2] << "], "
     << "hls::stream<" << arrayName << "_t" << arrayPart << "> &fifo_" << arrayName << "_in, "
     << "bool inter_trans_en) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx; // module id\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  if (!inter_trans_en) return;\n\n";
  os << "  for (ap_uint<2> c3 = p0; c3 <= " << (numPE - 1) << "; c3 += 1)\n";
  os << "    if (c3 == p0) {\n";
  os << "      // io_L2\n";
  os << "      for (ap_uint<3> c4 = 0; c4 <= " << (latency - 1) << "; c4 += 1) {\n";
      os << "      #pragma HLS PIPELINE II=1\n";
      os << "        // access_coalesce\n";
      os << "        {\n";
      os << "          " << arrayName << "_t" << arrayPart << " in_data;\n";
      os << "          " << arrayName << "_t" << arrayPart << " out_data;\n";
      os << "          in_data = fifo_" << arrayName << "_in.read();\n";
      os << "          out_data = in_data;\n";
      
      // 应用维度置换到写入索引
      SmallVector<std::string, 3> writeIdx = {"c4", "0", "0"};
      SmallVector<std::string, 3> permutedWriteIdx = applyAccessPermutation(arrayName, writeIdx);
      os << "          local_" << arrayName << "[" << permutedWriteIdx[0] << "]["
         << permutedWriteIdx[1] << "][" << permutedWriteIdx[2] << "] = out_data;\n";
      os << "        }\n";
      os << "      }\n";
    os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitIOL2In(StringRef arrayName) {
  auto dims = getArrayDims(arrayName);
  int64_t d0 = dims.size() > 0 ? dims[0] : (int64_t)latency;
  int64_t d1 = dims.size() > 1 ? dims[1] : 1;
  int64_t d2 = dims.size() > 2 ? dims[2] : (int64_t)arrayPart;
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_IO_L2_in(int idx, hls::stream<" << arrayName << "_t" << arrayPart 
     << "> &fifo_" << arrayName << "_in, hls::stream<" << arrayName << "_t" << arrayPart 
     << "> &fifo_" << arrayName << "_out, hls::stream<float> &fifo_" << arrayName << "_local_out) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx; // module id\n";
  os << "  " << arrayName << "_t" << arrayPart << " local_" << arrayName << "_ping[" << d0 << "][" << d1 << "][" << d2 << "];\n";
  os << "  #pragma HLS RESOURCE variable=local_" << arrayName << "_ping core=RAM_2P_BRAM\n";
  os << "  " << arrayName << "_t" << arrayPart << " local_" << arrayName << "_pong[" << d0 << "][" << d1 << "][" << d2 << "];\n";
  os << "  #pragma HLS RESOURCE variable=local_" << arrayName << "_pong core=RAM_2P_BRAM\n";
  os << "  bool arb = 0;\n";
  os << "  bool inter_trans_en = 1;\n";
  os << "  bool intra_trans_en = 1;\n";
  os << "  /* Variable Declaration */\n\n";
  bool doubleReduction = contraction.hasExtraReductionLoop();
  bool thirdReduction = contraction.hasThirdReductionLoop();
  os << "  {\n";
  os << "    for (ap_uint<" << bitsTiles << "> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "      for (ap_uint<" << bitsTiles << "> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  os << "        // array\n";
  os << "        // io_L3\n";
  os << "        if (arb == 0) {\n";
  os << "          for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_inter_trans(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_pong, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_in, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_out, \n";
  os << "              /* enable */ inter_trans_en\n";
  os << "            );\n";
  if (thirdReduction)
    os << "            for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
  if (doubleReduction)
    os << "            for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_intra_trans(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_pong, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_local_out, \n";
  os << "              /* enable */ intra_trans_en\n";
  os << "            );\n";
  if (doubleReduction)
    os << "            }\n";
  if (thirdReduction)
    os << "            }\n";
  os << "          }\n";
  os << "        } else {\n";
  os << "          for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_inter_trans(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_ping, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_in, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_out, \n";
  os << "              /* enable */ inter_trans_en\n";
  os << "            );\n";
  if (thirdReduction)
    os << "            for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
  if (doubleReduction)
    os << "            for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_intra_trans(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_ping, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_local_out, \n";
  os << "              /* enable */ intra_trans_en\n";
  os << "            );\n";
  if (doubleReduction)
    os << "            }\n";
  if (thirdReduction)
    os << "            }\n";
  os << "          }\n";
  os << "        }\n";
  os << "        arb = !arb;\n";
  os << "      }\n";
  os << "  }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitIOL2InBoundary(StringRef arrayName) {
  auto dims = getArrayDims(arrayName);
  int64_t d0 = dims.size() > 0 ? dims[0] : (int64_t)latency;
  int64_t d1 = dims.size() > 1 ? dims[1] : 1;
  int64_t d2 = dims.size() > 2 ? dims[2] : (int64_t)arrayPart;
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_IO_L2_in_boundary(int idx, hls::stream<" << arrayName << "_t" << arrayPart 
     << "> &fifo_" << arrayName << "_in, hls::stream<float> &fifo_" << arrayName << "_local_out) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx; // module id\n";
  os << "  " << arrayName << "_t" << arrayPart << " local_" << arrayName << "_ping[" << d0 << "][" << d1 << "][" << d2 << "];\n";
  os << "  #pragma HLS RESOURCE variable=local_" << arrayName << "_ping core=RAM_2P_BRAM\n";
  os << "  " << arrayName << "_t" << arrayPart << " local_" << arrayName << "_pong[" << d0 << "][" << d1 << "][" << d2 << "];\n";
  os << "  #pragma HLS RESOURCE variable=local_" << arrayName << "_pong core=RAM_2P_BRAM\n";
  os << "  bool arb = 0;\n";
  os << "  bool inter_trans_en = 1;\n";
  os << "  bool intra_trans_en = 1;\n";
  os << "  /* Variable Declaration */\n\n";
  bool doubleReduction = contraction.hasExtraReductionLoop();
  bool thirdReduction = contraction.hasThirdReductionLoop();
  os << "  {\n";
  os << "    for (ap_uint<" << bitsTiles << "> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "      for (ap_uint<" << bitsTiles << "> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  os << "        // array\n";
  os << "        // io_L3\n";
  os << "        if (arb == 0) {\n";
  os << "          for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_inter_trans_boundary(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_pong, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_in, \n";
  os << "              /* enable */ inter_trans_en\n";
  os << "            );\n";
  if (thirdReduction)
    os << "            for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
  if (doubleReduction)
    os << "            for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_intra_trans(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_pong, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_local_out, \n";
  os << "              /* enable */ intra_trans_en\n";
  os << "            );\n";
  if (doubleReduction)
    os << "            }\n";
  if (thirdReduction)
    os << "            }\n";
  os << "          }\n";
  os << "        } else {\n";
  os << "          for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_inter_trans_boundary(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_ping, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_in, \n";
  os << "              /* enable */ inter_trans_en\n";
  os << "            );\n";
  if (thirdReduction)
    os << "            for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
  if (doubleReduction)
    os << "            for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
  os << "            " << arrayName << "_IO_L2_in_intra_trans(\n";
  os << "              /* module id */ idx, \n";
  os << "              /* host iter */ c0, \n";
  os << "              /* host iter */ c1, \n";
  os << "              /* host iter */ c2, \n";
  os << "              /* array */ local_" << arrayName << "_ping, \n";
  os << "              /* fifo */ fifo_" << arrayName << "_local_out, \n";
  os << "              /* enable */ intra_trans_en\n";
  os << "            );\n";
  if (doubleReduction)
    os << "            }\n";
  if (thirdReduction)
    os << "            }\n";
  os << "          }\n";
  os << "        }\n";
  os << "        arb = !arb;\n";
  os << "      }\n";
  os << "  }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitPEInitCondition() {
  if (contraction.hasThirdReductionLoop())
    os << "r2 == 0 && r1 == 0 && c2 == 0 && c5 == 0";
  else if (contraction.hasExtraReductionLoop())
    os << "r1 == 0 && c2 == 0 && c5 == 0";
  else
    os << "c2 == 0 && c5 == 0";
}

void SystolicHLSEmitter::emitPEDrainCondition(unsigned c5Bound) {
  if (contraction.hasThirdReductionLoop())
    os << "r2 == " << (size - 1) << " && r1 == " << (size - 1) << " && c2 == " << (numTiles - 1) << " && c5 == " << (c5Bound - 1);
  else if (contraction.hasExtraReductionLoop())
    os << "r1 == " << (size - 1) << " && c2 == " << (numTiles - 1) << " && c5 == " << (c5Bound - 1);
  else
    os << "c2 == " << (numTiles - 1) << " && c5 == " << (c5Bound - 1);
}

void SystolicHLSEmitter::emitPE() {
  unsigned c5Bound = arrayPart / simd;
  const std::string &out = outputName;
  bool extraReduction = contraction.hasExtraReductionLoop();
  bool thirdReduction = contraction.hasThirdReductionLoop();
  os << "/* Module Definition */\n";
  os << "void PE(int idx, int idy, ";
  for (size_t i = 0; i < inputNames.size(); i++)
    os << (i ? " " : "") << "hls::stream<float> &fifo_" << inputNames[i] << "_in, hls::stream<float> &fifo_" << inputNames[i] << "_out,";
  os << " hls::stream<float> &fifo_" << out << "_drain_out) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  int p0 = idx, p1 = idy;\n";
  for (const auto &in : inputNames) {
    os << "  " << in << "_t1 local_" << in << "[1][1];\n";
    os << "  #pragma HLS ARRAY_PARTITION variable=local_" << in << " dim=0 complete\n";
  }
  os << "  " << out << "_t1 local_" << out << "[" << latency << "][" << latency << "];\n";
  os << "  #pragma HLS RESOURCE variable=local_" << out << " core=RAM_2P_BRAM\n";
  os << "  for (ap_uint<" << bitsTiles << "> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "    for (ap_uint<" << bitsTiles << "> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1)\n";
  os << "      for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1) {\n";
  if (thirdReduction)
    os << "        for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
  if (extraReduction)
    os << "        for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
  os << "        for (ap_uint<" << bitsC5Bound << "> c5 = 0; c5 <= " << (c5Bound - 1) << "; c5 += 1) {\n";
  os << "          for (ap_uint<" << bitsLatency << "> c6 = 0; c6 <= " << (latency - 1) << "; c6 += 1) {\n";
  os << "            for (ap_uint<" << bitsLatency << "> c7 = 0; c7 <= " << (latency - 1) << "; c7 += 1) {\n";
  os << "            #pragma HLS PIPELINE II=1\n";
  os << "              {\n";
  for (const auto &in : inputNames)
    os << "                local_" << in << "[0][0] = fifo_" << in << "_in.read();\n";
  os << "                if (";
  emitPEInitCondition();
  os << ")\n                  local_" << out << "[c7][c6] = 0;\n";
  os << "                local_" << out << "[c7][c6] = (local_" << out << "[c7][c6] + (";
  for (size_t i = 0; i < inputNames.size(); i++)
    os << (i ? " * " : "") << "local_" << inputNames[i] << "[0][0]";
  os << "));\n";
  os << "                if (";
  emitPEDrainCondition(c5Bound);
  os << ")\n                  fifo_" << out << "_drain_out.write(local_" << out << "[c7][c6]);\n";
  for (size_t i = inputNames.size(); i > 0; i--)
    os << "                fifo_" << inputNames[i - 1] << "_out.write(local_" << inputNames[i - 1] << "[0][0]);\n";
  os << "              }\n";
  os << "            }\n";
  os << "          }\n";
  os << "        }\n";
  if (extraReduction)
    os << "        }\n";
  if (thirdReduction)
    os << "        }\n";
  os << "      }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitPEWrapper() {
  const std::string &out = outputName;
  os << "/* Module Definition */\n";
  os << "void PE_wrapper(int idx, int idy, ";
  for (size_t i = 0; i < inputNames.size(); i++)
    os << (i ? ", " : "") << "hls::stream<float> &fifo_" << inputNames[i] << "_in, hls::stream<float> &fifo_" << inputNames[i] << "_out";
  os << ", hls::stream<float> &fifo_" << out << "_drain_out) {\n";
  os << "  PE(idx, idy, ";
  for (size_t i = 0; i < inputNames.size(); i++)
    os << (i ? ", " : "") << "fifo_" << inputNames[i] << "_in, fifo_" << inputNames[i] << "_out";
  os << ", fifo_" << out << "_drain_out);\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitDummyModules() {
  unsigned c5Bound = arrayPart / simd;
  bool doubleReduction = contraction.hasExtraReductionLoop();
  bool thirdReduction = contraction.hasThirdReductionLoop();
  for (const auto &in : inputNames) {
    os << "/* Module Definition */\n";
    os << "void " << in << "_PE_dummy_in(int idx, int idy, hls::stream<float> &fifo_" << in << "_in) {\n";
    os << "  int p0 = idx, p1 = idy;\n";
    os << "  for (ap_uint<" << bitsTiles << "> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
    os << "    for (ap_uint<" << bitsTiles << "> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1)\n";
    os << "      for (ap_uint<" << bitsTiles << "> c2 = 0; c2 <= " << (numTiles - 1) << "; c2 += 1) {\n";
    if (thirdReduction)
      os << "        for (ap_uint<" << bitsSize << "> r2 = 0; r2 <= " << (size - 1) << "; r2 += 1) {\n";
    if (doubleReduction)
      os << "        for (ap_uint<" << bitsSize << "> r1 = 0; r1 <= " << (size - 1) << "; r1 += 1) {\n";
    os << "        for (ap_uint<" << bitsC5Bound << "> c5 = 0; c5 <= " << (c5Bound - 1) << "; c5 += 1)\n";
    os << "          for (ap_uint<" << bitsLatency << "> c6 = 0; c6 <= " << (latency - 1) << "; c6 += 1)\n";
    os << "            for (ap_uint<" << bitsLatency << "> c7 = 0; c7 <= " << (latency - 1) << "; c7 += 1) {\n";
    os << "            #pragma HLS PIPELINE II=1\n";
    os << "              " << in << "_t1 fifo_data; fifo_data = fifo_" << in << "_in.read();\n";
    os << "            }\n";
    if (doubleReduction)
      os << "        }\n";
    if (thirdReduction)
      os << "        }\n";
    os << "      }\n";
    os << "}\n";
    os << "/* Module Definition */\n\n";
  }
}

void SystolicHLSEmitter::emitDrainIOL1(StringRef arrayName) {
  // C_drain_IO_L1_out_intra_trans
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, "
     << arrayName << "_t" << latency << " local_" << arrayName << "[" << latency << "][1], "
     << "hls::stream<float> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "#pragma HLS INLINE\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx, p1 = idy; // module id\n";
  os << "  ap_uint<32> data_split[" << latency << "];\n";
  os << "  #pragma HLS ARRAY_PARTITION variable=data_split complete\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  // io_L1\n";
  os << "  // pe\n";
  os << "  // latency\n";
  os << "  for (ap_uint<3> c6 = 0; c6 <= " << (latency - 1) << "; c6 += 1) {\n";
  os << "    // latency\n";
  os << "    for (ap_uint<3> c7 = 0; c7 <= " << (latency - 1) << "; c7 += 1) {\n";
  os << "    #pragma HLS PIPELINE II=1\n";
  os << "      {\n";
  os << "        " << arrayName << "_t1 in_data;\n";
  os << "        " << arrayName << "_t" << latency << " out_data;\n";
  os << "        in_data = fifo_" << arrayName << "_drain_local_in.read();\n";
  if (isPowerOf2(latency))
    os << "        int split_idx = (c6) & " << (latency - 1) << ";\n";
  else
    os << "        int split_idx = (c6) % " << latency << ";\n";
  if (isPowerOf2(latency))
    os << "        out_data = local_" << arrayName << "[c7][(c6) >> " << log2Po2(latency) << "];\n";
  else
    os << "        out_data = local_" << arrayName << "[c7][c6 / " << latency << "];\n";
  os << "        for (ap_uint<3> n = 0; n < " << latency << "; n++) {\n";
  os << "        #pragma HLS UNROLL\n";
  os << "          data_split[n] = out_data(31, 0);\n";
  os << "          out_data = out_data >> 32;\n";
  os << "        }\n";
  os << "        union {unsigned int ui; float ut;} u;\n";
  os << "        u.ut = in_data;\n";
  os << "        data_split[split_idx] = ap_uint<32>(u.ui);\n";
  os << "        out_data = (";
  for (unsigned i = latency - 1; i > 0; i--) {
    os << "data_split[" << i << "], ";
  }
  os << "data_split[0]);\n";
  if (isPowerOf2(latency))
    os << "        local_" << arrayName << "[c7][(c6) >> " << log2Po2(latency) << "] = out_data;\n";
  else
    os << "        local_" << arrayName << "[c7][c6 / " << latency << "] = out_data;\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
  
  // C_drain_IO_L1_out_inter_trans
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L1_out_inter_trans(int idx, int idy, int c0, int c1, "
     << arrayName << "_t" << latency << " local_" << arrayName << "[" << latency << "][1], "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName 
     << "_drain_in, hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName 
     << "_drain_out) {\n";
  os << "#pragma HLS INLINE\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx, p1 = idy; // module id\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  for (ap_uint<2> c4 = p1; c4 <= " << (numPE - 1) << "; c4 += 1) {\n";
  os << "    // io_L1\n";
  os << "    if (c4 == p1) {\n";
  os << "      for (ap_uint<3> c5 = 0; c5 <= " << (latency - 1) << "; c5 += 1) {\n";
  os << "      #pragma HLS PIPELINE II=1\n";
  os << "        // access_coalesce\n";
  os << "        {\n";
  os << "          " << arrayName << "_t" << latency << " in_data;\n";
  os << "          " << arrayName << "_t" << latency << " out_data;\n";
  os << "          in_data = local_" << arrayName << "[c5][0];\n";
  os << "          out_data = in_data;\n";
  os << "          fifo_" << arrayName << "_drain_out.write(out_data);\n";
  os << "        }\n";
  os << "      }\n";
  os << "    } else {\n";
  os << "      for (ap_uint<3> c5 = 0; c5 <= " << (latency - 1) << "; c5 += 1) {\n";
  os << "      #pragma HLS PIPELINE II=1\n";
  os << "        // access_coalesce\n";
  os << "        {\n";
  os << "          " << arrayName << "_t" << latency << " in_data;\n";
  os << "          " << arrayName << "_t" << latency << " out_data;\n";
  os << "          in_data = fifo_" << arrayName << "_drain_in.read();\n";
  os << "          out_data = in_data;\n";
  os << "          fifo_" << arrayName << "_drain_out.write(out_data);\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
  
  // C_drain_IO_L1_out_inter_trans_boundary
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int c0, int c1, "
     << arrayName << "_t" << latency << " local_" << arrayName << "[" << latency << "][1], "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName << "_drain_out) {\n";
  os << "#pragma HLS INLINE\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx, p1 = idy; // module id\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  for (ap_uint<2> c4 = p1; c4 <= " << (numPE - 1) << "; c4 += 1)\n";
  os << "    if (c4 == p1) {\n";
  os << "      // io_L1\n";
  os << "      for (ap_uint<3> c5 = 0; c5 <= " << (latency - 1) << "; c5 += 1) {\n";
  os << "      #pragma HLS PIPELINE II=1\n";
  os << "        // access_coalesce\n";
  os << "        {\n";
  os << "          " << arrayName << "_t" << latency << " in_data;\n";
  os << "          " << arrayName << "_t" << latency << " out_data;\n";
  os << "          in_data = local_" << arrayName << "[c5][0];\n";
  os << "          out_data = in_data;\n";
  os << "          fifo_" << arrayName << "_drain_out.write(out_data);\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
  
  // C_drain_IO_L1_out
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L1_out(int idx, int idy, "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName << "_drain_in, "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName << "_drain_out, "
     << "hls::stream<float> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx, p1 = idy; // module id\n";
  os << "  " << arrayName << "_t" << latency << " local_" << arrayName << "[" << latency << "][1];\n";
  os << "  #pragma HLS RESOURCE variable=local_" << arrayName << " core=RAM_2P_BRAM\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  for (ap_uint<3> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "    for (ap_uint<3> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  os << "      // array\n";
  os << "      // io_L3\n";
  os << "      // io_L2\n";
  os << "      " << arrayName << "_drain_IO_L1_out_intra_trans(\n";
  os << "        /* module id */ idx, \n";
  os << "        /* module id */ idy, \n";
  os << "        /* host iter */ c0, \n";
  os << "        /* host iter */ c1, \n";
  os << "        /* array */ local_" << arrayName << ", \n";
  os << "        /* fifo */ fifo_" << arrayName << "_drain_local_in\n";
  os << "      );\n";
  os << "      " << arrayName << "_drain_IO_L1_out_inter_trans(\n";
  os << "        /* module id */ idx, \n";
  os << "        /* module id */ idy, \n";
  os << "        /* host iter */ c0, \n";
  os << "        /* host iter */ c1, \n";
  os << "        /* array */ local_" << arrayName << ", \n";
  os << "        /* fifo */ fifo_" << arrayName << "_drain_in, \n";
  os << "        /* fifo */ fifo_" << arrayName << "_drain_out\n";
  os << "      );\n";
  os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
  
  // C_drain_IO_L1_out_wrapper
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L1_out_wrapper(int idx, int idy, "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName << "_drain_in, "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName << "_drain_out, "
     << "hls::stream<float> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "  " << arrayName << "_drain_IO_L1_out(\n";
  os << "    /* module id */ idx, \n";
  os << "    /* module id */ idy, \n";
  os << "    /* fifo */ fifo_" << arrayName << "_drain_in, \n";
  os << "    /* fifo */ fifo_" << arrayName << "_drain_out, \n";
  os << "    /* fifo */ fifo_" << arrayName << "_drain_local_in);\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
  
  // C_drain_IO_L1_out_boundary
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L1_out_boundary(int idx, int idy, "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName << "_drain_out, "
     << "hls::stream<float> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "#pragma HLS INLINE\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx, p1 = idy; // module id\n";
  os << "  " << arrayName << "_t" << latency << " local_" << arrayName << "[" << latency << "][1];\n";
  os << "  #pragma HLS RESOURCE variable=local_" << arrayName << " core=RAM_2P_BRAM\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  for (ap_uint<3> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "    for (ap_uint<3> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  os << "      // array\n";
  os << "      // io_L3\n";
  os << "      " << arrayName << "_drain_IO_L1_out_intra_trans(\n";
  os << "        /* module id */ idx, \n";
  os << "        /* module id */ idy, \n";
  os << "        /* host iter */ c0, \n";
  os << "        /* host iter */ c1, \n";
  os << "        /* array */ local_" << arrayName << ", \n";
  os << "        /* fifo */ fifo_" << arrayName << "_drain_local_in\n";
  os << "      );\n";
  os << "      " << arrayName << "_drain_IO_L1_out_inter_trans_boundary(\n";
  os << "        /* module id */ idx, \n";
  os << "        /* module id */ idy, \n";
  os << "        /* host iter */ c0, \n";
  os << "        /* host iter */ c1, \n";
  os << "        /* array */ local_" << arrayName << ", \n";
  os << "        /* fifo */ fifo_" << arrayName << "_drain_out\n";
  os << "      );\n";
  os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
  
  // C_drain_IO_L1_out_boundary_wrapper
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L1_out_boundary_wrapper(int idx, int idy, "
     << "hls::stream<" << arrayName << "_t" << latency << "> &fifo_" << arrayName << "_drain_out, "
     << "hls::stream<float> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "  " << arrayName << "_drain_IO_L1_out_boundary(\n";
  os << "    /* module id */ idx, \n";
  os << "    /* module id */ idy, \n";
  os << "    /* fifo */ fifo_" << arrayName << "_drain_out, \n";
  os << "    /* fifo */ fifo_" << arrayName << "_drain_local_in);\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitDrainIOL2(StringRef arrayName) {
  // C_drain_IO_L2_out
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L2_out(int idx, hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_in, hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_out, hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx; // module id\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  for (ap_uint<3> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "    for (ap_uint<3> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  os << "      // array\n";
  os << "      // io_L3\n";
  os << "      for (ap_uint<2> c3 = p0; c3 <= " << (numPE - 1) << "; c3 += 1) {\n";
  os << "        // io_L2\n";
  os << "        if (c3 == p0) {\n";
  os << "          for (ap_uint<2> c4 = 0; c4 <= " << (numPE - 1) << "; c4 += 1) {\n";
  os << "            // io_L1\n";
  os << "            for (ap_uint<3> c5 = 0; c5 <= " << (latency - 1) << "; c5 += 1) {\n";
  os << "            #pragma HLS PIPELINE II=1\n";
  os << "              // access_coalesce\n";
  os << "              {\n";
  os << "                " << arrayName << "_t" << latency << " in_data;\n";
  os << "                " << arrayName << "_t" << latency << " out_data;\n";
  os << "                in_data = fifo_" << arrayName << "_drain_local_in.read();\n";
  os << "                out_data = in_data;\n";
  os << "                fifo_" << arrayName << "_drain_out.write(out_data);\n";
  os << "              }\n";
  os << "            }\n";
  os << "          }\n";
  os << "        } else {\n";
  os << "          for (ap_uint<2> c4 = 0; c4 <= " << (numPE - 1) << "; c4 += 1) {\n";
  os << "            // io_L1\n";
  os << "            for (ap_uint<3> c5 = 0; c5 <= " << (latency - 1) << "; c5 += 1) {\n";
  os << "            #pragma HLS PIPELINE II=1\n";
  os << "              // access_coalesce\n";
  os << "              {\n";
  os << "                " << arrayName << "_t" << latency << " in_data;\n";
  os << "                " << arrayName << "_t" << latency << " out_data;\n";
  os << "                in_data = fifo_" << arrayName << "_drain_in.read();\n";
  os << "                out_data = in_data;\n";
  os << "                fifo_" << arrayName << "_drain_out.write(out_data);\n";
  os << "              }\n";
  os << "            }\n";
  os << "          }\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
  
  // C_drain_IO_L2_out_boundary
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L2_out_boundary(int idx, hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_out, hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  int p0 = idx; // module id\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  for (ap_uint<3> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "    for (ap_uint<3> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  os << "      // array\n";
  os << "      // io_L3\n";
  os << "      for (ap_uint<2> c3 = p0; c3 <= " << (numPE - 1) << "; c3 += 1)\n";
  os << "        if (c3 == p0) {\n";
  os << "          // io_L2\n";
  os << "          for (ap_uint<2> c4 = 0; c4 <= " << (numPE - 1) << "; c4 += 1) {\n";
  os << "            // io_L1\n";
  os << "            for (ap_uint<3> c5 = 0; c5 <= " << (latency - 1) << "; c5 += 1) {\n";
  os << "            #pragma HLS PIPELINE II=1\n";
  os << "              // access_coalesce\n";
  os << "              {\n";
  os << "                " << arrayName << "_t" << latency << " in_data;\n";
  os << "                " << arrayName << "_t" << latency << " out_data;\n";
  os << "                in_data = fifo_" << arrayName << "_drain_local_in.read();\n";
  os << "                out_data = in_data;\n";
  os << "                fifo_" << arrayName << "_drain_out.write(out_data);\n";
  os << "              }\n";
  os << "            }\n";
  os << "          }\n";
  os << "        }\n";
  os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitDrainIOL3(StringRef arrayName) {
  // C_drain_IO_L3_out
  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L3_out(hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_out, hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "#pragma HLS INLINE OFF\n";
  os << "  /* Variable Declaration */\n";
  os << "  /* Variable Declaration */\n\n";
  os << "  for (ap_uint<3> c0 = 0; c0 <= " << (numTiles - 1) << "; c0 += 1)\n";
  os << "    for (ap_uint<3> c1 = 0; c1 <= " << (numTiles - 1) << "; c1 += 1) {\n";
  os << "      // array\n";
  os << "      // io_L3\n";
  os << "      for (ap_uint<2> c3 = 0; c3 <= " << (numPE - 1) << "; c3 += 1) {\n";
  os << "        // io_L2\n";
  os << "        for (ap_uint<2> c4 = 0; c4 <= " << (numPE - 1) << "; c4 += 1) {\n";
  os << "          // io_L1\n";
  os << "          for (ap_uint<3> c5 = 0; c5 <= " << (latency - 1) << "; c5 += 1) {\n";
  os << "          #pragma HLS PIPELINE II=1\n";
  os << "            // access_coalesce\n";
  os << "            // access_serialize\n";
  os << "            {\n";
  os << "              " << arrayName << "_t" << latency << " in_data;\n";
  os << "              " << arrayName << "_t" << latency << " out_data;\n";
  os << "              in_data = fifo_" << arrayName << "_drain_local_in.read();\n";
  os << "              out_data = in_data;\n";
  os << "              fifo_" << arrayName << "_drain_out.write(out_data);\n";
  os << "            }\n";
  os << "          }\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitDrainSerialize(StringRef arrayName, unsigned totalSize,
                                            llvm::ArrayRef<int64_t> outShape) {
  // C_drain_IO_L3_out_serialize
  unsigned packFactor = kFloatsPerDramWord / latency;
  unsigned floatsPerWord = kFloatsPerDramWord;
  // 由 outputShape 推导写回字数；无 shape 时退化为 totalSize^2（2D 方阵）
  unsigned iterations;
  if (outShape.size() == 3) {
    uint64_t totalElements3 = (uint64_t)outShape[0] * outShape[1] * outShape[2];
    iterations = (unsigned)((totalElements3 * 4) / kDramWordBytes);
  } else if (outShape.size() >= 2) {
    uint64_t totalElements2 = (uint64_t)outShape[0] * outShape[1];
    iterations = (unsigned)((totalElements2 * 4) / kDramWordBytes);
  } else {
    iterations = (totalSize * totalSize * 4) / kDramWordBytes;
  }

  os << "/* Module Definition */\n";
  os << "void " << arrayName << "_drain_IO_L3_out_serialize(" << arrayName 
     << "_t16 *" << arrayName << ", hls::stream<" << arrayName << "_t" << latency 
     << "> &fifo_" << arrayName << "_drain_local_in) {\n";
  os << "#pragma HLS INLINE OFF\n";

  // Rank-3 output, no write-time reorder: sequential pack & write (drain order = row-major i,j,k).
  if (contraction.isRank3Output() && outShape.size() == 3 && !hasReordering3D(arrayName)) {
    os << "  /* Rank-3 output: pack fifo to DRAM (row-major " << outShape[0] << "x" << outShape[1] << "x" << outShape[2] << ") */\n";
    os << "  for (ap_uint<" << (unsigned)ceil(log2(iterations + 1)) << "> i = 0; i < " << iterations << "; i++) {\n";
    os << "  #pragma HLS PIPELINE II=1\n";
    os << "    " << arrayName << "_t" << latency << " fifo_data;\n";
    os << "    " << arrayName << "_t16 mem_data;\n";
    os << "    " << arrayName << "_t" << latency << " mem_data_split[" << packFactor << "];\n";
    os << "    #pragma HLS ARRAY_PARTITION variable=mem_data_split complete\n";
    os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++) {\n";
    os << "      fifo_data = fifo_" << arrayName << "_drain_local_in.read();\n";
    os << "      mem_data_split[p] = fifo_data;\n";
    os << "    }\n";
    os << "    mem_data = (";
    for (unsigned i = packFactor - 1; i > 0; i--) os << "mem_data_split[" << i << "], ";
    os << "mem_data_split[0]);\n";
    os << "    " << arrayName << "[i] = mem_data;\n";
    os << "  }\n";
    os << "}\n";
    os << "/* Module Definition */\n\n";
    return;
  }

  if (hasReordering3D(arrayName)) {
    auto it = arrayReordering.find(arrayName.str());
    const auto &info = it->second;
    unsigned s0 = (unsigned)info.originalDims[0];
    unsigned s1 = (unsigned)info.originalDims[1];
    unsigned s2 = (unsigned)info.originalDims[2];
    uint64_t totalElements = (uint64_t)s0 * s1 * s2;
    unsigned iter3 = (unsigned)((totalElements * 4) / kDramWordBytes);
    auto origExprs = getOriginalIndexExprs3D(arrayName, "d0", "d1", "d2");
    unsigned r0 = info.reorderedDims.size() >= 3 ? (unsigned)info.reorderedDims[0] : s0;
    unsigned r1 = info.reorderedDims.size() >= 3 ? (unsigned)info.reorderedDims[1] : s1;
    unsigned r2 = info.reorderedDims.size() >= 3 ? (unsigned)info.reorderedDims[2] : s2;
    unsigned loopBits0 = (unsigned)ceil(log2(r0 + 1));
    unsigned loopBits1 = (unsigned)ceil(log2(r1 + 1));
    unsigned loopBits2 = (unsigned)ceil(log2(r2 + 1));
    std::string linearExpr = "d0 * (" + std::to_string(r1 * r2) + ") + d1 * " + std::to_string(r2) + " + d2";
    if (origExprs.size() != 3) {
      os << "  /* Variable Declaration */\n\n";
      os << "  for (ap_uint<" << (unsigned)ceil(log2(iter3 + 1)) << "> i = 0; i < " << iter3 << "; i++) {\n";
      os << "  #pragma HLS PIPELINE II=1\n";
      os << "    " << arrayName << "_t" << latency << " fifo_data;\n";
      os << "    " << arrayName << "_t16 mem_data;\n";
      os << "    " << arrayName << "_t" << latency << " mem_data_split[" << packFactor << "];\n";
      os << "    #pragma HLS ARRAY_PARTITION variable=mem_data_split complete\n";
      os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++) {\n";
      os << "      fifo_data = fifo_" << arrayName << "_drain_local_in.read();\n";
      os << "      mem_data_split[p] = fifo_data;\n";
      os << "    }\n";
      os << "    mem_data = (";
      for (unsigned i = packFactor - 1; i > 0; i--) os << "mem_data_split[" << i << "], ";
      os << "mem_data_split[0]);\n";
      os << "    " << arrayName << "[i] = mem_data;\n";
      os << "  }\n";
      os << "}\n";
      os << "/* Module Definition */\n\n";
      return;
    }
    os << "  /* Write-time reordering (3D): buffer -> reorder -> pack & write */\n";
    os << "  float buffer[" << s0 << "][" << s1 << "][" << s2 << "];\n";
    os << "  float buffer_linear[" << totalElements << "];\n";
    os << "  union { float f; uint32_t u; } tmp;\n";
    os << "  /* Phase 1: unpack fifo into buffer (compute order) */\n";
    os << "  for (ap_uint<" << (unsigned)ceil(log2(iter3 + 1)) << "> i = 0; i < " << iter3 << "; i++) {\n";
    os << "  #pragma HLS PIPELINE II=1\n";
    os << "    " << arrayName << "_t" << latency << " fifo_data;\n";
    os << "    " << arrayName << "_t" << latency << " mem_data_split[" << packFactor << "];\n";
    os << "    #pragma HLS ARRAY_PARTITION variable=mem_data_split complete\n";
    os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++) {\n";
    os << "      fifo_data = fifo_" << arrayName << "_drain_local_in.read();\n";
    os << "      mem_data_split[p] = fifo_data;\n";
    os << "    }\n";
    os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++)\n";
    os << "      for (ap_uint<3> f = 0; f < " << latency << "; f++) {\n";
    os << "        unsigned idx = i * " << floatsPerWord << " + p * " << latency << " + f;\n";
    if (isPowerOf2(s1) && isPowerOf2(s2)) {
      unsigned s1s2 = s1 * s2;
      os << "        unsigned r0 = idx >> " << log2Po2(s1s2) << ", r1 = (idx >> " << log2Po2(s2) << ") & " << (s1 - 1) << ", r2 = idx & " << (s2 - 1) << ";\n";
    } else {
      os << "        unsigned r0 = idx / (" << s1 << " * " << s2 << "), r1 = (idx / " << s2 << ") % " << s1 << ", r2 = idx % " << s2 << ";\n";
    }
    os << "        tmp.u = mem_data_split[p].range(32*(int)(f+1)-1, 32*(int)f).to_uint();\n";
    os << "        buffer[r0][r1][r2] = tmp.f;\n";
    os << "      }\n";
    os << "  }\n";
    os << "  /* Phase 2: reorder into buffer_linear (reordered dimension order) */\n";
    os << "  for (ap_uint<" << loopBits0 << "> d0 = 0; d0 < " << r0 << "; d0++)\n";
    os << "    for (ap_uint<" << loopBits1 << "> d1 = 0; d1 < " << r1 << "; d1++)\n";
    os << "      for (ap_uint<" << loopBits2 << "> d2 = 0; d2 < " << r2 << "; d2++) {\n";
    os << "        unsigned linear = " << linearExpr << ";\n";
    os << "        buffer_linear[linear] = buffer[" << origExprs[0] << "][" << origExprs[1] << "][" << origExprs[2] << "];\n";
    os << "      }\n";
    os << "  /* Phase 3: pack and write to DRAM */\n";
    os << "  for (ap_uint<" << (unsigned)ceil(log2(iter3 + 1)) << "> i = 0; i < " << iter3 << "; i++) {\n";
    os << "  #pragma HLS PIPELINE II=1\n";
    os << "    " << arrayName << "_t16 mem_data;\n";
    os << "    float *ptr = (float *)&mem_data;\n";
    os << "    for (ap_uint<4> k = 0; k < " << floatsPerWord << "; k++)\n";
    os << "      ptr[k] = buffer_linear[i * " << floatsPerWord << " + k];\n";
    os << "    " << arrayName << "[i] = mem_data;\n";
    os << "  }\n";
  } else if (hasReordering2D(arrayName)) {
    auto it2d = arrayReordering.find(arrayName.str());
    unsigned s0 = it2d != arrayReordering.end() && it2d->second.originalDims.size() >= 2
                      ? (unsigned)it2d->second.originalDims[0]
                      : totalSize;
    unsigned s1 = it2d != arrayReordering.end() && it2d->second.originalDims.size() >= 2
                      ? (unsigned)it2d->second.originalDims[1]
                      : totalSize;
    unsigned totalElements2d = s0 * s1;
    unsigned iterations2d = (totalElements2d * 4) / kDramWordBytes;
    unsigned loopBits0 = (unsigned)ceil(log2(s0 + 1));
    unsigned loopBits1 = (unsigned)ceil(log2(s1 + 1));
    auto origExprs = getOriginalIndexExprs2D(arrayName, "d0", "d1");
    std::string linearExpr = getLinearIndexFromReordered2D(arrayName, "d0", "d1", s1);
    if (origExprs.size() != 2 || linearExpr.empty()) {
      // Fallback to non-reordered path if helpers failed
      os << "  /* Variable Declaration */\n\n";
      os << "  for (ap_uint<" << (unsigned)ceil(log2(iterations2d + 1)) << "> i = 0; i < " 
         << iterations2d << "; i++) {\n";
      os << "  #pragma HLS PIPELINE II=1\n";
      os << "    " << arrayName << "_t" << latency << " fifo_data;\n";
      os << "    " << arrayName << "_t16 mem_data;\n";
      os << "    " << arrayName << "_t" << latency << " mem_data_split[" << packFactor << "];\n";
      os << "    #pragma HLS ARRAY_PARTITION variable=mem_data_split complete\n";
      os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++) {\n";
      os << "      fifo_data = fifo_" << arrayName << "_drain_local_in.read();\n";
      os << "      mem_data_split[p] = fifo_data;\n";
      os << "    }\n";
      os << "    mem_data = (";
      for (unsigned i = packFactor - 1; i > 0; i--) os << "mem_data_split[" << i << "], ";
      os << "mem_data_split[0]);\n";
      os << "    " << arrayName << "[i] = mem_data;\n";
      os << "  }\n";
      os << "}\n";
      os << "/* Module Definition */\n\n";
      return;
    }
    os << "  /* Write-time reordering: buffer -> reorder -> pack & write */\n";
    os << "  float buffer[" << s0 << "][" << s1 << "];\n";
    os << "  float buffer_linear[" << totalElements2d << "];\n";
    os << "  union { float f; uint32_t u; } tmp;\n";
    os << "  /* Phase 1: unpack fifo into buffer (row-major compute order) */\n";
    os << "  for (ap_uint<" << (unsigned)ceil(log2(iterations2d + 1)) << "> i = 0; i < " << iterations2d << "; i++) {\n";
    os << "  #pragma HLS PIPELINE II=1\n";
    os << "    " << arrayName << "_t" << latency << " fifo_data;\n";
    os << "    " << arrayName << "_t" << latency << " mem_data_split[" << packFactor << "];\n";
    os << "    #pragma HLS ARRAY_PARTITION variable=mem_data_split complete\n";
    os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++) {\n";
    os << "      fifo_data = fifo_" << arrayName << "_drain_local_in.read();\n";
    os << "      mem_data_split[p] = fifo_data;\n";
    os << "    }\n";
    os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++)\n";
    os << "      for (ap_uint<3> f = 0; f < " << latency << "; f++) {\n";
    os << "        unsigned idx = i * " << floatsPerWord << " + p * " << latency << " + f;\n";
    if (isPowerOf2(s1))
      os << "        unsigned r = idx >> " << log2Po2(s1) << ", c = idx & " << (s1 - 1) << ";\n";
    else
      os << "        unsigned r = idx / " << s1 << ", c = idx % " << s1 << ";\n";
    os << "        tmp.u = mem_data_split[p].range(32*(int)(f+1)-1, 32*(int)f).to_uint();\n";
    os << "        buffer[r][c] = tmp.f;\n";
    os << "      }\n";
    os << "  }\n";
    os << "  /* Phase 2: reorder into buffer_linear (reordered dimension order) */\n";
    os << "  for (ap_uint<" << loopBits0 << "> d0 = 0; d0 < " << s0 << "; d0++)\n";
    os << "    for (ap_uint<" << loopBits1 << "> d1 = 0; d1 < " << s1 << "; d1++) {\n";
    os << "      unsigned linear = " << linearExpr << ";\n";
    os << "      buffer_linear[linear] = buffer[" << origExprs[0] << "][" << origExprs[1] << "];\n";
    os << "    }\n";
    os << "  /* Phase 3: pack and write to DRAM */\n";
    os << "  for (ap_uint<" << (unsigned)ceil(log2(iterations2d + 1)) << "> i = 0; i < " << iterations2d << "; i++) {\n";
    os << "  #pragma HLS PIPELINE II=1\n";
    os << "    " << arrayName << "_t16 mem_data;\n";
    os << "    float *ptr = (float *)&mem_data;\n";
    os << "    for (ap_uint<4> k = 0; k < " << floatsPerWord << "; k++)\n";
    os << "      ptr[k] = buffer_linear[i * " << floatsPerWord << " + k];\n";
    os << "    " << arrayName << "[i] = mem_data;\n";
    os << "  }\n";
  } else {
    os << "  /* Variable Declaration */\n\n";
    os << "  for (ap_uint<" << (unsigned)ceil(log2(iterations + 1)) << "> i = 0; i < " 
       << iterations << "; i++) {\n";
    os << "  #pragma HLS PIPELINE II=1\n";
    os << "    " << arrayName << "_t" << latency << " fifo_data;\n";
    os << "    " << arrayName << "_t16 mem_data;\n";
    os << "    " << arrayName << "_t" << latency << " mem_data_split[" << packFactor << "];\n";
    os << "    #pragma HLS ARRAY_PARTITION variable=mem_data_split complete\n";
    os << "    for (ap_uint<3> p = 0; p < " << packFactor << "; p++) {\n";
    os << "      fifo_data = fifo_" << arrayName << "_drain_local_in.read();\n";
    os << "      mem_data_split[p] = fifo_data;\n";
    os << "    }\n";
    os << "    mem_data = (";
    for (unsigned i = packFactor - 1; i > 0; i--) {
      os << "mem_data_split[" << i << "], ";
    }
    os << "mem_data_split[0]);\n";
    os << "    " << arrayName << "[i] = mem_data;\n";
    os << "  }\n";
  }
  os << "}\n";
  os << "/* Module Definition */\n\n";
}

void SystolicHLSEmitter::emitTopKernel(func::FuncOp funcOp) {
  const std::string &out = outputName;
  // 保留 kernel0 以兼容现有 host/e2e；后续可改为 funcOp.getName() 或 --kernel-name 选项
  os << "extern \"C\" {\n";
  os << "void kernel0(";
  for (size_t idx = 0; idx < inputNames.size(); idx++)
    os << (idx ? ", " : "") << inputNames[idx] << "_t16 *" << inputNames[idx];
  os << ", " << out << "_t16 *" << out << ") {\n";
  for (const auto &n : inputNames)
    os << "#pragma HLS INTERFACE m_axi port=" << n << " offset=slave bundle=gmem_" << n << "\n";
  os << "#pragma HLS INTERFACE m_axi port=" << out << " offset=slave bundle=gmem_" << out << "\n";
  for (const auto &n : inputNames)
    os << "#pragma HLS INTERFACE s_axilite port=" << n << " bundle=control\n";
  os << "#pragma HLS INTERFACE s_axilite port=" << out << " bundle=control\n";
  os << "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n";
  os << "#pragma HLS DATAFLOW\n\n";
  
  os << "  /* FIFO Declaration */\n";
  for (const auto &in : inputNames) {
    os << "  hls::stream<" << in << "_t" << arrayPart << "> fifo_" << in << "_" << in << "_IO_L3_in_serialize;\n";
    os << "  #pragma HLS STREAM variable=fifo_" << in << "_" << in << "_IO_L3_in_serialize depth=" << fifoDepth << "\n";
    os << "  #pragma HLS RESOURCE variable=fifo_" << in << "_" << in << "_IO_L3_in_serialize core=FIFO_SRL\n";
  }
  os << "  hls::stream<" << out << "_t" << latency << "> fifo_" << out << "_drain_" << out << "_drain_IO_L3_out_serialize;\n";
  os << "  #pragma HLS STREAM variable=fifo_" << out << "_drain_" << out << "_drain_IO_L3_out_serialize depth=" << fifoDepth << "\n";
  os << "  #pragma HLS RESOURCE variable=fifo_" << out << "_drain_" << out << "_drain_IO_L3_out_serialize core=FIFO_SRL\n\n";
  
  for (const auto &in : inputNames) {
    for (unsigned i = 0; i <= numPE; i++) {
      os << "  hls::stream<" << in << "_t" << arrayPart << "> fifo_" << in << "_" << in << "_IO_L2_in_" << i << ";\n";
      os << "  #pragma HLS STREAM variable=fifo_" << in << "_" << in << "_IO_L2_in_" << i << " depth=" << fifoDepth << "\n";
      os << "  #pragma HLS RESOURCE variable=fifo_" << in << "_" << in << "_IO_L2_in_" << i << " core=FIFO_SRL\n";
    }
  }
  for (size_t inIdx = 0; inIdx < inputNames.size(); inIdx++) {
    const std::string &in = inputNames[inIdx];
    if (inIdx == 0) {
      for (unsigned i = 0; i < numPE; i++) {
        for (unsigned j = 0; j <= numPE; j++) {
          os << "  hls::stream<float> fifo_" << in << "_PE_" << i << "_" << j << ";\n";
          os << "  #pragma HLS STREAM variable=fifo_" << in << "_PE_" << i << "_" << j << " depth=" << fifoDepth << "\n";
          os << "  #pragma HLS RESOURCE variable=fifo_" << in << "_PE_" << i << "_" << j << " core=FIFO_SRL\n";
        }
      }
    } else {
      for (unsigned i = 0; i <= numPE; i++) {
        for (unsigned j = 0; j < numPE; j++) {
          os << "  hls::stream<float> fifo_" << in << "_PE_" << i << "_" << j << ";\n";
          os << "  #pragma HLS STREAM variable=fifo_" << in << "_PE_" << i << "_" << j << " depth=" << fifoDepth << "\n";
          os << "  #pragma HLS RESOURCE variable=fifo_" << in << "_PE_" << i << "_" << j << " core=FIFO_SRL\n";
        }
      }
    }
  }
  for (unsigned i = 0; i < numPE; i++) {
    for (unsigned j = 0; j < numPE; j++) {
      os << "  hls::stream<float> fifo_" << out << "_drain_PE_" << i << "_" << j << ";\n";
      os << "  #pragma HLS STREAM variable=fifo_" << out << "_drain_PE_" << i << "_" << j << " depth=" << fifoDepth << "\n";
      os << "  #pragma HLS RESOURCE variable=fifo_" << out << "_drain_PE_" << i << "_" << j << " core=FIFO_SRL\n";
    }
  }
  for (unsigned i = 0; i < numPE; i++) {
    for (unsigned j = 0; j <= numPE; j++) {
      os << "  hls::stream<" << out << "_t" << latency << "> fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << i << "_" << j << ";\n";
      os << "  #pragma HLS STREAM variable=fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << i << "_" << j << " depth=" << fifoDepth << "\n";
      os << "  #pragma HLS RESOURCE variable=fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << i << "_" << j << " core=FIFO_SRL\n";
    }
  }
  for (unsigned i = 0; i <= numPE; i++) {
    os << "  hls::stream<" << out << "_t" << latency << "> fifo_" << out << "_drain_" << out << "_drain_IO_L2_out_" << i << ";\n";
    os << "  #pragma HLS STREAM variable=fifo_" << out << "_drain_" << out << "_drain_IO_L2_out_" << i << " depth=" << fifoDepth << "\n";
    os << "  #pragma HLS RESOURCE variable=fifo_" << out << "_drain_" << out << "_drain_IO_L2_out_" << i << " core=FIFO_SRL\n";
  }
  os << "  /* FIFO Declaration */\n\n";
  
  os << "  /* Module Call */\n";
  for (size_t inIdx = 0; inIdx < inputNames.size(); inIdx++) {
    const std::string &in = inputNames[inIdx];
    os << "  " << in << "_IO_L3_in_serialize(" << in << ", fifo_" << in << "_" << in << "_IO_L3_in_serialize);\n";
    os << "  " << in << "_IO_L3_in(fifo_" << in << "_" << in << "_IO_L3_in_serialize, fifo_" << in << "_" << in << "_IO_L2_in_0);\n";
    if (inIdx == 0) {
      for (unsigned i = 0; i < numPE - 1; i++) {
        os << "  " << in << "_IO_L2_in(" << i << ", fifo_" << in << "_" << in << "_IO_L2_in_" << i
           << ", fifo_" << in << "_" << in << "_IO_L2_in_" << (i + 1) << ", fifo_" << in << "_PE_" << i << "_0);\n";
      }
      os << "  " << in << "_IO_L2_in_boundary(" << (numPE - 1) << ", fifo_" << in << "_" << in << "_IO_L2_in_"
         << (numPE - 1) << ", fifo_" << in << "_PE_" << (numPE - 1) << "_0);\n\n";
    } else {
      for (unsigned j = 0; j < numPE - 1; j++) {
        os << "  " << in << "_IO_L2_in(" << j << ", fifo_" << in << "_" << in << "_IO_L2_in_" << j
           << ", fifo_" << in << "_" << in << "_IO_L2_in_" << (j + 1) << ", fifo_" << in << "_PE_0_" << j << ");\n";
      }
      os << "  " << in << "_IO_L2_in_boundary(" << (numPE - 1) << ", fifo_" << in << "_" << in << "_IO_L2_in_"
         << (numPE - 1) << ", fifo_" << in << "_PE_0_" << (numPE - 1) << ");\n\n";
    }
  }
  
  for (unsigned i = 0; i < numPE; i++) {
    for (unsigned j = 0; j < numPE; j++) {
      os << "  PE_wrapper(" << i << ", " << j << ", ";
      for (size_t inIdx = 0; inIdx < inputNames.size(); inIdx++) {
        const std::string &in = inputNames[inIdx];
        if (inIdx == 0)
          os << (inIdx ? ", " : "") << "fifo_" << in << "_PE_" << i << "_" << j << ", fifo_" << in << "_PE_" << i << "_" << (j + 1);
        else
          os << (inIdx ? ", " : "") << "fifo_" << in << "_PE_" << i << "_" << j << ", fifo_" << in << "_PE_" << (i + 1) << "_" << j;
      }
      os << ", fifo_" << out << "_drain_PE_" << i << "_" << j << ");\n";
    }
  }
  os << "\n";
  for (size_t inIdx = 0; inIdx < inputNames.size(); inIdx++) {
    const std::string &in = inputNames[inIdx];
    if (inIdx == 0) {
      for (unsigned i = 0; i < numPE; i++)
        os << "  " << in << "_PE_dummy_in(" << i << ", " << (numPE - 1) << ", fifo_" << in << "_PE_" << i << "_" << numPE << ");\n";
    } else {
      for (unsigned j = 0; j < numPE; j++)
        os << "  " << in << "_PE_dummy_in(" << (numPE - 1) << ", " << j << ", fifo_" << in << "_PE_" << numPE << "_" << j << ");\n";
    }
  }
  os << "\n";
  
  os << "  /* " << out << " drain modules */\n";
  for (unsigned j = numPE - 1; j > 0; j--) {
    for (unsigned i = 0; i < numPE; i++) {
      if (j == numPE - 1) {
        os << "  " << out << "_drain_IO_L1_out_boundary_wrapper(\n";
        os << "    " << i << ", " << j << ", fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << i << "_" << j << ", fifo_" << out << "_drain_PE_" << i << "_" << j << "\n  );\n";
      }
      os << "  " << out << "_drain_IO_L1_out_wrapper(\n";
      os << "    " << i << ", " << (j - 1) << ", fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << i << "_" << j
         << ", fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << i << "_" << (j - 1) << ", fifo_" << out << "_drain_PE_" << i << "_" << (j - 1) << "\n  );\n";
    }
  }
  os << "  " << out << "_drain_IO_L2_out_boundary(" << (numPE - 1) << ", fifo_" << out << "_drain_" << out << "_drain_IO_L2_out_" << (numPE - 1) << ", fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << (numPE - 1) << "_0);\n";
  for (unsigned i = 0; i < numPE - 1; i++) {
    os << "  " << out << "_drain_IO_L2_out(" << i << ", fifo_" << out << "_drain_" << out << "_drain_IO_L2_out_" << (i + 1)
       << ", fifo_" << out << "_drain_" << out << "_drain_IO_L2_out_" << i << ", fifo_" << out << "_drain_" << out << "_drain_IO_L1_out_" << i << "_0);\n";
  }
  os << "  " << out << "_drain_IO_L3_out(fifo_" << out << "_drain_" << out << "_drain_IO_L3_out_serialize, fifo_" << out << "_drain_" << out << "_drain_IO_L2_out_0);\n";
  os << "  " << out << "_drain_IO_L3_out_serialize(" << out << ", fifo_" << out << "_drain_" << out << "_drain_IO_L3_out_serialize);\n";
  os << "}\n";
  os << "}\n";
}

// Extract reordering information from function attributes
void SystolicHLSEmitter::extractReorderingInfo(func::FuncOp funcOp) {
  // 遍历所有函数参数，查找重排属性
  for (size_t argIdx = 0; argIdx < funcOp.getNumArguments(); ++argIdx) {
    auto arg = funcOp.getArgument(argIdx);
    
    if (auto memrefType = arg.getType().dyn_cast<MemRefType>()) {
      // 获取数组名称
      std::string arrayName = "arg" + std::to_string(argIdx);  // 默认值
      if (auto nameAttr = funcOp.getArgAttrOfType<StringAttr>(
              argIdx, "mlir.name")) {
        arrayName = nameAttr.getValue().str();
      } else {
        // 尝试从函数属性中查找（使用 arg0, arg1 等格式）
        std::string attrName = "systolic.reorder.arg" + std::to_string(argIdx) + ".dims";
        if (funcOp->getAttrOfType<ArrayAttr>(attrName)) {
          arrayName = "arg" + std::to_string(argIdx);
        }
      }
      
      // 查找重排维度属性
      std::string dimsAttrName = "systolic.reorder." + arrayName + ".dims";
      auto dimsAttr = funcOp->getAttrOfType<ArrayAttr>(dimsAttrName);
      
      // 如果使用 arg0 格式，也尝试查找
      if (!dimsAttr) {
        dimsAttrName = "systolic.reorder.arg" + std::to_string(argIdx) + ".dims";
        dimsAttr = funcOp->getAttrOfType<ArrayAttr>(dimsAttrName);
        if (dimsAttr) {
          arrayName = "arg" + std::to_string(argIdx);
        }
      }
      
      // 查找重排置换属性
      std::string permAttrName = "systolic.reorder." + arrayName + ".perm";
      auto permAttr = funcOp->getAttrOfType<ArrayAttr>(permAttrName);
      
      if (!permAttr) {
        permAttrName = "systolic.reorder.arg" + std::to_string(argIdx) + ".perm";
        permAttr = funcOp->getAttrOfType<ArrayAttr>(permAttrName);
      }
      
      // 如果两个属性都存在，提取信息
      if (dimsAttr && permAttr) {
        ArrayReorderingInfo info;
        info.arrayName = arrayName;
        
        // 原始维度
        for (int64_t dim : memrefType.getShape()) {
          info.originalDims.push_back(dim);
        }
        
        // 重排后维度
        for (auto attr : dimsAttr) {
          if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
            info.reorderedDims.push_back(intAttr.getInt());
          }
        }
        
        // 维度置换（限制在 [0, 2] 避免 3D 时越界）
        for (auto attr : permAttr) {
          if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
            int64_t v = intAttr.getInt();
            unsigned u = (unsigned)std::max(0, std::min(2, (int)v));
            info.dimPermutation.push_back(u);
          }
        }
        
        arrayReordering[arrayName] = info;
        
        LLVM_DEBUG({
          llvm::dbgs() << "Extracted reordering info for " << arrayName << ":\n";
          llvm::dbgs() << "  Original: [";
          for (size_t i = 0; i < info.originalDims.size(); i++)
            llvm::dbgs() << (i ? ", " : "") << info.originalDims[i];
          llvm::dbgs() << "]\n  Reordered: [";
          for (size_t i = 0; i < info.reorderedDims.size(); i++)
            llvm::dbgs() << (i ? ", " : "") << info.reorderedDims[i];
          llvm::dbgs() << "]\n  Permutation: [";
          for (size_t i = 0; i < info.dimPermutation.size(); i++)
            llvm::dbgs() << (i ? ", " : "") << info.dimPermutation[i];
          llvm::dbgs() << "]\n";
        });
      }
    }
  }
}

// Get array dimensions (considering reordering)
SmallVector<int64_t, 3> SystolicHLSEmitter::getArrayDims(StringRef arrayName) const {
  auto it = arrayReordering.find(arrayName.str());
  if (it != arrayReordering.end() && it->second.needsReordering()) {
    // 如果有重排，使用重排后的维度
    return it->second.reorderedDims;
  } else {
    // 否则使用默认维度（L2 缓冲区大小）
    return {static_cast<int64_t>(latency), 1, static_cast<int64_t>(arrayPart)};
  }
}

// Apply dimension permutation to access indices
SmallVector<std::string, 3> SystolicHLSEmitter::applyAccessPermutation(
    StringRef arrayName,
    const SmallVector<std::string, 3> &originalIndices) const {
  auto it = arrayReordering.find(arrayName.str());
  if (it != arrayReordering.end() && it->second.needsReordering()) {
    return it->second.applyPermutation(originalIndices);
  } else {
    // 无重排，返回原始索引
    return originalIndices;
  }
}

bool SystolicHLSEmitter::hasReordering2D(StringRef arrayName) const {
  auto it = arrayReordering.find(arrayName.str());
  return it != arrayReordering.end() && it->second.needsReordering() &&
         it->second.originalDims.size() == 2 && it->second.dimPermutation.size() == 2;
}

bool SystolicHLSEmitter::hasReordering3D(StringRef arrayName) const {
  auto it = arrayReordering.find(arrayName.str());
  if (it == arrayReordering.end() || it->second.originalDims.size() != 3 ||
      it->second.dimPermutation.size() != 3)
    return false;
  // 有 3D 的 dims/perm 属性即走 3D 路径（含恒等置换），确保 buffer_linear 被生成
  return true;
}

std::string SystolicHLSEmitter::getLinearIndexFromReordered2D(StringRef arrayName,
                                                            StringRef d0Var, StringRef d1Var,
                                                            unsigned sizeLiteral) const {
  auto it = arrayReordering.find(arrayName.str());
  if (it == arrayReordering.end() || it->second.originalDims.size() != 2 ||
      it->second.dimPermutation.size() != 2)
    return "";
  const auto &info = it->second;
  SmallVector<unsigned, 2> invPerm(2);
  for (unsigned i = 0; i < 2; i++)
    invPerm[info.dimPermutation[i]] = i;
  std::string d0s = d0Var.str(), d1s = d1Var.str();
  std::string orig0 = (invPerm[0] == 0) ? d0s : d1s;
  std::string orig1 = (invPerm[1] == 0) ? d0s : d1s;
  std::string dimStr = sizeLiteral ? std::to_string(sizeLiteral) : "size";
  return orig0 + " * " + dimStr + " + " + orig1;
}

SmallVector<std::string, 2> SystolicHLSEmitter::getOriginalIndexExprs2D(
    StringRef arrayName, StringRef d0Var, StringRef d1Var) const {
  SmallVector<std::string, 2> result;
  auto it = arrayReordering.find(arrayName.str());
  if (it == arrayReordering.end() || it->second.dimPermutation.size() != 2)
    return result;
  const auto &info = it->second;
  SmallVector<unsigned, 2> invPerm(2);
  for (unsigned i = 0; i < 2; i++)
    invPerm[info.dimPermutation[i]] = i;
  std::string d0s = d0Var.str(), d1s = d1Var.str();
  result.push_back((invPerm[0] == 0) ? d0s : d1s);
  result.push_back((invPerm[1] == 0) ? d0s : d1s);
  return result;
}

SmallVector<std::string, 3> SystolicHLSEmitter::getOriginalIndexExprs3D(
    StringRef arrayName, StringRef d0Var, StringRef d1Var, StringRef d2Var) const {
  SmallVector<std::string, 3> result;
  auto it = arrayReordering.find(arrayName.str());
  if (it == arrayReordering.end() || it->second.dimPermutation.size() != 3)
    return result;
  const auto &info = it->second;
  std::string d0s = d0Var.str(), d1s = d1Var.str(), d2s = d2Var.str();
  SmallVector<unsigned, 3> invPerm(3, 0);
  for (unsigned i = 0; i < 3; i++) {
    unsigned p = info.dimPermutation[i];
    if (p < 3)
      invPerm[p] = i;
  }
  for (unsigned i = 0; i < 3; i++)
    result.push_back((invPerm[i] == 0) ? d0s : ((invPerm[i] == 1) ? d1s : d2s));
  return result;
}

LogicalResult SystolicHLSEmitter::emitFunc(func::FuncOp funcOp) {
  emitTopKernel(funcOp);
  return success();
}

LogicalResult SystolicHLSEmitter::emit(ModuleOp module) {
  auto funcOps = module.getOps<func::FuncOp>();
  func::FuncOp kernelFunc;
  if (!funcOps.empty()) {
    // Prefer the kernel that has any systolic.reorder.*.dims (entry point after transform).
    for (func::FuncOp f : funcOps) {
      for (const auto &attr : f->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name.starts_with("systolic.reorder.") && name.ends_with(".dims")) {
          kernelFunc = f;
          break;
        }
      }
      if (kernelFunc) break;
    }
    if (!kernelFunc)
      for (func::FuncOp f : funcOps)
        if (!f.isPrivate())
          { kernelFunc = f; break; }
    if (!kernelFunc)
      kernelFunc = *funcOps.begin();
    extractReorderingInfo(kernelFunc);
    deriveArrayNamesFromFunction(kernelFunc);

    SmallVector<MemRefType, 4> memrefArgs;
    for (unsigned i = 0; i < kernelFunc.getNumArguments(); i++) {
      if (auto memrefTy = kernelFunc.getArgument(i).getType().dyn_cast<MemRefType>())
        memrefArgs.push_back(memrefTy);
    }
    inputShapes.clear();
    for (unsigned i = 0; i < inputNames.size() && i < memrefArgs.size(); i++) {
      SmallVector<int64_t, 3> sh;
      for (int64_t s : memrefArgs[i].getShape())
        sh.push_back(s);
      inputShapes.push_back(sh);
    }
    if (!memrefArgs.empty()) {
      auto outTy = memrefArgs.back();
      contraction.outputRank = outTy.getRank();
      outputShape.clear();
      for (int64_t s : outTy.getShape())
        outputShape.push_back(s);
    }

    // 由 Transform 侧写入的 time-loop 数量属性（近似“规约维数”）。
    if (auto numTime =
            kernelFunc->getAttrOfType<IntegerAttr>("systolic.num_time_loops")) {
      int64_t v = numTime.getInt();
      if (v < 1)
        contraction.numReductions = 1;
      else
        contraction.numReductions = static_cast<unsigned>(v);
    } else {
      contraction.numReductions = 1;
    }

    // 设置 contraction.kind，仅对 Unsupported 在下方报错返回。
    if (contraction.outputRank == 2) {
      contraction.kind = contraction.numReductions >= 2
          ? ContractionDesc::Kind::MttkrpLike
          : ContractionDesc::Kind::MatmulLike;
    } else if (contraction.outputRank == 3) {
      if (contraction.numReductions <= 3)
        contraction.kind = ContractionDesc::Kind::TtmcLike;
      else
        contraction.kind = ContractionDesc::Kind::Unsupported;
    } else {
      contraction.kind = ContractionDesc::Kind::Unsupported;
    }

    if (contraction.kind == ContractionDesc::Kind::Unsupported) {
      llvm::errs() << "systolic-translate error: unsupported output rank " << contraction.outputRank;
      if (contraction.outputRank == 3 && contraction.numReductions > 3)
        llvm::errs() << " (num_time_loops " << contraction.numReductions << " > 3 not supported)";
      llvm::errs() << ".\n  Function: " << kernelFunc.getName() << "\n";
      return failure();
    }
  } else {
    inputNames.assign({"A", "B"});
    outputName = "C";
  }

  {
    unsigned c5B = arrayPart / simd;
    if (c5B == 0) c5B = 1;
    bitsTiles = requiredLoopBits(numTiles > 0 ? (uint64_t)numTiles - 1 : 0);
    bitsPE = requiredLoopBits(numPE > 0 ? (uint64_t)numPE - 1 : 0);
    bitsSize = requiredLoopBits(size > 0 ? (uint64_t)size - 1 : 0);
    bitsLatency = requiredLoopBits(latency > 0 ? (uint64_t)latency - 1 : 0);
    bitsC5Bound = requiredLoopBits((uint64_t)c5B - 1);
  }

  emitFileHeader();
  emitTypeDefinitions();
  emitModuleDeclarations();
  
  for (size_t i = 0; i < inputNames.size(); i++) {
    llvm::ArrayRef<int64_t> arrShape =
        (i < inputShapes.size()) ? llvm::ArrayRef<int64_t>(inputShapes[i])
                                 : llvm::ArrayRef<int64_t>();
    emitIOL3InSerialize(inputNames[i], "float", size, arrShape);
    emitIOL3In(inputNames[i], "float");
    emitIOL2InIntraTrans(inputNames[i]);
    emitIOL2InInterTrans(inputNames[i]);
    emitIOL2InInterTransBoundary(inputNames[i]);
    emitIOL2In(inputNames[i]);
    emitIOL2InBoundary(inputNames[i]);
  }
  emitPE();
  emitPEWrapper();
  emitDummyModules();
  emitDrainIOL1(outputName);
  emitDrainIOL2(outputName);
  emitDrainIOL3(outputName);
  emitDrainSerialize(outputName, size, outputShape);
  
  if (!funcOps.empty() && kernelFunc && failed(emitFunc(kernelFunc)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Systolic HLS C++ Translator\n");
  
  // Set up context
  MLIRContext context;
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::systolic::dataflow::SystolicDataflowDialect>();
  
  // Parse input
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }
  
  // Set up output
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  
  // Emit HLS C++
  SystolicHLSEmitter emitter(output->os(), arrayPartSize, latencySize, simdFactor, problemSize, fifoDepth);
  if (failed(emitter.emit(*module))) {
    llvm::errs() << "Failed to emit HLS C++\n";
    return 1;
  }
  
  output->keep();
  return 0;
}
