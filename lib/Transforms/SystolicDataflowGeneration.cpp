//===----------------------------------------------------------------------===//
//
// MLIR-Systolic: SystolicDataflow Generation Pass
//
// This pass generates SystolicDataflow Dialect operations from Affine IR,
// creating multi-level IO modules, PE arrays, and double buffering structures.
//
// AutoSA Reference:
//   - sa_io_construct_optimize: Group array references, create I/O modules
//   - generate_hw_modules: Generate PE, I/O, and drain modules
//   - sa_io_module_gen: Create I/O module hierarchy (L3 -> L2 -> L1)
//
//===----------------------------------------------------------------------===//

#include "systolic/Transforms/Passes.h"
#include "systolic/Analysis/SpaceTimeAnalysis.h"
#include "systolic/Analysis/SystolicConfig.h"
#include "systolic/Analysis/WriteTimeReorderingAnalysis.h"
#include "systolic/Dialect/SystolicDataflow/SystolicDataflow.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <optional>

#define DEBUG_TYPE "systolic-dataflow-generation"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::systolic::dataflow;

namespace mlir {
namespace systolic {

//===----------------------------------------------------------------------===//
// Array Reference Group (AutoSA: autosa_array_ref_group)
//===----------------------------------------------------------------------===//

/// Represents a group of array references that should be handled together.
/// Similar to AutoSA's autosa_array_ref_group.
struct ArrayRefGroup {
  Value memref;
  std::string arrayName;
  
  // Access operations
  SmallVector<AffineLoadOp, 8> loads;
  SmallVector<AffineStoreOp, 8> stores;
  
  // Classification
  enum GroupType { IO_GROUP, PE_GROUP, DRAIN_GROUP } type;
  
  // IO level (1-3) - only for IO_GROUP
  int ioLevel = 0;
  
  // Direction
  bool isInput;   // True for input arrays (A, B)
  bool isOutput; // True for output arrays (C)
  
  // Phase 2: Parametric data flow direction
  SystolicFlowDir flowDirection = SystolicFlowDir::NONE;
  
  // Buffer information
  bool needsDoubleBuffer = false;
  SmallVector<int64_t, 3> bufferShape;
  
  // Tile information
  SmallVector<int64_t, 3> tileSizes;
  
  ArrayRefGroup(Value memref, StringRef name)
      : memref(memref), arrayName(name.str()), type(PE_GROUP),
        isInput(false), isOutput(false) {}
};

//===----------------------------------------------------------------------===//
// Array Reference Analysis
//===----------------------------------------------------------------------===//

/// Analyze array references in a function and group them.
/// This corresponds to AutoSA's group_array_references_io.
static LogicalResult analyzeArrayReferences(
    func::FuncOp func,
    std::vector<ArrayRefGroup> &groups) {
  
  groups.clear();
  
  // Map from memref to group
  llvm::DenseMap<Value, size_t> memrefToGroup;
  
  // Collect all load/store operations
  func.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      Value memref = loadOp.getMemRef();
      
      // Find or create group
      size_t groupIdx;
      if (auto it = memrefToGroup.find(memref); it != memrefToGroup.end()) {
        groupIdx = it->second;
      } else {
        // Create new group
        std::string name = "unknown";
        if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
          // Try to get name from function arguments
          for (auto arg : func.getArguments()) {
            if (arg == memref) {
              if (auto attr = func.getArgAttrOfType<StringAttr>(
                      arg.getArgNumber(), "mlir.name")) {
                name = attr.getValue().str();
              } else {
                name = "arg" + std::to_string(arg.getArgNumber());
              }
              break;
            }
          }
        }
        
        groupIdx = groups.size();
        groups.emplace_back(memref, name);
        memrefToGroup[memref] = groupIdx;
      }
      
      groups[groupIdx].loads.push_back(loadOp);
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      Value memref = storeOp.getMemRef();
      
      size_t groupIdx;
      if (auto it = memrefToGroup.find(memref); it != memrefToGroup.end()) {
        groupIdx = it->second;
      } else {
        std::string name = "unknown";
        if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
          for (auto arg : func.getArguments()) {
            if (arg == memref) {
              if (auto attr = func.getArgAttrOfType<StringAttr>(
                      arg.getArgNumber(), "mlir.name")) {
                name = attr.getValue().str();
              } else {
                name = "arg" + std::to_string(arg.getArgNumber());
              }
              break;
            }
          }
        }
        
        groupIdx = groups.size();
        groups.emplace_back(memref, name);
        memrefToGroup[memref] = groupIdx;
      }
      
      groups[groupIdx].stores.push_back(storeOp);
    }
  });
  
  // Classify groups as IO, PE, or Drain
  // Simple heuristic: if only loads -> IO, if only stores -> Drain,
  // if both -> PE (accumulator)
  for (auto &group : groups) {
    bool hasLoads = !group.loads.empty();
    bool hasStores = !group.stores.empty();
    
    if (hasLoads && !hasStores) {
      group.type = ArrayRefGroup::IO_GROUP;
      group.isInput = true;
    } else if (hasStores && !hasLoads) {
      group.type = ArrayRefGroup::DRAIN_GROUP;
      group.isOutput = true;
    } else if (hasLoads && hasStores) {
      // Read-modify-write pattern (e.g., C += A * B)
      group.type = ArrayRefGroup::PE_GROUP;
      group.isOutput = true;
    }
    
    // Determine IO level based on access pattern and loop nesting
    // Heuristic: Analyze the loop nesting depth where the array is accessed
    if (group.type == ArrayRefGroup::IO_GROUP || 
        group.type == ArrayRefGroup::DRAIN_GROUP) {
      
      // Find the minimum loop nesting depth for this array's accesses
      int minDepth = 1000;  // Large number
      int maxDepth = 0;
      
      for (auto loadOp : group.loads) {
        int depth = 0;
        Operation *parent = loadOp->getParentOp();
        while (parent) {
          if (isa<AffineForOp>(parent)) {
            depth++;
          }
          parent = parent->getParentOp();
        }
        minDepth = std::min(minDepth, depth);
        maxDepth = std::max(maxDepth, depth);
      }
      
      for (auto storeOp : group.stores) {
        int depth = 0;
        Operation *parent = storeOp->getParentOp();
        while (parent) {
          if (isa<AffineForOp>(parent)) {
            depth++;
          }
          parent = parent->getParentOp();
        }
        minDepth = std::min(minDepth, depth);
        maxDepth = std::max(maxDepth, depth);
      }
      
      // Determine IO level based on access depth:
      // - L1: Accesses at innermost loops (PE interface)
      // - L2: Accesses at middle loops (double buffering)
      // - L3: Accesses at outermost loops (global memory)
      // Typical MatMul: L3 (outermost) -> L2 (middle) -> L1 (innermost) -> PE
      if (minDepth >= 4) {
        // Access at very outer loops -> L3 (global memory interface)
        group.ioLevel = 3;
        group.needsDoubleBuffer = false;  // L3 typically doesn't need double buffer
      } else if (minDepth >= 2) {
        // Access at middle loops -> L2 (double buffering)
      group.ioLevel = 2;
        group.needsDoubleBuffer = true;  // L2 typically needs double buffering
      } else {
        // Access at inner loops -> L1 (PE interface)
        group.ioLevel = 1;
        group.needsDoubleBuffer = false;  // L1 typically doesn't need double buffer
      }
      
      LLVM_DEBUG(llvm::dbgs() << "  " << group.arrayName 
                              << ": minDepth=" << minDepth 
                              << ", maxDepth=" << maxDepth
                              << ", ioLevel=" << group.ioLevel << "\n");
    }
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "Array Reference Groups:\n";
    for (const auto &group : groups) {
      llvm::dbgs() << "  " << group.arrayName << ": type=";
      if (group.type == ArrayRefGroup::IO_GROUP) llvm::dbgs() << "IO";
      else if (group.type == ArrayRefGroup::PE_GROUP) llvm::dbgs() << "PE";
      else llvm::dbgs() << "DRAIN";
      llvm::dbgs() << ", level=" << group.ioLevel << "\n";
    }
  });
  
  return success();
}

//===----------------------------------------------------------------------===//
// SystolicDataflowGeneration Pass
//===----------------------------------------------------------------------===//

namespace {
struct SystolicDataflowGenerationPass
    : public PassWrapper<SystolicDataflowGenerationPass,
                        OperationPass<func::FuncOp>> {
  
  void runOnOperation() override;
  
  StringRef getArgument() const override {
    return "systolic-dataflow-generation";
  }
  
  StringRef getDescription() const override {
    return "Generate SystolicDataflow Dialect from Affine IR";
  }
};
} // namespace

void SystolicDataflowGenerationPass::runOnOperation() {
  func::FuncOp func = getOperation();
  
  // Step 1: Analyze array references
  std::vector<ArrayRefGroup> groups;
  if (failed(analyzeArrayReferences(func, groups))) {
    signalPassFailure();
    return;
  }
  
  if (groups.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No array references found, skipping\n");
    return;
  }
  
  // Step 1.5: Analyze write-time reordering opportunities
  // Run on kernel first; if kernel has no load/store (e.g. outlined to callee), run on callees and map attributes to kernel.
  auto setReorderAttrsFromPatterns = [&](const WriteTimeReorderingAnalyzer &analyzer, func::FuncOp targetFunc) {
    for (const auto &pattern : analyzer.getPatterns()) {
      if (!pattern.hasNonLinearAccess || pattern.reorderedDims.empty())
        continue;
      std::string arrayName = pattern.arrayName;
      std::string dimsAttrName = "systolic.reorder." + arrayName + ".dims";
      std::string permAttrName = "systolic.reorder." + arrayName + ".perm";
      SmallVector<Attribute, 3> dimAttrs;
      for (int64_t dim : pattern.reorderedDims)
        dimAttrs.push_back(IntegerAttr::get(IntegerType::get(func.getContext(), 64), dim));
      targetFunc->setAttr(dimsAttrName, ArrayAttr::get(func.getContext(), dimAttrs));
      SmallVector<Attribute, 3> permAttrs;
      for (unsigned perm : pattern.dimPermutation)
        permAttrs.push_back(IntegerAttr::get(IntegerType::get(func.getContext(), 32), perm));
      LLVM_DEBUG(llvm::dbgs() << "Stored reordering for " << arrayName << " (dims/perm)\n");
    }
  };

  WriteTimeReorderingAnalyzer reorderingAnalyzer(func);
  if (failed(reorderingAnalyzer.analyze())) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Failed to analyze write-time reordering on kernel\n");
  } else {
    setReorderAttrsFromPatterns(reorderingAnalyzer, func);
  }

  // If kernel has no reorder patterns (e.g. body is only calls after transform), analyze callees
  bool hasAnyReorder = false;
  for (const auto &it : llvm::make_early_inc_range(func->getAttrs())) {
    if (it.getName().getValue().starts_with("systolic.reorder."))
      hasAnyReorder = true;
  }
  if (!hasAnyReorder) {
    func.walk([&](func::CallOp callOp) {
      if (hasAnyReorder)
        return;
      auto callee = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
      if (!callee)
        return;
      func::FuncOp calleeFunc = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(func->getParentOfType<ModuleOp>(), callee.getRootReference()));
      if (!calleeFunc || calleeFunc.getBody().empty())
        return;
      WriteTimeReorderingAnalyzer calleeAnalyzer(calleeFunc);
      if (failed(calleeAnalyzer.analyze()))
        return;
      for (const auto &pattern : calleeAnalyzer.getPatterns()) {
        if (!pattern.hasNonLinearAccess || pattern.reorderedDims.empty())
          continue;
        hasAnyReorder = true;
        break;
      }
      if (!hasAnyReorder)
        return;
      // Map callee arg index -> kernel value for attribute name
      auto getKernelArgNameForCalleeArg = [&](Value calleeArg) -> Value {
        if (!calleeArg.isa<BlockArgument>())
          return nullptr;
        unsigned calleeIdx = calleeArg.cast<BlockArgument>().getArgNumber();
        if (calleeIdx >= callOp.getNumOperands())
          return nullptr;
        return callOp.getOperand(calleeIdx);
      };
      for (const auto &pattern : calleeAnalyzer.getPatterns()) {
        if (!pattern.hasNonLinearAccess || pattern.reorderedDims.empty())
          continue;
        std::string arrayName;
        if (pattern.memref.isa<BlockArgument>()) {
          Value kernelVal = getKernelArgNameForCalleeArg(pattern.memref);
          if (kernelVal && kernelVal.isa<BlockArgument>()) {
            unsigned kernelIdx = kernelVal.cast<BlockArgument>().getArgNumber();
            if (auto attr = func.getArgAttrOfType<StringAttr>(kernelIdx, "mlir.name"))
              arrayName = attr.getValue().str();
            else
              arrayName = "arg" + std::to_string(kernelIdx);
          } else {
            arrayName = pattern.arrayName;
          }
        } else {
          arrayName = pattern.arrayName;
        }
        if (arrayName.empty())
          continue;
        SmallVector<Attribute, 3> dimAttrs;
        for (int64_t dim : pattern.reorderedDims)
          dimAttrs.push_back(IntegerAttr::get(IntegerType::get(func.getContext(), 64), dim));
        func->setAttr("systolic.reorder." + arrayName + ".dims", ArrayAttr::get(func.getContext(), dimAttrs));
        SmallVector<Attribute, 3> permAttrs;
        for (unsigned perm : pattern.dimPermutation)
          permAttrs.push_back(IntegerAttr::get(IntegerType::get(func.getContext(), 32), perm));
        func->setAttr("systolic.reorder." + arrayName + ".perm", ArrayAttr::get(func.getContext(), permAttrs));
        LLVM_DEBUG(llvm::dbgs() << "Stored reordering from callee for " << arrayName << "\n");
      }
    });
  }
  
  // Step 2: Extract configuration from function attributes (set by SystolicTransform)
  SmallVector<int64_t, 2> peArraySize = {2, 2};  // Default
  SmallVector<int64_t, 2> tileSize = {8, 8};     // Default (latency factors)
  SmallVector<int64_t, 3> arrayPart = {16, 16, 16};  // Default
  SmallVector<int64_t, 2> latency = {8, 8};      // Default
  
  // Try to read from function attributes first (set by SystolicTransform)
  if (auto peArrayAttr = func->getAttrOfType<ArrayAttr>("systolic.pe_array_size")) {
    peArraySize.clear();
    for (auto attr : peArrayAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        peArraySize.push_back(intAttr.getInt());
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Read PE array size from attributes: ["
                            << peArraySize[0] << ", " << peArraySize[1] << "]\n");
  }
  
  if (auto latencyAttr = func->getAttrOfType<ArrayAttr>("systolic.latency")) {
    tileSize.clear();
    latency.clear();
    for (auto attr : latencyAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        int64_t val = intAttr.getInt();
        tileSize.push_back(val);
        latency.push_back(val);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Read latency/tile size from attributes: ["
                            << tileSize[0] << ", " << tileSize[1] << "]\n");
  }
  
  if (auto arrayPartAttr = func->getAttrOfType<ArrayAttr>("systolic.array_part")) {
    arrayPart.clear();
    for (auto attr : arrayPartAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        arrayPart.push_back(intAttr.getInt());
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Read array_part from attributes: ["
                            << arrayPart[0] << ", " << arrayPart[1] << ", "
                            << arrayPart[2] << "]\n");
  }
  
  // Phase 2: Parametric data flow analysis
  // Run space-time analysis to determine flow directions for each array
  DenseMap<Value, SystolicFlowDir> operandFlows;
  
  // Find outermost loop for analysis
  AffineForOp outermostLoop = nullptr;
  func.walk([&](AffineForOp forOp) {
    if (!forOp->getParentOfType<AffineForOp>()) {
      outermostLoop = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (outermostLoop) {
    // Get space-time mode from attributes (default to 3 for ST3)
    unsigned spaceTimeMode = 3;
    if (auto modeAttr = func->getAttrOfType<IntegerAttr>("systolic.space_time_mode")) {
      spaceTimeMode = modeAttr.getInt();
    }
    
    // Create parametric configuration from mode
    ParametricSpaceTime parametricConfig = ParametricSpaceTime::createFromMode(spaceTimeMode);
    
    if (parametricConfig.isValid()) {
      // Get loop nest
      SmallVector<AffineForOp, 8> loops;
      AffineForOp current = outermostLoop;
      while (current) {
        loops.push_back(current);
        AffineForOp inner = nullptr;
        for (auto &op : *current.getBody()) {
          if (auto nestedFor = dyn_cast<AffineForOp>(op)) {
            inner = nestedFor;
            break;
          }
        }
        current = inner;
      }
      
      // Run parametric data flow analysis
      if (loops.size() >= 3) {
        if (succeeded(analyzeOperandFlowsParametric(outermostLoop, loops,
                                                     parametricConfig, operandFlows))) {
          LLVM_DEBUG(llvm::dbgs() << "[DataflowGen] Parametric flow analysis succeeded\n");
          LLVM_DEBUG({
            llvm::dbgs() << "  Flow directions:\n";
            for (auto &entry : operandFlows) {
              llvm::dbgs() << "    Memref: ";
              switch (entry.second) {
                case SystolicFlowDir::HORIZONTAL:
                  llvm::dbgs() << "HORIZONTAL\n";
                  break;
                case SystolicFlowDir::VERTICAL:
                  llvm::dbgs() << "VERTICAL\n";
                  break;
                case SystolicFlowDir::NONE:
                  llvm::dbgs() << "NONE (local)\n";
                  break;
                default:
                  llvm::dbgs() << "UNKNOWN\n";
              }
            }
          });
          
          // Populate flow directions in groups
          for (auto &group : groups) {
            auto it = operandFlows.find(group.memref);
            if (it != operandFlows.end()) {
              group.flowDirection = it->second;
              LLVM_DEBUG(llvm::dbgs() << "  " << group.arrayName 
                                      << " flow: " << (int)group.flowDirection << "\n");
            }
          }
        } else {
          LLVM_DEBUG(llvm::dbgs() 
              << "[DataflowGen] Parametric flow analysis failed, using defaults\n");
        }
      }
    }
  }
  
  // Fallback: Try to infer from loop structure if attributes not available
  if (peArraySize[0] == 2 && peArraySize[1] == 2) {  // Still using defaults
  AffineForOp outermostLoop = nullptr;
  func.walk([&](AffineForOp forOp) {
    if (!forOp->getParentOfType<AffineForOp>()) {
      outermostLoop = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (outermostLoop) {
    SmallVector<AffineForOp, 3> loopNest;
    AffineForOp current = outermostLoop;
    while (current && loopNest.size() < 3) {
      loopNest.push_back(current);
      AffineForOp inner = nullptr;
      for (auto &op : *current.getBody()) {
        if (auto nestedFor = dyn_cast<AffineForOp>(op)) {
          inner = nestedFor;
          break;
        }
      }
      current = inner;
    }
    
    if (loopNest.size() >= 2) {
      if (loopNest[0].hasConstantUpperBound() && loopNest[1].hasConstantUpperBound()) {
        int64_t bound0 = loopNest[0].getConstantUpperBound();
        int64_t bound1 = loopNest[1].getConstantUpperBound();
        peArraySize[0] = std::max((int64_t)1, bound0 / 8);
        peArraySize[1] = std::max((int64_t)1, bound1 / 8);
          LLVM_DEBUG(llvm::dbgs() << "Inferred PE array size from loop bounds: ["
                                  << peArraySize[0] << ", " << peArraySize[1] << "]\n");
        }
      }
    }
  }
  
  // Find the innermost computation loop nest (for PE array)
  AffineForOp innermostLoop = nullptr;
  int maxDepth = -1;
  func.walk([&](AffineForOp forOp) {
    int depth = 0;
    Operation *parent = forOp->getParentOp();
    while (parent) {
      if (isa<AffineForOp>(parent)) {
        depth++;
      }
      parent = parent->getParentOp();
      }
    if (depth > maxDepth) {
      maxDepth = depth;
      innermostLoop = forOp;
    }
  });
  
  // Ensure all necessary dialects are loaded before creating operations
  MLIRContext *ctx = func.getContext();
  ctx->getOrLoadDialect<affine::AffineDialect>();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  ctx->getOrLoadDialect<memref::MemRefDialect>();
  ctx->getOrLoadDialect<dataflow::SystolicDataflowDialect>();
  
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&func.getBody().front());
  
  Location loc = func.getLoc();
  
  // Step 3: Generate IO modules for input arrays (L1, L2, L3)
  // Store created IO modules for later use
  llvm::DenseMap<Value, dataflow::IOModuleOp> ioModules;
  
  // Save the original insertion point (at function level)
  OpBuilder::InsertPoint funcInsertPoint = builder.saveInsertionPoint();
  
  for (const auto &group : groups) {
    if (group.type == ArrayRefGroup::IO_GROUP && group.ioLevel > 0) {
      // Restore insertion point to function level for each new IO module
      builder.restoreInsertionPoint(funcInsertPoint);
      
      // Create IO module
      auto ioModule = builder.create<dataflow::IOModuleOp>(
          loc,
          /*level=*/builder.getI32IntegerAttr(group.ioLevel),
          /*direction=*/builder.getStringAttr(group.isInput ? "in" : "out"),
          /*arrayName=*/builder.getStringAttr(group.arrayName),
          /*bufferShape=*/group.bufferShape.empty() 
              ? ArrayAttr() 
              : builder.getI64ArrayAttr(group.bufferShape),
          /*doubleBuffer=*/group.needsDoubleBuffer
              ? builder.getBoolAttr(true)
              : BoolAttr(),
          /*name=*/StringAttr());
      
      // Create body block
      Block *bodyBlock = builder.createBlock(&ioModule.getBody());
      builder.setInsertionPointToStart(bodyBlock);
      
      // Generate IO module content based on level
      if (group.ioLevel == 2 && group.needsDoubleBuffer) {
        // L2 with double buffering: Create DoubleBufferOp
        // Allocate ping and pong buffers
        MemRefType bufferType = group.memref.getType().cast<MemRefType>();
        SmallVector<int64_t, 3> bufferShape;
        for (int64_t dim : bufferType.getShape()) {
          bufferShape.push_back(dim);
        }
        
        // Adjust buffer shape based on tile size
        if (bufferShape.size() >= 2 && tileSize.size() >= 2) {
          bufferShape[0] = tileSize[0];
          bufferShape[1] = tileSize[1];
        }
        
        MemRefType pingType = MemRefType::get(bufferShape, bufferType.getElementType());
        MemRefType pongType = MemRefType::get(bufferShape, bufferType.getElementType());
        MemRefType arbiterType = MemRefType::get({}, builder.getI1Type());
        MemRefType intraEnableType = MemRefType::get({}, builder.getI1Type());
        
        // Allocate buffers (using memref.alloc for now, will be converted later)
        auto pingBuffer = builder.create<memref::AllocOp>(loc, pingType);
        auto pongBuffer = builder.create<memref::AllocOp>(loc, pongType);
        auto arbiter = builder.create<memref::AllocOp>(loc, arbiterType);
        auto intraEnable = builder.create<memref::AllocOp>(loc, intraEnableType);
        
        // Initialize arbiter to false (ping is active)
        auto falseVal = builder.create<arith::ConstantOp>(
            loc, builder.getBoolAttr(false));
        builder.create<memref::StoreOp>(loc, falseVal, arbiter);
        
        // Initialize intraEnable to false
        builder.create<memref::StoreOp>(loc, falseVal, intraEnable);
        
        // Create DoubleBufferOp
        auto doubleBuffer = builder.create<dataflow::DoubleBufferOp>(
            loc,
            pingBuffer, pongBuffer, arbiter, intraEnable);
        
        // Create inter-transfer region (loading data)
        Block *interBlock = builder.createBlock(&doubleBuffer.getInterTransfer());
        builder.setInsertionPointToStart(interBlock);
        
        // Generate load operations for the IO group
        // For now, create a simple loop to load data
        // TODO: Generate proper load logic based on access pattern
        builder.create<dataflow::DoubleBufferYieldOp>(loc);
        
        // Create intra-transfer region (sending data to PE)
        Block *intraBlock = builder.createBlock(&doubleBuffer.getIntraTransfer());
        builder.setInsertionPointToStart(intraBlock);
        
        // Generate send operations
        // TODO: Generate proper send logic
        builder.create<dataflow::DoubleBufferYieldOp>(loc);
        
        // Explicitly add terminator to IOModule body block
        builder.setInsertionPointToEnd(bodyBlock);
        builder.create<dataflow::IOModuleYieldOp>(loc);
        
        LLVM_DEBUG(llvm::dbgs() << "Created IO module with double buffering for " 
                                << group.arrayName << " at level " << group.ioLevel << "\n");
      } else {
        // L1 or L3: Simple IO module without double buffering
        // Generate load/store operations based on access pattern
        if (group.isInput && !group.loads.empty()) {
          // For input arrays, generate load operations
          // TODO: Generate proper load logic based on access pattern
        }
        
        // Explicitly add terminator to IOModule body block
        builder.setInsertionPointToEnd(bodyBlock);
        builder.create<dataflow::IOModuleYieldOp>(loc);
      
        LLVM_DEBUG(llvm::dbgs() << "Created IO module for " << group.arrayName
                                << " at level " << group.ioLevel << "\n");
      }
      
      ioModules[group.memref] = ioModule;
      
      // Restore insertion point to function level after completing this IO module
      builder.restoreInsertionPoint(funcInsertPoint);
    }
  }
  
  // Step 4: Generate PE Array
  // Find PE group (accumulator arrays with both loads and stores)
  bool hasPEGroup = false;
  for (auto &group : groups) {
    if (group.type == ArrayRefGroup::PE_GROUP) {
      hasPEGroup = true;
      break;
    }
  }
  
  dataflow::PEArrayOp peArray;
  if (hasPEGroup) {
    peArray = builder.create<dataflow::PEArrayOp>(
        loc,
        /*arraySize=*/builder.getI64ArrayAttr(peArraySize),
        /*tileSize=*/builder.getI64ArrayAttr(tileSize),
        /*name=*/StringAttr());
    
    // Create body block
    Block *bodyBlock = builder.createBlock(&peArray.getBody());
    builder.setInsertionPointToStart(bodyBlock);
    
    // Migrate computation loop body to PE array
    if (innermostLoop) {
      // Find the outermost loop that contains the computation
      // Typically, after SystolicTransform, we have a tiled loop nest
      // We want to clone the innermost computation loops (typically the k-loop for MatMul)
      
      // Find the loop nest starting from innermost
      SmallVector<AffineForOp, 4> loopNest;
      AffineForOp current = innermostLoop;
      while (current && loopNest.size() < 4) {
        loopNest.push_back(current);
        // Find parent loop
        Operation *parent = current->getParentOp();
        while (parent && !isa<AffineForOp>(parent)) {
          parent = parent->getParentOp();
        }
        if (auto parentFor = dyn_cast_or_null<AffineForOp>(parent)) {
          current = parentFor;
        } else {
          break;
        }
      }
      
      // Reverse to get from outermost to innermost
      std::reverse(loopNest.begin(), loopNest.end());
      
      // Clone the innermost 1-2 loops (computation loops, typically k-loop for MatMul)
      // Skip the outer tile loops as they are handled by the PE array structure
      IRMapping mapping;
      int startIdx = std::max(0, (int)loopNest.size() - 2);  // Clone last 2 loops
      
      AffineForOp lastClonedLoop;
      for (int i = startIdx; i < (int)loopNest.size(); ++i) {
        AffineForOp srcLoop = loopNest[i];
        
        // Clone the loop
        auto clonedLoop = builder.create<AffineForOp>(
            loc,
            srcLoop.getLowerBoundOperands(),
            srcLoop.getLowerBoundMap(),
            srcLoop.getUpperBoundOperands(),
            srcLoop.getUpperBoundMap(),
            srcLoop.getStep());
        
        mapping.map(srcLoop.getInductionVar(), clonedLoop.getInductionVar());
        
        // Clone loop body operations (excluding nested loops and yield)
        builder.setInsertionPointToStart(clonedLoop.getBody());
        for (auto &op : srcLoop.getBody()->getOperations()) {
          if (isa<AffineYieldOp>(op)) {
            // Skip yield, will be added at the end
            continue;
          }
          if (auto nestedFor = dyn_cast<AffineForOp>(op)) {
            // Skip nested loops that we're not cloning
            if (i + 1 < (int)loopNest.size() && nestedFor == loopNest[i + 1]) {
              // This is the next loop we'll clone, skip it here
              continue;
            }
          }
          builder.clone(op, mapping);
        }
        
        lastClonedLoop = clonedLoop;
      }
      
      // If no loops were cloned, clone the innermost loop body operations
      if (!lastClonedLoop) {
        for (auto &op : innermostLoop.getBody()->getOperations()) {
          if (!isa<AffineYieldOp>(op) && !isa<AffineForOp>(op)) {
            builder.clone(op, mapping);
          }
        }
      }
    } else {
      // No loop found, create empty PE array body
      LLVM_DEBUG(llvm::dbgs() << "Warning: No innermost loop found for PE array\n");
    }
    
    // Create yield terminator
    builder.setInsertionPointToEnd(bodyBlock);
    builder.create<dataflow::PEArrayYieldOp>(loc);
    
    LLVM_DEBUG(llvm::dbgs() << "Created PE array with size [" 
                            << peArraySize[0] << ", " << peArraySize[1] << "]\n");
  }
  
  // Step 5: Generate Drain modules for output arrays
  for (const auto &group : groups) {
    if (group.type == ArrayRefGroup::DRAIN_GROUP) {
      // Determine drain level (typically L2 for output)
      int drainLevel = group.ioLevel > 0 ? group.ioLevel : 2;
      
      auto drainModule = builder.create<dataflow::DrainModuleOp>(
          loc,
          /*level=*/builder.getI32IntegerAttr(drainLevel),
          /*arrayName=*/builder.getStringAttr(group.arrayName),
          /*bufferShape=*/group.bufferShape.empty()
              ? ArrayAttr()
              : builder.getI64ArrayAttr(group.bufferShape),
          /*name=*/StringAttr());
      
      // Create body block
      Block *bodyBlock = builder.createBlock(&drainModule.getBody());
      builder.setInsertionPointToStart(bodyBlock);
      
      // Generate drain logic: collect results from PE array and write to memory
      // For now, create a placeholder
      // TODO: Generate proper drain logic based on store operations
      if (!group.stores.empty()) {
        // TODO: Map values correctly and clone store operations
        LLVM_DEBUG(llvm::dbgs() << "  Drain module will handle "
                                << group.stores.size() << " store(s) for "
                                << group.arrayName << "\n");
      }
      
      // Create yield terminator
      builder.create<dataflow::DrainModuleYieldOp>(loc);
      
      LLVM_DEBUG(llvm::dbgs() << "Created drain module for " << group.arrayName
                              << " at level " << drainLevel << "\n");
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createSystolicDataflowGenerationPass() {
  return std::make_unique<SystolicDataflowGenerationPass>();
}

// Register the pass
static PassRegistration<SystolicDataflowGenerationPass> passRegistration;

//===----------------------------------------------------------------------===//
// SystolicWriteReorderAnalysisPass - run before transform so store indices
// (e.g. affine.apply) are still visible; attributes are preserved across transform.
//===----------------------------------------------------------------------===//
struct SystolicWriteReorderAnalysisPass
    : public PassWrapper<SystolicWriteReorderAnalysisPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SystolicWriteReorderAnalysisPass)
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    WriteTimeReorderingAnalyzer analyzer(func);
    if (failed(analyzer.analyze()))
      return;
    for (const auto &pattern : analyzer.getPatterns()) {
      if (!pattern.hasNonLinearAccess || pattern.reorderedDims.empty())
        continue;
      std::string dimsAttrName = "systolic.reorder." + pattern.arrayName + ".dims";
      std::string permAttrName = "systolic.reorder." + pattern.arrayName + ".perm";
      SmallVector<Attribute, 3> dimAttrs;
      for (int64_t dim : pattern.reorderedDims)
        dimAttrs.push_back(IntegerAttr::get(IntegerType::get(func.getContext(), 64), dim));
      func->setAttr(dimsAttrName, ArrayAttr::get(func.getContext(), dimAttrs));
      SmallVector<Attribute, 3> permAttrs;
      for (unsigned perm : pattern.dimPermutation)
        permAttrs.push_back(IntegerAttr::get(IntegerType::get(func.getContext(), 32), perm));
      func->setAttr(permAttrName, ArrayAttr::get(func.getContext(), permAttrs));
    }
  }
  StringRef getArgument() const override { return "systolic-write-reorder-analysis"; }
  StringRef getDescription() const override {
    return "Analyze write-time reordering and set systolic.reorder.* attributes (run before transform)";
  }
};

static PassRegistration<SystolicWriteReorderAnalysisPass> writeReorderAnalysisRegistration;

std::unique_ptr<Pass> createSystolicWriteReorderAnalysisPass() {
  return std::make_unique<SystolicWriteReorderAnalysisPass>();
}

} // namespace systolic
} // namespace mlir

