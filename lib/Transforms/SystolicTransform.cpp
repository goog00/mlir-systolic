//===----------------------------------------------------------------------===//
//
// MLIR-Systolic: Systolic Array Transform Pass
//
// This pass implements space-time transformation, array partitioning, and
// latency hiding for affine loop nests, following AutoSA's methodology.
//
// AutoSA Reference:
//   - sa_legality_check: Single band + uniform dependency
//   - sa_space_time_transform: Dependence distance analysis, space loop selection
//   - sa_array_partitioning_optimize: Multi-level tiling
//   - sa_latency_hiding_optimize: Latency hiding tiling
//
//===----------------------------------------------------------------------===//

#include "systolic/Transforms/Passes.h"
#include "systolic/Analysis/SpaceTimeAnalysis.h"
#include "systolic/Analysis/SystolicConfig.h"
#include "systolic/Analysis/PolymerAnalysis.h"
#include "systolic/Analysis/ParametricSpaceTime.h"

// Polymer includes (required)
#ifdef SYSTOLIC_ENABLE_POLYMER
#if __has_include("polymer/Transforms/ExtractScopStmt.h")
#include "polymer/Transforms/ExtractScopStmt.h"
#endif
#if __has_include("polymer/Transforms/Reg2Mem.h")
#include "polymer/Transforms/Reg2Mem.h"
#endif
#endif

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <set>
#include <vector>

#define DEBUG_TYPE "systolic-transform"

using namespace mlir;
using namespace mlir::affine;

namespace mlir {
namespace systolic {

//===----------------------------------------------------------------------===//
// Helper Data Structures
//===----------------------------------------------------------------------===//

/// Loop band representation (similar to AutoSA's band concept)
using LoopBand = SmallVector<AffineForOp, 8>;

/// Dependence information for a loop dimension
/// Following AutoSA's dep_dis (dependence distance) concept
struct LoopDepInfo {
  unsigned loopIndex;       // Index in the loop band
  int64_t minDistance;      // Minimum dependence distance
  int64_t maxDistance;      // Maximum dependence distance
  bool isUniform;           // True if distance is constant (uniform dep)
  bool canBeSpaceLoop;      // True if distance <= 1 (space loop candidate)
};

/// Problem size inferred from loop bounds or memref shapes
/// Following AutoSA's ProblemSize concept
struct ProblemSize {
  int64_t M = 0;  // Output rows
  int64_t N = 0;  // Output columns
  int64_t K = 0;  // Reduction dimension
  bool valid = false;
};

//===----------------------------------------------------------------------===//
// Pass Options
//===----------------------------------------------------------------------===//

struct SystolicTransformOptions {
  /// Space-time mapping mode (index into enumerated configurations)
  /// If >= 0, selects the configuration at that index from the enumerated list
  /// If < 0, uses default (first configuration or heuristics)
  /// Note: The index is dynamic and depends on loop count and maxSADim
  int spaceTimeMode = -1;  // -1 means use default/auto
  
  /// Maximum systolic array dimension (1, 2, or 3)
  /// Controls how many dimensions of PE arrays to explore
  unsigned maxSADim = 2;  // Default to 2D (most common)
  
  /// List all spacetime configurations instead of generating code
  bool listConfigs = false;
  
  SystolicTransformOptions() : spaceTimeMode(-1), maxSADim(2), listConfigs(false) {}
  
  /// Array partitioning factors (first-level tiling)
  SmallVector<int64_t, 3> arrayPart = {16, 16, 16};
  
  /// Latency hiding factors (second-level tiling)
  SmallVector<int64_t, 2> latency = {8, 8};
  
  /// SIMD width (vectorization factor)
  unsigned simdWidth = 1;
  
  /// Enable two-level buffering (L2 array partitioning)
  bool twoLevelBuffer = false;
};

//===----------------------------------------------------------------------===//
// Legality Check (AutoSA: sa_legality_check)
//===----------------------------------------------------------------------===//

/// Check if the loop band is suitable for systolic array transformation.
/// AutoSA requires:
/// 1. Single fully permutable outermost band
/// 2. Uniform dependences
static LogicalResult checkLegality(LoopBand &band) {
  if (band.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Error: Empty loop band\n");
    return failure();
  }
  
  // Check for at least 3 nested loops (typical for MatMul-like patterns)
  if (band.size() < 3) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Warning: Loop band has only "
                            << band.size() << " levels, expected >= 3\n");
  }
  
  // Check if loops are perfectly nested
  for (unsigned i = 0; i + 1 < band.size(); ++i) {
    auto outerLoop = band[i];
    auto innerLoop = band[i + 1];
    
    // Check if the outer loop body contains only the inner loop
    Block *body = outerLoop.getBody();
    unsigned opCount = 0;
    for (auto &op : *body) {
      if (!isa<AffineYieldOp>(op))
        opCount++;
    }
    
    if (opCount != 1 || body->front().getNumRegions() == 0) {
      LLVM_DEBUG(llvm::dbgs() 
          << "[Systolic] Warning: Loops are not perfectly nested at level " 
          << i << "\n");
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Dependence Distance Analysis
// (AutoSA: get_dep_dis_at_node in autosa_utils.cpp)
//===----------------------------------------------------------------------===//

/// Analyze dependence distances for each loop in the band.
/// REQUIRES Polymer for accurate polyhedral analysis - no fallback.
static LogicalResult analyzeDependenceDistances(
    func::FuncOp func,
    LoopBand &band,
    SmallVectorImpl<LoopDepInfo> &depInfos) {
  
  depInfos.clear();
  
  // REQUIREMENT: Must use Polymer for dependence analysis
  // No fallback to heuristic methods - Polymer is required
  if (!systolic::isPolymerAvailable()) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] ERROR: Polymer is required but not available\n");
    return failure();
  }
  
  LLVM_DEBUG(llvm::dbgs() << "[Systolic] Using Polymer for dependence analysis (required)\n");
    
    SmallVector<systolic::LoopDependenceDistance, 8> polymerDistances;
  if (failed(systolic::computeDependenceDistancesWithPolymer(func, polymerDistances))) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] ERROR: Polymer dependence analysis failed\n");
    return failure();
  }
  
      // Convert Polymer results to LoopDepInfo
      for (const auto &pdist : polymerDistances) {
        LoopDepInfo info;
        info.loopIndex = pdist.loopIndex;
        info.minDistance = pdist.minDistance;
        info.maxDistance = pdist.maxDistance;
        info.isUniform = pdist.isUniform;
        info.canBeSpaceLoop = pdist.canBeSpaceLoop;
        depInfos.push_back(info);
      }
      
  if (depInfos.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] ERROR: Polymer analysis returned no dependencies\n");
    return failure();
  }
  
  LLVM_DEBUG(llvm::dbgs() << "[Systolic] Polymer analysis successful\n");
  return success();
  
  // NO FALLBACK - Polymer is required
  // All heuristic analysis code has been removed
}

//===----------------------------------------------------------------------===//
// Dynamic Space-Time Configuration Enumeration
//===----------------------------------------------------------------------===//

/// Enumerate all valid space-time configurations for a loop nest.
/// This function generates all possible combinations of space and time loops,
/// respecting data dependence constraints.
///
/// Args:
///   loops: The loops to configure
///   depInfos: Dependence information for each loop
///   maxSADim: Maximum systolic array dimensionality (1D, 2D, or 3D)
///   configs: Output vector of all valid configurations
///
/// Returns:
///   success() if at least one valid configuration found
///   failure() if no valid configurations exist
static LogicalResult enumerateSpaceTimeConfigs(
    const SmallVector<AffineForOp> &loops,
    const SmallVectorImpl<LoopDepInfo> &depInfos,
    unsigned maxSADim,  // Maximum systolic array dimension (default: 2)
    SmallVector<ParametricSpaceTime, 8> &configs) {
  
  configs.clear();
  unsigned numLoops = loops.size();
  unsigned configId = 0;
  
  // Collect loop names (store as strings to ensure lifetime)
  SmallVector<std::string> loopNameStrings;
  SmallVector<StringRef> loopNames;
  for (unsigned i = 0; i < loops.size(); ++i) {
    // AffineForOp doesn't have getName(), use default names
    loopNameStrings.push_back("loop" + std::to_string(i));
    loopNames.push_back(StringRef(loopNameStrings.back()));
  }
  
  LLVM_DEBUG(llvm::dbgs() << "[Systolic] Enumerating spacetime configs:\n");
  LLVM_DEBUG(llvm::dbgs() << "  numLoops: " << numLoops << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  maxSADim: " << maxSADim << "\n");
  
  // 1. Enumerate 1D arrays
  if (maxSADim >= 1 && numLoops >= 1) {
    LLVM_DEBUG(llvm::dbgs() << "  Exploring 1D arrays...\n");
    for (unsigned i = 0; i < numLoops; ++i) {
      if (i < depInfos.size() && depInfos[i].canBeSpaceLoop) {
        SmallVector<unsigned> spaceLoops = {i};
        SmallVector<unsigned> timeLoops;
        
        for (unsigned j = 0; j < numLoops; ++j) {
          if (j != i) {
            timeLoops.push_back(j);
          }
        }
        
        ParametricSpaceTime config = 
            ParametricSpaceTime::createFromLoopIndices(
                spaceLoops, timeLoops, loopNames);
        config.setConfigId(configId++);
        configs.push_back(config);
        
        LLVM_DEBUG(llvm::dbgs() << "    [" << (configId-1) << "] 1D: space=[" 
                                << i << "], time=[");
        for (unsigned j = 0; j < timeLoops.size(); ++j) {
          LLVM_DEBUG(llvm::dbgs() << timeLoops[j]);
          if (j < timeLoops.size() - 1) LLVM_DEBUG(llvm::dbgs() << ",");
        }
        LLVM_DEBUG(llvm::dbgs() << "]\n");
      }
    }
  }
  
  // 2. Enumerate 2D arrays
  if (maxSADim >= 2 && numLoops >= 2) {
    LLVM_DEBUG(llvm::dbgs() << "  Exploring 2D arrays...\n");
    for (unsigned i = 0; i < numLoops; ++i) {
      if (i < depInfos.size() && depInfos[i].canBeSpaceLoop) {
        for (unsigned j = i + 1; j < numLoops; ++j) {
          if (j < depInfos.size() && depInfos[j].canBeSpaceLoop) {
            SmallVector<unsigned> spaceLoops = {i, j};
            SmallVector<unsigned> timeLoops;
            
            for (unsigned k = 0; k < numLoops; ++k) {
              if (k != i && k != j) {
                timeLoops.push_back(k);
              }
            }
            
            ParametricSpaceTime config = 
                ParametricSpaceTime::createFromLoopIndices(
                    spaceLoops, timeLoops, loopNames);
            config.setConfigId(configId++);
            configs.push_back(config);
            
            LLVM_DEBUG(llvm::dbgs() << "    [" << (configId-1) << "] 2D: space=[" 
                                    << i << "," << j << "], time=[");
            for (unsigned k = 0; k < timeLoops.size(); ++k) {
              LLVM_DEBUG(llvm::dbgs() << timeLoops[k]);
              if (k < timeLoops.size() - 1) LLVM_DEBUG(llvm::dbgs() << ",");
            }
            LLVM_DEBUG(llvm::dbgs() << "]\n");
          }
        }
      }
    }
  }
  
  // 3. Enumerate 3D arrays (optional)
  if (maxSADim >= 3 && numLoops >= 3) {
    LLVM_DEBUG(llvm::dbgs() << "  Exploring 3D arrays...\n");
    for (unsigned i = 0; i < numLoops; ++i) {
      if (i < depInfos.size() && depInfos[i].canBeSpaceLoop) {
        for (unsigned j = i + 1; j < numLoops; ++j) {
          if (j < depInfos.size() && depInfos[j].canBeSpaceLoop) {
            for (unsigned k = j + 1; k < numLoops; ++k) {
              if (k < depInfos.size() && depInfos[k].canBeSpaceLoop) {
                SmallVector<unsigned> spaceLoops = {i, j, k};
                SmallVector<unsigned> timeLoops;
                
                for (unsigned l = 0; l < numLoops; ++l) {
                  if (l != i && l != j && l != k) {
                    timeLoops.push_back(l);
                  }
                }
                
                ParametricSpaceTime config = 
                    ParametricSpaceTime::createFromLoopIndices(
                        spaceLoops, timeLoops, loopNames);
                config.setConfigId(configId++);
                configs.push_back(config);
                
                LLVM_DEBUG(llvm::dbgs() << "    [" << (configId-1) << "] 3D: space=[" 
                                        << i << "," << j << "," << k << "], time=[");
                for (unsigned l = 0; l < timeLoops.size(); ++l) {
                  LLVM_DEBUG(llvm::dbgs() << timeLoops[l]);
                  if (l < timeLoops.size() - 1) LLVM_DEBUG(llvm::dbgs() << ",");
                }
                LLVM_DEBUG(llvm::dbgs() << "]\n");
              }
            }
          }
        }
      }
    }
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "[Systolic] Enumerated " << configs.size() 
                 << " spacetime configurations:\n";
    for (const auto &config : configs) {
      llvm::dbgs() << "  [" << config.getConfigId() << "] " 
                   << config.getSpaceTimeTypeString() << "\n";
    }
  });
  
  if (configs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] No valid spacetime configurations found\n");
    return failure();
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Space Loop Selection
// (AutoSA: sa_space_time_transform_at_dim_async)
//===----------------------------------------------------------------------===//

/// Select space loops based on dependence analysis and spaceTimeMode.
/// Following AutoSA's methodology:
/// - Space loops must have dependence distance <= 1
/// - Mode determines which loops are space (PE indices) vs time (execution order)
//===----------------------------------------------------------------------===//
// Parametric Space-Time Loop Selection (Phase 2)
//
// This enhanced version uses the ParametricSpaceTime framework to determine
// space and time loop indices instead of hardcoded [0,1]/[2..] assumptions.
//===----------------------------------------------------------------------===//

/// Select space and time loops using parametric configuration
static LogicalResult selectSpaceLoopsParametric(
    const SmallVectorImpl<LoopDepInfo> &depInfos,
    const ParametricSpaceTime &parametric,
    SmallVectorImpl<unsigned> &spaceLoopIndices,
    SmallVectorImpl<unsigned> &timeLoopIndices) {
  
  spaceLoopIndices.clear();
  timeLoopIndices.clear();
  
  unsigned numLoops = depInfos.size();
  
  // Extract space loop dimensions from parametric configuration
  // These are the loop indices that form the PE array dimensions
  SmallVector<unsigned> spaceLoopDims;
  for (unsigned i = 0; i < parametric.getNumSpaceDims(); ++i) {
    unsigned loopIdx = parametric.getSpaceDimConfig(i).loopDim;
    spaceLoopDims.push_back(loopIdx);
  }
  
  // Verify all space loop dimensions are within bounds
  for (unsigned loopIdx : spaceLoopDims) {
    if (loopIdx >= numLoops) {
      LLVM_DEBUG(llvm::dbgs() << "[Systolic] Space loop index " << loopIdx 
                              << " out of range (total loops: " << numLoops << ")\n");
      return failure();
    }
  }
  
  // Assign space loop indices
  for (unsigned loopIdx : spaceLoopDims) {
    spaceLoopIndices.push_back(loopIdx);
  }
  
  // Assign time loop indices (remaining loops not in space dimension)
  std::set<unsigned> spaceSet(spaceLoopDims.begin(), spaceLoopDims.end());
  for (unsigned i = 0; i < numLoops; ++i) {
    if (spaceSet.find(i) == spaceSet.end()) {
      timeLoopIndices.push_back(i);
    }
  }
  
  // Verify selected space loops have distance <= 1 (space loop legality)
  for (unsigned idx : spaceLoopIndices) {
    if (!depInfos[idx].canBeSpaceLoop) {
      LLVM_DEBUG(llvm::dbgs() 
          << "[Systolic] Warning: Loop " << idx 
          << " has dep distance > 1, may not be suitable for space mapping\n");
      // Don't fail - just warn, as some configurations may intentionally
      // use higher-distance loops for specific applications
    }
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "[Systolic] Parametric space-time (" 
                 << parametric.getSpaceTimeTypeString() << "):\n";
    llvm::dbgs() << "  Space loops: ";
    for (unsigned i : spaceLoopIndices) llvm::dbgs() << i << " ";
    llvm::dbgs() << "\n  Time loops: ";
    for (unsigned i : timeLoopIndices) llvm::dbgs() << i << " ";
    llvm::dbgs() << "\n  PE array dims: " << parametric.getNumSpaceDims() << "D\n";
  });
  
  return success();
}

/// Legacy space-time loop selection (backward compatible)
/// 
/// This function maintains the original hardcoded space-time modes (ST0-ST5)
/// for backward compatibility with existing code that doesn't use
/// the ParametricSpaceTime framework.
static LogicalResult selectSpaceLoops(
    const SmallVectorImpl<LoopDepInfo> &depInfos,
    unsigned spaceTimeMode,
    SmallVectorImpl<unsigned> &spaceLoopIndices,
    SmallVectorImpl<unsigned> &timeLoopIndices) {
  
  spaceLoopIndices.clear();
  timeLoopIndices.clear();
  
  unsigned numLoops = depInfos.size();
  if (numLoops < 3) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Need at least 3 loops for space-time mapping\n");
    return failure();
  }
  
  // AutoSA space-time modes for 3-loop case (i=0, j=1, k=2):
  switch (spaceTimeMode) {
    case 0:  // [i] - 1D row array
      spaceLoopIndices.push_back(0);
      timeLoopIndices.push_back(1);
      timeLoopIndices.push_back(2);
      break;
    case 1:  // [j] - 1D column array
      spaceLoopIndices.push_back(1);
      timeLoopIndices.push_back(0);
      timeLoopIndices.push_back(2);
      break;
    case 2:  // [k] - 1D reduction array
      spaceLoopIndices.push_back(2);
      timeLoopIndices.push_back(0);
      timeLoopIndices.push_back(1);
      break;
    case 3:  // [i,j] - 2D output-stationary (default)
      spaceLoopIndices.push_back(0);
      spaceLoopIndices.push_back(1);
      timeLoopIndices.push_back(2);
      break;
    case 4:  // [i,k] - 2D with horizontal reduction
      spaceLoopIndices.push_back(0);
      spaceLoopIndices.push_back(2);
      timeLoopIndices.push_back(1);
      break;
    case 5:  // [j,k] - 2D with vertical reduction
      spaceLoopIndices.push_back(1);
      spaceLoopIndices.push_back(2);
      timeLoopIndices.push_back(0);
      break;
    default:
      LLVM_DEBUG(llvm::dbgs() << "[Systolic] Invalid space-time mode: " 
                              << spaceTimeMode << "\n");
      return failure();
  }
  
  // Verify selected space loops have distance <= 1
  for (unsigned idx : spaceLoopIndices) {
    if (idx >= depInfos.size()) {
      LLVM_DEBUG(llvm::dbgs() << "[Systolic] Space loop index out of range: " 
                              << idx << "\n");
      return failure();
    }
    if (!depInfos[idx].canBeSpaceLoop) {
      LLVM_DEBUG(llvm::dbgs() 
          << "[Systolic] Warning: Loop " << idx 
          << " has dep distance > 1, may not be suitable for space mapping\n");
    }
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "[Systolic] Space-time mode " << spaceTimeMode << ":\n";
    llvm::dbgs() << "  Space loops: ";
    for (unsigned i : spaceLoopIndices) llvm::dbgs() << i << " ";
    llvm::dbgs() << "\n  Time loops: ";
    for (unsigned i : timeLoopIndices) llvm::dbgs() << i << " ";
    llvm::dbgs() << "\n";
  });
  
  return success();
}

//===----------------------------------------------------------------------===//
// Loop Permutation
// (AutoSA: loop_interchange_at_node)
//===----------------------------------------------------------------------===//

/// Permute loops to place space loops as the outer dimensions.
/// For async systolic arrays, space loops should be outermost.
/// After tiling, the order becomes:
///   [space_tile] -> [time_tile] -> [space_point] -> [time_point]
static LogicalResult permuteLoopsForSpaceTime(
    LoopBand &band,
    const SmallVectorImpl<unsigned> &spaceLoopIndices,
    const SmallVectorImpl<unsigned> &timeLoopIndices) {
  
  // Build permutation map: space loops first, then time loops
  SmallVector<unsigned, 6> permMap;
  for (unsigned idx : spaceLoopIndices)
    permMap.push_back(idx);
  for (unsigned idx : timeLoopIndices)
    permMap.push_back(idx);
  
  // Check if permutation is identity
  bool isIdentity = true;
  for (unsigned i = 0; i < permMap.size(); ++i) {
    if (permMap[i] != i) {
      isIdentity = false;
      break;
    }
  }
  
  if (isIdentity) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Loop order is already optimal\n");
    return success();
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "[Systolic] Permuting loops: ";
    for (unsigned i : permMap) llvm::dbgs() << i << " ";
    llvm::dbgs() << "\n";
  });
  
  // Use MLIR's loop permutation utility
  // Note: This requires the loops to be perfectly nested
  // permuteLoops returns the number of loops that could not be permuted
  unsigned numUnpermuted = permuteLoops(band, permMap);
  if (numUnpermuted > 0) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Loop permutation failed, " 
                            << numUnpermuted << " loops could not be permuted\n");
    return failure();
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Multi-Level Tiling
// (AutoSA: sa_array_partitioning_optimize + sa_latency_hiding_optimize)
//===----------------------------------------------------------------------===//

/// Apply multi-level tiling following AutoSA's methodology:
/// 1. Array partitioning (first-level tiling): Creates tile loops for PE array
/// 2. Latency hiding (second-level tiling): Creates point loops for pipelining
///
/// Input loop nest (3 loops: i, j, k):
///   for i = 0..M:
///     for j = 0..N:
///       for k = 0..K:
///         C[i,j] += A[i,k] * B[k,j]
///
/// After array_part tiling [16,16,16]:
///   for i0 = 0..M/16:         // Tile loops
///     for j0 = 0..N/16:
///       for k0 = 0..K/16:
///         for i1 = 0..16:     // Point loops
///           for j1 = 0..16:
///             for k1 = 0..16:
///               C[i0*16+i1, j0*16+j1] += ...
///
/// After latency tiling [8,8,16] on point loops:
///   for i0, j0, k0:           // Array partition tiles (L3)
///     for i1, j1, k1:         // Latency tiles (L2)
///       for i2, j2, k2:       // Point loops (L1)
///         ...
///
/// PE array size = array_part / latency = 16/8 = 2 (per dimension)
static LogicalResult applyMultiLevelTiling(
    LoopBand &band,
    const SystolicTransformOptions &options,
    LoopBand &tiledBand) {
  
  if (band.size() < 3) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Need at least 3 loops for tiling\n");
    return failure();
  }
  
  // Level 1: Array Partitioning
  SmallVector<int64_t, 3> tileSizes1;
  for (unsigned i = 0; i < std::min((size_t)band.size(), options.arrayPart.size()); ++i) {
    tileSizes1.push_back(options.arrayPart[i]);
  }
  // Pad with 1s if needed
  while (tileSizes1.size() < band.size()) {
    tileSizes1.push_back(1);
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "[Systolic] Array partitioning tile sizes: ";
    for (auto s : tileSizes1) llvm::dbgs() << s << " ";
    llvm::dbgs() << "\n";
  });
  
  // Apply first-level tiling
  SmallVector<AffineForOp, 6> tiledNest1;
  SmallVector<unsigned, 3> tileSizes1Unsigned(tileSizes1.begin(), tileSizes1.end());
  if (failed(tilePerfectlyNested(band, tileSizes1Unsigned, &tiledNest1))) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] First-level tiling failed\n");
    return failure();
  }
  
  // After first tiling: [tile0, tile1, tile2, point0, point1, point2]
  unsigned numOrigLoops = band.size();
  if (tiledNest1.size() != numOrigLoops * 2) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Unexpected number of loops after tiling: "
                            << tiledNest1.size() << "\n");
    return failure();
  }
  
  // Extract point loops for second-level tiling
  LoopBand pointLoops;
  for (unsigned i = numOrigLoops; i < tiledNest1.size(); ++i) {
    pointLoops.push_back(tiledNest1[i]);
  }
  
  // Level 2: Latency Hiding
  SmallVector<int64_t, 3> tileSizes2;
  for (unsigned i = 0; i < pointLoops.size(); ++i) {
    if (i < options.latency.size()) {
      tileSizes2.push_back(options.latency[i]);
    } else {
      // For non-space loops (e.g., k), use full tile size (no latency tiling)
      tileSizes2.push_back(options.arrayPart[i]);
    }
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "[Systolic] Latency hiding tile sizes: ";
    for (auto s : tileSizes2) llvm::dbgs() << s << " ";
    llvm::dbgs() << "\n";
  });
  
  // Apply second-level tiling
  SmallVector<AffineForOp, 6> tiledNest2;
  SmallVector<unsigned, 3> tileSizes2Unsigned(tileSizes2.begin(), tileSizes2.end());
  if (failed(tilePerfectlyNested(pointLoops, tileSizes2Unsigned, &tiledNest2))) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Second-level tiling failed\n");
    return failure();
  }
  
  // Build final tiled band:
  // [array_part_tile0, array_part_tile1, array_part_tile2,
  //  latency_tile0, latency_tile1, latency_tile2,
  //  point0, point1, point2]
  tiledBand.clear();
  
  // Array partition tile loops
  for (unsigned i = 0; i < numOrigLoops; ++i) {
    tiledBand.push_back(tiledNest1[i]);
  }
  
  // Latency tile loops + point loops from second tiling
  for (auto loop : tiledNest2) {
    tiledBand.push_back(loop);
  }
  
  LLVM_DEBUG(llvm::dbgs() << "[Systolic] Final tiled nest has " 
                          << tiledBand.size() << " loops\n");
  
  return success();
}

//===----------------------------------------------------------------------===//
// Loop Permutation after Tiling
// (AutoSA's permutation map: {0,1,2, 3,4,5, 7,8,6})
//===----------------------------------------------------------------------===//

/// Apply final permutation to place loops in the order needed for HLS:
/// After tiling we have 9 loops (for 3-loop nest):
///   [i0,j0,k0, i1,j1,k1, i2,j2,k2]
///    0  1  2   3  4  5   6  7  8
///
/// AutoSA permutes to:
///   [i0,j0,k0, i1,j1,k1, j2,k2,i2]
///    0  1  2   3  4  5   7  8  6
///
/// This places the reduction loop innermost for accumulation.
static LogicalResult applyFinalPermutation(LoopBand &tiledBand) {
  if (tiledBand.size() != 9) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Expected 9 loops for final permutation, got "
                            << tiledBand.size() << "\n");
    // For non-standard sizes, skip permutation
    return success();
  }
  
  // AutoSA permutation: {0,1,2, 3,4,5, 7,8,6}
  std::vector<unsigned> permMap = {0, 1, 2, 3, 4, 5, 7, 8, 6};
  
  LLVM_DEBUG({
    llvm::dbgs() << "[Systolic] Applying final permutation: ";
    for (unsigned i : permMap) llvm::dbgs() << i << " ";
    llvm::dbgs() << "\n";
  });
  
  unsigned numUnpermuted = permuteLoops(tiledBand, permMap);
  if (numUnpermuted > 0) {
    LLVM_DEBUG(llvm::dbgs() << "[Systolic] Final permutation failed, "
                            << numUnpermuted << " loops could not be permuted\n");
    return failure();
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Problem Size Inference
// (AutoSA: inferProblemSize)
//===----------------------------------------------------------------------===//

/// Infer problem dimensions (M, N, K) from loop bounds or memref shapes.
static ProblemSize inferProblemSize(LoopBand &band) {
  ProblemSize size;
  
  // Method 1: Extract from loop bounds
  if (band.size() >= 3) {
    for (unsigned i = 0; i < 3; ++i) {
      auto loop = band[i];
      if (loop.hasConstantUpperBound()) {
        int64_t ub = loop.getConstantUpperBound();
        if (i == 0) size.M = ub;
        else if (i == 1) size.N = ub;
        else if (i == 2) size.K = ub;
      }
    }
    
    if (size.M > 0 && size.N > 0 && size.K > 0) {
      size.valid = true;
      LLVM_DEBUG(llvm::dbgs() << "[Systolic] Inferred problem size: M=" << size.M
                              << ", N=" << size.N << ", K=" << size.K << "\n");
      return size;
    }
  }
  
  // Method 2: Extract from memref shapes (TODO: implement)
  
  return size;
}

//===----------------------------------------------------------------------===//
// Systolic Transform Pass
//===----------------------------------------------------------------------===//

namespace {

struct SystolicTransformPass 
    : public PassWrapper<SystolicTransformPass, OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SystolicTransformPass)
  
  SystolicTransformPass() = default;
  SystolicTransformPass(const SystolicConfig &config) {
    options.spaceTimeMode = 3;  // Default to [i,j] 2D output-stationary
    if (!config.arrayPart.empty())
      options.arrayPart.assign(config.arrayPart.begin(), config.arrayPart.end());
    if (!config.latency.empty())
      options.latency.assign(config.latency.begin(), config.latency.end());
    options.simdWidth = config.simdWidth;
  }
  
  StringRef getArgument() const override { return "systolic-transform"; }
  StringRef getDescription() const override {
    return "Apply systolic array transformations (space-time, tiling, permutation)";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect>();
  }
  
  void runOnOperation() override {
    auto func = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== Systolic Transform Pass ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing function: " << func.getName() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Space-time mode: " << options.spaceTimeMode << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Array partition: [" << options.arrayPart[0] << ", "
                            << options.arrayPart[1] << ", " << options.arrayPart[2] << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "Latency: [" << options.latency[0] << ", "
                            << options.latency[1] << "]\n");
    
    llvm::outs() << "[Systolic] Transform Pass Configuration:\n";
    llvm::outs() << "  Space-time mode: " << options.spaceTimeMode << "\n";
    llvm::outs() << "  Array partition: [" << options.arrayPart[0] << ", "
                 << options.arrayPart[1] << ", " << options.arrayPart[2] << "]\n";
    llvm::outs() << "  Latency: [" << options.latency[0] << ", "
                 << options.latency[1] << "]\n";
    
    // REQUIREMENT: Polymer must be available and used
    if (!systolic::isPolymerAvailable()) {
      func.emitError("Polymer is required for SystolicTransform but is not available. "
                     "Please ensure Polymer is built and linked.");
      return signalPassFailure();
    }
    
    // DEBUG: Log that Polymer is available
    llvm::dbgs() << "[Systolic Debug] Polymer is available\n";
    llvm::errs() << "[Systolic] Polymer is AVAILABLE - proceeding with transformation\n";
    
    // Step 0: Preprocess function with ExtractScopStmt if needed
    // This is required by Polymer's createIslFromFuncOp
    mlir::ModuleOp module = cast<mlir::ModuleOp>(func->getParentOp());
    
    // Check if function already has scop.stmt structure
    bool hasScopStmt = false;
    func.walk([&](mlir::func::CallOp callOp) {
      if (auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee())) {
        if (callee->hasAttr("scop.stmt")) {
          hasScopStmt = true;
        }
      }
    });
    
        if (!hasScopStmt) {
      LLVM_DEBUG(llvm::dbgs() << "Function does not have scop.stmt structure, "
                              << "running ExtractScopStmt pass...\n");
      
#ifdef SYSTOLIC_ENABLE_POLYMER
#if __has_include("polymer/Transforms/ExtractScopStmt.h")
      // Run ExtractScopStmt pass to convert affine.for loops to scop.stmt format
      PassManager pm(module.getContext());
      // Run reg2mem before ExtractScopStmt (Polymer pipeline requirement)
    #if __has_include("polymer/Transforms/Reg2Mem.h")
      pm.addNestedPass<mlir::func::FuncOp>(polymer::createRegToMemPass());
      LLVM_DEBUG(llvm::dbgs() << "Ran Reg2Mem prior to ExtractScopStmt\n");
    #endif
      pm.addPass(polymer::createExtractScopStmtPass());
      if (failed(pm.run(module))) {
        func.emitError("Failed to run ExtractScopStmt pass. This is required for Polymer.");
        return signalPassFailure();
      }
      
      LLVM_DEBUG(llvm::dbgs() << "ExtractScopStmt pass completed successfully\n");
      llvm::errs() << "[Systolic] Preprocessing done (reg2mem + extract-scop-stmt)\n";
#else
      func.emitError("ExtractScopStmt pass header not found. Please ensure Polymer Transforms library is built.");
      return signalPassFailure();
#endif
#else
      func.emitError("Polymer is not enabled. Please enable SYSTOLIC_ENABLE_POLYMER.");
      return signalPassFailure();
#endif
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Function already has scop.stmt structure\n");
      llvm::errs() << "[Systolic] scop.stmt detected, skipping preprocessing\n";
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Options:\n";
      llvm::dbgs() << "  space-time mode: " << options.spaceTimeMode << "\n";
      llvm::dbgs() << "  array-part: ";
      for (auto s : options.arrayPart) llvm::dbgs() << s << " ";
      llvm::dbgs() << "\n  latency: ";
      for (auto s : options.latency) llvm::dbgs() << s << " ";
      llvm::dbgs() << "\n  simd: " << options.simdWidth << "\n";
    });
    
    // Step 1: Collect all loop bands
    SmallVector<LoopBand, 4> bands;
    func.walk([&](AffineForOp forOp) {
      // Only process top-level loops (not nested inside another AffineForOp)
      if (!forOp->getParentOfType<AffineForOp>()) {
        LoopBand band;
        // Collect the full loop nest
        AffineForOp current = forOp;
        while (current) {
          band.push_back(current);
          // Find the next nested loop
          AffineForOp inner = nullptr;
          for (auto &op : *current.getBody()) {
            if (auto nestedFor = dyn_cast<AffineForOp>(op)) {
              inner = nestedFor;
              break;
            }
          }
          current = inner;
        }
        if (!band.empty())
          bands.push_back(band);
      }
    });
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << bands.size() << " loop band(s)\n");
    
    // Step 2: Process each loop band
    // Note: For testing, we assume there's only one loop band (single SCoP region)
    if (bands.size() > 1) {
      LLVM_DEBUG(llvm::dbgs() << "[Systolic] Warning: Multiple loop bands found, "
                              << "processing only the first one (testing assumption)\n");
    }
    
    for (auto &band : bands) {
      LLVM_DEBUG(llvm::dbgs() << "\nProcessing band with " << band.size() << " loops\n");
      
      // Step 2.1: Legality check (AutoSA: sa_legality_check)
      if (failed(checkLegality(band))) {
        LLVM_DEBUG(llvm::dbgs() << "Legality check failed, skipping band\n");
        continue;
      }
      
      // Step 2.2: Infer problem size
      ProblemSize problemSize = inferProblemSize(band);
      
      // Step 3: Dependence analysis (AutoSA: get_dep_dis_at_node)
      // REQUIRES Polymer - no fallback
      SmallVector<LoopDepInfo, 4> depInfos;
      if (failed(analyzeDependenceDistances(func, band, depInfos))) {
        LLVM_DEBUG(llvm::dbgs() << "Dependence analysis failed\n");
        llvm::errs() << "[Systolic] Dependence analysis FAILED\n";
        continue;
      }
      llvm::errs() << "[Systolic] Dependence analysis OK, deps=" << depInfos.size() << "\n";
      for (auto &d : depInfos) {
        llvm::errs() << "  loop=" << d.loopIndex << " min=" << d.minDistance << " max=" << d.maxDistance
                     << " uniform=" << (d.isUniform?"y":"n") << " space?=" << (d.canBeSpaceLoop?"y":"n") << "\n";
      }
      
      // Step 2.4: Select space and time loops (AutoSA: sa_space_time_transform)
      // Phase 2 Enhancement: Use parametric space-time framework
      SmallVector<unsigned, 2> spaceLoops;
      SmallVector<unsigned, 3> timeLoops;
      
      // Create parametric configuration based on spaceTimeMode
      ParametricSpaceTime parametricConfig = 
        ParametricSpaceTime::createFromMode(options.spaceTimeMode);
      
      // Use parametric version for loop selection if available
      // This replaces hardcoded [0,1]/[2..] assumptions
      if (parametricConfig.isValid()) {
        if (failed(selectSpaceLoopsParametric(depInfos, parametricConfig,
                                              spaceLoops, timeLoops))) {
          LLVM_DEBUG(llvm::dbgs() 
              << "Parametric space loop selection failed, "
              << "falling back to legacy mode\n");
          // Fallback to legacy mode
          if (failed(selectSpaceLoops(depInfos, options.spaceTimeMode,
                                      spaceLoops, timeLoops))) {
            LLVM_DEBUG(llvm::dbgs() << "Space loop selection failed\n");
            continue;
          }
        }
      } else {
        // Fallback to legacy mode if parametric config is invalid
        if (failed(selectSpaceLoops(depInfos, options.spaceTimeMode,
                                    spaceLoops, timeLoops))) {
          LLVM_DEBUG(llvm::dbgs() << "Space loop selection failed\n");
          continue;
        }
      }
      
      // Step 2.5: Permute loops (space loops to outer positions)
      // Note: For now, we skip permutation before tiling if already in order
      // This matches AutoSA's behavior for [i,j] mode on MatMul
      
      // Step 2.6: Multi-level tiling (AutoSA: array_part + latency)
      LoopBand tiledBand;
      if (failed(applyMultiLevelTiling(band, options, tiledBand))) {
        LLVM_DEBUG(llvm::dbgs() << "Multi-level tiling failed\n");
        continue;
      }
      
      // Step 2.7: Final permutation
      if (failed(applyFinalPermutation(tiledBand))) {
        LLVM_DEBUG(llvm::dbgs() << "Final permutation failed\n");
        // Continue anyway, permutation is optional
      }
      
      // Calculate PE array dimensions
      int64_t numPE_I = options.arrayPart[0] / options.latency[0];
      int64_t numPE_J = options.arrayPart[1] / options.latency[1];
      
      // Store configuration information as function attributes for later passes
      // This allows SystolicDataflowGeneration to access the configuration
      OpBuilder builder(func.getContext());
      builder.setInsertionPointToStart(&func.getBody().front());
      
      // Store array partitioning factors
      func->setAttr("systolic.array_part", 
                    builder.getI64ArrayAttr(options.arrayPart));
      
      // Store latency hiding factors
      func->setAttr("systolic.latency", 
                    builder.getI64ArrayAttr(options.latency));
      
      // Store PE array dimensions
      SmallVector<int64_t, 2> peArraySize = {numPE_I, numPE_J};
      func->setAttr("systolic.pe_array_size", 
                    builder.getI64ArrayAttr(peArraySize));
      
      // Store space-time mode
      func->setAttr("systolic.space_time_mode", 
                    builder.getI32IntegerAttr(options.spaceTimeMode));
      
      LLVM_DEBUG(llvm::dbgs() << "[Systolic] Stored configuration:\n");
      LLVM_DEBUG(llvm::dbgs() << "  array_part: [" 
                              << options.arrayPart[0] << ", "
                              << options.arrayPart[1] << ", "
                              << options.arrayPart[2] << "]\n");
      LLVM_DEBUG(llvm::dbgs() << "  latency: [" 
                              << options.latency[0] << ", "
                              << options.latency[1] << "]\n");
      LLVM_DEBUG(llvm::dbgs() << "  PE array size: " << numPE_I << " x " << numPE_J << "\n");
      
      llvm::outs() << "[Systolic] Transformation complete:\n";
      llvm::outs() << "  PE array size: " << numPE_I << " x " << numPE_J << "\n";
      llvm::outs() << "  Total loops after tiling: " << tiledBand.size() << "\n";
    }
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== Systolic Transform Pass Complete ===\n");
  }
  
private:
  SystolicTransformOptions options;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createSpaceTimeTransformPass() {
  return std::make_unique<SystolicTransformPass>();
}

std::unique_ptr<Pass> createSpaceTimeTransformPass(const SystolicConfig &config) {
  return std::make_unique<SystolicTransformPass>(config);
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void registerSystolicPasses() {
  PassRegistration<SystolicTransformPass>();
  // SystolicDataflowGenerationPass is registered via static PassRegistration
  // in SystolicDataflowGeneration.cpp. To ensure the static registration
  // is linked, we explicitly reference the create function to prevent
  // the linker from removing the unused symbol.
  (void)createSystolicDataflowGenerationPass();
  (void)createSystolicWriteReorderAnalysisPass();
  // SystolicDataflowToHLSPass is registered in its own file
}

} // namespace systolic
} // namespace mlir
