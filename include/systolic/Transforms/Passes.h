//===----------------------------------------------------------------------===//
//
// MLIR-Systolic: Transform Passes
//
//===----------------------------------------------------------------------===//

#ifndef SYSTOLIC_TRANSFORMS_PASSES_H
#define SYSTOLIC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "systolic/Analysis/SystolicConfig.h"

namespace mlir {
namespace systolic {

//===----------------------------------------------------------------------===//
// Pass Declarations
//===----------------------------------------------------------------------===//

/// Create a pass that performs space-time transformation.
/// This includes:
/// - Analyzing dependences (via Polymer)
/// - Selecting space loops
/// - Loop permutation and tiling
/// AutoSA: sa_space_time_transform, sa_array_partitioning, sa_latency_hiding
std::unique_ptr<Pass> createSpaceTimeTransformPass();
std::unique_ptr<Pass> createSpaceTimeTransformPass(const SystolicConfig &config);

/// Create a pass that generates stream channels and dataflow tasks.
/// This creates the complete systolic array dataflow structure:
/// - I/O modules (L3, L2, L1) for data feeding
/// - PE modules for computation
/// - Drain modules for output
/// AutoSA: sa_io_construct_optimize, generate_hw_modules
std::unique_ptr<Pass> createStreamGenerationPass();
std::unique_ptr<Pass> createDataflowGenerationPass();
std::unique_ptr<Pass> createDataflowGenerationPass(const SystolicConfig &config);

/// Create a pass that generates I/O modules (L1, L2, L3).
/// AutoSA: sa_io_module_gen
std::unique_ptr<Pass> createIOModuleGenerationPass();

/// Create a pass that generates SystolicDataflow Dialect from Affine IR.
/// This creates multi-level IO modules, PE arrays, and double buffering.
/// AutoSA: sa_io_construct_optimize, generate_hw_modules
std::unique_ptr<Pass> createSystolicDataflowGenerationPass();

/// Run write-time reordering analysis and set systolic.reorder.* attributes.
/// Must run before systolic-transform so store indices (e.g. affine.apply) are visible.
std::unique_ptr<Pass> createSystolicWriteReorderAnalysisPass();

/// Create a pass that lowers SystolicDataflow Dialect to HLS Dialect.
/// This converts high-level systolic abstractions to concrete HLS structures.
std::unique_ptr<Pass> createSystolicDataflowToHLSPass();

/// Create a pass that applies SIMD vectorization.
/// AutoSA: sa_simd_vectorization_optimize
std::unique_ptr<Pass> createSIMDVectorizationPass(unsigned simdWidth = 2);

//===----------------------------------------------------------------------===//
// Pass Options
//===----------------------------------------------------------------------===//

struct SpaceTimeTransformOptions {
  /// Space-time mapping mode (0-5, same as AutoSA)
  /// 0: [i]   - 1D row array
  /// 1: [j]   - 1D column array  
  /// 2: [k]   - 1D reduction array
  /// 3: [i,j] - 2D output-stationary (default)
  /// 4: [i,k] - 2D with horizontal reduction
  /// 5: [j,k] - 2D with vertical reduction
  unsigned spaceTimeMode = 3;
  
  /// Array partitioning factors (first-level tiling)
  llvm::SmallVector<int64_t, 3> arrayPart = {16, 16, 16};
  
  /// Latency hiding factors (second-level tiling)
  llvm::SmallVector<int64_t, 2> latency = {8, 8};
  
  /// SIMD width
  unsigned simdWidth = 1;
  
  /// Enable two-level buffering
  bool twoLevelBuffer = false;
  
  /// Enable double buffering (ping-pong)
  bool doubleBuffer = true;
};

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
// #include "systolic/Transforms/Passes.h.inc"  // Will be generated

void registerSystolicPasses();

} // namespace systolic
} // namespace mlir

#endif // SYSTOLIC_TRANSFORMS_PASSES_H
