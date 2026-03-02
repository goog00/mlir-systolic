// Minimal 4-loop MTTKRP-like kernel: D[i,j,l] += A[i,j,k]*B[k,l]
// Loops: i, j, k, l. A 3D, B 2D, D 3D. Reduction on k.
// Small bounds (8) for pipeline testing; see docs/design/SYSTOLIC_OPTIMIZATION_IMPROVEMENT_PLAN.md Phase 2.
#map_i = affine_map<() -> (8)>
#map_j = affine_map<() -> (8)>
#map_k = affine_map<() -> (8)>
#map_l = affine_map<() -> (8)>
module {
  func.func @mttkrp(%A: memref<8x8x8xf32>, %B: memref<8x8xf32>, %D: memref<8x8x8xf32>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.for %k = 0 to 8 {
          affine.for %l = 0 to 8 {
            %a = affine.load %A[%i, %j, %k] : memref<8x8x8xf32>
            %b = affine.load %B[%k, %l] : memref<8x8xf32>
            %d = affine.load %D[%i, %j, %l] : memref<8x8x8xf32>
            %p = arith.mulf %a, %b : f32
            %s = arith.addf %d, %p : f32
            affine.store %s, %D[%i, %j, %l] : memref<8x8x8xf32>
          }
        }
      }
    }
    return
  }
}
