// Minimal 3-loop kernel that triggers write-time reordering:
// Output C is 1024x32; store at C[i*32+j, k] so the first index has Mul (non-linear).
// Analyzer will set systolic.reorder.C.dims/perm; translate will emit 2D write-time reorder path.
#map_ij = affine_map<(i, j) -> (i * 32 + j)>
module {
  func.func @reorder_write(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<1024x32xf32>) {
    affine.for %i = 0 to 32 {
      affine.for %j = 0 to 32 {
        affine.for %k = 0 to 32 {
          %ij = affine.apply #map_ij(%i, %j)
          %a = affine.load %A[%i, %j] : memref<32x32xf32>
          %b = affine.load %B[%j, %k] : memref<32x32xf32>
          %p = arith.mulf %a, %b : f32
          affine.store %p, %C[%ij, %k] : memref<1024x32xf32>
        }
      }
    }
    return
  }
}
