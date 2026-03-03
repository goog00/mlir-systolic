// Minimal 4-loop kernel that triggers 3D write-time reordering:
// Output D is 64x8x8 (logically 8x8x8 in i,j,k); store at D[i*8+j, k, l] so dim0 is non-linear.
// Analyzer sets systolic.reorder.arg2.dims/perm for 3D; translate emits 3D buffer -> buffer_linear path.
#map_ij = affine_map<(i, j) -> (i * 8 + j)>
module {
  func.func @reorder_write_3d(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %D: memref<64x8x8xf32>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.for %k = 0 to 8 {
          affine.for %l = 0 to 8 {
            %ij = affine.apply #map_ij(%i, %j)
            %a = affine.load %A[%i, %k] : memref<8x8xf32>
            %b = affine.load %B[%j, %l] : memref<8x8xf32>
            %p = arith.mulf %a, %b : f32
            affine.store %p, %D[%ij, %k, %l] : memref<64x8x8xf32>
          }
        }
      }
    }
    return
  }
}
