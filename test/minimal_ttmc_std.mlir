// Standard TTMc:
// D(i,j,k) += A(i,l,m) * B(l,j) * C(m,k)
// A: [I,L,M], B: [L,J], C: [M,K], D: [I,J,K]
module {
  func.func @ttmc_std(%A: memref<8x8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>, %D: memref<8x8x8xf32>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.for %k = 0 to 8 {
          %zero = arith.constant 0.0 : f32
          affine.store %zero, %D[%i, %j, %k] : memref<8x8x8xf32>
          affine.for %l = 0 to 8 {
            affine.for %m = 0 to 8 {
              %a = affine.load %A[%i, %l, %m] : memref<8x8x8xf32>
              %b = affine.load %B[%l, %j] : memref<8x8xf32>
              %c = affine.load %C[%m, %k] : memref<8x8xf32>
              %d = affine.load %D[%i, %j, %k] : memref<8x8x8xf32>
              %ab = arith.mulf %a, %b : f32
              %abc = arith.mulf %ab, %c : f32
              %sum = arith.addf %d, %abc : f32
              affine.store %sum, %D[%i, %j, %k] : memref<8x8x8xf32>
            }
          }
        }
      }
    }
    return
  }
}
