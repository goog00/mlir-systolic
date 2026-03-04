// Standard MTTKRP (mode-1 style):
// D(i,j) += A(i,k,l) * B(k,j) * C(l,j)
// A: [I,K,L], B: [K,J], C: [L,J], D: [I,J]
module {
  func.func @mttkrp_std(%A: memref<8x8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>, %D: memref<8x8xf32>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %zero = arith.constant 0.0 : f32
        affine.store %zero, %D[%i, %j] : memref<8x8xf32>
        affine.for %k = 0 to 8 {
          affine.for %l = 0 to 8 {
            %a = affine.load %A[%i, %k, %l] : memref<8x8x8xf32>
            %b = affine.load %B[%k, %j] : memref<8x8xf32>
            %c = affine.load %C[%l, %j] : memref<8x8xf32>
            %d = affine.load %D[%i, %j] : memref<8x8xf32>
            %ab = arith.mulf %a, %b : f32
            %abc = arith.mulf %ab, %c : f32
            %sum = arith.addf %d, %abc : f32
            affine.store %sum, %D[%i, %j] : memref<8x8xf32>
          }
        }
      }
    }
    return
  }
}
