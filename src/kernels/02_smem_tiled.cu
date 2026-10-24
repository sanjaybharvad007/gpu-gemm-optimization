// Kernel 2 - shared-memory tiling.  Kernels 0/1 read every element of A N times and every
// element of B M times from global memory (L2 absorbs some of it, not enough).  A block now
// cooperatively loads a BLOCK x BLOCK tile of A and of B into shared memory, synchronises, and
// every thread does BLOCK multiply-adds from the tile before the next tile is fetched.
//
// Arithmetic intensity per block tile rises from 1/2 FLOP per byte to BLOCK/4 = 8 FLOP/byte for
// BLOCK=32 (2*BLOCK^2*BLOCK FLOPs per 2*BLOCK^2*4 bytes).  Still memory bound on every modern GPU
// (ridge point ~ 50-100 FLOP/byte); kernel 3 raises it further by giving each thread more outputs.
#include "gemm/common.h"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

template <int BLOCK>
__global__ void sgemm_smem_tiled_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                                        const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[BLOCK][BLOCK];
  __shared__ float Bs[BLOCK][BLOCK];
  const int tx = threadIdx.x % BLOCK, ty = threadIdx.x / BLOCK;
  const int row = blockIdx.y * BLOCK + ty, col = blockIdx.x * BLOCK + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += BLOCK) {
    // Each thread loads one element of each tile; loads are coalesced along tx.  Out-of-range
    // elements are zero so ragged M/N/K need no special casing in the inner loop.
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) acc += As[ty][kk] * Bs[kk][tx];   // As: broadcast, Bs: conflict-free
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// explicit instantiation of the configuration the launcher uses (also lets NVRTC/ptxas report it)
template __global__ void sgemm_smem_tiled_kernel<32>(int, int, int, float, const float*, const float*, float, float*);

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_smem_tiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                              cudaStream_t s) {
  constexpr int BLOCK = 32;
  dim3 block(BLOCK * BLOCK), grid(cdiv(N, BLOCK), cdiv(M, BLOCK));
  auto kfn = &sgemm_smem_tiled_kernel<BLOCK>;
  KERNEL_LAUNCH(kfn, grid, block, 0, s)(M, N, K, alpha, A, B, beta, C);
  return {grid, block, 2 * BLOCK * BLOCK * sizeof(float)};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
