// Kernel 8 - strided batched GEMM.  Transformers run attention scores / projections as many
// small GEMMs; launching them one by one leaves the GPU idle between launches.  blockIdx.z
// indexes the batch and every block offsets its pointers - the main loop is untouched.
#include "sgemm_tile.cuh"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

__global__ void __launch_bounds__(Tile128::THREADS)
sgemm_batched_kernel(int M, int N, int K, float alpha, const float* __restrict__ A, long long sA,
                     const float* __restrict__ B, long long sB, float beta, float* __restrict__ C, long long sC) {
  extern __shared__ float smem[];
  const int bm = blockIdx.y * 128, bn = blockIdx.x * 128, z = blockIdx.z;
  float acc[8][8] = {};
  Tile128::run(smem, A + z * sA, B + z * sB, M, N, K, bm, bn, 0, K, acc);
  Tile128::epilogue(C + z * sC, M, N, bm, bn, alpha, beta, acc);
}

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_batched(int M, int N, int K, float alpha, const float* A, long long sA, const float* B,
                           long long sB, float beta, float* C, long long sC, int batch, cudaStream_t s) {
  dim3 block(Tile128::THREADS), grid(cdiv(N, 128), cdiv(M, 128), batch);
  auto kfn = &sgemm_batched_kernel;
  KERNEL_LAUNCH(kfn, grid, block, Tile128::SMEM_BYTES, s)(M, N, K, alpha, A, sA, B, sB, beta, C, sC);
  return {grid, block, Tile128::SMEM_BYTES};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
