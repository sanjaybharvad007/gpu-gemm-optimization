// Kernel 1 - same arithmetic as kernel 0, different thread mapping.  Consecutive threads now
// own consecutive COLUMNS of C, so within a warp:
//   * the A element A[row][k] is the same address for all 32 threads -> one broadcast load;
//   * the B elements B[k][col..col+31] are contiguous -> one 128-byte transaction;
//   * the C stores are contiguous.
// Nothing else changed and the kernel is typically 5-8x faster than the naive one.  The lesson of
// the whole repository in one commit: on a GPU, the memory access pattern IS the algorithm.
#include "gemm/common.h"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

template <int BLOCK>
__global__ void sgemm_coalesced_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                                       const float* __restrict__ B, float beta, float* __restrict__ C) {
  const int col = blockIdx.x * BLOCK + (threadIdx.x % BLOCK);   // fastest index along N
  const int row = blockIdx.y * BLOCK + (threadIdx.x / BLOCK);
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// explicit instantiation of the configuration the launcher uses (also lets NVRTC/ptxas report it)
template __global__ void sgemm_coalesced_kernel<32>(int, int, int, float, const float*, const float*, float, float*);

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_coalesced(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                             cudaStream_t s) {
  constexpr int BLOCK = 32;
  dim3 block(BLOCK * BLOCK), grid(cdiv(N, BLOCK), cdiv(M, BLOCK));
  auto kfn = &sgemm_coalesced_kernel<BLOCK>;
  KERNEL_LAUNCH(kfn, grid, block, 0, s)(M, N, K, alpha, A, B, beta, C);
  return {grid, block, 0};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
