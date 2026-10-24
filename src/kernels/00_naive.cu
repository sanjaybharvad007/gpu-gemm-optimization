// Kernel 0 - the textbook GEMM.  One thread per output element, three nested loops collapsed
// into one: each thread walks a row of A and a column of B.
//
// Thread mapping is the "obvious" one (threadIdx.x -> row), which is the worst possible: the 32
// threads of a warp touch 32 different rows of A (stride K floats) and 32 different rows of C.
// Every warp-level load becomes 32 separate 32-byte sectors.  Nsight shows ~3% global load
// efficiency; this kernel exists so the next one has something to be measured against.
#include "gemm/common.h"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                                   const float* __restrict__ B, float beta, float* __restrict__ C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;   // adjacent threads -> adjacent rows (bad)
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                         cudaStream_t s) {
  dim3 block(32, 32), grid(cdiv(M, 32), cdiv(N, 32));
  auto kfn = &sgemm_naive_kernel;
  KERNEL_LAUNCH(kfn, grid, block, 0, s)(M, N, K, alpha, A, B, beta, C);
  return {grid, block, 0};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
