// Kernel 4 - register blocking + double buffering (see sgemm_tile.cuh for the main loop).
// Compared with kernel 3 the only change is WHEN loads are issued: the next tile's global loads
// are in flight during the current tile's 512 FMAs per thread, and one of the two barriers per
// tile disappears.  On Ampere this is typically the step that takes the SIMT kernel from ~55% to
// ~75-80% of cuBLAS SGEMM.
#include "sgemm_tile.cuh"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

__global__ void __launch_bounds__(Tile128::THREADS)
sgemm_double_buffered_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                             const float* __restrict__ B, float beta, float* __restrict__ C) {
  extern __shared__ float smem[];
  const int bm = blockIdx.y * 128, bn = blockIdx.x * 128;
  float acc[8][8] = {};
  Tile128::run(smem, A, B, M, N, K, bm, bn, 0, K, acc);
  Tile128::epilogue(C, M, N, bm, bn, alpha, beta, acc);
}

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_double_buffered(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                                   float* C, cudaStream_t s) {
  dim3 block(Tile128::THREADS), grid(cdiv(N, 128), cdiv(M, 128));
  auto kfn = &sgemm_double_buffered_kernel;
  KERNEL_LAUNCH(kfn, grid, block, Tile128::SMEM_BYTES, s)(M, N, K, alpha, A, B, beta, C);
  return {grid, block, Tile128::SMEM_BYTES};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
