// GEMV  y = alpha*A*x + beta*y  (A: MxN row-major).  Memory-bound by construction: 2 FLOP per
// 4-byte element of A, so the only goal is to stream A at full bandwidth.  One warp per row,
// float4 loads (16 B per lane, 512 B per warp per step), warp-shuffle reduction, no shared memory.
#include "gemm/common.h"

namespace gemm {

__global__ void sgemv_kernel(int M, int N, float alpha, const float* __restrict__ A, const float* __restrict__ x,
                             float beta, float* __restrict__ y) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= M) return;
  const float* row = A + (size_t)warp * N;
  float acc = 0.f;
  const int n4 = N / 4;
  for (int i = lane; i < n4; i += 32) {
    const float4 a = reinterpret_cast<const float4*>(row)[i];
    const float4 b = reinterpret_cast<const float4*>(x)[i];
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    acc = fmaf(a.w, b.w, acc);
  }
  for (int i = n4 * 4 + lane; i < N; i += 32) acc = fmaf(row[i], x[i], acc);   // ragged tail
  acc = warp_sum(acc);
  if (lane == 0) y[warp] = alpha * acc + (beta != 0.f ? beta * y[warp] : 0.f);
}

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
void sgemv(int M, int N, float alpha, const float* A, const float* x, float beta, float* y, cudaStream_t s) {
  constexpr int WARPS_PER_BLOCK = 4;
  const dim3 grid(cdiv(M, WARPS_PER_BLOCK)), block(WARPS_PER_BLOCK * 32);
  auto kfn = &sgemv_kernel;
  KERNEL_LAUNCH(kfn, grid, block, 0, s)(M, N, alpha, A, x, beta, y);
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
