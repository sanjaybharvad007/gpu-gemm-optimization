// Kernel 3 - register blocking (2-D thread tiling).  Each thread now computes an 8x8 patch of C
// instead of one element.  Per K step it reads 8 values of A and 8 of B from shared memory and
// performs 64 FMAs: the shared-memory traffic per FLOP drops 8x compared with kernel 2 and the
// FMAs are issued back to back from registers, which is what the FP32 pipes need to stay busy.
//
// Block tile 128x128x8 with 256 threads (16x16 threads, 8x8 outputs each):
//   * A tile (128x8) is loaded with float4 along K and stored TRANSPOSED (As[k][m]) so the inner
//     loop reads 8 consecutive m values for a fixed k as two float4 - conflict-free, no gather.
//   * B tile (8x128) is loaded with float4 along N and stored as is (Bs[k][n]).
//   * 64 fp32 accumulators + 16 operand registers per thread; ~100 registers total, which still
//     allows 2 blocks (512 threads) per SM.
// Arithmetic intensity of the block tile: 2*128*128*8 / ((128+128)*8*4) = 32 FLOP/byte.
#include "gemm/common.h"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemm_register_blocked_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                              const float* __restrict__ B, float beta, float* __restrict__ C) {
  constexpr int THREADS = (BM / TM) * (BN / TN);
  static_assert(BM * BK / 4 == THREADS && BK * BN / 4 == THREADS, "one float4 of A and of B per thread");
  __shared__ float As[BK][BM];   // transposed
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN), ty = tid / (BN / TN);
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

  // global -> smem mapping for the loads
  const int a_row = tid / (BK / 4), a_col = (tid % (BK / 4)) * 4;   // A[bm+a_row][k0+a_col .. +3]
  const int b_row = tid / (BN / 4), b_col = (tid % (BN / 4)) * 4;   // B[k0+b_row][bn+b_col .. +3]

  float acc[TM][TN] = {};
  float a_frag[TM], b_frag[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f), b4 = a4;
    if (bm + a_row < M && k0 + a_col + 3 < K)
      a4 = *reinterpret_cast<const float4*>(A + (size_t)(bm + a_row) * K + k0 + a_col);
    if (k0 + b_row < K && bn + b_col + 3 < N)
      b4 = *reinterpret_cast<const float4*>(B + (size_t)(k0 + b_row) * N + bn + b_col);
    As[a_col + 0][a_row] = a4.x;
    As[a_col + 1][a_row] = a4.y;
    As[a_col + 2][a_row] = a4.z;
    As[a_col + 3][a_row] = a4.w;
    *reinterpret_cast<float4*>(&Bs[b_row][b_col]) = b4;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int i = 0; i < TM; i += 4) *reinterpret_cast<float4*>(&a_frag[i]) = *reinterpret_cast<const float4*>(&As[kk][ty * TM + i]);
#pragma unroll
      for (int j = 0; j < TN; j += 4) *reinterpret_cast<float4*>(&b_frag[j]) = *reinterpret_cast<const float4*>(&Bs[kk][tx * TN + j]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads();
  }

  // epilogue: C = alpha*acc + beta*C, float4 stores (N % 4 == 0 is a kernel constraint)
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = bm + ty * TM + i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int col = bn + tx * TN + j;
      if (col >= N) continue;
      float4* cp = reinterpret_cast<float4*>(C + (size_t)row * N + col);
      float4 r = make_float4(alpha * acc[i][j], alpha * acc[i][j + 1], alpha * acc[i][j + 2], alpha * acc[i][j + 3]);
      if (beta != 0.f) {
        float4 c = *cp;
        r.x += beta * c.x; r.y += beta * c.y; r.z += beta * c.z; r.w += beta * c.w;
      }
      *cp = r;
    }
  }
}

// explicit instantiation of the configuration the launcher uses (also lets NVRTC/ptxas report it)
template __global__ void sgemm_register_blocked_kernel<128, 128, 8, 8, 8>(int, int, int, float, const float*, const float*, float, float*);

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_register_blocked(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                                    float* C, cudaStream_t s) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 block((BM / TM) * (BN / TN)), grid(cdiv(N, BN), cdiv(M, BM));
  auto kfn = &sgemm_register_blocked_kernel<BM, BN, BK, TM, TN>;
  KERNEL_LAUNCH(kfn, grid, block, 0, s)(M, N, K, alpha, A, B, beta, C);
  return {grid, block, (BM + BN) * BK * sizeof(float)};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
