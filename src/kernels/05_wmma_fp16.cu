// Kernel 5 - Tensor Cores through the WMMA API.  nvcuda::wmma exposes 16x16x16 matrix fragments:
// a warp loads an A and a B fragment from shared memory and issues one mma_sync, which the hardware
// executes on the Tensor Cores as a 16x16x16 fp16 product with fp32 accumulation (~8x the FP32
// pipe's FLOP rate on Ampere).
//
// Block tile 128x128x32, 8 warps as 2 (M) x 4 (N), warp tile 64x32 = 4x2 fragments, 8 fp32
// accumulator fragments per warp.  Shared-memory rows are padded by 8 halves (16 B) so that the
// fragment loads - which touch 16 rows at a stride - do not all hit the same bank.
//
// Mixed precision contract: fp16 operands (exact products in fp32), fp32 accumulation, fp32 output.
// The error analysis in docs/numerics.md shows this is within ~2x of full fp32 SGEMM's error for
// well-scaled inputs, while bf16 would need the inputs' dynamic range and fp16 would overflow past
// 65504 if accumulated in fp16.
#include <mma.h>
#include "gemm/common.h"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

using namespace nvcuda;

template <int BM, int BN, int BK, int WM, int WN>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
hgemm_wmma_kernel(int M, int N, int K, float alpha, const half* __restrict__ A, const half* __restrict__ B,
                  float beta, float* __restrict__ C) {
  constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, THREADS = WARPS_M * WARPS_N * 32;
  constexpr int FRAG = 16, FM = WM / FRAG, FN = WN / FRAG;
  constexpr int PAD = 8;   // halves
  constexpr int LDA = BK + PAD, LDB = BN + PAD;
  __shared__ __align__(32) half As[BM * LDA];
  __shared__ __align__(32) half Bs[BK * LDB];

  const int tid = threadIdx.x, warp = tid / 32;
  const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  constexpr int A_CHUNKS = BM * BK / 8, B_CHUNKS = BK * BN / 8;   // 16-byte chunks per tile
  static_assert(A_CHUNKS % THREADS == 0 && B_CHUNKS % THREADS == 0, "chunks must divide evenly");

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int c = tid; c < A_CHUNKS; c += THREADS) {
      const int row = c / (BK / 8), col = (c % (BK / 8)) * 8;
      *reinterpret_cast<int4*>(&As[row * LDA + col]) =
          *reinterpret_cast<const int4*>(A + (size_t)(bm + row) * K + k0 + col);
    }
#pragma unroll
    for (int c = tid; c < B_CHUNKS; c += THREADS) {
      const int row = c / (BN / 8), col = (c % (BN / 8)) * 8;
      *reinterpret_cast<int4*>(&Bs[row * LDB + col]) =
          *reinterpret_cast<const int4*>(B + (size_t)(k0 + row) * N + bn + col);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += FRAG) {
      wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, half, wmma::row_major> a_frag[FM];
      wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, half, wmma::row_major> b_frag[FN];
#pragma unroll
      for (int i = 0; i < FM; ++i) wmma::load_matrix_sync(a_frag[i], &As[(warp_m * WM + i * FRAG) * LDA + kk], LDA);
#pragma unroll
      for (int j = 0; j < FN; ++j) wmma::load_matrix_sync(b_frag[j], &Bs[kk * LDB + warp_n * WN + j * FRAG], LDB);
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads();
  }

  // Epilogue.  Accumulator fragments of the same shape share an element order, so alpha/beta can
  // be applied element-wise against a fragment loaded from C without knowing the mapping.
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      float* cp = C + (size_t)(bm + warp_m * WM + i * FRAG) * N + bn + warp_n * WN + j * FRAG;
      if (beta != 0.f) {
        wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> c_frag;
        wmma::load_matrix_sync(c_frag, cp, N, wmma::mem_row_major);
#pragma unroll
        for (int e = 0; e < acc[i][j].num_elements; ++e) acc[i][j].x[e] = alpha * acc[i][j].x[e] + beta * c_frag.x[e];
      } else if (alpha != 1.f) {
#pragma unroll
        for (int e = 0; e < acc[i][j].num_elements; ++e) acc[i][j].x[e] *= alpha;
      }
      wmma::store_matrix_sync(cp, acc[i][j], N, wmma::mem_row_major);
    }
}

// explicit instantiation of the configuration the launcher uses (also lets NVRTC/ptxas report it)
template __global__ void hgemm_wmma_kernel<128, 128, 32, 64, 32>(int, int, int, float, const half*, const half*, float, float*);

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_wmma_fp16(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C,
                             cudaStream_t s) {
  constexpr int BM = 128, BN = 128, BK = 32, WM = 64, WN = 32;
  dim3 block((BM / WM) * (BN / WN) * 32), grid(cdiv(N, BN), cdiv(M, BM));
  auto kfn = &hgemm_wmma_kernel<BM, BN, BK, WM, WN>;
  KERNEL_LAUNCH(kfn, grid, block, 0, s)(M, N, K, alpha, A, B, beta, C);
  return {grid, block, (size_t)(BM * (BK + 8) + BK * (BN + 8)) * sizeof(half)};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
