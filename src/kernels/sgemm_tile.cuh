// The double-buffered 128x128x8 fp32 main loop, shared by kernels 4 (plain), 7 (split-K) and
// 8 (batched).  A block computes the tile C[bm:bm+BM, bn:bn+BN] over K range [k_begin, k_end).
//
// Double buffering: while the FMAs run on smem buffer `cur`, the NEXT K tile is already in flight
// from global memory into registers (issued before the compute loop); after the loop it is written
// into buffer `cur ^ 1`.  One __syncthreads per K tile instead of two, and global-memory latency is
// hidden behind BK*TM*TN = 512 FMAs per thread.
#pragma once
#include "gemm/common.h"

namespace gemm {

template <int BM, int BN, int BK, int TM, int TN>
struct SgemmTile {
  static constexpr int THREADS = (BM / TM) * (BN / TN);
  static constexpr size_t SMEM_BYTES = 2 * (BM + BN) * BK * sizeof(float);
  static_assert(BM * BK / 4 == THREADS && BK * BN / 4 == THREADS, "one float4 of A and of B per thread");

  __device__ __forceinline__ static void load_global(const float* __restrict__ A, const float* __restrict__ B, int M,
                                                     int N, int K, int bm, int bn, int k0, int a_row, int a_col,
                                                     int b_row, int b_col, float4& a4, float4& b4) {
    a4 = make_float4(0.f, 0.f, 0.f, 0.f);
    b4 = a4;
    if (bm + a_row < M && k0 + a_col + 3 < K)
      a4 = *reinterpret_cast<const float4*>(A + (size_t)(bm + a_row) * K + k0 + a_col);
    if (k0 + b_row < K && bn + b_col + 3 < N)
      b4 = *reinterpret_cast<const float4*>(B + (size_t)(k0 + b_row) * N + bn + b_col);
  }

  __device__ __forceinline__ static void store_smem(float (*As)[BM], float (*Bs)[BN], int a_row, int a_col, int b_row,
                                                    int b_col, const float4& a4, const float4& b4) {
    As[a_col + 0][a_row] = a4.x;
    As[a_col + 1][a_row] = a4.y;
    As[a_col + 2][a_row] = a4.z;
    As[a_col + 3][a_row] = a4.w;
    *reinterpret_cast<float4*>(&Bs[b_row][b_col]) = b4;
  }

  // Runs the main loop; acc must be zero-initialised by the caller.  smem points at SMEM_BYTES of
  // dynamic shared memory laid out as As[2][BK][BM] then Bs[2][BK][BN].
  __device__ __forceinline__ static void run(float* smem, const float* __restrict__ A, const float* __restrict__ B,
                                             int M, int N, int K, int bm, int bn, int k_begin, int k_end,
                                             float (&acc)[TM][TN]) {
    float (*As)[BK][BM] = reinterpret_cast<float (*)[BK][BM]>(smem);
    float (*Bs)[BK][BN] = reinterpret_cast<float (*)[BK][BN]>(smem + 2 * BK * BM);
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int a_row = tid / (BK / 4), a_col = (tid % (BK / 4)) * 4;
    const int b_row = tid / (BN / 4), b_col = (tid % (BN / 4)) * 4;

    float4 a4, b4;
    load_global(A, B, M, N, K, bm, bn, k_begin, a_row, a_col, b_row, b_col, a4, b4);
    store_smem(As[0], Bs[0], a_row, a_col, b_row, b_col, a4, b4);
    __syncthreads();

    int cur = 0;
    float a_frag[TM], b_frag[TN];
    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
      const int k_next = k0 + BK;
      const bool has_next = k_next < k_end;
      if (has_next) load_global(A, B, M, N, K, bm, bn, k_next, a_row, a_col, b_row, b_col, a4, b4);   // in flight

#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
        for (int i = 0; i < TM; i += 4)
          *reinterpret_cast<float4*>(&a_frag[i]) = *reinterpret_cast<const float4*>(&As[cur][kk][ty * TM + i]);
#pragma unroll
        for (int j = 0; j < TN; j += 4)
          *reinterpret_cast<float4*>(&b_frag[j]) = *reinterpret_cast<const float4*>(&Bs[cur][kk][tx * TN + j]);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
      }

      if (has_next) {
        store_smem(As[cur ^ 1], Bs[cur ^ 1], a_row, a_col, b_row, b_col, a4, b4);   // other buffer: no hazard
        cur ^= 1;
      }
      __syncthreads();   // next tile visible to all threads / this tile done before it is overwritten
    }
  }

  // C = alpha*acc + beta*C for this thread's TMxTN patch.
  __device__ __forceinline__ static void epilogue(float* __restrict__ C, int M, int N, int bm, int bn, float alpha,
                                                  float beta, const float (&acc)[TM][TN]) {
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
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
};

using Tile128 = SgemmTile<128, 128, 8, 8, 8>;

}  // namespace gemm
