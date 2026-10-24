// Kernel 6 - Tensor Cores at the PTX level: mma.sync.m16n8k16 fed by ldmatrix, with a 3-stage
// cp.async pipeline.  This is the instruction mix cuBLAS / CUTLASS emit for Ampere fp16 GEMM, and
// it removes the three things WMMA hides from you:
//   * ldmatrix loads a 16x16 A tile for a whole warp in ONE instruction, straight into the register
//     layout mma.sync expects (WMMA's load_matrix_sync compiles to several shared loads + shuffles);
//   * cp.async copies global -> shared without passing through registers, so the warp can keep
//     issuing MMAs while the next two K tiles are in flight (multi-stage software pipeline);
//   * the m16n8k16 shape lets the warp tile be 64x32 with 4x4 MMAs per k16 step: 16 MMAs per
//     2 ldmatrix.x4 (A) + 2 ldmatrix.x4.trans (B), i.e. the Tensor pipe is the bottleneck, not smem.
//
// Shared-memory layout per stage: A as [128][32+8] halves, B as [32][128+8] halves (16-byte row
// padding keeps the 8 row addresses of each ldmatrix phase in distinct banks - see docs/tensor_cores.md
// for the bank arithmetic and for the XOR-swizzle alternative that avoids the padding).
// 3 stages x 18.5 KB = 55.5 KB dynamic shared memory -> opt-in above 48 KB.
#include "gemm/common.h"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

namespace {
constexpr int BM = 128, BN = 128, BK = 32, STAGES = 3;
constexpr int WARPS_M = 2, WARPS_N = 4, THREADS = WARPS_M * WARPS_N * 32;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;   // 64 x 32
constexpr int MT = WM / 16, NT = WN / 8;              // 4 x 4 mma tiles per warp
constexpr int PAD = 8;
constexpr int LDA = BK + PAD, LDB = BN + PAD;         // halves
constexpr int A_STAGE = BM * LDA, B_STAGE = BK * LDB;  // halves
[[maybe_unused]] constexpr size_t SMEM_BYTES = (size_t)STAGES * (A_STAGE + B_STAGE) * sizeof(half);
constexpr int A_CHUNKS = BM * BK / 8, B_CHUNKS = BK * BN / 8;   // 512 each -> 2 per thread
static_assert(A_CHUNKS == 2 * THREADS && B_CHUNKS == 2 * THREADS, "two 16-byte cp.async per operand per thread");
}  // namespace

__device__ __forceinline__ void load_stage(half* As, half* Bs, const half* __restrict__ A, const half* __restrict__ B,
                                           int N, int K, int bm, int bn, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int c = tid + i * THREADS;
    const int row = c / (BK / 8), col = (c % (BK / 8)) * 8;
    cp_async_16(smem_u32(As + row * LDA + col), A + (size_t)(bm + row) * K + k0 + col);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int c = tid + i * THREADS;
    const int row = c / (BN / 8), col = (c % (BN / 8)) * 8;
    cp_async_16(smem_u32(Bs + row * LDB + col), B + (size_t)(k0 + row) * N + bn + col);
  }
}

__global__ void __launch_bounds__(THREADS)
hgemm_mma_ptx_kernel(int M, int N, int K, float alpha, const half* __restrict__ A, const half* __restrict__ B,
                     float beta, float* __restrict__ C) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  half* smem = reinterpret_cast<half*>(smem_raw);
  auto As = [&](int s) { return smem + s * (A_STAGE + B_STAGE); };
  auto Bs = [&](int s) { return smem + s * (A_STAGE + B_STAGE) + A_STAGE; };

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
  const int KT = K / BK;

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  // prologue: stages 0 .. STAGES-2 in flight
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(As(s), Bs(s), A, B, N, K, bm, bn, s * BK, tid);
    cp_async_commit();
  }

  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();   // tile kt has landed (for this thread's copies)
    __syncthreads();               // ... and everyone else's; also: all reads of stage (kt-1)%STAGES are done
    {
      const int nxt = kt + STAGES - 1;
      if (nxt < KT) load_stage(As(nxt % STAGES), Bs(nxt % STAGES), A, B, N, K, bm, bn, nxt * BK, tid);
      cp_async_commit();           // always commit so the wait_group arithmetic stays uniform
    }
    const half* a_s = As(kt % STAGES);
    const half* b_s = Bs(kt % STAGES);

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t a_frag[MT][4], b_frag[NT][2];
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) {
        // lanes 0-15 -> rows 0-15 at k=kk, lanes 16-31 -> rows 0-15 at k=kk+8  (matrices a0,a1,a2,a3)
        const half* p = a_s + (warp_m * WM + mi * 16 + (lane % 16)) * LDA + kk + (lane / 16) * 8;
        ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], smem_u32(p));
      }
#pragma unroll
      for (int nj = 0; nj < NT; nj += 2) {
        // B is stored [k][n]; .trans delivers the k-contiguous "col" fragment.  One x4 covers two
        // n8 tiles: lanes 0-15 -> k rows kk..kk+15 at n0, lanes 16-31 -> the same rows at n0+8.
        const half* p = b_s + (kk + (lane % 16)) * LDB + warp_n * WN + nj * 8 + (lane / 16) * 8;
        ldmatrix_x4_trans(b_frag[nj][0], b_frag[nj][1], b_frag[nj + 1][0], b_frag[nj + 1][1], smem_u32(p));
      }
#pragma unroll
      for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) mma_m16n8k16_f16_f32(acc[mi][nj], a_frag[mi], b_frag[nj]);
    }
  }
  cp_async_wait<0>();

  // epilogue: c0,c1 -> (row = lane/4, cols 2*(lane%4)+{0,1}); c2,c3 -> row + 8
#pragma unroll
  for (int mi = 0; mi < MT; ++mi)
#pragma unroll
    for (int nj = 0; nj < NT; ++nj) {
      const int row = bm + warp_m * WM + mi * 16 + lane / 4;
      const int col = bn + warp_n * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float2* cp = reinterpret_cast<float2*>(C + (size_t)(row + h * 8) * N + col);
        float2 r = make_float2(alpha * acc[mi][nj][2 * h], alpha * acc[mi][nj][2 * h + 1]);
        if (beta != 0.f) {
          float2 c = *cp;
          r.x += beta * c.x;
          r.y += beta * c.y;
        }
        *cp = r;
      }
    }
}

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_mma_ptx(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C,
                           cudaStream_t s) {
  static bool attr_set = false;
  if (!attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(hgemm_mma_ptx_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
    attr_set = true;
  }
  dim3 block(THREADS), grid(cdiv(N, BN), cdiv(M, BM));
  auto kfn = &hgemm_mma_ptx_kernel;
  KERNEL_LAUNCH(kfn, grid, block, SMEM_BYTES, s)(M, N, K, alpha, A, B, beta, C);
  return {grid, block, SMEM_BYTES};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
