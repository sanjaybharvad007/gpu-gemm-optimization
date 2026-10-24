// Kernel 7 - split-K.  A 128x128 block tile over C needs M*N/16384 blocks; for M=N=512 that is
// 16 blocks, i.e. 16 of the 108-132 SMs busy and the rest idle, however large K is.  Split-K
// gives blockIdx.z a slice of K: each z-slice computes a partial product into a workspace
// [split][M][N] and a second kernel reduces the slices and applies alpha/beta.
//
// Cost: an extra M*N*split*4 bytes written and read; worth it exactly when the grid would
// otherwise under-fill the GPU - default_split_k() encodes that rule.
#include "sgemm_tile.cuh"
#ifndef __CUDACC_RTC__
#include "kernels.h"
#endif

namespace gemm {

__global__ void __launch_bounds__(Tile128::THREADS)
sgemm_split_k_kernel(int M, int N, int K, int k_per_split, const float* __restrict__ A,
                     const float* __restrict__ B, float* __restrict__ workspace) {
  extern __shared__ float smem[];
  const int bm = blockIdx.y * 128, bn = blockIdx.x * 128;
  const int k_begin = blockIdx.z * k_per_split;
  const int k_end = min(K, k_begin + k_per_split);
  float acc[8][8] = {};
  if (k_begin < k_end) Tile128::run(smem, A, B, M, N, K, bm, bn, k_begin, k_end, acc);
  float* slice = workspace + (size_t)blockIdx.z * M * N;
  Tile128::epilogue(slice, M, N, bm, bn, 1.f, 0.f, acc);
}

__global__ void split_k_reduce_kernel(int MN, int split, float alpha, const float* __restrict__ workspace, float beta,
                                      float* __restrict__ C) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= MN) return;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int z = 0; z < split; ++z) {
    float4 p = *reinterpret_cast<const float4*>(workspace + (size_t)z * MN + i);
    sum.x += p.x; sum.y += p.y; sum.z += p.z; sum.w += p.w;
  }
  float4* cp = reinterpret_cast<float4*>(C + i);
  float4 r = make_float4(alpha * sum.x, alpha * sum.y, alpha * sum.z, alpha * sum.w);
  if (beta != 0.f) {
    float4 c = *cp;
    r.x += beta * c.x; r.y += beta * c.y; r.z += beta * c.z; r.w += beta * c.w;
  }
  *cp = r;
}

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
LaunchStats launch_split_k(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                           float* workspace, int split_k, cudaStream_t s) {
  split_k = max(1, split_k);
  // slices are whole multiples of the K tile so no tile straddles two slices
  int k_per_split = cdiv(cdiv(K, 8), split_k) * 8;
  split_k = cdiv(K, k_per_split);
  dim3 block(Tile128::THREADS), grid(cdiv(N, 128), cdiv(M, 128), split_k);
  auto kfn = &sgemm_split_k_kernel;
  KERNEL_LAUNCH(kfn, grid, block, Tile128::SMEM_BYTES, s)(M, N, K, k_per_split, A, B, workspace);
  const int MN = M * N;
  auto rfn = &split_k_reduce_kernel;
  KERNEL_LAUNCH(rfn, dim3(cdiv(MN / 4, 256)), dim3(256), 0, s)(MN, split_k, alpha, workspace, beta, C);
  return {grid, block, Tile128::SMEM_BYTES};
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
