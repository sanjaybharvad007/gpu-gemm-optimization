// Reductions and row softmax - the building blocks around GEMM in a transformer layer.
//   reduce_sum : grid-stride float4 loads, warp shuffles, one partial per block, second tiny kernel
//                (no atomics -> deterministic, bit-reproducible across runs).
//   softmax    : one block per row, online max/sum in a single pass over the row, exp2f with the
//                log2(e) folded into the subtraction - the same trick FlashAttention uses.
#include "gemm/common.h"

namespace gemm {

__device__ __forceinline__ float block_sum(float v, float* shared /* >= 32 floats */) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32, tid = threadIdx.x;
  v = warp_sum(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = (tid < warps) ? shared[tid] : 0.f;
  if (warp == 0) v = warp_sum(v);
  __syncthreads();   // shared may be reused by the caller
  return v;
}

__device__ __forceinline__ float block_max(float v, float* shared) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32, tid = threadIdx.x;
  v = warp_max(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = (tid < warps) ? shared[tid] : neg_inf();
  if (warp == 0) v = warp_max(v);
  __syncthreads();
  return v;
}

__global__ void reduce_sum_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
  __shared__ float shared[32];
  float acc = 0.f;
  const int n4 = n / 4;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    const float4 v = reinterpret_cast<const float4*>(x)[i];
    acc += (v.x + v.y) + (v.z + v.w);
  }
  if (blockIdx.x == 0)
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) acc += x[i];
  acc = block_sum(acc, shared);
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

template <int THREADS>
__global__ void softmax_rows_kernel(const float* __restrict__ x, float* __restrict__ y, int cols) {
  __shared__ float shared[32];
  const float* row = x + (size_t)blockIdx.x * cols;
  float* out = y + (size_t)blockIdx.x * cols;
  // online (max, sum) in one pass: when a bigger max arrives, rescale the running sum
  float m = neg_inf(), l = 0.f;
  for (int i = threadIdx.x; i < cols; i += THREADS) {
    const float v = row[i];
    const float m_new = fmaxf(m, v);
    l = l * exp2f((m - m_new) * 1.4426950408889634f) + exp2f((v - m_new) * 1.4426950408889634f);
    m = m_new;
  }
  const float m_blk = block_max(m, shared);
  l = block_sum(l * exp2f((m - m_blk) * 1.4426950408889634f), shared);
  const float inv = 1.f / l;
  for (int i = threadIdx.x; i < cols; i += THREADS) out[i] = exp2f((row[i] - m_blk) * 1.4426950408889634f) * inv;
}

// explicit instantiation of the configuration the launcher uses (also lets NVRTC/ptxas report it)
template __global__ void softmax_rows_kernel<256>(const float*, float*, int);

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
void reduce_sum(const float* x, float* partial, float* out, int n, cudaStream_t s) {
  const int blocks = min(1024, cdiv(n, 256 * 4 * 4));
  auto kfn = &reduce_sum_kernel;
  KERNEL_LAUNCH(kfn, dim3(max(1, blocks)), dim3(256), 0, s)(x, partial, n);
  KERNEL_LAUNCH(kfn, dim3(1), dim3(1024), 0, s)(partial, out, max(1, blocks));
}

void softmax_rows(const float* x, float* y, int rows, int cols, cudaStream_t s) {
  auto kfn = &softmax_rows_kernel<256>;
  KERNEL_LAUNCH(kfn, dim3(rows), dim3(256), 0, s)(x, y, cols);
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
