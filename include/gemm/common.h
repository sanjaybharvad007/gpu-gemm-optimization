// Shared helpers for every kernel: error checking, ceil-div, vector types, warp reductions,
// cp.async / ldmatrix / mma.sync wrappers (PTX), and the KERNEL_LAUNCH macro that lets the
// .cu files type-check under plain g++ in CI (see src/host_stub/).
#pragma once
#include <cuda_fp16.h>
#ifdef __CUDACC_RTC__
// NVRTC (tools/nvrtc_compile.py) compiles the device code only: no runtime API, no host helpers.
#include <cuda/std/cstdint>
using cuda::std::uint32_t;
using cuda::std::uintptr_t;
#else
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#endif

#ifndef __CUDACC_RTC__
#define CUDA_CHECK(expr)                                                                      \
  do {                                                                                        \
    cudaError_t err__ = (expr);                                                               \
    if (err__ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,         \
                   cudaGetErrorString(err__));                                                \
      std::exit(1);                                                                           \
    }                                                                                         \
  } while (0)

#ifdef GEMM_HOST_STUB
#define KERNEL_LAUNCH(kernel, grid, block, smem, stream) if (false) (*(kernel))
#else
#define KERNEL_LAUNCH(kernel, grid, block, smem, stream) (*(kernel))<<<(grid), (block), (smem), (stream)>>>
#endif
#endif  // !__CUDACC_RTC__

namespace gemm {

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000u); }

// ---- warp-level reductions -------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// ---- PTX wrappers (sm_80+) -------------------------------------------------------------------
// Address conversion for shared memory operands of cp.async / ldmatrix.
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
#ifdef GEMM_HOST_STUB
  return (uint32_t)(uintptr_t)p;
#else
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
#endif
}

// 16-byte asynchronous global->shared copy (bypasses registers, L1 allocate .cg = cache global only).
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
}
// Zero-filling variant: copies `src_bytes` (0 or 16) and fills the rest with zeros - used for ragged tails.
__device__ __forceinline__ void cp_async_16_zfill(uint32_t smem_addr, const void* gmem_ptr, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ldmatrix: each thread supplies one row address (16 B); four 8x8 b16 matrices land in r0..r3
// distributed in the exact layout mma.sync expects for its A / B fragments.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// D(16x8,f32) += A(16x16,f16,row) * B(16x8,f16,col)
__device__ __forceinline__ void mma_m16n8k16_f16_f32(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace gemm
