// Public API of the kernel library.  All matrices are row-major:  C[MxN] = alpha * A[MxK] * B[KxN] + beta * C.
// FP32 kernels take float A/B/C.  Tensor-Core kernels take half A/B and accumulate/store in fp32
// (the usual "mixed precision" contract: fp16 operands, fp32 math).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace gemm {

enum class Kernel : int {
  Naive = 0,         // one thread per C element, row-major thread mapping (uncoalesced)
  Coalesced,         // same, threads mapped along N so warps read/write contiguous memory
  SmemTiled,         // 32x32 block tile staged through shared memory, one output per thread
  RegisterBlocked,   // 128x128x8 block tile, 8x8 outputs per thread, float4 loads, A transposed in smem
  DoubleBuffered,    // + register prefetch of the next K tile, two smem buffers, one barrier per tile
  SplitK,            // DoubleBuffered with K split across blockIdx.z + reduction (for small M*N, large K)
  WmmaFp16,          // Tensor Cores via nvcuda::wmma 16x16x16 fragments, fp16 in / fp32 accumulate
  MmaPtx,            // Tensor Cores via mma.sync.m16n8k16 + ldmatrix + 3-stage cp.async pipeline
  Cublas,            // cublasSgemm / cublasGemmEx reference
  Count
};

const char* kernel_name(Kernel k);
Kernel kernel_from_name(const char* name);   // exits with a message on an unknown name
bool kernel_is_fp16(Kernel k);                // true for the Tensor-Core kernels (half inputs)

// Shape constraints a kernel imposes (alignment of M/N/K).  The harness skips unsupported shapes.
struct Constraints { int m_mult, n_mult, k_mult; };
Constraints kernel_constraints(Kernel k);
inline bool kernel_supports(Kernel k, int M, int N, int K) {
  Constraints c = kernel_constraints(k);
  return M % c.m_mult == 0 && N % c.n_mult == 0 && K % c.k_mult == 0;
}

// Workspace needed by split-K (bytes); 0 for every other kernel.
size_t workspace_bytes(Kernel k, int M, int N, int K, int split_k);
int default_split_k(int M, int N, int K, int num_sms);

struct LaunchStats { dim3 grid, block; size_t smem_bytes; };

// FP32 SIMT kernels.
LaunchStats sgemm(Kernel k, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                  cudaStream_t stream, void* workspace = nullptr, int split_k = 1);
// Tensor-Core kernels: half inputs, fp32 output.
LaunchStats hgemm(Kernel k, int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C,
                  cudaStream_t stream);
// Strided batched fp32 GEMM (DoubleBuffered body, blockIdx.z = batch index).
LaunchStats sgemm_batched(int M, int N, int K, float alpha, const float* A, long long strideA, const float* B,
                          long long strideB, float beta, float* C, long long strideC, int batch, cudaStream_t stream);

// ---- Other numerical kernels (benchmarks/tests reuse the same harness) ------------------------
void sgemv(int M, int N, float alpha, const float* A, const float* x, float beta, float* y, cudaStream_t stream);
void reduce_sum(const float* x, float* partial, float* out, int n, cudaStream_t stream);   // partial: >= 1024 floats
void softmax_rows(const float* x, float* y, int rows, int cols, cudaStream_t stream);
void transpose(const float* in, float* out, int rows, int cols, cudaStream_t stream);

// ---- cuBLAS reference -------------------------------------------------------------------------
void cublas_sgemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t s);
void cublas_hgemm(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C, cudaStream_t s);
void cublas_shutdown();

}  // namespace gemm
