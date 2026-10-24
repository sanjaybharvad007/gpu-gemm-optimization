// cuBLAS reference.  cuBLAS is column-major; a row-major C = A*B is the column-major C^T = B^T*A^T,
// so the call is sgemm(N, N, n, m, k, B, ldb=n, A, lda=k, C, ldc=n) with no transposes or copies.
// FP16 uses cublasGemmEx with fp32 compute (the same mixed-precision contract as our kernels) and
// lets cuBLAS pick its Tensor-Core algorithm.
#include <cublas_v2.h>
#include "gemm/common.h"
#include "gemm/gemm.h"

namespace gemm {

namespace {
cublasHandle_t g_handle = nullptr;
cublasHandle_t handle(cudaStream_t s) {
  if (!g_handle) {
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublasCreate failed\n"); std::exit(1); }
  }
  cublasSetStream(g_handle, s);
  return g_handle;
}
}  // namespace

void cublas_sgemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t s) {
  cublasSgemm(handle(s), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void cublas_hgemm(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C, cudaStream_t s) {
  cublasGemmEx(handle(s), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
               CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void cublas_shutdown() {
  if (g_handle) cublasDestroy(g_handle);
  g_handle = nullptr;
}

}  // namespace gemm
