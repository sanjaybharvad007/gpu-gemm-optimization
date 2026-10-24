// Internal launch entry points, one per kernel file.  registry.cu maps gemm::Kernel onto these.
#pragma once
#include "gemm/gemm.h"

namespace gemm {
LaunchStats launch_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t s);
LaunchStats launch_coalesced(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t s);
LaunchStats launch_smem_tiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t s);
LaunchStats launch_register_blocked(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t s);
LaunchStats launch_double_buffered(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t s);
LaunchStats launch_split_k(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                           float* workspace, int split_k, cudaStream_t s);
LaunchStats launch_batched(int M, int N, int K, float alpha, const float* A, long long sA, const float* B, long long sB,
                           float beta, float* C, long long sC, int batch, cudaStream_t s);
LaunchStats launch_wmma_fp16(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C, cudaStream_t s);
LaunchStats launch_mma_ptx(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C, cudaStream_t s);
}  // namespace gemm
