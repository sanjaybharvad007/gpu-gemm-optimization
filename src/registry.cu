// Maps gemm::Kernel onto the per-file launchers, and holds the per-kernel metadata the harness,
// the tests and the Python binding rely on (names, operand type, shape constraints, workspace).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "gemm/common.h"
#include "gemm/gemm.h"
#include "kernels/kernels.h"

namespace gemm {

namespace {
struct Meta { Kernel k; const char* name; bool fp16; Constraints c; };
const Meta kMeta[] = {
    {Kernel::Naive, "naive", false, {1, 1, 1}},
    {Kernel::Coalesced, "coalesced", false, {1, 1, 1}},
    {Kernel::SmemTiled, "smem_tiled", false, {1, 1, 1}},
    {Kernel::RegisterBlocked, "register_blocked", false, {1, 4, 4}},
    {Kernel::DoubleBuffered, "double_buffered", false, {1, 4, 4}},
    {Kernel::SplitK, "split_k", false, {1, 4, 4}},
    {Kernel::WmmaFp16, "wmma_fp16", true, {128, 128, 32}},
    {Kernel::MmaPtx, "mma_ptx", true, {128, 128, 32}},
    {Kernel::Cublas, "cublas", false, {1, 1, 1}},
};
const Meta& meta(Kernel k) { return kMeta[static_cast<int>(k)]; }
}  // namespace

const char* kernel_name(Kernel k) { return meta(k).name; }
bool kernel_is_fp16(Kernel k) { return meta(k).fp16; }
Constraints kernel_constraints(Kernel k) { return meta(k).c; }

Kernel kernel_from_name(const char* name) {
  for (const Meta& m : kMeta)
    if (std::strcmp(m.name, name) == 0) return m.k;
  std::fprintf(stderr, "unknown kernel '%s'; known:", name);
  for (const Meta& m : kMeta) std::fprintf(stderr, " %s", m.name);
  std::fprintf(stderr, "\n");
  std::exit(2);
}

size_t workspace_bytes(Kernel k, int M, int N, int /*K*/, int split_k) {
  return k == Kernel::SplitK ? (size_t)split_k * M * N * sizeof(float) : 0;
}

// Split until the 128x128 tile grid fills every SM at least twice, but never below 256 K per slice.
int default_split_k(int M, int N, int K, int num_sms) {
  const int tiles = cdiv(M, 128) * cdiv(N, 128);
  int split = 1;
  while (tiles * split < 2 * num_sms && (K / (split * 2)) >= 256) split *= 2;
  return split;
}

LaunchStats sgemm(Kernel k, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                  cudaStream_t s, void* workspace, int split_k) {
  switch (k) {
    case Kernel::Naive: return launch_naive(M, N, K, alpha, A, B, beta, C, s);
    case Kernel::Coalesced: return launch_coalesced(M, N, K, alpha, A, B, beta, C, s);
    case Kernel::SmemTiled: return launch_smem_tiled(M, N, K, alpha, A, B, beta, C, s);
    case Kernel::RegisterBlocked: return launch_register_blocked(M, N, K, alpha, A, B, beta, C, s);
    case Kernel::DoubleBuffered: return launch_double_buffered(M, N, K, alpha, A, B, beta, C, s);
    case Kernel::SplitK: return launch_split_k(M, N, K, alpha, A, B, beta, C, static_cast<float*>(workspace), split_k, s);
    case Kernel::Cublas: cublas_sgemm(M, N, K, alpha, A, B, beta, C, s); return {dim3(), dim3(), 0};
    default: std::fprintf(stderr, "sgemm: %s is an fp16 kernel\n", kernel_name(k)); std::exit(2);
  }
}

LaunchStats hgemm(Kernel k, int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C,
                  cudaStream_t s) {
  switch (k) {
    case Kernel::WmmaFp16: return launch_wmma_fp16(M, N, K, alpha, A, B, beta, C, s);
    case Kernel::MmaPtx: return launch_mma_ptx(M, N, K, alpha, A, B, beta, C, s);
    case Kernel::Cublas: cublas_hgemm(M, N, K, alpha, A, B, beta, C, s); return {dim3(), dim3(), 0};
    default: std::fprintf(stderr, "hgemm: %s is an fp32 kernel\n", kernel_name(k)); std::exit(2);
  }
}

LaunchStats sgemm_batched(int M, int N, int K, float alpha, const float* A, long long sA, const float* B, long long sB,
                          float beta, float* C, long long sC, int batch, cudaStream_t s) {
  return launch_batched(M, N, K, alpha, A, sA, B, sB, beta, C, sC, batch, s);
}

}  // namespace gemm
