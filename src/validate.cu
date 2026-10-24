// Numerical accuracy validation against an fp64 host reference.
//
//   ./validate                      # all kernels, a mix of aligned and ragged shapes, alpha/beta
//   ./validate --k-sweep            # error growth with K for fp32 SIMT, fp16 Tensor Core, cuBLAS
//   ./validate --quick              # small shapes only (used by CTest)
//
// Reported per kernel: max |err|, max relative err, relative Frobenius error ||C-C*||/||C*||, and
// the pass/fail against a bound derived from the rounding model (docs/numerics.md):
//   fp32 accumulate over K terms: |err| <= c * K * eps32 * sum|a||b|   -> rel_fro ~ sqrt(K) * eps32 in practice
//   fp16 operands, fp32 accumulate: dominated by operand rounding eps16 = 2^-11 -> rel_fro ~ eps16
#include <algorithm>
#include <cstring>
#include "harness.h"

using namespace gemm;

namespace {

struct Case { int M, N, K; float alpha, beta; };

bool check_kernel(Kernel k, const Case& c, cudaStream_t s, bool verbose) {
  const int M = c.M, N = c.N, K = c.K;
  if (!kernel_supports(k, M, N, K)) return true;
  const bool fp16 = kernel_is_fp16(k);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
  fill_random(hA, 11 + K); fill_random(hB, 23 + N); fill_random(hC, 37 + M);
  std::vector<half> hA16, hB16;
  std::vector<float> A_used = hA, B_used = hB;
  if (fp16) { hA16 = to_half(hA); hB16 = to_half(hB); A_used = from_half(hA16); B_used = from_half(hB16); }   // reference sees the rounded operands

  std::vector<double> ref;
  gemm_ref_f64(M, N, K, c.alpha, A_used.data(), B_used.data(), c.beta, hC.data(), ref);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *ws = nullptr;
  half *dA16 = nullptr, *dB16 = nullptr;
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * 4));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * 4, cudaMemcpyHostToDevice));
  int split = 1;
  if (fp16) {
    CUDA_CHECK(cudaMalloc(&dA16, hA16.size() * 2)); CUDA_CHECK(cudaMalloc(&dB16, hB16.size() * 2));
    CUDA_CHECK(cudaMemcpy(dA16, hA16.data(), hA16.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB16, hB16.data(), hB16.size() * 2, cudaMemcpyHostToDevice));
    hgemm(k, M, N, K, c.alpha, dA16, dB16, c.beta, dC, s);
  } else {
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * 4)); CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    if (k == Kernel::SplitK) { split = std::max(2, default_split_k(M, N, K, 108)); CUDA_CHECK(cudaMalloc(&ws, workspace_bytes(k, M, N, K, split))); }
    sgemm(k, M, N, K, c.alpha, dA, dB, c.beta, dC, s, ws, split);
  }
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> got(hC.size());
  CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * 4, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dA16); cudaFree(dB16); cudaFree(ws);

  ErrorStats e = compare(got, ref);
  // bounds: fp32 -> 16 * sqrt(K) * eps32 on the Frobenius norm (very loose vs. observed ~1-2 sqrt(K) eps);
  //         fp16 operands -> 4 * eps16 (operand rounding dominates and does not grow with K thanks to fp32 accumulate)
  const double eps32 = 5.96e-8, eps16 = 4.88e-4;
  const double bound = fp16 ? 4 * eps16 : 16 * std::sqrt((double)K) * eps32;
  const bool ok = e.rel_fro <= bound && std::isfinite(e.max_abs);
  if (verbose || !ok)
    std::printf("  %-17s %5dx%5dx%5d a=%.2f b=%.2f  max|err| %.3e  max rel %.3e  rel_fro %.3e  bound %.1e  %s\n",
                kernel_name(k), M, N, K, c.alpha, c.beta, e.max_abs, e.max_rel, e.rel_fro, bound, ok ? "ok" : "FAIL");
  return ok;
}

void k_sweep(cudaStream_t s) {
  std::printf("\nError growth with K (M=N=256, relative Frobenius error):\n%8s %14s %14s %14s %14s\n", "K", "fp32 simt", "fp32 cublas", "fp16 wmma", "fp16 mma_ptx");
  for (int K : {32, 128, 512, 2048, 8192}) {
    std::printf("%8d", K);
    for (Kernel k : {Kernel::DoubleBuffered, Kernel::Cublas, Kernel::WmmaFp16, Kernel::MmaPtx}) {
      // reuse check_kernel's machinery by recomputing here (kept simple: same code path, verbose off)
      const int M = 256, N = 256;
      std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N, 0.f);
      fill_random(hA, 5 + K); fill_random(hB, 7 + K);
      const bool fp16 = kernel_is_fp16(k);
      std::vector<float> A_used = hA, B_used = hB;
      std::vector<half> hA16, hB16;
      if (fp16) { hA16 = to_half(hA); hB16 = to_half(hB); A_used = from_half(hA16); B_used = from_half(hB16); }
      std::vector<double> ref;
      gemm_ref_f64(M, N, K, 1.0, A_used.data(), B_used.data(), 0.0, nullptr, ref);
      float *dC = nullptr; CUDA_CHECK(cudaMalloc(&dC, hC.size() * 4)); CUDA_CHECK(cudaMemset(dC, 0, hC.size() * 4));
      if (fp16) {
        half *dA = nullptr, *dB = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, hA16.size() * 2)); CUDA_CHECK(cudaMalloc(&dB, hB16.size() * 2));
        CUDA_CHECK(cudaMemcpy(dA, hA16.data(), hA16.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB16.data(), hB16.size() * 2, cudaMemcpyHostToDevice));
        hgemm(k, M, N, K, 1.f, dA, dB, 0.f, dC, s);
        CUDA_CHECK(cudaStreamSynchronize(s)); cudaFree(dA); cudaFree(dB);
      } else {
        float *dA = nullptr, *dB = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * 4)); CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
        sgemm(k, M, N, K, 1.f, dA, dB, 0.f, dC, s);
        CUDA_CHECK(cudaStreamSynchronize(s)); cudaFree(dA); cudaFree(dB);
      }
      std::vector<float> got(hC.size());
      CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * 4, cudaMemcpyDeviceToHost));
      cudaFree(dC);
      std::printf(" %14.3e", compare(got, ref).rel_fro);
    }
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false, sweep = false, verbose = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--quick")) quick = true;
    else if (!std::strcmp(argv[i], "--k-sweep")) sweep = true;
    else if (!std::strcmp(argv[i], "--quiet")) verbose = false;
  }
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  std::vector<Case> cases = {
      {128, 128, 32, 1.f, 0.f}, {256, 256, 256, 1.f, 0.f}, {256, 384, 128, 0.5f, 2.f},
      {100, 68, 36, 1.f, 1.f}, {129, 132, 44, 1.f, 0.f}, {1, 4, 4, 1.f, 0.f},
      {512, 512, 1024, 1.f, 0.f}, {384, 256, 2048, 1.f, -1.f},
  };
  if (!quick) cases.insert(cases.end(), {{1024, 1024, 1024, 1.f, 0.f}, {2048, 1024, 512, 1.f, 0.5f}, {640, 896, 4096, 1.f, 0.f}});
  int failed = 0, run = 0;
  std::printf("validating %zu shapes x %d kernels against fp64 reference\n", cases.size(), (int)Kernel::Count);
  for (int ki = 0; ki < (int)Kernel::Count; ++ki)
    for (const Case& c : cases) {
      const Kernel k = (Kernel)ki;
      if (!kernel_supports(k, c.M, c.N, c.K)) continue;
      if ((k == Kernel::Naive || k == Kernel::Coalesced) && (double)c.M * c.N * c.K > 1e9) continue;
      ++run;
      if (!check_kernel(k, c, s, verbose)) ++failed;
    }
  if (sweep) k_sweep(s);
  cublas_shutdown();
  std::printf("%d checks, %d failed\n", run, failed);
  return failed ? 1 : 0;
}
