// CTest suite: every kernel against the fp64 reference on small shapes, plus the non-GEMM kernels
// (GEMV, reductions, softmax, transpose) and batched GEMM.  Run with `cd build && ctest`.
#include <cstring>
#include "../../src/harness.h"

using namespace gemm;

namespace {
int g_failed = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_failed; } } while (0)

template <typename T> T* upload(const std::vector<T>& v) { T* d = nullptr; CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T))); CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <typename T> std::vector<T> download(const T* d, size_t n) { std::vector<T> v(n); CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }

void test_gemm_kernels(cudaStream_t s) {
  struct Sh { int M, N, K; float alpha, beta; };
  const Sh shapes[] = {{128, 128, 32, 1.f, 0.f}, {256, 128, 64, 0.5f, 1.5f}, {64, 96, 40, 1.f, 0.f}, {33, 20, 12, 2.f, -1.f}, {256, 256, 512, 1.f, 0.f}};
  for (int ki = 0; ki < (int)Kernel::Count; ++ki) {
    const Kernel k = (Kernel)ki;
    for (const Sh& sh : shapes) {
      if (!kernel_supports(k, sh.M, sh.N, sh.K)) continue;
      std::vector<float> A((size_t)sh.M * sh.K), B((size_t)sh.K * sh.N), C((size_t)sh.M * sh.N);
      fill_random(A, 1); fill_random(B, 2); fill_random(C, 3);
      std::vector<float> Au = A, Bu = B;
      std::vector<half> A16, B16;
      if (kernel_is_fp16(k)) { A16 = to_half(A); B16 = to_half(B); Au = from_half(A16); Bu = from_half(B16); }
      std::vector<double> ref; gemm_ref_f64(sh.M, sh.N, sh.K, sh.alpha, Au.data(), Bu.data(), sh.beta, C.data(), ref);
      float* dC = upload(C);
      if (kernel_is_fp16(k)) { half *dA = upload(A16), *dB = upload(B16); hgemm(k, sh.M, sh.N, sh.K, sh.alpha, dA, dB, sh.beta, dC, s); CUDA_CHECK(cudaStreamSynchronize(s)); cudaFree(dA); cudaFree(dB); }
      else {
        float *dA = upload(A), *dB = upload(B), *ws = nullptr; int split = 1;
        if (k == Kernel::SplitK) { split = 4; CUDA_CHECK(cudaMalloc(&ws, workspace_bytes(k, sh.M, sh.N, sh.K, split))); }
        sgemm(k, sh.M, sh.N, sh.K, sh.alpha, dA, dB, sh.beta, dC, s, ws, split);
        CUDA_CHECK(cudaStreamSynchronize(s)); cudaFree(dA); cudaFree(dB); cudaFree(ws);
      }
      CUDA_CHECK(cudaGetLastError());
      ErrorStats e = compare(download(dC, C.size()), ref);
      cudaFree(dC);
      const double tol = kernel_is_fp16(k) ? 2e-3 : 1e-5 * std::sqrt((double)sh.K);
      EXPECT(e.rel_fro <= tol, "%s %dx%dx%d alpha=%g beta=%g rel_fro=%.3e > %.1e", kernel_name(k), sh.M, sh.N, sh.K, sh.alpha, sh.beta, e.rel_fro, tol);
    }
  }
  std::printf("gemm kernels: done\n");
}

void test_batched(cudaStream_t s) {
  const int M = 64, N = 128, K = 96, batch = 5;
  std::vector<float> A((size_t)batch * M * K), B((size_t)batch * K * N), C((size_t)batch * M * N);
  fill_random(A, 4); fill_random(B, 5); fill_random(C, 6);
  float *dA = upload(A), *dB = upload(B), *dC = upload(C);
  sgemm_batched(M, N, K, 1.f, dA, (long long)M * K, dB, (long long)K * N, 0.5f, dC, (long long)M * N, batch, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  auto got = download(dC, C.size());
  for (int z = 0; z < batch; ++z) {
    std::vector<double> ref;
    gemm_ref_f64(M, N, K, 1.0, A.data() + (size_t)z * M * K, B.data() + (size_t)z * K * N, 0.5, C.data() + (size_t)z * M * N, ref);
    std::vector<float> slice(got.begin() + (size_t)z * M * N, got.begin() + (size_t)(z + 1) * M * N);
    EXPECT(compare(slice, ref).rel_fro < 1e-4, "batched z=%d", z);
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  std::printf("batched: done\n");
}

void test_gemv(cudaStream_t s) {
  const int M = 301, N = 1027;   // ragged N exercises the scalar tail
  std::vector<float> A((size_t)M * N), x(N), y(M);
  fill_random(A, 7); fill_random(x, 8); fill_random(y, 9);
  float *dA = upload(A), *dx = upload(x), *dy = upload(y);
  sgemv(M, N, 2.f, dA, dx, 1.f, dy, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  auto got = download(dy, M);
  for (int i = 0; i < M; ++i) {
    double acc = 0; for (int j = 0; j < N; ++j) acc += (double)A[(size_t)i * N + j] * x[j];
    const double want = 2.0 * acc + y[i];
    EXPECT(std::fabs(got[i] - want) < 1e-3 * (1 + std::fabs(want)), "gemv row %d: %g vs %g", i, got[i], want);
  }
  cudaFree(dA); cudaFree(dx); cudaFree(dy);
  std::printf("gemv: done\n");
}

void test_reduce_softmax(cudaStream_t s) {
  const int n = 1 << 20;
  std::vector<float> x(n + 3);   // non-multiple of 4
  fill_random(x, 10);
  float *dx = upload(x), *partial = nullptr, *out = nullptr;
  CUDA_CHECK(cudaMalloc(&partial, 1024 * 4)); CUDA_CHECK(cudaMalloc(&out, 4));
  reduce_sum(dx, partial, out, (int)x.size(), s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  double want = 0; for (float v : x) want += v;
  const float got = download(out, 1)[0];
  EXPECT(std::fabs(got - want) < 1e-2, "reduce_sum %g vs %g", got, want);
  // determinism: run twice, bit-identical
  reduce_sum(dx, partial, out, (int)x.size(), s); CUDA_CHECK(cudaStreamSynchronize(s));
  EXPECT(download(out, 1)[0] == got, "reduce_sum not deterministic");
  cudaFree(dx); cudaFree(partial); cudaFree(out);

  const int rows = 17, cols = 1000;
  std::vector<float> m((size_t)rows * cols);
  fill_random(m, 11, -30.f, 30.f);   // large range: a naive exp() overflows without the max subtraction
  float *dm = upload(m), *dy = nullptr; CUDA_CHECK(cudaMalloc(&dy, m.size() * 4));
  softmax_rows(dm, dy, rows, cols, s); CUDA_CHECK(cudaStreamSynchronize(s));
  auto y = download(dy, m.size());
  for (int r = 0; r < rows; ++r) {
    double mx = -1e30; for (int c = 0; c < cols; ++c) mx = std::max(mx, (double)m[(size_t)r * cols + c]);
    double sum = 0; for (int c = 0; c < cols; ++c) sum += std::exp(m[(size_t)r * cols + c] - mx);
    double rowsum = 0, maxerr = 0;
    for (int c = 0; c < cols; ++c) { const double want = std::exp(m[(size_t)r * cols + c] - mx) / sum; maxerr = std::max(maxerr, std::fabs(want - y[(size_t)r * cols + c])); rowsum += y[(size_t)r * cols + c]; }
    EXPECT(maxerr < 1e-6 && std::fabs(rowsum - 1) < 1e-5, "softmax row %d maxerr %g rowsum %g", r, maxerr, rowsum);
  }
  cudaFree(dm); cudaFree(dy);
  std::printf("reduce/softmax: done\n");
}

void test_transpose(cudaStream_t s) {
  const int rows = 70, cols = 45;
  std::vector<float> in((size_t)rows * cols);
  for (size_t i = 0; i < in.size(); ++i) in[i] = (float)i;
  float *din = upload(in), *dout = nullptr; CUDA_CHECK(cudaMalloc(&dout, in.size() * 4));
  transpose(din, dout, rows, cols, s); CUDA_CHECK(cudaStreamSynchronize(s));
  auto out = download(dout, in.size());
  bool ok = true;
  for (int r = 0; r < rows && ok; ++r) for (int c = 0; c < cols; ++c) if (out[(size_t)c * rows + r] != in[(size_t)r * cols + c]) { ok = false; break; }
  EXPECT(ok, "transpose mismatch");
  cudaFree(din); cudaFree(dout);
  std::printf("transpose: done\n");
}
}  // namespace

int main() {
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  test_gemm_kernels(s);
  test_batched(s);
  test_gemv(s);
  test_reduce_softmax(s);
  test_transpose(s);
  cublas_shutdown();
  std::printf(g_failed ? "%d FAILURES\n" : "all tests passed\n", g_failed);
  return g_failed ? 1 : 0;
}
