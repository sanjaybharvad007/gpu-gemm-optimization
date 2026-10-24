// Host-side helpers shared by the benchmark, the validator and the tests.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include "gemm/common.h"
#include "gemm/gemm.h"

namespace gemm {

struct DeviceInfo {
  std::string name;
  int sms = 0, major = 0, minor = 0;
  double fp32_tflops = 0, fp16_tc_tflops = 0, hbm_gbs = 0, l2_mb = 0, smem_per_sm_kb = 0;
};

// Peak numbers derived from the device properties (clock * SMs * lanes), fp16 Tensor-Core peak
// as the architecture multiple of FP32 (sm80: 8x dense, sm89/90: 8x; older: 4x via hmma).
inline DeviceInfo query_device() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  DeviceInfo d;
  d.name = p.name;
  d.sms = p.multiProcessorCount;
  d.major = p.major;
  d.minor = p.minor;
  const int fp32_lanes = (p.major >= 8) ? 128 : 64;        // Ampere+ : 128 FP32 lanes per SM
  const double ghz = p.clockRate * 1e-6;
  d.fp32_tflops = 2.0 * fp32_lanes * d.sms * ghz * 1e-3;    // 2 FLOP per FMA
  const double tc_mult = (p.major >= 8) ? 8.0 : (p.major == 7 ? 4.0 : 0.0);
  d.fp16_tc_tflops = d.fp32_tflops * tc_mult;
  d.hbm_gbs = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) * 1e-9;   // DDR: x2
  d.l2_mb = p.l2CacheSize / (1024.0 * 1024.0);
  d.smem_per_sm_kb = p.sharedMemPerMultiprocessor / 1024.0;
  return d;
}

inline void fill_random(std::vector<float>& v, uint32_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : v) x = dist(rng);
}

inline std::vector<half> to_half(const std::vector<float>& v) {
  std::vector<half> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
  return h;
}
inline std::vector<float> from_half(const std::vector<half>& h) {
  std::vector<float> v(h.size());
  for (size_t i = 0; i < h.size(); ++i) v[i] = __half2float(h[i]);
  return v;
}

// fp64 reference GEMM on the host (the "ground truth" for every accuracy number in the repo).
inline void gemm_ref_f64(int M, int N, int K, double alpha, const float* A, const float* B, double beta,
                         const float* C_in, std::vector<double>& C) {
  C.assign((size_t)M * N, 0.0);
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      const double a = A[(size_t)i * K + k];
      const float* brow = B + (size_t)k * N;
      double* crow = C.data() + (size_t)i * N;
      for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
    }
    for (int j = 0; j < N; ++j) C[(size_t)i * N + j] = alpha * C[(size_t)i * N + j] + beta * (C_in ? C_in[(size_t)i * N + j] : 0.0);
  }
}

struct ErrorStats { double max_abs = 0, max_rel = 0, rel_fro = 0, mean_abs = 0; };

inline ErrorStats compare(const std::vector<float>& got, const std::vector<double>& ref) {
  ErrorStats e;
  double num = 0, den = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::fabs((double)got[i] - ref[i]);
    e.max_abs = std::max(e.max_abs, d);
    e.mean_abs += d;
    e.max_rel = std::max(e.max_rel, d / std::max(1e-6, std::fabs(ref[i])));
    num += d * d;
    den += ref[i] * ref[i];
  }
  e.mean_abs /= std::max<size_t>(1, got.size());
  e.rel_fro = std::sqrt(num / std::max(den, 1e-300));
  return e;
}

struct GpuTimer {
  cudaEvent_t start{}, stop{};
  GpuTimer() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
  ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  void begin(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(start, s)); }
  float end_ms(cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// Times fn() with warm-up, reports the median of `reps` timed batches of `iters` launches each.
template <typename F>
float time_kernel_ms(F fn, cudaStream_t s, int iters = 20, int reps = 5, int warmup = 3) {
  GpuTimer t;
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<float> samples;
  for (int r = 0; r < reps; ++r) {
    t.begin(s);
    for (int i = 0; i < iters; ++i) fn();
    samples.push_back(t.end_ms(s) / iters);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

inline double gemm_flops(int M, int N, int K) { return 2.0 * M * N * K; }
// Minimum compulsory traffic: read A and B once, write C once (plus read C when beta != 0).
inline double gemm_min_bytes(int M, int N, int K, int in_bytes, int out_bytes, bool read_c) {
  return ((double)M * K + (double)K * N) * in_bytes + (double)M * N * out_bytes * (read_c ? 2 : 1);
}

inline std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t comma = s.find(',', pos);
    if (comma == std::string::npos) comma = s.size();
    out.push_back(std::stoi(s.substr(pos, comma - pos)));
    pos = comma + 1;
  }
  return out;
}

}  // namespace gemm
