// Benchmark harness: every kernel vs cuBLAS across a sweep of square (or explicit MxNxK) shapes.
//
//   ./bench_gemm --sizes 256,512,1024,2048,4096,8192 --kernels all --csv results/bench.csv
//   ./bench_gemm --shape 4096x4096x4096 --kernels double_buffered,mma_ptx,cublas
//   ./bench_gemm --splitk-sweep                 # M=N=512, K=16384 with split 1,2,4,8,16
//   ./bench_gemm --batched 64 --sizes 256       # strided batched GEMM, 64 x (256x256x256)
//
// Columns: time, TFLOP/s, effective GB/s against the compulsory traffic, % of cuBLAS, % of device
// peak (fp32 or Tensor-Core peak for the fp16 kernels), arithmetic intensity, grid/block/smem.
// tools/roofline.py plots the CSV.
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include "harness.h"

using namespace gemm;

namespace {

struct Args {
  std::vector<int> sizes{256, 512, 1024, 2048, 4096};
  int M = 0, N = 0, K = 0;
  std::vector<std::string> kernels{"all"};
  std::string csv;
  int iters = 20, batched = 0, split_k = 0;
  bool splitk_sweep = false;
  float beta = 0.f;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (k == "--sizes") a.sizes = parse_int_list(next());
    else if (k == "--shape") { std::string s = next(); std::replace(s.begin(), s.end(), 'x', ','); auto v = parse_int_list(s); a.M = v[0]; a.N = v[1]; a.K = v[2]; }
    else if (k == "--kernels") { std::string s = next(); a.kernels.clear(); size_t p = 0; while (p <= s.size()) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size(); a.kernels.push_back(s.substr(p, c - p)); p = c + 1; } }
    else if (k == "--csv") a.csv = next();
    else if (k == "--iters") a.iters = std::stoi(next());
    else if (k == "--batched") a.batched = std::stoi(next());
    else if (k == "--split-k") a.split_k = std::stoi(next());
    else if (k == "--splitk-sweep") a.splitk_sweep = true;
    else if (k == "--beta") a.beta = std::stof(next());
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  return a;
}

std::vector<Kernel> select_kernels(const std::vector<std::string>& names) {
  std::vector<Kernel> ks;
  for (const auto& n : names) {
    if (n == "all") { for (int i = 0; i < (int)Kernel::Count; ++i) ks.push_back((Kernel)i); }
    else if (n == "fp32") { for (int i = 0; i < (int)Kernel::Count; ++i) if (!kernel_is_fp16((Kernel)i)) ks.push_back((Kernel)i); }
    else if (n == "fp16") { ks = {Kernel::WmmaFp16, Kernel::MmaPtx, Kernel::Cublas}; }
    else ks.push_back(kernel_from_name(n.c_str()));
  }
  return ks;
}

struct Buffers {
  float *A32 = nullptr, *B32 = nullptr, *C = nullptr, *C0 = nullptr, *ws = nullptr;
  half *A16 = nullptr, *B16 = nullptr;
  size_t ws_bytes = 0;
};

Buffers alloc(int M, int N, int K, int batch, size_t ws_bytes) {
  Buffers b;
  const size_t a = (size_t)M * K * batch, bb = (size_t)K * N * batch, c = (size_t)M * N * batch;
  CUDA_CHECK(cudaMalloc(&b.A32, a * 4)); CUDA_CHECK(cudaMalloc(&b.B32, bb * 4));
  CUDA_CHECK(cudaMalloc(&b.C, c * 4)); CUDA_CHECK(cudaMalloc(&b.C0, c * 4));
  CUDA_CHECK(cudaMalloc(&b.A16, a * 2)); CUDA_CHECK(cudaMalloc(&b.B16, bb * 2));
  if (ws_bytes) CUDA_CHECK(cudaMalloc(&b.ws, ws_bytes));
  b.ws_bytes = ws_bytes;
  std::vector<float> hA(a), hB(bb), hC(c);
  fill_random(hA, 1); fill_random(hB, 2); fill_random(hC, 3);
  CUDA_CHECK(cudaMemcpy(b.A32, hA.data(), a * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.B32, hB.data(), bb * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.C0, hC.data(), c * 4, cudaMemcpyHostToDevice));
  auto hA16 = to_half(hA), hB16 = to_half(hB);
  CUDA_CHECK(cudaMemcpy(b.A16, hA16.data(), a * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.B16, hB16.data(), bb * 2, cudaMemcpyHostToDevice));
  return b;
}
void release(Buffers& b) { cudaFree(b.A32); cudaFree(b.B32); cudaFree(b.C); cudaFree(b.C0); cudaFree(b.A16); cudaFree(b.B16); if (b.ws) cudaFree(b.ws); }

void print_header(std::FILE* f, bool /*csv*/) {
  std::fprintf(f, "%-18s %-5s %6s %6s %6s %10s %9s %9s %8s %8s %8s  %s\n", "kernel", "dtype", "M", "N", "K", "ms", "TFLOP/s", "GB/s", "%cuBLAS", "%peak", "AI", "launch");
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  DeviceInfo dev = query_device();
  std::printf("# %s: %d SMs, sm_%d%d, fp32 peak %.1f TFLOP/s, fp16 TC peak %.1f TFLOP/s, HBM %.0f GB/s, L2 %.0f MB\n",
              dev.name.c_str(), dev.sms, dev.major, dev.minor, dev.fp32_tflops, dev.fp16_tc_tflops, dev.hbm_gbs, dev.l2_mb);
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  std::ofstream csv;
  print_header(stdout, false);
  if (!args.csv.empty()) { csv.open(args.csv); csv << "kernel,dtype,M,N,K,batch,split_k,ms,tflops,gbs,pct_cublas,pct_peak,arith_intensity,grid,block,smem_bytes\n"; }

  std::vector<std::array<int, 3>> shapes;
  if (args.M) shapes.push_back({args.M, args.N, args.K});
  else for (int n : args.sizes) shapes.push_back({n, n, n});
  if (args.splitk_sweep) shapes = {{512, 512, 16384}};

  std::vector<Kernel> kernels = select_kernels(args.kernels);
  if (args.splitk_sweep) kernels = {Kernel::DoubleBuffered, Kernel::SplitK, Kernel::Cublas};

  for (auto& sh : shapes) {
    const int M = sh[0], N = sh[1], K = sh[2];
    const int batch = std::max(1, args.batched);
    std::vector<int> splits{args.split_k ? args.split_k : default_split_k(M, N, K, dev.sms)};
    if (args.splitk_sweep) splits = {1, 2, 4, 8, 16};
    const size_t ws = workspace_bytes(Kernel::SplitK, M, N, K, *std::max_element(splits.begin(), splits.end()));
    Buffers buf = alloc(M, N, K, batch, ws);
    const double flops = gemm_flops(M, N, K) * batch;
    float cublas32_ms = 0.f, cublas16_ms = 0.f;
    const bool read_c = args.beta != 0.f;

    auto run = [&](Kernel k, bool fp16, int split) -> float {
      const size_t cbytes = (size_t)M * N * batch * 4;
      auto fn = [&]() {
        if (args.batched) {
          if (k == Kernel::Cublas) { for (int z = 0; z < batch; ++z) cublas_sgemm(M, N, K, 1.f, buf.A32 + (size_t)z * M * K, buf.B32 + (size_t)z * K * N, args.beta, buf.C + (size_t)z * M * N, s); }
          else sgemm_batched(M, N, K, 1.f, buf.A32, (long long)M * K, buf.B32, (long long)K * N, args.beta, buf.C, (long long)M * N, batch, s);
        } else if (fp16) hgemm(k, M, N, K, 1.f, buf.A16, buf.B16, args.beta, buf.C, s);
        else sgemm(k, M, N, K, 1.f, buf.A32, buf.B32, args.beta, buf.C, s, buf.ws, split);
      };
      CUDA_CHECK(cudaMemcpyAsync(buf.C, buf.C0, cbytes, cudaMemcpyDeviceToDevice, s));
      fn();
      CUDA_CHECK(cudaStreamSynchronize(s));
      CUDA_CHECK(cudaGetLastError());
      return time_kernel_ms(fn, s, args.iters);
    };

    // cuBLAS first so every other row can be expressed as a percentage of it
    const bool tc_shape = M % 128 == 0 && N % 128 == 0 && K % 32 == 0;
    cublas32_ms = run(Kernel::Cublas, false, 1);
    if (tc_shape && !args.batched) cublas16_ms = run(Kernel::Cublas, true, 1);

    for (Kernel k : kernels) {
      const bool fp16 = kernel_is_fp16(k);
      if (args.batched && fp16) continue;
      if (!kernel_supports(k, M, N, K)) { std::printf("%-18s %-5s %6d %6d %6d   (shape not supported: needs M%%%d N%%%d K%%%d)\n", kernel_name(k), fp16 ? "f16" : "f32", M, N, K, kernel_constraints(k).m_mult, kernel_constraints(k).n_mult, kernel_constraints(k).k_mult); continue; }
      // naive kernels take seconds at 4096+; cap them
      if ((k == Kernel::Naive || k == Kernel::Coalesced) && (double)M * N * K > 1.2e11) { std::printf("%-18s %-5s %6d %6d %6d   (skipped: too slow at this size)\n", kernel_name(k), "f32", M, N, K); continue; }
      const std::vector<int> ks_splits = (k == Kernel::SplitK) ? splits : std::vector<int>{1};
      for (int split : ks_splits) {
        const float ms = (k == Kernel::Cublas) ? cublas32_ms : run(k, fp16, split);
        const float ms16 = (k == Kernel::Cublas) ? cublas16_ms : 0.f;
        auto report = [&](const char* dtype, float t, bool is16) {
          LaunchStats st{};
          if (k != Kernel::Cublas) st = is16 ? hgemm(k, M, N, K, 0.f, buf.A16, buf.B16, 1.f, buf.C, s) : sgemm(k, M, N, K, 0.f, buf.A32, buf.B32, 1.f, buf.C, s, buf.ws, split);
          const double tflops = flops / (t * 1e-3) / 1e12;
          const double bytes = gemm_min_bytes(M, N, K, is16 ? 2 : 4, 4, read_c) * batch;
          const double gbs = bytes / (t * 1e-3) / 1e9;
          const float ref = is16 ? cublas16_ms : cublas32_ms;
          const double pct_cublas = ref > 0 ? 100.0 * ref / t : 0.0;
          const double peak = is16 ? dev.fp16_tc_tflops : dev.fp32_tflops;
          const double pct_peak = peak > 0 ? 100.0 * tflops / peak : 0.0;
          const double ai = flops / bytes;
          char launch[96];
          std::snprintf(launch, sizeof launch, "%ux%ux%u/%u/%zuB", st.grid.x, st.grid.y, st.grid.z, st.block.x, st.smem_bytes);
          std::string name = kernel_name(k);
          if (k == Kernel::SplitK) name += "(" + std::to_string(split) + ")";
          std::printf("%-18s %-5s %6d %6d %6d %10.4f %9.2f %9.1f %8.1f %8.1f %8.1f  %s\n", name.c_str(), dtype, M, N, K, t, tflops, gbs, pct_cublas, pct_peak, ai, k == Kernel::Cublas ? "-" : launch);
          if (csv.is_open()) csv << kernel_name(k) << ',' << dtype << ',' << M << ',' << N << ',' << K << ',' << batch << ',' << split << ',' << t << ',' << tflops << ',' << gbs << ',' << pct_cublas << ',' << pct_peak << ',' << ai << ',' << st.grid.x << 'x' << st.grid.y << 'x' << st.grid.z << ',' << st.block.x << ',' << st.smem_bytes << '\n';
        };
        report(fp16 ? "f16" : "f32", ms, fp16);
        if (ms16 > 0) report("f16", ms16, true);
      }
    }
    release(buf);
    std::printf("\n");
  }
  cublas_shutdown();
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
