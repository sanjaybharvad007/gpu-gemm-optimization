// PyTorch binding: run every kernel on torch tensors so the Python tests can compare against
// torch.matmul and the kernels can be dropped into a model for profiling.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "gemm/gemm.h"

namespace {

gemm::Kernel parse(const std::string& name) { return gemm::kernel_from_name(name.c_str()); }

torch::Tensor matmul(torch::Tensor a, torch::Tensor b, const std::string& kernel, double alpha, int64_t split_k) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "shape mismatch");
  a = a.contiguous();
  b = b.contiguous();
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  gemm::Kernel k = parse(kernel);
  TORCH_CHECK(gemm::kernel_supports(k, M, N, K), kernel, " does not support ", M, "x", N, "x", K,
              " (needs M%", gemm::kernel_constraints(k).m_mult, " N%", gemm::kernel_constraints(k).n_mult, " K%",
              gemm::kernel_constraints(k).k_mult, ")");
  auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (gemm::kernel_is_fp16(k)) {
    TORCH_CHECK(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf, kernel, " needs fp16 inputs");
    gemm::hgemm(k, M, N, K, (float)alpha, reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(b.data_ptr<at::Half>()), 0.f, c.data_ptr<float>(), stream);
  } else {
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, kernel, " needs fp32 inputs");
    torch::Tensor ws;
    int split = (int)split_k;
    if (k == gemm::Kernel::SplitK) {
      if (split <= 0) split = gemm::default_split_k(M, N, K, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
      ws = torch::empty({(int64_t)gemm::workspace_bytes(k, M, N, K, split) / 4}, c.options());
    }
    gemm::sgemm(k, M, N, K, (float)alpha, a.data_ptr<float>(), b.data_ptr<float>(), 0.f, c.data_ptr<float>(), stream,
                ws.defined() ? ws.data_ptr<float>() : nullptr, split);
  }
  return c;
}

torch::Tensor bmm(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) && a.size(2) == b.size(1), "shape mismatch");
  a = a.contiguous().to(torch::kFloat32);
  b = b.contiguous().to(torch::kFloat32);
  const int batch = a.size(0), M = a.size(1), K = a.size(2), N = b.size(2);
  auto c = torch::empty({batch, M, N}, a.options());
  gemm::sgemm_batched(M, N, K, 1.f, a.data_ptr<float>(), (long long)M * K, b.data_ptr<float>(), (long long)K * N, 0.f,
                      c.data_ptr<float>(), (long long)M * N, batch, at::cuda::getCurrentCUDAStream());
  return c;
}

torch::Tensor softmax(torch::Tensor x) {
  x = x.contiguous().to(torch::kFloat32);
  auto y = torch::empty_like(x);
  gemm::softmax_rows(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), at::cuda::getCurrentCUDAStream());
  return y;
}

std::vector<std::string> kernels() {
  std::vector<std::string> out;
  for (int i = 0; i < (int)gemm::Kernel::Count; ++i) out.push_back(gemm::kernel_name((gemm::Kernel)i));
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul, py::arg("a"), py::arg("b"), py::arg("kernel") = "double_buffered", py::arg("alpha") = 1.0, py::arg("split_k") = 0);
  m.def("bmm", &bmm);
  m.def("softmax", &softmax);
  m.def("kernels", &kernels);
}
