"""GPU tests through the torch extension (skipped when it is not built)."""
import pytest

torch = pytest.importorskip("torch")
gemm_ext = pytest.importorskip("gemm_ext")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and gemm_ext.available()), reason="needs CUDA + built extension")

FP32 = ["naive", "coalesced", "smem_tiled", "register_blocked", "double_buffered", "split_k", "cublas"]
FP16 = ["wmma_fp16", "mma_ptx"]


@pytest.mark.parametrize("kernel", FP32)
@pytest.mark.parametrize("shape", [(128, 128, 32), (256, 384, 512), (100, 68, 36)])
def test_fp32_matches_torch(kernel, shape):
    M, N, K = shape
    if kernel in ("register_blocked", "double_buffered", "split_k") and (N % 4 or K % 4):
        pytest.skip("alignment constraint")
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = gemm_ext.matmul(a, b, kernel)
    torch.testing.assert_close(c, a @ b, rtol=1e-4, atol=1e-3 * K**0.5)


@pytest.mark.parametrize("kernel", FP16)
def test_fp16_matches_torch(kernel):
    a = torch.randn(256, 512, device="cuda", dtype=torch.half)
    b = torch.randn(512, 384, device="cuda", dtype=torch.half)
    c = gemm_ext.matmul(a, b, kernel)
    ref = (a.float() @ b.float())
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=2e-2)


def test_bmm_and_softmax():
    a = torch.randn(4, 64, 96, device="cuda")
    b = torch.randn(4, 96, 128, device="cuda")
    torch.testing.assert_close(gemm_ext.bmm(a, b), a @ b, rtol=1e-4, atol=1e-3)
    x = torch.randn(17, 1000, device="cuda") * 30
    torch.testing.assert_close(gemm_ext.softmax(x), torch.softmax(x, dim=1), rtol=1e-5, atol=1e-6)
