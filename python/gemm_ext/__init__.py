"""Thin Python surface over the CUDA kernels (built by `pip install -e python/`)."""
from __future__ import annotations

try:
    from . import _C  # type: ignore[attr-defined]
except ImportError:  # extension not built (CPU-only CI)
    _C = None

KERNELS = ["naive", "coalesced", "smem_tiled", "register_blocked", "double_buffered", "split_k", "wmma_fp16", "mma_ptx", "cublas"]


def available() -> bool:
    return _C is not None


def matmul(a, b, kernel: str = "double_buffered", alpha: float = 1.0, split_k: int = 0):
    """C = alpha * a @ b with the named kernel.  fp32 kernels take fp32 inputs, Tensor-Core kernels fp16; output is fp32."""
    if _C is None:
        raise RuntimeError("gemm_ext._C is not built; run `pip install -e python/` on a machine with nvcc")
    return _C.matmul(a, b, kernel, alpha, split_k)


def bmm(a, b):
    if _C is None:
        raise RuntimeError("gemm_ext._C is not built")
    return _C.bmm(a, b)


def softmax(x):
    if _C is None:
        raise RuntimeError("gemm_ext._C is not built")
    return _C.softmax(x)
