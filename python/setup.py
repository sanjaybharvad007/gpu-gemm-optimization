"""pip install -e python/   (needs nvcc; builds the kernels as a torch extension `gemm_ext._C`)."""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = [
    "registry.cu", "cublas_ref.cu", "kernels/00_naive.cu", "kernels/01_coalesced.cu", "kernels/02_smem_tiled.cu",
    "kernels/03_register_blocked.cu", "kernels/04_double_buffered.cu", "kernels/05_wmma_fp16.cu",
    "kernels/06_mma_ptx.cu", "kernels/07_split_k.cu", "kernels/08_batched.cu", "kernels/gemv.cu",
    "kernels/reductions.cu", "kernels/transpose.cu",
]
arch_flags = os.environ.get("GEMM_ARCH_FLAGS", "-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89").split()

setup(
    name="gemm_ext",
    version="1.0.0",
    packages=["gemm_ext"],
    ext_modules=[
        CUDAExtension(
            "gemm_ext._C",
            sources=[os.path.join(ROOT, "python", "csrc", "bindings.cpp")] + [os.path.join(ROOT, "src", k) for k in KERNELS],
            include_dirs=[os.path.join(ROOT, "include"), os.path.join(ROOT, "src")],
            libraries=["cublas"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--expt-relaxed-constexpr", "-lineinfo"] + arch_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
