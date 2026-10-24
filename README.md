# cuda-gemm-optimization

**A GEMM written nine times, from a naive triple loop to `mma.sync` Tensor-Core PTX with `ldmatrix` and a multi-stage `cp.async` pipeline — each step measured against cuBLAS, placed on a roofline, validated against an fp64 reference, and explained.  Plus the kernels that surround GEMM in a transformer layer: split-K and batched GEMM, GEMV, deterministic reductions, online softmax and a bank-conflict-free transpose.**

[![CI](https://github.com/sanjaybharvad007/cuda-gemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjaybharvad007/cuda-gemm-optimization/actions/workflows/ci.yml)
[![GPU](https://github.com/sanjaybharvad007/cuda-gemm-optimization/actions/workflows/gpu.yml/badge.svg)](https://github.com/sanjaybharvad007/cuda-gemm-optimization/actions/workflows/gpu.yml)
![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%20sm__80%2B-76B900)
![cuBLAS](https://img.shields.io/badge/cuBLAS-reference-76B900)
![Tensor Cores](https://img.shields.io/badge/Tensor%20Cores-WMMA%20%7C%20mma.sync-76B900)
![Nsight](https://img.shields.io/badge/Nsight-Compute%20%7C%20Systems-76B900)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C)
![CMake](https://img.shields.io/badge/CMake-3.18%2B-064F8C)
![PyTorch](https://img.shields.io/badge/PyTorch-extension-ee4c2c)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```bash
cmake -S . -B build && cmake --build build -j
./build/validate                                   # every kernel vs fp64 reference, aligned + ragged shapes, alpha/beta
./build/bench_gemm --sizes 512,1024,2048,4096,8192 --csv results/bench.csv
python tools/roofline.py results/bench.csv --device "A100-SXM4-80GB" --out results/roofline.png
```

```
kernel             dtype      M      N      K         ms   TFLOP/s      GB/s  %cuBLAS    %peak       AI  launch
naive              f32     4096   4096   4096      ...        ...       ...      ~2      ...    682.7  128x128x1/1024/0B
coalesced          f32     4096   4096   4096      ...        ...       ...     ~12      ...    682.7  128x128x1/1024/0B
smem_tiled         f32     4096   4096   4096      ...        ...       ...     ~30      ...    682.7  128x128x1/1024/8192B
register_blocked   f32     4096   4096   4096      ...        ...       ...     ~60      ...    682.7  32x32x1/256/8192B
double_buffered    f32     4096   4096   4096      ...        ...       ...     ~80      ...    682.7  32x32x1/256/16384B
wmma_fp16          f16     4096   4096   4096      ...        ...       ...     ~50      ...   1024.0  32x32x1/256/18944B
mma_ptx            f16     4096   4096   4096      ...        ...       ...  75-90      ...   1024.0  32x32x1/256/56832B
cublas             f32/f16 4096   4096   4096      ...        ...       ...    100      ...
```

The percentages are the typical staircase on an A100-class GPU (see [docs/optimization_journey.md](docs/optimization_journey.md) for what each step changes and why); the harness prints the measured numbers for the GPU it runs on.

---

## Contents

- [Highlights](#highlights)
- [The kernels](#the-kernels)
- [Architecture](#architecture)
- [Build and run](#build-and-run)
- [Design](#design)
- [Performance methodology](#performance-methodology)
- [Numerical validation](#numerical-validation)
- [Testing and CI](#testing-and-ci)
- [Profiling](#profiling)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [References](#references)

---

## Highlights

- **Nine GEMM kernels, one change at a time.** Coalescing, shared-memory tiling, 8x8 register blocking with `float4` loads and a transposed A tile, register-prefetch double buffering, split-K, strided batched, WMMA Tensor Cores, and `mma.sync.m16n8k16` with `ldmatrix`(`.trans`) and a 3-stage `cp.async` pipeline.  Every kernel's header comment states what changed and which hardware limit it addresses.
- **Measured against cuBLAS, not against itself.** The harness reports TFLOP/s, effective bandwidth, arithmetic intensity, % of cuBLAS (`cublasSgemm` / `cublasGemmEx` fp16 with fp32 compute) and % of device peak, with launch geometry, across a size sweep; CSV out, roofline and Markdown tables from `tools/`.
- **Roofline done properly.** Compulsory-traffic intensity for the DRAM roof, plus the hierarchical version that actually governs GEMM: block-tile intensity against L2 and thread-tile intensity against shared memory.  `tools/tile_model.py` computes both for any tile configuration together with registers, occupancy limiter and wave count.
- **Mixed precision with a correct reference.** fp16 operands, fp32 accumulation; the validator rounds the operands first so it measures the kernel's error, not the input rounding, and shows the `sqrt(K)` growth for every kernel with `validate --k-sweep`.
- **Warp-level and block-level numerics beyond GEMM.** Warp-shuffle GEMV at full bandwidth, two-pass deterministic reductions (bit-identical across runs, asserted), single-pass online softmax, and the padded-tile transpose with the bank-conflict arithmetic written out.
- **Three compilers, one source.** The same `.cu` files build with nvcc (CMake), type-check with g++ through a host stub, and compile to SASS on a GPU-less CI runner with NVRTC + `ptxas` installed from PyPI - so register counts, shared memory and spills are checked on every push.
- **PyTorch binding.** `gemm_ext.matmul(a, b, kernel="mma_ptx")` runs any kernel on torch tensors for drop-in profiling and `torch.matmul` parity tests.

## The kernels

| # | file | kernel | dtype | tile | key technique | constraints |
|---|---|---|---|---|---|---|
| 0 | `00_naive.cu` | `naive` | f32 | 1 output/thread | nothing (uncoalesced baseline) | any |
| 1 | `01_coalesced.cu` | `coalesced` | f32 | 1 output/thread | threads along N: coalesced B and C, broadcast A | any |
| 2 | `02_smem_tiled.cu` | `smem_tiled` | f32 | 32x32x32 | cooperative smem tiles, predicated loads | any |
| 3 | `03_register_blocked.cu` | `register_blocked` | f32 | 128x128x8, 8x8/thread | `float4` loads, A transposed in smem, 64 accumulators | N%4, K%4 |
| 4 | `04_double_buffered.cu` | `double_buffered` | f32 | 128x128x8, 8x8/thread | register prefetch, two smem buffers, one barrier per tile | N%4, K%4 |
| 7 | `07_split_k.cu` | `split_k` | f32 | as 4, `blockIdx.z` = K slice | workspace partials + vectorised reduce; `default_split_k()` | N%4, K%4 |
| 8 | `08_batched.cu` | (`sgemm_batched`) | f32 | as 4, `blockIdx.z` = batch | strided batched, one launch | N%4, K%4 |
| 5 | `05_wmma_fp16.cu` | `wmma_fp16` | f16 -> f32 | 128x128x32, 64x32/warp | `nvcuda::wmma` 16x16x16, padded smem, fragment-wise alpha/beta | 128/128/32 |
| 6 | `06_mma_ptx.cu` | `mma_ptx` | f16 -> f32 | 128x128x32, 64x32/warp, 3 stages | `mma.sync.m16n8k16`, `ldmatrix.x4(.trans)`, `cp.async.cg`, 55.5 KB dynamic smem | 128/128/32 |
| - | `gemv.cu` | `sgemv` | f32 | warp/row | `float4` + warp shuffle | any |
| - | `reductions.cu` | `reduce_sum`, `softmax_rows` | f32 | block | shuffle tree, two-pass deterministic; online max/sum with `exp2f` | any |
| - | `transpose.cu` | `transpose` | f32 | 32x32 (+1 pad) | conflict-free smem transpose | any |

Compile-time footprint (ptxas, sm_80): `register_blocked` 113 regs / 8 KB static smem, `double_buffered` 128 regs / 16 KB dynamic, `wmma_fp16` 123 regs / 18.5 KB, `mma_ptx` 128 regs / 55.5 KB dynamic - **no spills in any kernel** (`python tools/nvrtc_compile.py`).

## Architecture

```
include/gemm/gemm.h      public API: Kernel enum, sgemm / hgemm / sgemm_batched, constraints, workspace, cuBLAS ref
include/gemm/common.h    CUDA_CHECK, cdiv, warp reductions, PTX wrappers: cp.async, ldmatrix(.trans), mma.m16n8k16
        │
src/registry.cu          name <-> Kernel, constraints, default_split_k, dispatch to the per-file launchers
src/kernels/*.cu         one kernel per file; sgemm_tile.cuh = shared 128x128x8 double-buffered main loop (4, 7, 8)
src/cublas_ref.cu        row-major cuBLAS calls (C^T = B^T A^T), GemmEx fp16 with fp32 compute
        │
src/bench_gemm.cu        sweep harness -> table + CSV          src/validate.cu   fp64 reference, bounds, K sweep
tests/cuda/              CTest suite                           tests/python/     tools' unit tests
tools/                   roofline.py  tile_model.py  occupancy.py  report.py  plot_bench.py  nvrtc_compile.py  syntax_check.sh
python/                  torch extension (gemm_ext) over the same kernels
```

## Build and run

Requirements: CUDA 12.x, CMake 3.18+, a GPU with compute capability 8.0+ for the Tensor-Core kernels (the SIMT kernels run on anything the toolkit targets).

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80   # or 86 / 89 / 90; default builds all four
cmake --build build -j
(cd build && ctest --output-on-failure)             # kernels + validate --quick

./build/bench_gemm --sizes 256,512,1024,2048,4096,8192 --csv results/bench.csv
./build/bench_gemm --shape 4096x4096x4096 --kernels double_buffered,mma_ptx,cublas --beta 1
./build/bench_gemm --splitk-sweep                    # 512x512x16384 with split 1..16
./build/bench_gemm --batched 64 --sizes 256,512      # strided batched vs looped cuBLAS
./build/validate --k-sweep                           # error growth table

python tools/report.py results/bench.csv             # Markdown tables (% cuBLAS per kernel per size)
python tools/plot_bench.py results/bench.csv --out results/tflops.png
python tools/roofline.py results/bench.csv --device "RTX 4090" --out results/roofline.png
```

PyTorch extension (optional):

```bash
pip install -e python/ --no-build-isolation
python -c "import torch, gemm_ext; a=torch.randn(512,512,device='cuda').half(); b=torch.randn(512,512,device='cuda').half(); print((gemm_ext.matmul(a,b,'mma_ptx') - a.float()@b.float()).abs().max())"
pytest -q tests/python/test_torch_ext.py
```

## Design

### One change per kernel

The repository is organised as an argument, not a library: kernel *n+1* differs from kernel *n* by exactly one idea, and the harness measures what that idea bought.  The ideas, in order: memory access pattern, reuse through shared memory, reuse through registers, latency hiding by software pipelining, filling the machine (split-K / batching), and finally the Tensor-Core instruction set, first through the WMMA abstraction and then at the PTX level where the fragment layout, the shared-memory bank mapping and the asynchronous copy pipeline are all explicit.  [docs/optimization_journey.md](docs/optimization_journey.md) is the narrative version with the Nsight counters that change at each step.

### A shared main loop for the "production" fp32 kernels

Kernels 4, 7 and 8 are the same 128x128x8 double-buffered main loop (`sgemm_tile.cuh`) with three thin wrappers: plain, `blockIdx.z` = K slice writing partials to a workspace, `blockIdx.z` = batch index offsetting pointers.  The point of split-K and batching is scheduling, not arithmetic, so they share the arithmetic.

### Tensor Cores at two levels

`wmma_fp16` shows how little code Tensor Cores need; `mma_ptx` shows what the abstraction costs.  Same tile, same numerics; the PTX kernel replaces `load_matrix_sync` (several LDS + shuffles) by one `ldmatrix.x4` per 16x16 operand, uses `.trans` to consume row-major B without a transposed copy, streams tiles with `cp.async` so loads never occupy registers, and keeps two tiles in flight.  The 16-byte row padding that makes the `ldmatrix` phases conflict-free is derived, not guessed, in [docs/tensor_cores.md](docs/tensor_cores.md), with the XOR-swizzle alternative.

### Correct before fast

Every kernel is checked against fp64 on aligned *and* ragged shapes with non-trivial alpha/beta (a kernel that forgets to read C passes every beta = 0 test).  The fp16 kernels are compared against a reference computed from the *rounded* operands so the measured error is the kernel's own.  Reductions are deterministic by construction and the test asserts bit-identity across runs.  Details and the rounding model: [docs/numerics.md](docs/numerics.md).

### Verifiable without a GPU

The hosted CI has no GPU, so it does the next best things: `ruff` + pytest on the analysis tools, a g++ type-check of every `.cu` through `src/host_stub/`, a real device compile of every kernel with NVRTC + `ptxas` (register / shared-memory / spill report), and a full nvcc build inside the `nvidia/cuda` container.  A manual `gpu` workflow runs the tests, the sweeps and the roofline on a self-hosted GPU runner and uploads `results/`.

## Performance methodology

- **Timing**: CUDA events around batches of 20 launches, median of 5 batches after 3 warm-ups; C is reset from a copy before timing so `beta != 0` paths are measured on real data.
- **FLOPs**: `2MNK` (the standard GEMM count; alpha/beta epilogue not counted).
- **Bytes**: compulsory traffic `(MK + KN) * in + MN * out` (`x2` on C when `beta != 0`) - this is what the arithmetic-intensity axis of the roofline uses, so kernels that move more than the compulsory bytes show up *below* the roof rather than as inflated bandwidth.
- **Peaks**: derived from the device properties at run time (`clock x SMs x lanes`), with datasheet values for known GPUs in `tools/device_peaks.json`; `% peak` uses the fp32 peak for SIMT kernels and the dense fp16 Tensor-Core peak for the fp16 kernels.
- **cuBLAS baseline**: `cublasSgemm` for fp32; `cublasGemmEx` (fp16 in, fp32 out, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`) for fp16 - the same mixed-precision contract as the kernels, so the comparison is like for like.
- **Sizes**: squares 256 ... 8192 by default; `--shape MxNxK` for anything else; `--splitk-sweep` for the tall-skinny regime; `--batched`.

[docs/roofline.md](docs/roofline.md) explains how to read the plot, why large GEMM is never DRAM-bound and why the shared-memory roof is the one that decides the thread-tile size (8x8 is the smallest fp32 thread tile that is not smem-bound on an SM with 128 B/clk of shared-memory bandwidth and 128 FMA/clk of FP32 throughput).

## Numerical validation

```
$ ./build/validate
validating 11 shapes x 9 kernels against fp64 reference
  naive               128x  128x   32 a=1.00 b=0.00  max|err| ...  max rel ...  rel_fro ...  bound 5.4e-06  ok
  ...
  mma_ptx             640x  896x 4096 a=1.00 b=0.00  max|err| ...  max rel ...  rel_fro ...  bound 2.0e-03  ok
NN checks, 0 failed
```

Bounds: fp32 kernels `rel_fro <= 16 sqrt(K) eps32`; fp16-operand kernels `rel_fro <= 4 eps16` against the rounded-operand reference.  `--k-sweep` prints the growth table for K = 32 ... 8192 for the SIMT, cuBLAS, WMMA and `mma.sync` kernels side by side.

## Testing and CI

| what | where | runs on |
|---|---|---|
| all GEMM kernels vs fp64, 5 shapes incl. ragged, alpha/beta; batched; GEMV; reductions (+determinism); softmax (large dynamic range); transpose | `tests/cuda/test_kernels.cu` (CTest `kernels`) | GPU |
| validator, quick shape set | CTest `validate_quick` | GPU |
| torch parity for every kernel, bmm, softmax | `tests/python/test_torch_ext.py` | GPU (skipped otherwise) |
| occupancy calculator, tile model, roofline maths, report tables | `tests/python/test_*.py` (11 tests) | CPU |
| g++ type-check of every `.cu` via host stub | `tools/syntax_check.sh` | CPU |
| NVRTC + ptxas device compile, spill check, sm_80/86/89/90 | `tools/nvrtc_compile.py` | CPU |
| full nvcc build, `-Xptxas -v` report | `.github/workflows/ci.yml` (`nvidia/cuda` container) | CPU |

## Profiling

```bash
ncu --set full -k regex:mma_ptx -c 1 ./build/bench_gemm --shape 4096x4096x4096 --kernels mma_ptx
nsys profile -t cuda -o bench ./build/bench_gemm --sizes 1024,4096
```

[docs/profiling.md](docs/profiling.md) maps each optimisation step to the Nsight Compute metrics it moves (sector efficiency, LSU:FMA ratio, barrier and long-scoreboard stalls, Tensor-pipe utilisation, LDSM/LDGSTS counts, bank conflicts) and the stall reason -> fix table.

## Repository layout

```
include/gemm/        gemm.h  common.h
src/                 registry.cu  cublas_ref.cu  harness.h  bench_gemm.cu  validate.cu
src/kernels/         00_naive 01_coalesced 02_smem_tiled 03_register_blocked 04_double_buffered
                     05_wmma_fp16 06_mma_ptx 07_split_k 08_batched  gemv reductions transpose  sgemm_tile.cuh kernels.h
src/host_stub/       cuda_runtime.h cuda_fp16.h mma.h cublas_v2.h  (g++ type-check only)
tests/cuda/          test_kernels.cu            tests/python/  test_occupancy test_tile_model test_roofline test_report test_torch_ext
tools/               roofline.py tile_model.py occupancy.py report.py plot_bench.py nvrtc_compile.py syntax_check.sh device_peaks.json
python/              setup.py  gemm_ext/  csrc/bindings.cpp
docs/                optimization_journey.md roofline.md tensor_cores.md numerics.md profiling.md
results/             sample_bench.csv (synthetic fixture for the tools)  + your bench.csv / roofline.png
.github/workflows/   ci.yml (CPU: lint, tests, host stub, NVRTC+ptxas, nvcc build)   gpu.yml (self-hosted GPU runner)
```

## Roadmap

- fp32 main loop with BK=32 and a 3-stage `cp.async` pipeline (close the last ~15% to cuBLAS SGEMM)
- Threadblock swizzle for L2 locality; smem-staged vectorised epilogue for the Tensor-Core kernels
- XOR-swizzled smem layouts as a compile-time alternative to padding
- Stream-K scheduling; persistent kernel
- Hopper path: `wgmma.mma_async` + TMA with producer/consumer warp specialisation
- BF16 and FP8 (`e4m3`) operand variants with per-tensor scaling
- Fused epilogues (bias + GELU, residual add) and INT8 `mma.m16n8k32`

## References

- NVIDIA. *CUDA C++ Programming Guide*; *CUDA C++ Best Practices Guide*; *PTX ISA* (mma, ldmatrix, cp.async).
- NVIDIA. *CUTLASS: CUDA Templates for Linear Algebra Subroutines* (github.com/NVIDIA/cutlass) and the CUTLASS GEMM API documentation.
- Williams, S., Waterman, A., Patterson, D. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 2009.
- Volkov, V. *Better Performance at Lower Occupancy.* GTC 2010.
- Lai, J., Seznec, A. *Performance Upper Bound Analysis and Optimization of SGEMM on Fermi and Kepler GPUs.* CGO 2013.
- Markidis, S. et al. *NVIDIA Tensor Core Programmability, Performance & Precision.* IPDPSW 2018.
- Higham, N. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002 (ch. 3, inner products).
- Harris, M. *An Efficient Matrix Transpose in CUDA C/C++* and *Optimizing Parallel Reduction in CUDA.* NVIDIA Developer Blog.
- Osama, M. et al. *Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU.* PPoPP 2023.

## License

MIT. See [LICENSE](LICENSE).
