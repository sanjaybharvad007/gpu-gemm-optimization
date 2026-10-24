# The optimisation journey: from 2% to ~80% of cuBLAS

Every kernel in `src/kernels/` is one step; each step changes one thing and the benchmark harness
measures what that one thing bought.  This page is the narrative; the code comments at the top of each
file carry the details.  Numbers quoted are the typical fraction of cuBLAS SGEMM at M=N=K=4096 on an
A100-class part - run `bench_gemm --csv results/bench.csv && python tools/report.py results/bench.csv`
for your own GPU.

| # | kernel | what changed | why it matters | typical % cuBLAS |
|---|---|---|---|---|
| 0 | `naive` | one thread per output, `threadIdx.x` along M | adjacent lanes hit different rows: every warp load is 32 sectors | 1-3 |
| 1 | `coalesced` | `threadIdx.x` along N | one 128 B transaction per warp for B and C, broadcast for A | 10-15 |
| 2 | `smem_tiled` | 32x32 tiles staged in shared memory | A/B elements loaded once per block instead of once per thread; AI 0.5 -> 8 FLOP/B | 25-35 |
| 3 | `register_blocked` | 128x128x8 tile, 8x8 outputs per thread, float4, A transposed in smem | 64 FMAs per 16 smem reads; AI 32 FLOP/B; FMAs issue back to back from registers | 55-65 |
| 4 | `double_buffered` | next tile prefetched to registers during compute, two smem buffers, one barrier | memory latency hidden behind 512 FMAs/thread; half the barriers | 75-85 |
| 7 | `split_k` | K sliced across `blockIdx.z` + reduction | fills the GPU when M*N is small (16 tiles on 108 SMs otherwise) | 2-6x over kernel 4 at 512x512x16384 |
| 5 | `wmma_fp16` | Tensor Cores via `nvcuda::wmma` | fp16 x fp16 -> fp32 at ~8x the FP32 pipe rate | 45-60 (of cuBLAS fp16) |
| 6 | `mma_ptx` | `mma.sync.m16n8k16` + `ldmatrix` + 3-stage `cp.async` | one ldmatrix per 16x16 operand, loads bypass registers, two tiles in flight | 75-90 (of cuBLAS fp16) |

## What each step teaches

**0 -> 1: the access pattern is the algorithm.**  Same FLOPs, same instruction count, 5-8x faster.
Nsight Compute's `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` goes from ~12% to ~100%.
Nothing else in this repository matters if this is wrong.

**1 -> 2: reuse through shared memory.**  Each block loads a BLOCK x BLOCK tile once and uses every element
BLOCK times.  The arithmetic intensity of the block (FLOP per global byte) becomes BLOCK/4; the kernel
is still memory-bound because the ridge point of an A100 is ~10 FLOP/B for fp32 and the tile only reaches 8.
Also the first appearance of `__syncthreads()` - two per tile, and each costs the whole block.

**2 -> 3: register blocking.**  The real lever.  A thread computing an 8x8 patch reads 16 values from smem
per k-step and issues 64 FMAs: the smem bandwidth per FLOP drops 8x (`tools/tile_model.py` prints
`smem_bytes_per_flop`), which is what lets the FP32 pipes stay busy; A100 smem delivers 128 B/clk/SM
while the FP32 pipes consume 128 FMA/clk/SM, so anything above ~1 B/FLOP is smem-bound.  Two layout
decisions carry the step: A is stored **transposed** in smem (`As[k][m]`) so the per-thread 8 values along
M are a contiguous `float4` pair, and loads are `float4` so each thread issues one 16 B load per operand
per tile.  Cost: 64 accumulators + operands = ~110 registers, 2 blocks per SM, 25% occupancy - and that
is fine, because latency is now hidden by ILP within the thread, not by warp count.

**3 -> 4: software pipelining.**  The global loads for tile k+1 are issued before the FMAs for tile k and
land in registers while the math runs; they are written to the *other* smem buffer afterwards.  One
barrier per tile instead of two.  On an A100 this takes the SIMT kernel to ~80% of cuBLAS SGEMM; the
remaining gap is cuBLAS's larger K tile, swizzled block ordering for L2 locality and a vectorised
epilogue (see "What is left" below).

**4 -> 7: split-K.**  Not faster in general - it adds a workspace round trip - but at small M*N the
128x128 tile grid cannot fill the machine.  512x512 is 16 blocks for 108 SMs: 15% of the GPU.
Splitting K by 8 gives 128 blocks and a 5x speed-up.  `default_split_k()` applies the rule "split until
there are two blocks per SM, never below 256 K per slice"; `bench_gemm --splitk-sweep` shows the curve
and where it turns over.

**4 -> 5: Tensor Cores.**  The same tiling, with the inner product replaced by `wmma::mma_sync` on
16x16x16 fp16 fragments accumulating in fp32.  WMMA is easy and already ~8x the FP32 FLOP rate, but it
hides the fragment layout, so the loads from smem are compiled into several LDS + shuffles and the kernel
sits at ~50% of cuBLAS fp16, bound on shared-memory instructions rather than on the Tensor pipe.

**5 -> 6: PTX-level control.**  `ldmatrix.x4` loads a whole 16x16 operand for the warp in one
instruction, directly in `mma.sync`'s register layout; `cp.async` streams global -> shared without
registers so the 3-stage pipeline keeps two tiles in flight while the Tensor pipe works on the third.
Per k16 step a warp issues 2 ldmatrix (A) + 2 ldmatrix.trans (B) and 16 MMAs.  [`tensor_cores.md`](tensor_cores.md)
walks through the fragment layouts, the `.trans` trick for row-major B and the bank arithmetic behind
the 16-byte row padding.

## What is left (the last 10-20%)

In the order cuBLAS/CUTLASS do them:

1. **Larger K tile for fp32** (BK=16 or 32 with a 2-3 stage `cp.async` pipeline instead of register prefetch).
2. **Threadblock swizzle**: launch tiles in an order that keeps a 128x128 neighbourhood of A/B rows in L2
   (grouped M-raster), worth 5-10% at 8192+.
3. **Vectorised, smem-staged epilogue** so C stores are full 128 B lines per warp (both tensor-core kernels
   currently store `float2` per lane).
4. **Warp specialisation / TMA on Hopper** (`cp.async.bulk.tensor`, `wgmma.mma_async`): a different kernel,
   not a tweak - see the roadmap in the README.
5. **Stream-K** instead of split-K: assigns K iterations to persistent blocks so no wave is partial.

## How to look at it yourself

```bash
ncu --set full -k regex:sgemm_double_buffered -c 1 ./build/bench_gemm --shape 4096x4096x4096 --kernels double_buffered
```

The "GPU Speed Of Light", "Memory Workload Analysis" and "Warp State Statistics" sections tell you which
of the three regimes you are in - bytes, FLOPs, or latency - and [`profiling.md`](profiling.md) lists the
specific metrics each kernel in this repository changes.
