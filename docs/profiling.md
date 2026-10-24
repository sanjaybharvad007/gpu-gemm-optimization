# Profiling the kernels with Nsight

Build with line info (default, `-lineinfo`) so Nsight Compute can attribute stalls to source lines.

## Nsight Compute: one kernel, every counter

```bash
# one launch of a specific kernel, full metric set
ncu --set full -k regex:double_buffered -c 1 -o db ./build/bench_gemm --shape 4096x4096x4096 --kernels double_buffered
ncu --import db.ncu-rep --page details | less

# the staircase in one run: one launch of each kernel at 2048^3
ncu --set full -k regex:"sgemm|hgemm" -c 9 -o all ./build/bench_gemm --sizes 2048 --iters 1
```

Metrics that change at each step of [`optimization_journey.md`](optimization_journey.md):

| step | metric | expect |
|---|---|---|
| 0 -> 1 coalescing | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | ~12% -> ~100% |
| | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` per request | 32 -> 4 |
| 1 -> 2 smem tiling | `dram__bytes_read.sum` | drops ~BLOCK x; `lts__t_bytes` (L2) too |
| | `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | appears (two barriers per tile) |
| 2 -> 3 register blocking | `smsp__inst_executed_pipe_lsu.sum` / `pipe_fma.sum` | LSU:FMA ratio 1:1 -> 1:4 |
| | `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | ~25% -> ~70% |
| | `launch__registers_per_thread`, `sm__warps_active.avg.pct_of_peak_sustained_active` | ~110 regs, occupancy ~25% - and faster |
| 3 -> 4 double buffering | `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | halves (global latency hidden) |
| | `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | halves (one barrier per tile) |
| 4 -> 7 split-K (512x512x16384) | `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | 15% -> 90%+ (SMs no longer idle) |
| 4 -> 5 WMMA | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | from 0 to ~40%; `inst_executed_pipe_lsu` high |
| 5 -> 6 mma.sync + ldmatrix + cp.async | `sm__pipe_tensor_cycles_active` | ~40% -> ~75-85% |
| | `smsp__inst_executed_op_ldsm.sum` vs `op_shared_ld.sum` | LDSM replaces LDS |
| | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | 0 (padding) |
| | `smsp__inst_executed_op_ldgsts.sum` | cp.async present (LDGSTS) |

The "Warp State Statistics" section names the dominant stall reason; the mapping to fixes is:

* `long_scoreboard` - waiting on global/L2 loads: prefetch deeper (more stages), larger tiles.
* `barrier` - `__syncthreads` cost: fewer barriers per tile (double buffering), fewer, bigger tiles.
* `mio_throttle` / `short_scoreboard` - shared-memory instruction pressure: register blocking,
  `ldmatrix`, vector LDS, conflict-free layout.
* `math_pipe_throttle` on the tensor pipe - you are done; look at tails and the epilogue.
* `not_selected` with high issue rate - also done (plenty of eligible warps).

## Nsight Systems: the timeline

```bash
nsys profile -t cuda,nvtx -o bench ./build/bench_gemm --sizes 1024,4096 --kernels double_buffered,mma_ptx,cublas
nsys stats --report cuda_gpu_kern_sum bench.nsys-rep
```

Useful for the split-K and batched cases: gaps between launches, the reduce kernel's share, and whether
the batched launch really replaced N small launches with one.

## Compile-time numbers without a GPU

```bash
cmake -S . -B build -DGEMM_PTXAS_VERBOSE=ON && cmake --build build     # nvcc: registers, smem, spills per kernel
python tools/nvrtc_compile.py --arch sm_80                              # same via NVRTC + ptxas from PyPI wheels
python tools/occupancy.py --threads 256 --regs 128 --smem 56832 --arch sm80
```

`ptxas -v` is the first check after any change to a tile size: a kernel that spills (non-zero "bytes
spill stores") has lost before it runs.  All kernels in this repository compile with zero spills at
their default configurations for sm_80/86/89/90.
