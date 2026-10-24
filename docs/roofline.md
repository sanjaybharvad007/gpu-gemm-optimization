# Roofline analysis

The roofline model bounds the attainable throughput of a kernel by two device properties and one kernel
property:

```
attainable FLOP/s = min( peak FLOP/s,  peak bytes/s x arithmetic intensity )
arithmetic intensity (AI) = FLOPs executed / bytes moved from DRAM
```

The ridge point `peak FLOP/s / peak bytes/s` separates memory-bound kernels (left) from compute-bound
ones (right).  For an A100-80GB: fp32 19.5 TFLOP/s / 2.04 TB/s = **~10 FLOP/B**; fp16 Tensor Cores
312 TFLOP/s / 2.04 TB/s = **~153 FLOP/B**.  `tools/device_peaks.json` lists datasheet peaks for common GPUs;
`bench_gemm` also derives peaks from the device properties at run time.

## Arithmetic intensity of GEMM

With compulsory traffic only (A and B read once, C written once):

```
AI(M,N,K) = 2MNK / ((MK + KN) * in_bytes + MN * out_bytes)
square fp32:  n/6       (n=4096 -> 683 FLOP/B)
square fp16 in / fp32 out:  n/4    (n=4096 -> 1024 FLOP/B)
```

So a large square GEMM is deeply compute-bound: a 4096^3 SGEMM sits 70x to the right of the fp32 ridge,
and even the Tensor-Core ridge is 7x to its left.  **For large GEMM, DRAM bandwidth is never the
bottleneck** - which is why kernels 0-2 are so slow: they do not move the compulsory bytes, they move
N or BLOCK times more, because the reuse happens (or fails to happen) in the caches, not in DRAM.

The roofline that actually governs GEMM kernels is therefore a *hierarchical* one.  Replace DRAM with
the level the kernel is really streaming from:

| level | A100 bandwidth (approx.) | what sets the kernel's effective AI |
|---|---|---|
| DRAM | 2.0 TB/s | whole-problem reuse (always fine for GEMM) |
| L2 | ~5-7 TB/s | block tile: `2*BM*BN*BK / ((BM+BN)*BK*bytes)` -> 128x128 fp32: 32 FLOP/B |
| shared memory | 128 B/clk/SM = ~19 TB/s aggregate | thread tile: `2*TM*TN / ((TM+TN)*bytes)` -> 8x8 fp32: 2 FLOP/B |
| registers | - | accumulator reuse: each operand register feeds TM or TN FMAs |

`tools/tile_model.py` prints both the block-tile AI and the smem bytes per FLOP for a configuration:

```
$ python tools/tile_model.py --bm 128 --bn 128 --bk 8 --tm 8 --tn 8 --dtype f32
arith_intensity_flop_per_byte    32.000     # global/L2 side
smem_bytes_per_flop              0.500      # shared-memory side (1x1 thread tile would be 4.0)
```

Against the 128 B/clk smem port and the 128 FMA/clk (= 256 FLOP/clk) FP32 pipes of an SM, the smem
budget is 0.5 B/FLOP exactly - the 8x8 thread tile is the smallest that is not smem-bound, which is why
every fast SIMT GEMM uses 8x8 or 8x16.  Tensor Cores raise the FLOP rate 8x, so the operand feed must
drop below 0.06 B/FLOP: `ldmatrix` feeding 16x8x16 MMAs from a 64x32 warp tile gets there, WMMA's
load pattern does not - the quantitative reason behind the kernel 5 -> 6 step.

## Reading the plot

```bash
./build/bench_gemm --sizes 512,1024,2048,4096,8192 --csv results/bench.csv
python tools/roofline.py results/bench.csv --device "A100-SXM4-80GB" --out results/roofline.png
```

Every point is a (kernel, shape).  The table lists, per point, the binding roof and the percentage of it
achieved.  What to expect and what it means:

* **naive / coalesced** far below both roofs at huge AI: the compulsory-traffic AI is irrelevant, the
  kernels are bound by L1/L2 transactions.  Nsight: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` versus
  requests.
* **smem_tiled** at ~30% of the fp32 roof: bound by shared-memory bandwidth and barriers (two per tile).
* **register_blocked / double_buffered** at 60-85% of the fp32 roof: compute-bound; the gap is issue
  overhead (address arithmetic, barriers, epilogue) and partial waves.
* **wmma_fp16** at ~40-50% of the Tensor-Core roof: smem-instruction bound (see above).
* **mma_ptx / cuBLAS** at 70-90% of the Tensor-Core roof: Tensor-pipe bound; the rest is pipeline
  bubbles, epilogue and tail effects.  (cuBLAS fp16 on an A100 reaches ~250-280 of the 312 TFLOP/s peak.)

## Waves and tails

The achieved number also depends on how the grid maps to the SMs.  4096^2 / 128^2 = 1024 tiles over
108 SMs x 2 blocks = 4.74 waves: the last wave is 74% full, so even a perfect kernel loses ~5%.  At
2048^2 it is 256 tiles = 1.19 waves: 19% of the second wave is used and the kernel runs at ~60% of its
large-size throughput however good its inner loop is.  `tools/tile_model.py` reports `waves`;
split-K, stream-K and smaller tiles are the levers when it is small.

## Reference

Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore
Architectures.* CACM 2009.  NVIDIA Nsight Compute's "GPU Speed Of Light Roofline" section automates
the DRAM-level version of this chart for a single kernel launch.
