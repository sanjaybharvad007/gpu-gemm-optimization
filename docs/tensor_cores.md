# Tensor Cores: WMMA, `mma.sync`, `ldmatrix`, `cp.async`

Two kernels use the Tensor Cores.  `05_wmma_fp16.cu` goes through the `nvcuda::wmma` API; `06_mma_ptx.cu`
issues the PTX instructions directly.  Same tile (128x128x32, 8 warps, warp tile 64x32), same numerics
(fp16 operands, fp32 accumulate), ~1.6x apart in throughput.  This page explains where the 1.6x comes from.

## The instruction: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

One warp computes `D[16x8] += A[16x16] * B[16x8]` in a single instruction.  Operands live in registers
with a fixed per-lane distribution (PTX ISA, "Matrix Fragments for mma.m16n8k16"):

```
lane l, groupID g = l / 4, t = l % 4

A (16x16, row):  a0 = A[g   ][2t..2t+1]   a1 = A[g+8][2t..2t+1]
                 a2 = A[g   ][2t+8..2t+9] a3 = A[g+8][2t+8..2t+9]     (4 x b32, 2 halves each)
B (16x8, col):   b0 = B[2t..2t+1][g]      b1 = B[2t+8..2t+9][g]       (2 x b32: consecutive k for one n)
C/D (16x8 f32):  c0,c1 = C[g][2t], C[g][2t+1]    c2,c3 = C[g+8][2t], C[g+8][2t+1]
```

Filling these registers from shared memory with ordinary loads takes 4 LDS.32 for A plus shuffles -
that is what WMMA's `load_matrix_sync` compiles to - and the shared-memory instruction rate, not the
Tensor pipe, becomes the limit.

## `ldmatrix`: the load that matches the fragment

`ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr]` loads four 8x8 b16 matrices.  Each
lane supplies the address of one 16-byte row (lanes 0-7 -> rows of matrix 0, 8-15 -> matrix 1, ...),
and each lane receives, in `r_i`, the two consecutive elements of matrix i that the `mma` fragment
layout wants it to hold.  For a 16x16 A tile at `(row0, k0)` in smem:

```
p = &As[row0 + (lane % 16)][k0 + (lane / 16) * 8];   // lanes 0-15: k0, lanes 16-31: k0+8
ldmatrix_x4(a0, a1, a2, a3, p);
```

gives matrices (rows 0-7,k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15) = exactly
a0..a3.  One instruction per 16x16 operand per warp, no shuffles.

### Row-major B and `.trans`

`mma`'s B fragment is k-contiguous per column ("col" layout), but our B tile is stored `[k][n]` with n
contiguous, so each lane needs a *column* of the stored tile.  `ldmatrix ... .trans` transposes each
8x8 matrix on the way to the registers: with row addresses `&Bs[k0 + (lane % 16)][n0 + (lane / 16) * 8]`
the x4 form returns `b0,b1` for the n8 tile at `n0` and `b0,b1` for the tile at `n0 + 8`.  Two
`ldmatrix.x4.trans` cover the four n8 tiles of a 64x32 warp tile; no transposed copy of B is ever made.

## Shared-memory bank conflicts and the 16-byte padding

Shared memory has 32 banks x 4 bytes; a 128 B line spans all banks.  `ldmatrix` is executed in
phases of 8 lanes, each phase reading 8 rows x 16 B; it is conflict-free iff those 8 row segments fall
in 8 distinct 16-byte bank groups, i.e. their byte offsets are distinct modulo 128.

* A tile stored as `[128][32]` halves: row stride 64 B.  Rows r and r+2 are 128 B apart -> same banks ->
  **4-way conflict**.
* Padded to `[128][40]` (stride 80 B): offsets of 8 consecutive rows mod 128 are 0,80,32,112,64,16,96,48 ->
  all distinct -> **conflict-free**.
* B tile `[32][128]` halves: stride 256 B -> every row in the same banks -> **8-way conflict** for
  `.trans` reads.  Padded to `[32][136]` (stride 272 B): offsets mod 128 = 0,16,32,...,112 -> conflict-free.

The alternative to padding is an **XOR swizzle**: store 16-byte chunk `c` of row `r` at chunk index
`c ^ f(r)` (for the A tile, `f(r) = (r >> 1) & 3`; for B, `f(k) = k & 7`) and apply the same XOR when
forming `ldmatrix` addresses.  It saves the padding (10% of the tile) and is what CUTLASS does; padding
was chosen here because the bank arithmetic is visible in the layout instead of hidden in the address
function.  Either way, Nsight Compute's `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` should
read zero for kernel 6.

`cp.async` writes also need the destination 16-byte aligned: 80 B and 272 B row strides are both
multiples of 16, and each stage base (10240 B + 8704 B) is too.

## `cp.async` and the multi-stage pipeline

`cp.async.cg.shared.global [smem], [gmem], 16` copies 16 bytes from global to shared memory without
staging through registers (`.cg`: cache in L2 only - the data is consumed once).  Copies are grouped
with `commit_group` and awaited with `wait_group N` ("all but the newest N groups are complete").

With `STAGES = 3`:

```
prologue:  issue tile 0 -> stage 0, commit;  issue tile 1 -> stage 1, commit
for kt:    wait_group<1>          // tile kt landed (this thread's copies)
           __syncthreads()        // everyone's copies; and all reads of stage (kt-1)%3 are finished
           issue tile kt+2 -> stage (kt+2)%3, commit    // overwrites the stage freed by the barrier
           ldmatrix / mma on stage kt%3
```

Two tiles are always in flight while one is being consumed, so a 32-deep K tile (16 MMAs x 2 k-steps x
4x4 per warp = 32 `mma.sync`, ~1 us of Tensor-pipe work per block at 4096^3) completely hides the
~600-800 cycle global latency.  The empty `commit_group` after the last real tile keeps the
`wait_group` arithmetic uniform so the loop has no tail special-case.

Cost: 3 x 18.5 KB = 55.5 KB of dynamic shared memory per block, above the 48 KB default - hence
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` in the launcher.  On an
A100 (164 KB/SM) two blocks fit; `python tools/occupancy.py --threads 256 --regs 128 --smem 56832`.

## Why the fp32 accumulator matters (mixed precision)

fp16 has an 11-bit significand and a maximum of 65504.  Products of two fp16 values are exact in fp32;
*sums* of K such products in fp16 lose ~`sqrt(K) * 2^-11` relative accuracy and overflow for large K
or poorly scaled data.  Accumulating in fp32 - which the Tensor Cores do natively - keeps the error at
the level of the operand rounding (`2^-11 ~ 4.9e-4` relative), independent of K.  [`numerics.md`](numerics.md)
measures it; `validate --k-sweep` prints the table for your GPU.

## Beyond Ampere

* **sm_89 (Ada)**: same instructions, plus FP8 (`e4m3`/`e5m2`) `mma` shapes.
* **sm_90 (Hopper)**: `wgmma.mma_async` operates on a whole warpgroup (4 warps) with operands in
  shared memory described by matrix descriptors, and TMA (`cp.async.bulk.tensor`) moves whole tiles with
  one instruction.  The pipeline becomes producer/consumer warp-specialised.  The kernels here run on
  Hopper unchanged (`mma.sync` is supported) but reach a smaller fraction of its peak.

## References

* NVIDIA. *PTX ISA* - "Warp Level Matrix Multiply-Accumulate Instructions" (`mma`, `ldmatrix`), "Data
  Movement and Conversion Instructions" (`cp.async`).
* NVIDIA. *CUDA C++ Programming Guide* - Warp Matrix Functions (WMMA); Asynchronous Data Copies.
* Thakkar et al. *CUTLASS* (github.com/NVIDIA/cutlass) - `mma_sm80.h`, `ldmatrix`-based iterators, swizzle functors.
* Markidis et al. *NVIDIA Tensor Core Programmability, Performance & Precision.* IPDPSW 2018.
