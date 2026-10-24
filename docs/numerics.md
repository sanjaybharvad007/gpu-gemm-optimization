# Numerical accuracy

Fast is only useful if it is right.  Every kernel is validated against an **fp64 host reference**
(`src/harness.h: gemm_ref_f64`); the validator reports three numbers per kernel and shape and checks
them against a bound derived from the rounding model.

```
max |err|      largest absolute deviation
max rel err    largest |err| / max(|ref|, 1e-6)        (dominated by near-zero outputs; informational)
rel_fro        ||C - C*||_F / ||C*||_F                  (the pass/fail criterion)
```

## Rounding model

A dot product of length K in floating point with unit round-off `eps` has a forward error bounded by
(Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 3):

```
|fl(sum a_k b_k) - sum a_k b_k|  <=  gamma_K * sum |a_k b_k|,     gamma_K = K eps / (1 - K eps)
```

That is the worst case; for random data with independent rounding errors the observed error grows like
`sqrt(K) * eps` (random walk), and the **relative Frobenius error is close to `sqrt(K) * eps`** for
uniform inputs.  The bounds used by `validate`:

| kernel class | eps | bound on rel_fro | typical observed |
|---|---|---|---|
| fp32 SIMT (0-4, 7, 8) and cuBLAS SGEMM | 2^-24 = 6.0e-8 | `16 sqrt(K) eps` | `~1 sqrt(K) eps` (K=4096: ~4e-6) |
| fp16 operands, fp32 accumulate (5, 6, cuBLAS GemmEx) | operand 2^-11 = 4.9e-4; accumulate 2^-24 | `4 * 4.9e-4` | reference uses the *rounded* operands, so observed error is again `~sqrt(K) * 6e-8` |

Two subtleties that the validator handles explicitly:

* **What is the reference for an fp16 kernel?**  Rounding fp32 inputs to fp16 is not the kernel's fault.
  The validator rounds A and B to fp16 first and feeds those values (exactly representable in fp64) to the
  reference, so the measured error is the *kernel's* error: fp32 accumulation of exact fp16 products.
  Comparing against the unrounded fp32 inputs instead would show the 4.9e-4 operand error and tell you
  nothing about the kernel.
* **Summation order.**  Kernels 3/4 accumulate per thread in K order; split-K sums `split` partial
  sums in a fixed order; the tensor-core kernels accumulate 16 products per `mma` in hardware-defined
  order.  All are deterministic run to run (no atomics anywhere), but they are *not* bit-identical to
  each other or to cuBLAS.  Tests therefore compare against the fp64 reference with a tolerance, never
  kernel to kernel bitwise.

## Error growth with K

`./build/validate --k-sweep` prints `rel_fro` at M=N=256 for K = 32 ... 8192:

```
       K      fp32 simt    fp32 cublas      fp16 wmma   fp16 mma_ptx
      32      ~3e-7          ~3e-7           ~3e-7          ~3e-7
     512      ~1e-6          ~1e-6           ~1e-6          ~1e-6
    8192      ~5e-6          ~5e-6           ~5e-6          ~5e-6
```

(the exact digits vary with the data; the `sqrt(K)` slope - 2x per 4x K - is the thing to check.)  The
fp16 columns match the fp32 ones because the products are exact and the accumulator is fp32: **mixed
precision costs operand precision, not accumulation precision**.  Had the accumulation been fp16 the
third and fourth columns would start near 1e-3 and overflow to `inf` by K ~ 4096 for inputs in [-1, 1].

## alpha / beta and the epilogue

`C = alpha * acc + beta * C` is applied once, in fp32, after the full-K accumulation (split-K: after the
reduction).  Shapes with `beta != 0` are part of every test set because a kernel that forgets to read C,
or reads it after overwriting it (a real hazard when the same thread stores a `float4` it later reads),
passes every `beta = 0` test.

## Ragged shapes

Kernels 0-2 handle any M, N, K through predication.  The 128x128 fp32 kernels require `N % 4 == 0` and
`K % 4 == 0` (16-byte vector loads) and predicate M, N tails; the tensor-core kernels require
`M % 128 == N % 128 == K % 32 == 0`.  `kernel_constraints()` exposes this and the test suite includes
`33x20x12`, `100x68x36`, `129x132x44` so the predication paths are exercised, not just the aligned ones.

## Bit-reproducibility of the reductions

`reduce_sum` is a two-pass tree (block partials, then one block over the partials) with no atomics, so
the result is bit-identical across runs - the test asserts it.  Atomics would be shorter and faster for
one number but non-deterministic, which matters the moment the sum feeds a loss or a norm that is
compared between runs.
