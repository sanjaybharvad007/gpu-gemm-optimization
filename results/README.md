# results/

`bench_gemm --csv results/bench.csv` writes measurements here; `tools/roofline.py`, `tools/plot_bench.py`
and `tools/report.py` consume them.  The GPU workflow uploads this directory as a build artifact.

`sample_bench.csv` is a **synthetic fixture** used by the CPU CI to exercise the analysis tools when no
GPU is available.  Its numbers are illustrative placeholders, not measurements - replace it with a real
`bench.csv` from your hardware.
