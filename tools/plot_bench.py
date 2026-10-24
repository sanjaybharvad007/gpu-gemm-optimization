"""TFLOP/s vs problem size for every kernel, with cuBLAS as the reference line.

    python tools/plot_bench.py results/bench.csv --out results/tflops.png
"""
from __future__ import annotations

import argparse
import csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results/tflops.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dtype in zip(axes, ("f32", "f16")):
        sub = [r for r in rows if r["dtype"] == dtype and int(r["M"]) == int(r["N"]) == int(r["K"])]
        for k in sorted({r["kernel"] for r in sub}):
            pts = sorted((int(r["M"]), float(r["tflops"])) for r in sub if r["kernel"] == k)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=2.5 if k == "cublas" else 1.2, ls="-" if k != "cublas" else "--", label=k)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("M = N = K")
        ax.set_ylabel("TFLOP/s")
        ax.set_title(f"{dtype} GEMM")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
