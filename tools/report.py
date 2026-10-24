"""Render the benchmark CSV as the Markdown tables used in README / docs.

    python tools/report.py results/bench.csv            # % of cuBLAS per kernel per size
    python tools/report.py results/bench.csv --metric tflops
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict


def load(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def table(rows: list[dict], metric: str, dtype: str) -> str:
    rows = [r for r in rows if r["dtype"] == dtype]
    sizes = sorted({(int(r["M"]), int(r["N"]), int(r["K"])) for r in rows})
    kernels: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        kernels.setdefault(r["kernel"], {})[(int(r["M"]), int(r["N"]), int(r["K"]))] = float(r[metric])
    head = "| kernel | " + " | ".join(f"{m}x{n}x{k}" if not (m == n == k) else str(m) for m, n, k in sizes) + " |"
    sep = "|---|" + "---:|" * len(sizes)
    lines = [head, sep]
    for k, vals in kernels.items():
        cells = [f"{vals[s]:.1f}" if s in vals else "-" for s in sizes]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--metric", default="pct_cublas", choices=["pct_cublas", "tflops", "ms", "gbs", "pct_peak"])
    args = ap.parse_args()
    rows = load(args.csv)
    for dtype in ("f32", "f16"):
        if any(r["dtype"] == dtype for r in rows):
            print(f"\n**{dtype}, {args.metric}**\n")
            print(table(rows, args.metric, dtype))


if __name__ == "__main__":
    main()
