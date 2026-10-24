"""Roofline analysis of the benchmark CSV.

    python tools/roofline.py results/bench.csv --device "A100-SXM4-80GB" --out results/roofline.png
    python tools/roofline.py results/bench.csv --peak-fp32 19.5 --peak-fp16 312 --peak-bw 1555

For each measured kernel the arithmetic intensity (FLOP per byte of compulsory traffic) and the
achieved TFLOP/s are plotted against the device's memory roof (bandwidth * intensity) and compute
roofs (fp32 and fp16 Tensor Core).  The table printed alongside gives, per point, the binding roof
and the fraction of it achieved - the number that tells you whether to optimise bytes or FLOPs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Peaks:
    fp32_tflops: float
    fp16_tc_tflops: float
    hbm_gbs: float

    @property
    def ridge_fp32(self) -> float:  # FLOP/byte where the memory roof meets the fp32 roof
        return self.fp32_tflops * 1e12 / (self.hbm_gbs * 1e9)

    @property
    def ridge_fp16(self) -> float:
        return self.fp16_tc_tflops * 1e12 / (self.hbm_gbs * 1e9)


def lookup_device(name: str) -> Peaks | None:
    with open(os.path.join(HERE, "device_peaks.json")) as f:
        table = json.load(f)
    name_l = name.lower()
    for key, v in table.items():
        if key.startswith("_"):
            continue
        if key.lower() in name_l or name_l in key.lower():
            return Peaks(v["fp32_tflops"], v["fp16_tc_tflops"], v["hbm_gbs"])
    return None


def gemm_intensity(M: int, N: int, K: int, in_bytes: int, out_bytes: int = 4, read_c: bool = False) -> float:
    """FLOP per byte assuming A, B are read once and C written once (the compulsory traffic)."""
    flops = 2.0 * M * N * K
    byts = (M * K + K * N) * in_bytes + M * N * out_bytes * (2 if read_c else 1)
    return flops / byts


def attainable(ai: float, peaks: Peaks, fp16: bool) -> tuple[float, str]:
    """Roofline bound in TFLOP/s for a point and which roof binds."""
    mem = peaks.hbm_gbs * 1e9 * ai / 1e12
    comp = peaks.fp16_tc_tflops if fp16 else peaks.fp32_tflops
    return (mem, "memory") if mem < comp else (comp, "fp16 tensor core" if fp16 else "fp32")


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("M", "N", "K", "batch", "split_k"):
            r[k] = int(r[k])
        for k in ("ms", "tflops", "gbs", "pct_cublas", "pct_peak", "arith_intensity"):
            r[k] = float(r[k])
    return rows


def analyse(rows: list[dict], peaks: Peaks) -> list[dict]:
    out = []
    for r in rows:
        fp16 = r["dtype"] == "f16"
        ai = r["arith_intensity"] or gemm_intensity(r["M"], r["N"], r["K"], 2 if fp16 else 4)
        bound, roof = attainable(ai, peaks, fp16)
        out.append({**r, "ai": ai, "bound_tflops": bound, "roof": roof, "pct_roofline": 100.0 * r["tflops"] / bound if bound else float("nan")})
    return out


def print_table(points: list[dict]) -> None:
    print(f"{'kernel':<18}{'dtype':<6}{'M':>6}{'N':>6}{'K':>6}{'AI':>8}{'TFLOP/s':>10}{'roof':>18}{'bound':>9}{'%roof':>7}{'%cuBLAS':>9}")
    for p in points:
        print(f"{p['kernel']:<18}{p['dtype']:<6}{p['M']:>6}{p['N']:>6}{p['K']:>6}{p['ai']:>8.1f}{p['tflops']:>10.2f}{p['roof']:>18}{p['bound_tflops']:>9.1f}{p['pct_roofline']:>7.1f}{p['pct_cublas']:>9.1f}")


def plot(points: list[dict], peaks: Peaks, out: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ais = [2 ** i for i in range(-2, 13)]
    mem = [peaks.hbm_gbs * 1e9 * a / 1e12 for a in ais]
    ax.plot(ais, [min(m, peaks.fp32_tflops) for m in mem], color="#555", lw=2, label=f"fp32 roof ({peaks.fp32_tflops:.0f} TFLOP/s, ridge {peaks.ridge_fp32:.0f} FLOP/B)")
    ax.plot(ais, [min(m, peaks.fp16_tc_tflops) for m in mem], color="#1a7f37", lw=2, ls="--", label=f"fp16 Tensor Core roof ({peaks.fp16_tc_tflops:.0f} TFLOP/s, ridge {peaks.ridge_fp16:.0f} FLOP/B)")
    kernels = sorted({p["kernel"] for p in points})
    cmap = plt.get_cmap("tab10")
    for i, k in enumerate(kernels):
        pts = [p for p in points if p["kernel"] == k]
        ax.scatter([p["ai"] for p in pts], [p["tflops"] for p in pts], label=k, color=cmap(i % 10), s=36, edgecolor="k", lw=0.4, zorder=3)
        for p in pts:
            ax.annotate(f"{p['M']}", (p["ai"], p["tflops"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("arithmetic intensity (FLOP / byte of compulsory traffic)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--device", default="")
    ap.add_argument("--peak-fp32", type=float)
    ap.add_argument("--peak-fp16", type=float)
    ap.add_argument("--peak-bw", type=float)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    peaks = lookup_device(args.device) if args.device else None
    if args.peak_fp32 and args.peak_fp16 and args.peak_bw:
        peaks = Peaks(args.peak_fp32, args.peak_fp16, args.peak_bw)
    if peaks is None:
        raise SystemExit("unknown device: pass --device <name in tools/device_peaks.json> or --peak-fp32/--peak-fp16/--peak-bw")
    points = analyse(load_rows(args.csv), peaks)
    print_table(points)
    if args.out:
        plot(points, peaks, args.out, args.device or "roofline")
    if not math.isfinite(sum(p["pct_roofline"] for p in points)):
        raise SystemExit("non-finite numbers in CSV")


if __name__ == "__main__":
    main()
