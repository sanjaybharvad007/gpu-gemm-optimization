"""Analytical model of a GEMM block tile - the arithmetic you do before writing a kernel.

    python tools/tile_model.py --bm 128 --bn 128 --bk 8 --tm 8 --tn 8 --dtype f32
    python tools/tile_model.py --bm 128 --bn 128 --bk 32 --warps-m 2 --warps-n 4 --dtype f16 --stages 3

Reports, for a tile configuration:
  * global-memory arithmetic intensity of the block tile (FLOP per byte loaded into smem),
  * shared-memory bytes per stage and in total,
  * registers needed for accumulators + operand fragments per thread,
  * smem bandwidth required per FLOP (the quantity register blocking reduces),
  * blocks per SM limited by smem / registers / threads (via occupancy.py) and the resulting waves
    for a given problem size.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from occupancy import SM80, ArchSpec, occupancy


@dataclass
class TileConfig:
    bm: int
    bn: int
    bk: int
    tm: int = 8          # per-thread outputs (SIMT) or ignored for tensor-core tiles
    tn: int = 8
    dtype_bytes: int = 4
    stages: int = 1
    pad_elems: int = 0   # smem row padding (tensor-core kernels)
    threads: int | None = None

    @property
    def num_threads(self) -> int:
        return self.threads or (self.bm // self.tm) * (self.bn // self.tn)

    def flops_per_tile_step(self) -> float:
        return 2.0 * self.bm * self.bn * self.bk

    def global_bytes_per_step(self) -> float:
        return (self.bm + self.bn) * self.bk * self.dtype_bytes

    def arithmetic_intensity(self) -> float:
        """FLOP per byte moved from global memory into shared memory for one K step."""
        return self.flops_per_tile_step() / self.global_bytes_per_step()

    def smem_bytes_per_stage(self) -> int:
        a = self.bm * (self.bk + self.pad_elems) * self.dtype_bytes
        b = self.bk * (self.bn + self.pad_elems) * self.dtype_bytes
        return a + b

    def smem_bytes(self) -> int:
        return self.stages * self.smem_bytes_per_stage()

    def accumulator_regs(self) -> int:
        return self.bm * self.bn // self.num_threads  # fp32 accumulators per thread

    def smem_bytes_per_flop(self) -> float:
        """SIMT: each thread reads tm + tn values per k from smem for 2*tm*tn FLOPs."""
        return (self.tm + self.tn) * self.dtype_bytes / (2.0 * self.tm * self.tn)


def report(cfg: TileConfig, arch: ArchSpec, M: int, N: int, regs_per_thread: int | None = None) -> dict:
    regs = regs_per_thread or (cfg.accumulator_regs() + cfg.tm + cfg.tn + 24)
    occ = occupancy(arch, threads_per_block=cfg.num_threads, regs_per_thread=regs, smem_per_block=cfg.smem_bytes())
    tiles = -(-M // cfg.bm) * -(-N // cfg.bn)
    waves = tiles / max(1, occ.blocks_per_sm * arch.sms)
    return {
        "threads": cfg.num_threads,
        "arith_intensity_flop_per_byte": cfg.arithmetic_intensity(),
        "smem_bytes": cfg.smem_bytes(),
        "accumulators_per_thread": cfg.accumulator_regs(),
        "est_regs_per_thread": regs,
        "smem_bytes_per_flop": cfg.smem_bytes_per_flop(),
        "blocks_per_sm": occ.blocks_per_sm,
        "occupancy_limiter": occ.limiter,
        "achieved_occupancy_pct": occ.occupancy_pct,
        "tiles": tiles,
        "waves": waves,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=8)
    ap.add_argument("--tm", type=int, default=8)
    ap.add_argument("--tn", type=int, default=8)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--regs", type=int)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    args = ap.parse_args()
    cfg = TileConfig(args.bm, args.bn, args.bk, args.tm, args.tn, 4 if args.dtype == "f32" else 2, args.stages, args.pad, args.threads)
    for k, v in report(cfg, SM80, args.M, args.N, args.regs).items():
        print(f"{k:32s} {v:.3f}" if isinstance(v, float) else f"{k:32s} {v}")


if __name__ == "__main__":
    main()
