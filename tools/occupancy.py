"""A CUDA occupancy calculator: blocks per SM given threads, registers and shared memory per block.

    python tools/occupancy.py --threads 256 --regs 128 --smem 56832 --arch sm80

Implements the same limits the CUDA Occupancy Calculator applies: max threads / warps / blocks
per SM, register file size with per-warp allocation granularity, and shared memory per SM with
the per-block allocation granularity (and the 1 KB per block the driver reserves on sm_80+).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    name: str
    sms: int
    max_threads_per_sm: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int
    reg_alloc_granularity: int      # registers per warp are rounded up to this
    max_regs_per_thread: int
    smem_per_sm: int                # bytes usable by blocks
    smem_alloc_granularity: int
    smem_reserved_per_block: int


SM80 = ArchSpec("sm80 (A100)", 108, 2048, 64, 32, 65536, 256, 255, 164 * 1024, 128, 1024)
SM86 = ArchSpec("sm86 (RTX 30xx)", 82, 1536, 48, 16, 65536, 256, 255, 100 * 1024, 128, 1024)
SM89 = ArchSpec("sm89 (RTX 40xx / L4)", 128, 1536, 48, 24, 65536, 256, 255, 100 * 1024, 128, 1024)
SM90 = ArchSpec("sm90 (H100)", 132, 2048, 64, 32, 65536, 256, 255, 228 * 1024, 128, 1024)
ARCHS = {"sm80": SM80, "sm86": SM86, "sm89": SM89, "sm90": SM90}


@dataclass
class Occupancy:
    blocks_per_sm: int
    warps_per_sm: int
    occupancy_pct: float
    limiter: str
    limits: dict


def _ceil_to(x: int, g: int) -> int:
    return -(-x // g) * g


def occupancy(arch: ArchSpec, threads_per_block: int, regs_per_thread: int, smem_per_block: int) -> Occupancy:
    warps_per_block = -(-threads_per_block // 32)
    limits: dict[str, int] = {}
    limits["threads"] = arch.max_threads_per_sm // threads_per_block
    limits["warps"] = arch.max_warps_per_sm // warps_per_block
    limits["blocks"] = arch.max_blocks_per_sm
    if regs_per_thread > arch.max_regs_per_thread:
        limits["registers"] = 0
    else:
        regs_per_warp = _ceil_to(max(1, regs_per_thread) * 32, arch.reg_alloc_granularity)
        limits["registers"] = (arch.regs_per_sm // regs_per_warp) // warps_per_block
    smem = _ceil_to(smem_per_block + arch.smem_reserved_per_block, arch.smem_alloc_granularity)
    limits["shared memory"] = arch.smem_per_sm // smem if smem else arch.max_blocks_per_sm
    limiter = min(limits, key=lambda k: (limits[k], k))
    blocks = max(0, min(limits.values()))
    warps = blocks * warps_per_block
    return Occupancy(blocks, warps, 100.0 * warps / arch.max_warps_per_sm, limiter, limits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--regs", type=int, required=True)
    ap.add_argument("--smem", type=int, default=0)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="sm80")
    args = ap.parse_args()
    arch = ARCHS[args.arch]
    o = occupancy(arch, args.threads, args.regs, args.smem)
    print(f"{arch.name}: {o.blocks_per_sm} blocks/SM, {o.warps_per_sm} warps/SM = {o.occupancy_pct:.1f}% occupancy, limited by {o.limiter}")
    for k, v in o.limits.items():
        print(f"  {k:14s} allows {v} blocks")


if __name__ == "__main__":
    main()
