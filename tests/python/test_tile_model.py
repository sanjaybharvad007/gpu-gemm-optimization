import pytest
from occupancy import SM80
from tile_model import TileConfig, report


def test_arithmetic_intensity_grows_with_tile():
    small = TileConfig(32, 32, 32, tm=1, tn=1)
    big = TileConfig(128, 128, 8, tm=8, tn=8)
    assert small.arithmetic_intensity() == pytest.approx(8.0)    # 2*32^3 / ((32+32)*32*4)
    assert big.arithmetic_intensity() == pytest.approx(32.0)     # 2*128*128*8 / (256*8*4)


def test_register_blocking_cuts_smem_traffic():
    one = TileConfig(32, 32, 32, tm=1, tn=1)
    eight = TileConfig(128, 128, 8, tm=8, tn=8)
    # bytes of smem read per FLOP: (tm+tn)*4 / (2*tm*tn)
    assert one.smem_bytes_per_flop() == pytest.approx(4.0)
    assert eight.smem_bytes_per_flop() == pytest.approx(0.5)
    assert eight.accumulator_regs() == 64 and eight.num_threads == 256


def test_tensor_core_tile_smem_and_waves():
    cfg = TileConfig(128, 128, 32, dtype_bytes=2, stages=3, pad_elems=8, threads=256)
    assert cfg.smem_bytes_per_stage() == 128 * 40 * 2 + 32 * 136 * 2
    assert cfg.smem_bytes() == 56832
    r = report(cfg, SM80, 4096, 4096, regs_per_thread=128)
    assert r["tiles"] == 1024
    assert r["blocks_per_sm"] == 2
    assert r["waves"] == pytest.approx(1024 / (2 * 108))
