from occupancy import SM80, SM86, occupancy


def test_register_limited():
    # 256 threads x 128 regs = 32768 regs per block -> 2 blocks in a 64K register file
    o = occupancy(SM80, threads_per_block=256, regs_per_thread=128, smem_per_block=8192)
    assert o.blocks_per_sm == 2 and o.limiter == "registers"
    assert abs(o.occupancy_pct - 25.0) < 1e-9


def test_smem_limited_mma_kernel():
    # the 3-stage mma.sync kernel: 56832 B dynamic smem + 1 KB reserved -> 2 blocks on A100 (164 KB)
    o = occupancy(SM80, threads_per_block=256, regs_per_thread=64, smem_per_block=56832)
    assert o.blocks_per_sm == 2 and o.limiter == "shared memory"
    # on a 100 KB-smem consumer part only one block fits
    o86 = occupancy(SM86, threads_per_block=256, regs_per_thread=64, smem_per_block=56832)
    assert o86.blocks_per_sm == 1


def test_thread_and_block_limits():
    o = occupancy(SM80, threads_per_block=1024, regs_per_thread=32, smem_per_block=0)
    assert o.blocks_per_sm == 2 and o.occupancy_pct == 100.0
    o = occupancy(SM80, threads_per_block=32, regs_per_thread=16, smem_per_block=0)
    assert o.blocks_per_sm == 32 and o.limiter == "blocks"


def test_too_many_registers():
    assert occupancy(SM80, 256, 300, 0).blocks_per_sm == 0
