// Matrix transpose through a padded shared-memory tile.  Reads and writes are both coalesced
// (each warp reads a contiguous row segment of `in` and writes a contiguous row segment of `out`);
// the [32][33] padding turns the column read of the tile - 32 threads hitting addresses 32 floats
// apart, i.e. the same bank - into a conflict-free diagonal.  A GEMM with a transposed operand is
// usually better served by changing the smem layout (kernel 3 stores A transposed for free), but a
// standalone transpose is the canonical bank-conflict demonstration and Nsight shows the
// `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` counter dropping to zero with the +1.
#include "gemm/common.h"

namespace gemm {

template <int TILE, int ROWS_PER_ITER>
__global__ void transpose_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];
  const int x = blockIdx.x * TILE + threadIdx.x, y = blockIdx.y * TILE + threadIdx.y;
#pragma unroll
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * cols + x];
  __syncthreads();
  const int tx = blockIdx.y * TILE + threadIdx.x, ty = blockIdx.x * TILE + threadIdx.y;   // transposed block
#pragma unroll
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (tx < rows && ty + j < cols) out[(size_t)(ty + j) * rows + tx] = tile[threadIdx.x][threadIdx.y + j];
}

// explicit instantiation of the configuration the launcher uses (also lets NVRTC/ptxas report it)
template __global__ void transpose_kernel<32, 8>(const float*, float*, int, int);

#ifndef __CUDACC_RTC__  // host launchers (not compiled by NVRTC)
void transpose(const float* in, float* out, int rows, int cols, cudaStream_t s) {
  constexpr int TILE = 32, ROWS = 8;
  auto kfn = &transpose_kernel<TILE, ROWS>;
  KERNEL_LAUNCH(kfn, dim3(cdiv(cols, TILE), cdiv(rows, TILE)), dim3(TILE, ROWS), 0, s)(in, out, rows, cols);
}

#endif  // !__CUDACC_RTC__

}  // namespace gemm
