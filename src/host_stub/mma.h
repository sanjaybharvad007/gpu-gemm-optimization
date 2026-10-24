// Minimal nvcuda::wmma surface for syntax checking (shapes and layouts are not validated here).
#pragma once
#include "cuda_fp16.h"
namespace nvcuda { namespace wmma {
struct matrix_a {}; struct matrix_b {}; struct accumulator {};
struct row_major {}; struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment { static constexpr int num_elements = 8; T x[num_elements]; };
template <typename F> inline void fill_fragment(F& f, float v) { for (int i = 0; i < F::num_elements; ++i) f.x[i] = (decltype(f.x[0]))v; }
template <typename F, typename T> inline void load_matrix_sync(F&, const T*, unsigned) {}
template <typename F, typename T> inline void load_matrix_sync(F&, const T*, unsigned, layout_t) {}
template <typename F, typename T> inline void store_matrix_sync(T*, const F&, unsigned, layout_t) {}
template <typename D, typename A, typename B, typename C> inline void mma_sync(D&, const A&, const B&, const C&) {}
}}  // namespace nvcuda::wmma
