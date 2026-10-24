cmake_minimum_required(VERSION 3.18)
project(cuda_gemm_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# sm_80 (A100) is the floor: cp.async, ldmatrix and the mma.m16n8k16 shape need Ampere.
# Add 86/89/90 for consumer Ampere / Ada / Hopper; the SIMT kernels run on anything >= sm_70.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

option(GEMM_LINEINFO "Embed line info for Nsight Compute source view" ON)
option(GEMM_PTXAS_VERBOSE "Print registers / smem per kernel at compile time" OFF)
option(GEMM_BUILD_TESTS "Build the CTest suite" ON)

find_package(CUDAToolkit REQUIRED)

add_library(gemm_kernels STATIC
  src/registry.cu
  src/cublas_ref.cu
  src/kernels/00_naive.cu
  src/kernels/01_coalesced.cu
  src/kernels/02_smem_tiled.cu
  src/kernels/03_register_blocked.cu
  src/kernels/04_double_buffered.cu
  src/kernels/05_wmma_fp16.cu
  src/kernels/06_mma_ptx.cu
  src/kernels/07_split_k.cu
  src/kernels/08_batched.cu
  src/kernels/gemv.cu
  src/kernels/reductions.cu
  src/kernels/transpose.cu)
target_include_directories(gemm_kernels PUBLIC include PRIVATE src)
target_link_libraries(gemm_kernels PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(gemm_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${GEMM_LINEINFO}>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${GEMM_PTXAS_VERBOSE}>>:-Xptxas=-v>)

add_executable(bench_gemm src/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE gemm_kernels)
target_include_directories(bench_gemm PRIVATE src)

add_executable(validate src/validate.cu)
target_link_libraries(validate PRIVATE gemm_kernels)
target_include_directories(validate PRIVATE src)

if(GEMM_BUILD_TESTS)
  enable_testing()
  add_executable(test_kernels tests/cuda/test_kernels.cu)
  target_link_libraries(test_kernels PRIVATE gemm_kernels)
  target_include_directories(test_kernels PRIVATE src)
  add_test(NAME kernels COMMAND test_kernels)
  add_test(NAME validate_quick COMMAND validate --quick --quiet)
endif()
