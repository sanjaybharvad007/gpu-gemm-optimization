// Host-side stand-ins for the CUDA runtime so every .cu file type-checks with a plain C++ compiler
// in CI (no GPU, no nvcc):
//   g++ -fsyntax-only -std=c++17 -x c++ -Iinclude -Isrc/host_stub -DGEMM_HOST_STUB src/kernels/06_mma_ptx.cu
// Nothing here is meant to run; it catches the large class of errors that have nothing to do with
// the GPU (types, templates, constexpr arithmetic, signatures) before the code reaches a runner.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cstring>
#ifndef GEMM_HOST_STUB
#define GEMM_HOST_STUB 1
#endif
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
using std::min;
using std::max;

struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
static dim3 threadIdx, blockIdx, blockDim, gridDim;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
enum cudaError_t { cudaSuccess = 0, cudaErrorUnknown = 999 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize };
struct cudaDeviceProp {
  char name[256]; int multiProcessorCount, major, minor, clockRate, memoryClockRate, memoryBusWidth, l2CacheSize;
  size_t sharedMemPerMultiprocessor; int regsPerMultiprocessor, maxThreadsPerMultiProcessor;
};
struct cudaFuncAttributes { int numRegs; size_t sharedSizeBytes; int maxThreadsPerBlock; };

inline void __syncthreads() {}
inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
inline int __shfl_xor_sync(unsigned, int v, int) { return v; }
inline float __shfl_down_sync(unsigned, float v, int) { return v; }
inline float __shfl_sync(unsigned, float v, int) { return v; }

struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int4 { int x, y, z, w; };
struct uint4 { unsigned x, y, z, w; };
inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
template <typename T> inline T __ldg(const T* p) { return *p; }
inline float __int_as_float(unsigned v) { float f; __builtin_memcpy(&f, &v, 4); return f; }
inline float atomicAdd(float* p, float v) { float o = *p; *p += v; return o; }

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaMalloc(void** p, size_t) { *p = nullptr; return cudaSuccess; }
template <typename T> inline cudaError_t cudaMalloc(T** p, size_t) { *p = nullptr; return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t = nullptr) { return cudaSuccess; }
inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = cudaDeviceProp{}; return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 0; return cudaSuccess; }
template <typename F> inline cudaError_t cudaFuncSetAttribute(F, cudaFuncAttribute, int) { return cudaSuccess; }
template <typename F> inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, F) { *a = cudaFuncAttributes{}; return cudaSuccess; }
template <typename F> inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, F, int, size_t) { *n = 0; return cudaSuccess; }
