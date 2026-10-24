#pragma once
#include "cuda_runtime.h"
struct __half { unsigned short v; };
typedef __half half;
struct __half2 { __half x, y; };
inline float __half2float(__half) { return 0.f; }
inline __half __float2half(float) { return __half{}; }
inline __half2 __floats2half2_rn(float, float) { return __half2{}; }
inline float2 __half22float2(__half2) { return float2{}; }
