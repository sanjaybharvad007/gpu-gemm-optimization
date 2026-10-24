#pragma once
#include "cuda_fp16.h"
typedef struct cublasContext* cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
enum cudaDataType { CUDA_R_16F, CUDA_R_32F };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT_TENSOR_OP };
inline cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = nullptr; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t) { return CUBLAS_STATUS_SUCCESS; }
