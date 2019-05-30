
using namespace ML;
#pragma once

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <iostream>
#include <math.h>
#include "sparse/coo.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define cuda_free(x)    CUDA_CHECK(cudaFree(x))
#define EXP(x)          MLCommon::myExp(x)
#define LOG(x)          MLCommon::myLog(x)
#define MAX(a, b)       ((a > b) ? a : b)
#define MIN(a, b)       ((a < b) ? a : b)
#define kernel_sync     __syncthreads
#define kernel_fence    __threadfence


#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

#define GPU_ARCH "PASCAL"
#define THREADS1 512
#define THREADS2 512
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024
#define THREADS7 1024

#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 1
#define FACTOR4 4
#define FACTOR5 2
#define FACTOR6 2
#define FACTOR7 1

#define WARPSIZE 32
#define MAXDEPTH 32


//
extern __device__ volatile int stepd, bottomd, maxdepthd;
extern __device__ unsigned int blkcntd;
extern __device__ volatile float radiusd;
//


// Malloc some space
template <typename Type>
void cuda_malloc(Type *&ptr, const size_t n) {
    // From ml-prims / src / utils.h
    CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * n));
}

// Memset
template <typename Type>
void cuda_memset(Type *&ptr, const size_t n) {
    // From ml-prims / src / utils.h
    CUDA_CHECK(cudaMemset(ptr, 0, sizeof(Type) * n));
}


// Malloc and memset some space
template <typename Type>
void cuda_calloc(Type *&ptr, const size_t n, const Type val, cudaStream_t stream) {
    // From ml-prims / src / utils.h
    // Just allows easier memsetting
    CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * n));

    if (val == 0) {
        // Notice memset stes BYTES to val. Cannot use to fill array
        // with another value other than 0
        CUDA_CHECK(cudaMemset(ptr, 0, sizeof(Type) * n));
    }
    else {
        thrust::device_ptr<Type> begin = thrust::device_pointer_cast(ptr);
        // IS THIS CORRECT??
        thrust::fill(thrust::cuda::par.on(stream), begin, begin+n, val);
    }
}


//
namespace Utils_ {

// Determines number of blocks
// Similar to page size determination --> need to round up
inline int ceildiv(const int a, const int b) {
    if (a % b != 0)
        return a/b + 1;
    return a/b;
}


// Finds minimum(array) from UMAP
template <typename Type>
inline Type min_array(  thrust::device_ptr<const Type> start, 
                        thrust::device_ptr<const Type> end,
                        cudaStream_t stream) 
{
    return *(thrust::min_element(thrust::cuda::par.on(stream), start, end))
}


// Finds maximum(array) from UMAP
template <typename Type>
inline Type max_array(  thrust::device_ptr<const Type> start, 
                        thrust::device_ptr<const Type> end,
                        cudaStream_t stream) 
{
    return *(thrust::max_element(thrust::cuda::par.on(stream), start, end))
}


// Finds sum(array)
template <typename Type>
inline Type sum_array(  thrust::device_ptr<const Type> start, 
                        thrust::device_ptr<const Type> end,
                        cudaStream_t stream) 
{
    return thrust::reduce(thrust::cuda::par.on(stream), start, end, 0.0f, thrust::plus<float>());
}



#include "stats/sum.h"
// Does row_sum on C contiguous data
inline void row_sum(float *out, const float *in, 
                    const int n, const int p,
                    cudaStream_t stream)
{
    // Since sum is for F-Contiguous columns, then C-contiguous rows
    // is also fast since it is transposed.
    //                  dim, rows, rowMajor
    Stats::sum(out, in,  n,   p,    false, stream);
}




//
#include <cusparse.h>
// Handles sparse matrix operations

// From https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-spmat-create-csr
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed with error (%d) at line %d\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

//
typedef cusparseSpMatDescr_t CSR_t;
typedef cusparseDnMatDescr_t Dense_t;
typedef cusparseHandle_t Sparse_handle_t;


// Create cuSPARSE handle
#define createHandle(handle) CHECK_CUSPARSE( cusparseCreate(handle) );
#define destroyHandle(handle) CHECK_CUSPARSE( cusparseDestroy(handle) );


// Creates a CSR matrix
void createCSR(
    CSR_t *CSR_Matrix,
    const int n,
    const int p,
    const int NNZ,
    float * __restrict__ VAL,
    float * __restrict__ COL,
    float * __restrict__ ROW)
{
    CHECK_CUSPARSE(
        cusparseCreateCsr(
            CSR_Matrix, 
            (int64_t)n, (int64_t)p, (int64_t)NNZ,
            ROW, COL, VAL,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F)
        );
}

// Deletes CSR matrix
#define destroyCSR(CSR_Matrix) CHECK_CUSPARSE( cusparseDestroySpMat(CSR_Matrix) )


// Creates a Dense Matrix descriptor for use in sparse operations
void createDense(
    Dense_t *Dense_Matrix,
    const int n,
    const int p,
    float * __restrict__ data)
{
    CHECK_CUSPARSE(
        cusparseCreateDnMat(
            Dense_Matrix,
            (int64_t)n, (int64_t)p, (int64_t)n,
            data, CUDA_R_32F, 
            CUSPARSE_ORDER_COL)
        );
}

// Deletes Dense Matrix
#define destroyDense(Dense_Matrix) CHECK_CUSPARSE( cusparseDestroyDnMat(Dense_Matrix) )


// Create Buffer for spMM
void *createBuffer(
    Sparse_handle_t handle,
    const CSR_t CSR_Matrix,
    const Dense_t Dense_Matrix,
    Dense_t Output_Matrix)
{
    const float alpha = 1.0;
    const float beta = 1.0;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(
        cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            (const void*) &alpha,
            CSR_Matrix,
            Dense_Matrix,
            (const void*) &beta,
            Output_Matrix,
            CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT,
            &bufferSize)
        );

    void *buffer;
    cuda_malloc(buffer, bufferSize);
    return buffer;
}

// Destory buffer
#define destoryBuffer(buffer) cuda_free(buffer);



// Sparse Matrix @ Dense Matrix Multiply
void spMM(
    Sparse_handle_t handle,
    const CSR_t CSR_Matrix,
    const Dense_t Dense_Matrix,
    const float alpha,
    const float beta,
    Dense_t Output_Matrix,
    void *buffer)
{
    CHECK_CUSPARSE(
        cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            (const void*) &alpha,
            CSR_Matrix,
            Dense_Matrix,
            (const void*) &beta,
            Output_Matrix,
            CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT,
            buffer)
        );
}


} // end namespace
