
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
// thread count
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 512
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024
#define THREADS7 1024

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 4  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2
#define FACTOR7 1

#define WARPSIZE 32



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


} // end namespace
