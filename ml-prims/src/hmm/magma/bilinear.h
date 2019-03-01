#include "cublas_v2.h"     // if you need CUBLAS v2, include before magma.h
// #include "magma.h"
// #include "magma_lapack.h"  // if you need BLAS & LAPACK

#include "magma/magma_test_utils.h"
// #include "cuda_utils.h"

namespace MLCommon {

template <typename T>
__host__ __device__
T dot(int n, T* x, T* y){
        T res = 0;
        for (size_t i = 0; i < n; i++) {
                res += x[i] * y[i];
        }
        return res;
}

template <typename T>
__global__
void dot_batched_kernel(int n, T **dX_array, T **dY_array, T *dO,
                        magma_int_t batchCount, int numThreads){
        int idxThread = threadIdx.x + blockDim.x * blockIdx.x;
        for (size_t i = idxThread; i < batchCount; i+=numThreads) {
                dO[i] = dot(n, dX_array[i], dY_array[i]);
        }
}

template <typename T>
void dot_batched(int n, T **dX_array, T **dY_array, T *dO,
                 magma_int_t batchCount){
        dim3 block(32, 1, 1);
        dim3 grid(ceildiv(batchCount, (int)block.x), 1, 1);
        int numThreads = grid.x * block.x;
        dot_batched_kernel<T> <<< grid, block >>>(n, dX_array, dY_array, dO,
                                                  batchCount, numThreads);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__host__ __device__
T bilinear_naive(int m, int n, T* x, T* y, T* A, magma_int_t lda){
        T res = 0;
        for (size_t j = 0; j < m; j++) {
                for (size_t i = 0; i < n; i++) {
                        res += x[i] * y[j] * A[IDX(i, j, lda)];
                }
        }
        return res;
}


// Loop over all elements
template <typename T>
__global__
void bilinear_batched_kernel(magma_int_t m, magma_int_t n,
                             T **dX_array, T** dA_array, magma_int_t ldda,
                             T **dY_array, T *dO, magma_int_t batchCount,
                             int numThreads){
        int idxThread = threadIdx.x + blockDim.x * blockIdx.x;
        for (size_t i = idxThread; i < batchCount; i+=numThreads) {
                dO[i] = bilinear_naive(m, n, dX_array[i], dY_array[i], dA_array[i], ldda);
        }
}

template <typename T>
void naive_bilinear_batched(magma_int_t m, magma_int_t n,
                            T **dX_array, T** dA_array, magma_int_t ldda,
                            T **dY_array, T *dO, magma_int_t batchCount){
        dim3 block(32, 1, 1);
        dim3 grid(ceildiv(batchCount, (int)block.x), 1, 1);
        int numThreads = grid.x * block.x;
        bilinear_batched_kernel<T> <<< grid, block >>>(m, n, dX_array, dA_array,
                                                       ldda, dY_array, dO, batchCount,
                                                       numThreads);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
void bilinear_batched(magma_int_t m, magma_int_t n,
                      T **dX_array, T** dA_array, magma_int_t ldda,
                      T **dY_array, T *dO, magma_int_t batchCount,  magma_queue_t queue)
{
        // // Allocate temporary memory
        T **dT_array;
        allocate_pointer_array(dT_array, n, batchCount);

        T alpha = 1, beta = 0;
        magma_int_t incx = 1, incy = 1;

        // Batched gemv
        magmablas_dgemv_batched(MagmaTrans, m, n, alpha, dA_array, ldda, dX_array, incx, beta, dT_array, incy, batchCount, queue);

        // Batched dot
        dot_batched(n, dT_array, dY_array, dO, batchCount);

        free_pointer_array(dT_array, batchCount);

}
}
