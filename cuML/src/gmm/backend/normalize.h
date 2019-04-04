#pragma once

#include <cuda.h>
#include <stats/sum.h>

namespace gmm {

template <typename T>
__global__
void normalizeMatrixKernel(size_t m, size_t n,
                           T *dA, size_t ldda,
                           bool colwise,
                           int numThreads_x){
        int start = threadIdx.x + blockDim.x * blockIdx.x;
        T sum;

        if (colwise) {
                for (size_t j = start; j < n; j+=numThreads_x) {
                        sum = 0;
                        for (size_t i = 0; i < m; i++) {
                                sum += dA[IDX(i, j, ldda)];
                        }
                        for (size_t i = 0; i < m; i++) {
                                dA[IDX(i, j, ldda)] /= sum;
                        }
                }
        }
        else {
                // TODO : Requires testing
                for (size_t i = start; i < m; i+=numThreads_x) {
                        sum = 0;
                        for (size_t j = 0; j < m; j++) {
                                sum += dA[IDX(i, j, ldda)];
                        }
                        for (size_t j = 0; j < m; j++) {
                                dA[IDX(i, j, ldda)] /= sum;
                        }
                }
        }
}

template <typename T>
void normalize_matrix(size_t m, size_t n,
                      T *dA, size_t ldda,
                      bool colwise,
                      cudaStream_t stream = 0)
{
        dim3 block(32);
        dim3 grid;
        if (colwise) {
                grid.x = ceildiv((int)n, (int)block.x);
        }
        else  {
                grid.x = ceildiv((int)m, (int)block.x);
        }

        int numThreads_x = grid.x * block.x;

        normalizeMatrixKernel<T> <<< grid, block, 0, stream >>>(m, n,
                                                                dA, ldda,
                                                                colwise,
                                                                numThreads_x);
        CUDA_CHECK(cudaPeekAtLastError());
}

}
