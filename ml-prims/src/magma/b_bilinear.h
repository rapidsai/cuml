/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cublas_v2.h"     // if you need CUBLAS v2, include before magma.h
// #include "magma.h"
// #include "magma_lapack.h"  // if you need BLAS & LAPACK

#include "magma/magma_utils.h"
#include "magma/magma_batched_wrappers.h"
#include "magma/b_handles.h"
#include "magma/b_allocate.h"
#include "magma/b_split.h"

// #include "cuda_utils.h"

using namespace MLCommon::LinAlg;

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
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__device__
T bilinear_naive(int m, int n, T* x, T* y, T* A, magma_int_t lda){
        T res = 0;
        for (size_t j = 0; j < n; j++) {
                for (size_t i = 0; i < m; i++) {
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
                            T **dY_array, T *dO, magma_int_t batchCount,
                            cudaStream_t stream=0){
        dim3 block(32, 1, 1);
        dim3 grid(ceildiv(batchCount, (int)block.x), 1, 1);
        int numThreads = grid.x * block.x;
        bilinear_batched_kernel<T> <<< grid, block, 0, stream>>>(m, n, dX_array,
                                                                 dA_array, ldda,
                                                                 dY_array, dO, batchCount,
                                                                 numThreads);
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void createBilinearHandle_t(bilinearHandle_t<T>& handle, int n, int batchCount){
        allocate_pointer_array(handle.dT_array, n, batchCount);
}

template <typename T>
void createBilinearHandle_t_new(bilinearHandle_t<T>& handle, void* workspace){
        handle.dT_array = (T **)((size_t)handle.dT_array + (size_t)workspace);
        handle.dT = (T *)((size_t)handle.dT + (size_t)workspace);

        split_to_batches(handle.batchCount, handle.dT_array, handle.dT, handle.lddt);
}

template <typename T>
void bilinear_bufferSize(bilinearHandle_t<T>& handle,
                         int n, int lddt, int batchCount,
                         size_t& workspaceSize){
        workspaceSize = 0;
        const size_t granularity = 256;

        handle.dT_array = (T **)workspaceSize;
        workspaceSize += alignTo(batchCount *sizeof(T*), granularity);
        handle.dT = (T *)workspaceSize;
        workspaceSize += alignTo(batchCount * n * sizeof(T*), granularity);

        handle.lddt = lddt;
        handle.batchCount = batchCount;
}

template <typename T>
void bilinear_batched(magma_int_t m, magma_int_t n,
                      T **dX_array, T** dA_array, magma_int_t ldda,
                      T **dY_array, T *dO, magma_int_t batchCount,
                      magma_queue_t queue, bilinearHandle_t<T> handle)
{
        T alpha = 1, beta = 0;
        magma_int_t incx = 1, incy = 1;

        // Batched gemv
        magmablas_gemv_batched(MagmaTrans, m, n,
                               alpha, dA_array, ldda,
                               dX_array, incx, beta, handle.dT_array, incy,
                               batchCount, queue);

        // Batched dot
        dot_batched(n, handle.dT_array, dY_array, dO, batchCount);
}
}
