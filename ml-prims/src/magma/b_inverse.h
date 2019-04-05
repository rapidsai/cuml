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

#include "magma/magma_utils.h"
#include "magma/magma_batched_wrappers.h"

#include "magma/b_handles.h"
#include "magma/b_copy.h"
#include "magma/b_allocate.h"
#include "magma/b_split.h"


using namespace MLCommon::LinAlg;

// TODO : ADD batched cublas
// Using cublas for large batch sizes and magma otherwise
// https://github.com/pytorch/pytorch/issues/13546

namespace MLCommon {


template <typename T>
__global__ void ID_kernel (int n, T *A, int ldda,
                           int nThreads_x, int nThreads_y) {
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t j = j_start; j < n; j+=nThreads_y) {
                for (size_t i = i_start; i <  n; i+=nThreads_x) {
                        if (i == j)
                                A[IDX(i, j, ldda)] = 1.0;
                        else
                                A[IDX(i, j, ldda)] = 0.0;
                }
        }
}


template <typename T>
void make_ID_matrix(int n, T *A, int ldda, cudaStream_t stream=0) {
        dim3 block(32,32);
        dim3 grid(ceildiv(n, (int)block.x),
                  ceildiv(n, (int)block.y),
                  1);
        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;

        ID_kernel<T> <<< grid, block, 0, stream >>>(n, A, ldda,
                                                    nThreads_x, nThreads_y);
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void createInverseHandle_t(inverseHandle_t<T>& handle,
                           int batchCount, int n, int ldda){
        allocate_pointer_array(handle.dipiv_array, n, batchCount);
        allocate_pointer_array(handle.dA_cpy_array, ldda * n, batchCount);
        allocate(handle.info_array, batchCount);
}

template <typename T>
void createInverseHandle_t_new(inverseHandle_t<T>& handle, void* workspace){
        handle.dipiv_array = (int **)((size_t)handle.dipiv_array + (size_t)workspace);
        handle.dA_cpy_array = (T **)((size_t)handle.dA_cpy_array + (size_t)workspace);

        handle.dipiv = (int *)((size_t)handle.dipiv + (size_t)workspace);
        handle.dA_cpy = (T *)((size_t)handle.dA_cpy + (size_t)workspace);
        handle.info_array = (int *)((size_t)handle.info_array + (size_t)workspace);

        split_to_batches(handle.batchCount, handle.dipiv_array, handle.dipiv, handle.n);
        split_to_batches(handle.batchCount, handle.dA_cpy_array, handle.dA_cpy, handle.ldda * handle.n);
}

template <typename T>
void inverse_bufferSize(inverseHandle_t<T>& handle,
                        int n, int ldda, int batchCount,
                        size_t& workspaceSize){
        workspaceSize = 0;
        const size_t granularity = 256;

        handle.dipiv_array = (int **)workspaceSize;
        workspaceSize += alignTo(batchCount * sizeof(int*), granularity);
        handle.dA_cpy_array = (T **)workspaceSize;
        workspaceSize += alignTo(batchCount *sizeof(T*), granularity);
        handle.dipiv = (int *)workspaceSize;
        workspaceSize += alignTo(n * batchCount * sizeof(int), granularity);
        handle.info_array = (int *)workspaceSize;
        workspaceSize += alignTo(batchCount * sizeof(int), granularity);
        handle.dA_cpy = (T *)workspaceSize;
        workspaceSize += alignTo(batchCount * ldda * n * sizeof(T), granularity);

        handle.ldda = ldda;
        handle.batchCount = batchCount;
        handle.n = n;
}

template <typename T>
void inverse_batched_magma(magma_int_t n, T** dA_array, magma_int_t ldda,
                           T**& dinvA_array, magma_int_t batchCount,
                           magma_queue_t queue, inverseHandle_t<T> handle){
        b_copy(handle.dA_cpy_array, dA_array, ldda * n, batchCount);

        magma_getrf_batched(n, n, handle.dA_cpy_array, ldda,
                            handle.dipiv_array, handle.info_array,
                            batchCount, queue);
        // assert_batched(batchCount, info_array);

        magma_getri_outofplace_batched(n, handle.dA_cpy_array, ldda,
                                       handle.dipiv_array,
                                       dinvA_array, ldda, handle.info_array,
                                       batchCount, queue);
        // assert_batched(batchCount, info_array);


}

template <typename T>
void inverse_batched(magma_int_t n, T** dA_array, magma_int_t ldda,
                     T** dinvA_array, magma_int_t batchCount,
                     magma_queue_t queue, inverseHandle_t<T> handle){
        inverse_batched_magma(n, dA_array, ldda, dinvA_array, batchCount,
                              queue, handle);

}
}
