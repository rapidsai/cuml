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

#include "utils.h"

using namespace MLCommon;
using namespace MLCommon::LinAlg;

namespace MLCommon {


template <typename T>
__global__
void diag_batched_kernel(magma_int_t n, T** dU_array, magma_int_t lddu,
                         T* dDet_array, magma_int_t batchCount, int numThreads){
        int idxThread = threadIdx.x + blockDim.x * blockIdx.x;
        for (size_t i = idxThread; i < batchCount; i+=numThreads) {
                dDet_array[i] = 0;
                for (size_t j = 0; j < n; j++) {
                        dDet_array[i] += std::log(std::abs(dU_array[i][IDX(j, j, lddu)]));
                }
                dDet_array[i] = std::exp(dDet_array[i]);
        }
}

template <typename T>
void diag_product_batched(magma_int_t n, T** dU_array, magma_int_t lddu,
                          T* dDet_array, magma_int_t batchCount){
        dim3 block(32);
        dim3 grid(ceildiv(batchCount, (int)block.x));
        int numThreads = grid.x * block.x;

        diag_batched_kernel<T> <<< grid, block >>>(n, dU_array, lddu,
                                                   dDet_array, batchCount,
                                                   numThreads);
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void createDeterminantHandle_t(determinantHandle_t<T>& handle,
                               int n, int ldda, int batchCount){
        allocate_pointer_array(handle.dipiv_array, n, batchCount);
        allocate_pointer_array(handle.dA_cpy_array, ldda * n, batchCount);
        allocate(handle.info_array, batchCount);
}

template <typename T>
void createDeterminantHandle_t_new(determinantHandle_t<T>& handle, void* workspace){
        handle.dipiv_array = (int **)((size_t)handle.dipiv_array + (size_t)workspace);
        handle.dA_cpy_array = (T **)((size_t)handle.dA_cpy_array + (size_t)workspace);

        handle.dipiv = (int *)((size_t)handle.dipiv + (size_t)workspace);
        handle.dA_cpy = (T *)((size_t)handle.dA_cpy + (size_t)workspace);
        handle.info_array = (int *)((size_t)handle.info_array + (size_t)workspace);

        split_to_batches(handle.batchCount, handle.dipiv_array, handle.dipiv, handle.n);
        split_to_batches(handle.batchCount, handle.dA_cpy_array, handle.dA_cpy, handle.ldda * handle.n);
}

template <typename T>
void determinant_bufferSize(determinantHandle_t<T>& handle,
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
void det_batched(magma_int_t n, T** dA_array, magma_int_t ldda,
                 T* dDet_array, magma_int_t batchCount,
                 magma_queue_t queue, determinantHandle_t<T> handle){
        b_copy(handle.dA_cpy_array, dA_array, ldda * n, batchCount);

        magma_getrf_batched(n, n, handle.dA_cpy_array, ldda,
                            handle.dipiv_array, handle.info_array,
                            batchCount, queue);

        // magma_potrf_batched(MagmaLower,
        //                     n,
        //                     dA_array,
        //                     ldda,
        //                     info_array,
        //                     batchCount,
        //                     queue
        //                     );
        // assert_batched(batchCount, info_array);

        diag_product_batched(n, handle.dA_cpy_array, ldda,
                             dDet_array, batchCount);
}


}
