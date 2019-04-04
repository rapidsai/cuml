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

#include "utils.h"

using namespace MLCommon;
using namespace MLCommon::LinAlg;

namespace MLCommon {

template <typename T>
__device__
T sign(T x){
        if (x > 0)
                return (T) 1;
        else if (x < 0)
                return (T) -1;
        else
                return 0;
}

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
        allocate_pointer_array(handle.dA_array_cpy, ldda * n, batchCount);
        allocate(handle.info_array, batchCount);
}

template <typename T>
void destroyDeterminantHandle_t(determinantHandle_t<T>& handle,
                                int batchCount){
        free_pointer_array(handle.dipiv_array, batchCount);
        free_pointer_array(handle.dA_array_cpy, batchCount);
        CUDA_CHECK(cudaFree(handle.info_array));
}

template <typename T>
void det_batched(magma_int_t n, T** dA_array, magma_int_t ldda,
                 T* dDet_array, magma_int_t batchCount,
                 magma_queue_t queue, determinantHandle_t<T> handle){
        copy_batched(handle.dA_array_cpy, dA_array, ldda * n, batchCount);

        magma_getrf_batched(n, n, handle.dA_array_cpy, ldda,
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

        diag_product_batched(n, handle.dA_array_cpy, ldda,
                             dDet_array, batchCount);
}


}
