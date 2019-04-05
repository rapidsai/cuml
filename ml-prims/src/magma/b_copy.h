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

#include <magma_v2.h>
#include "cuda_utils.h"

namespace MLCommon {

template <typename T>
__global__
void copy_batched_kernel(T **dA_dest_array, T **dA_src_array,
                         size_t len, magma_int_t batchCount,
                         int nThreads_x, int nThreads_y ){
        int bId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int i_start = threadIdx.y + blockDim.y * blockIdx.y;
        for (size_t bId = bId_start; bId < batchCount; bId+=nThreads_x) {
                for (size_t i = i_start; i < len; i+=nThreads_y) {
                        dA_dest_array[bId][i] = dA_src_array[bId][i];
                }
        }
}

template <typename T>
void b_copy(T **dA_dest_array, T **dA_src_array,
            int len, int batchCount,
            cudaStream_t stream = 0){
        dim3 block(32, 32);
        dim3 grid(ceildiv(batchCount, (int)block.x),
                  ceildiv(len, (int)block.y));
        int nThreads_x= grid.x * block.x;
        int nThreads_y= grid.y * block.y;
        copy_batched_kernel<T> <<< grid, block, 0, stream >>>(dA_dest_array, dA_src_array,
                                                              len, batchCount,
                                                              nThreads_x, nThreads_y );
        CUDA_CHECK(cudaPeekAtLastError());
}

}
