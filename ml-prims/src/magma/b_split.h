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

#include <stdio.h>
#include <stdlib.h>

#include <magma_v2.h>
#include "cuda_utils.h"

namespace MLCommon {

template <typename T>
__global__
void split_to_batchesKernel(magma_int_t batchCount,
                            T **dA_array, T *dA, magma_int_t ldda,
                            int nThreads_x){
        int start = threadIdx.x + blockDim.x * blockIdx.x;
        for (size_t bId = start; bId < batchCount; bId+=nThreads_x) {
                dA_array[bId] = dA + IDX(0, bId, ldda);
        }
}

template <typename T>
void split_to_batches(magma_int_t batchCount,
                      T **&dA_array, T *&dA, magma_int_t ldda,
                      cudaStream_t stream=0){
        dim3 block(32);
        dim3 grid(ceildiv((int)batchCount, (int) block.x));

        int nThreads_x = grid.x * block.x;

        split_to_batchesKernel<T> <<< grid, block, 0, stream >>>(batchCount,
                                                                 dA_array,
                                                                 dA, ldda,
                                                                 nThreads_x);
        CUDA_CHECK(cudaPeekAtLastError());
}
}
