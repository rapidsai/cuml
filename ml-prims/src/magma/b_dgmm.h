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

namespace MLCommon {

template <typename T>
__global__
void dgmmBatchedKernel(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                       T** dO_array, magma_int_t lddO,
                       T** dX_array, magma_int_t lddx,
                       T* dD_array, magma_int_t lddd,
                       int nThreads_x, int nThreads_y, int nThreads_z){

        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        int idxO, idxX, idxD;

        for (size_t bId = k_start; bId < batchCount; bId+=nThreads_z) {
                for (size_t j = j_start; j < n; j+=nThreads_y) {
                        for (size_t i = i_start; i < m; i+=nThreads_x) {
                                idxO = IDX(i, j, lddO);
                                idxX = IDX(i, j, lddx);
                                idxD = IDX(bId, j, lddd);
                                dO_array[bId][idxO] = dX_array[bId][idxX] * dD_array[idxD];
                        }
                }
        }
}

template <typename T>
void dgmm_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                  T** dO_array, magma_int_t lddO,
                  T** dX_array, magma_int_t lddx,
                  T* dD_array, magma_int_t lddd){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)m, (int)block.x),
                  ceildiv((int)n, (int)block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        dgmmBatchedKernel<T> <<< grid, block >>>(m, n, batchCount,
                                                 dO_array, lddO,
                                                 dX_array, lddx,
                                                 dD_array, lddd,
                                                 nThreads_x, nThreads_y, nThreads_z);
        CUDA_CHECK(cudaPeekAtLastError());
}

}
