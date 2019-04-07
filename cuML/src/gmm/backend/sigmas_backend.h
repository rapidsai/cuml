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

#include <magma/magma_utils.h>

namespace gmm {

template <typename T>
__global__
void regularizeKernel (int n, int batchCount,
                       T **dA_array, int ldda,
                       T reg,
                       int nThreads_x, int nThreads_y) {
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t bId = j_start; bId < batchCount; bId+=nThreads_y) {
                for (size_t i = i_start; i <  n; i+=nThreads_x) {
                        dA_array[bId][IDX(i, i, ldda)] += reg;
                }
        }
}


template <typename T>
void regularize_sigmas(int n, int batchCount,
                       T** dA_array, int ldda,
                       T reg) {
        dim3 block(32,32);
        dim3 grid(ceildiv(n, (int)block.x),
                  ceildiv(batchCount, (int)block.y));
        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        regularizeKernel<T><<< grid, block >>>(n, batchCount,
                                               dA_array, ldda,
                                               reg,
                                               nThreads_x, nThreads_y);
        CUDA_CHECK(cudaPeekAtLastError());
}




template <typename T>
__global__
void createSigmasBatchesKernel(int nCl,
                               T **dX_batches, T **dmu_batches, T **dsigma_batches,
                               T *dX, magma_int_t lddx,
                               T *dmu,  magma_int_t lddmu,
                               T *dsigma,  magma_int_t lddsigma, magma_int_t lddsigma_full,
                               int nThreads_x){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;

        for (size_t clId = i_start; clId < nCl; clId+=nThreads_x) {
                dX_batches[clId] = dX;
                dmu_batches[clId] = dmu + IDX(0, clId, lddmu);
                dsigma_batches[clId] = dsigma + IDX(0, clId, lddsigma_full);
        }
}

template <typename T>
void create_sigmas_batches(int nCl,
                           T **&dX_batches, T **&dmu_batches, T **&dsigma_batches,
                           T *&dX, magma_int_t lddx,
                           T *&dmu,  magma_int_t lddmu,
                           T *&dsigma,  magma_int_t lddsigma, magma_int_t lddsigma_full){
        dim3 block(32, 1, 1);
        dim3 grid(ceildiv((int) nCl, (int) block.x), 1, 1);

        int nThreads_x = grid.x * block.x;

        createSigmasBatchesKernel<T> <<< grid, block >>>(nCl,
                                                         dX_batches, dmu_batches, dsigma_batches,
                                                         dX, lddx,
                                                         dmu,  lddmu,
                                                         dsigma, lddsigma, lddsigma_full,
                                                         nThreads_x);

        CUDA_CHECK(cudaPeekAtLastError());
}

}
