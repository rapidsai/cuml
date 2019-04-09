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

namespace gmm {

template <typename T>
__global__
void subtractBatchedKernel(magma_int_t nDim, magma_int_t nObs, magma_int_t nCl,
                           T* dO, magma_int_t lddO,
                           T* dX, magma_int_t lddx,
                           T* dmu, magma_int_t lddmu,
                           int nThreads_x, int nThreads_y, int nThreads_z){

        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        int idxO, idxX, idxmu;
        int batchldd = nObs * lddx;

        for (size_t clId = k_start; clId < nCl; clId+=nThreads_z) {
                for (size_t obsId = j_start; obsId < nObs; obsId+=nThreads_y) {
                        for (size_t dimId = i_start; dimId < nDim; dimId+=nThreads_x) {
                                idxO = IDX2(dimId, obsId, clId, lddO, batchldd);
                                idxX = IDX(dimId, obsId, lddx);
                                idxmu = IDX(dimId, clId, lddmu);
                                dO[idxO] = dX[idxX] - dmu[idxmu];
                        }
                }
        }
}

template <typename T>
void subtract_batched(magma_int_t nDim, magma_int_t nObs, magma_int_t nCl,
                      T* dO, magma_int_t lddO,
                      T* dX, magma_int_t lddx,
                      T* dmu, magma_int_t lddmu){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)nDim, (int)block.x),
                  ceildiv((int)nObs, (int)block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        subtractBatchedKernel<T> <<< grid, block >>>(nDim, nObs, nCl,
                                                     dO, lddO,
                                                     dX, lddx,
                                                     dmu, lddmu,
                                                     nThreads_x, nThreads_y, nThreads_z);
        CUDA_CHECK(cudaPeekAtLastError());
}

}
