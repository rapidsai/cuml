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

#include <stats/sum.h>

#include "magma/b_bilinear.h"
#include "magma/b_inverse.h"
#include "magma/b_determinant.h"

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>

using namespace MLCommon;
using namespace MLCommon::LinAlg;

namespace MLCommon {
template <typename math_t>
void log(math_t *out, const math_t *in, int len,
         cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return std::log(in);
                },
                stream);
}



template <typename T>
__global__
void createLlhdBatchesKernel(int nObs, int nCl,
                             T **dX_batches, T **dmu_batches, T **dInvSigma_batches,
                             T **dX_array, T **dmu_array, T **dInvSigma_array,
                             int nThreads_x, int nThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t obsId = j_start; obsId < nObs; obsId+=nThreads_y) {
                for (size_t clId = i_start; clId < nCl; clId+=nThreads_x) {

                        size_t idx = IDX(clId, obsId, nCl);
                        dX_batches[idx] = dX_array[obsId];
                        dmu_batches[idx] = dmu_array[clId];
                        dInvSigma_batches[idx] = dInvSigma_array[clId];
                }
        }
}

template <typename T>
void create_llhd_batches(int nObs, int nCl,
                         T **&dX_batches, T **&dmu_batches, T **&dInvSigma_batches,
                         T **&dX_array, T **&dmu_array, T **&dInvSigma_array){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int) nCl, (int) block.x),
                  ceildiv((int) nObs, (int) block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;

        createLlhdBatchesKernel<T> <<< grid, block >>>(nObs, nCl,
                                                       dX_batches, dmu_batches, dInvSigma_batches,
                                                       dX_array, dmu_array, dInvSigma_array,
                                                       nThreads_x, nThreads_y);
        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
__global__
void subtractBatchedKernel(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                           T** dO_array, magma_int_t lddO,
                           T** dX_array, magma_int_t lddx,
                           T** dY_array, magma_int_t lddy,
                           int nThreads_x, int nThreads_y, int nThreads_z){

        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        int idxO, idxX, idxY;
        // TODO : Check the difference

        for (size_t bId = k_start; bId < batchCount; bId+=nThreads_z) {
                for (size_t j = j_start; j < n; j+=nThreads_x) {
                        for (size_t i = i_start; i < m; i+=nThreads_y) {
                                idxO = IDX(i, j, lddO);
                                idxX = IDX(i, j, lddx);
                                idxY = IDX(i, 0, lddy);
                                // printf("%f\n", (float) dX_array[bId][idxX] - dY_array[bId][idxY]);
                                // printf("%d\n", idxO);
                                dO_array[bId][idxO] = dX_array[bId][idxX] - dY_array[bId][idxY];
                        }
                }
        }
}

template <typename T>
void subtract_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                      T** dO_array, magma_int_t lddO,
                      T** dX_array, magma_int_t lddx,
                      T** dY_array, magma_int_t lddy){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)m, (int)block.x),
                  ceildiv((int)n, (int)block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        subtractBatchedKernel<T> <<< grid, block >>>(m, n, batchCount,
                                                     dO_array, lddO,
                                                     dX_array, lddx,
                                                     dY_array, lddy,
                                                     nThreads_x, nThreads_y, nThreads_z);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__host__ __device__
T lol_llhd_atomic(T inv_det, T bil, int nDim){
        return -0.5 * (-std::log(inv_det) + nDim * std::log(2 * M_PI) + bil);
}

template <typename T>
__global__
void LogLikelihoodKernel(int nObs, int nCl, int nDim,
                         T* dInvdet_array, T* dBil_batches,
                         T* dLlhd, int lddLlhd, bool isLog,
                         int nThreads_x, int nThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int idxL, idxB;
        for (size_t clId = i_start; clId < nCl; clId+=nThreads_x) {
                for (size_t oId = j_start; oId < nObs; oId+=nThreads_y) {
                        idxL = IDX(clId, oId, lddLlhd);
                        idxB = IDX(clId, oId, nCl);

                        dLlhd[idxL] = lol_llhd_atomic(dInvdet_array[clId],
                                                      dBil_batches[idxB],
                                                      nDim);

                        if (!isLog) {
                                dLlhd[idxL] = std::exp(dLlhd[idxL]);
                        }
                }
        }
}


template <typename T>
void _likelihood_batched(int nObs, int nCl, int nDim,
                         T* dInvdet_array, T* dBil_batches,
                         T* dLlhd, int lddLlhd, bool isLog){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv(nCl, (int)block.x),
                  ceildiv(nObs, (int)block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;

        LogLikelihoodKernel<T> <<< grid, block >>>(nObs, nCl, nDim,
                                                   dInvdet_array, dBil_batches,
                                                   dLlhd, lddLlhd, isLog,
                                                   nThreads_x, nThreads_y);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

}

template <typename T>
void likelihood_batched(magma_int_t nCl, magma_int_t nDim,
                        magma_int_t nObs,
                        T** &dX_array, int lddx,
                        T** &dmu_array, int lddmu,
                        T** &dsigma_array, int lddsigma_full, int lddsigma,
                        T* dLlhd, int lddLlhd,
                        bool isLog){
        // Allocate
        T **dInvSigma_array=NULL, *dInvdet_array=NULL;
        T **dX_batches=NULL, **dmu_batches=NULL,
        **dInvSigma_batches=NULL, **dDiff_batches=NULL;
        T *dBil_batches=NULL;


        magma_int_t batchCount = nObs * nCl;

        allocate_pointer_array(dInvSigma_array, lddsigma_full, nCl);
        allocate(dInvdet_array, nCl);

        allocate(dBil_batches, batchCount);
        allocate(dX_batches, batchCount);
        allocate(dmu_batches, batchCount);
        allocate(dInvSigma_batches, batchCount);
        allocate_pointer_array(dDiff_batches, lddx, batchCount);


        int device = 0;      // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        // Compute sigma inverses

        inverse_batched(nDim, dsigma_array, lddsigma, dInvSigma_array, nCl, queue);

        // Compute sigma inv dets
        det_batched(nDim, dInvSigma_array, lddsigma, dInvdet_array, nCl, queue);

        // Create batches
        create_llhd_batches(nObs, nCl,
                            dX_batches, dmu_batches, dInvSigma_batches,
                            dX_array, dmu_array, dInvSigma_array);

        // Compute diffs
        subtract_batched(nDim, 1, batchCount,
                         dDiff_batches, lddx,
                         dX_batches, lddx,
                         dmu_batches, lddmu);

        // Compute bilinears
        bilinear_batched(nDim, nDim,
                         dDiff_batches, dInvSigma_batches, lddsigma,
                         dDiff_batches, dBil_batches, batchCount, queue);


        // Compute log likelihoods
        _likelihood_batched(nObs, nCl, nDim,
                            dInvdet_array, dBil_batches, dLlhd, lddLlhd, isLog);

        // free
        free_pointer_array(dInvSigma_array, nCl);
        free_pointer_array(dDiff_batches, batchCount);

        CUDA_CHECK(cudaFree(dBil_batches));
        CUDA_CHECK(cudaFree(dInvdet_array));
        CUDA_CHECK(cudaFree(dX_batches));
        CUDA_CHECK(cudaFree(dmu_batches));
        CUDA_CHECK(cudaFree(dInvSigma_batches));

}

}
