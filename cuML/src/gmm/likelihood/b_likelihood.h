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

// #include "magma/b_bilinear.h"
#include "magma/b_inverse.h"
#include "magma/b_determinant.h"
#include "magma/b_allocate.h"
#include <magma/b_split.h>
#include <magma/b_subtract.h>
#include <magma/b_dgmm.h>

#include "gmm/likelihood/handle.h"
#include "gmm/likelihood/bilinear_backend.h"

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>

#include <magma/magma_utils.h>

using namespace MLCommon;
using namespace MLCommon::LinAlg;

namespace gmm {
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
float to_Mb(T size){
        return ((float) size / (1024 * 1024));
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
        CUDA_CHECK(cudaPeekAtLastError());

}



template <typename T>
void createllhdHandle_t(llhdHandle_t<T>& llhd_handle,
                        int nCl, int nObs, int nDim,
                        int lddx, int lddsigma, int lddsigma_full){
        // magma_int_t batchCount = nObs * nCl;
        //
        // allocate_pointer_array(llhd_handle.dInvSigma_array, lddsigma_full, nCl);
        // allocate_pointer_array(llhd_handle.dDiff_batches, lddx, batchCount);
        //
        // allocate(llhd_handle.dInvdet_array, nCl);
        // allocate(llhd_handle.dBil_batches, batchCount);
        // allocate(llhd_handle.dX_batches, batchCount);
        // allocate(llhd_handle.dmu_batches, batchCount);
        // allocate(llhd_handle.dInvSigma_batches, batchCount);
        //
        // createBilinearHandle_t(llhd_handle.bilinearHandle, nDim, batchCount);
        // createDeterminantHandle_t(llhd_handle.determinantHandle, nDim, lddsigma, batchCount);
        // createInverseHandle_t(llhd_handle.inverseHandle, nCl, nDim, lddsigma);
}

template <typename T>
void createLlhdHandle_t_new(llhdHandle_t<T>& handle, void* workspace){
        handle.dInvSigma_array = (T **)((size_t)handle.dInvSigma_array + (size_t)workspace);
        handle.dInvSigma = (T *)((size_t)handle.dInvSigma + (size_t)workspace);
        handle.dInvdet_array = (T *)((size_t)handle.dInvdet_array + (size_t)workspace);
        handle.dX_batches = (T **)((size_t)handle.dX_batches + (size_t)workspace);
        handle.dmu_batches = (T **)((size_t)handle.dmu_batches + (size_t)workspace);
        handle.dInvSigma_batches = (T **)((size_t)handle.dInvSigma_batches + (size_t)workspace);
        handle.dDiff = (T *)((size_t)handle.dDiff + (size_t)workspace);
        handle.dDiff_batches = (T **)((size_t)handle.dDiff_batches + (size_t)workspace);
        handle.dBil_batches = (T *)((size_t)handle.dBil_batches + (size_t)workspace);

        handle.bilinearWs = (T *)((size_t)handle.bilinearWs + (size_t)workspace);
        handle.determinantWs = (T *)((size_t)handle.determinantWs + (size_t)workspace);
        handle.inverseWs = (T *)((size_t)handle.inverseWs + (size_t)workspace);

        split_to_batches(handle.nCl, handle.dInvSigma_array, handle.dInvSigma, handle.lddsigma_full);
        split_to_batches(handle.batchCount, handle.dDiff_batches, handle.dDiff, handle.lddx);

        createDeterminantHandle_t_new(handle.determinantHandle, handle.determinantWs);
        createBilinearHandle_t_new(handle.bilinearHandle, handle.bilinearWs);
        createInverseHandle_t_new(handle.inverseHandle, handle.inverseWs);
}

template <typename T>
void llhd_bufferSize(llhdHandle_t<T>& handle,
                     int nCl, int nObs, int nDim,
                     int lddx, int lddsigma, int lddsigma_full, int lddDiff,
                     size_t& workspaceSize){
        workspaceSize = 0;
        const size_t granularity = 256;
        size_t tempWsSize;
        magma_int_t batchCount = nObs * nCl;

        handle.dInvSigma_array = (T **)workspaceSize;
        workspaceSize += alignTo(nCl * sizeof(T*), granularity);

        handle.dInvSigma = (T *)workspaceSize;
        workspaceSize += alignTo(lddsigma_full * nCl * sizeof(T), granularity);

        handle.dDiff_batches = (T **)workspaceSize;
        workspaceSize += alignTo(batchCount * sizeof(T*), granularity);

        handle.dDiff = (T *)workspaceSize;
        workspaceSize += alignTo(lddx * batchCount * sizeof(T), granularity);
        handle.dDiff_size = batchCount;

        handle.dBil_batches = (T *)workspaceSize;
        workspaceSize += alignTo(batchCount * sizeof(T), granularity);

        handle.dInvdet_array = (T *)workspaceSize;
        workspaceSize += alignTo(nCl * sizeof(T), granularity);

        handle.dX_batches = (T **)workspaceSize;
        workspaceSize += alignTo(batchCount * sizeof(T*), granularity);

        handle.dmu_batches = (T **)workspaceSize;
        workspaceSize += alignTo(batchCount * sizeof(T*), granularity);

        handle.dInvSigma_batches = (T **)workspaceSize;
        workspaceSize += alignTo(batchCount * sizeof(T*), granularity);

        handle.inverseWs = (T *)workspaceSize;
        inverse_bufferSize(handle.inverseHandle,
                           nDim, lddsigma, nCl,
                           tempWsSize);
        workspaceSize += alignTo(tempWsSize, granularity);
        printf("inverse workspace size %f Mb\n", to_Mb(tempWsSize) );

        handle.bilinearWs = (T *)workspaceSize;
        bilinear_bufferSize(handle.bilinearHandle,
                            nDim, nCl, nObs, lddDiff,
                            tempWsSize);
        workspaceSize += alignTo(tempWsSize, granularity);
        printf("bilinear workspace size %f Mb\n", to_Mb(tempWsSize) );

        handle.determinantWs = (T *)workspaceSize;
        determinant_bufferSize(handle.determinantHandle,
                               nDim, lddx, nCl,
                               tempWsSize);
        workspaceSize += alignTo(tempWsSize, granularity);
        printf("determinant workspace size %f Mb\n", to_Mb(tempWsSize) );
        printf("\n");

        handle.nCl = nCl;
        handle.lddx = lddx;
        handle.lddsigma_full = lddsigma_full;
        handle.batchCount = batchCount;
}


template <typename T>
void likelihood_batched(magma_int_t nCl, magma_int_t nDim,
                        magma_int_t nObs,
                        T** &dX_array, int lddx,
                        T** &dmu_array, int lddmu,
                        T** &dsigma_array, int lddsigma_full, int lddsigma,
                        T* dLlhd, int lddLlhd,
                        bool isLog,
                        magma_queue_t queue,
                        llhdHandle_t<T>& llhd_handle){
        int batch_nObs, batchCount, batch_obs_offset, nBatches;

        // Compute sigma inverses
        inverse_batched(nDim, dsigma_array, lddsigma,
                        llhd_handle.dInvSigma_array, nCl,
                        queue, llhd_handle.inverseHandle);

        // Compute sigma inv dets
        det_batched(nDim, llhd_handle.dInvSigma_array, lddsigma,
                    llhd_handle.dInvdet_array, nCl,
                    queue, llhd_handle.determinantHandle);

        // Create batches
        create_llhd_batches(nObs, nCl,
                            llhd_handle.dX_batches,
                            llhd_handle.dmu_batches,
                            llhd_handle.dInvSigma_batches,
                            dX_array, dmu_array, llhd_handle.dInvSigma_array);

        // nBatches = ceil((float) (nObs * nCl) / (float) llhd_handle.dDiff_size);
        // printf("nbatches %d\n", nBatches);
        // printf("nObs %d\n", nObs);

        // batch_obs_offset = 0;
        // for (size_t batchId = 0; batchId < nBatches; batchId++) {
        //         if (batchId == nBatches - 1) {
        //                 batch_nObs = nObs % llhd_handle.dDiff_size;
        //                 if (batch_nObs == 0) {
        //                         batch_nObs = llhd_handle.dDiff_size;
        //                 }
        //         }
        //         else
        //         {
        //                 batch_nObs = llhd_handle.dDiff_size;
        //         }

        batchCount = nObs * nCl;

        // print_matrix_batched(1, nDim, batchCount,
        //                      llhd_handle.dX_batches, lddx,
        //                      "dX_batches");

        subtract_batched(nDim, 1, batchCount,
                         llhd_handle.dDiff_batches, lddx,
                         llhd_handle.dX_batches, lddx,
                         llhd_handle.dmu_batches, lddmu);

        // print_matrix_batched(1, nDim, batchCount,
        //                      llhd_handle.dDiff_batches, lddx,
        //                      "dDiff_batches");

        // Compute bilinears
        bilinear_batched(nDim,
                         llhd_handle.dInvSigma_batches, lddsigma,
                         llhd_handle.dDiff_batches,
                         llhd_handle.dBil_batches,
                         batchCount,
                         queue, llhd_handle.bilinearHandle);
        // bilinear_batched(nDim, nDim,
        //                  llhd_handle.dDiff_batches,
        //                  llhd_handle.dInvSigma_batches, lddsigma,
        //                  llhd_handle.dDiff_batches,
        //                  llhd_handle.dBil_batches,
        //                  batchCount,
        //                  queue, llhd_handle.bilinearHandle);

        // print_matrix_device(nCl, nObs,
        //                     llhd_handle.dBil_batches, nCl,
        //                     "dBil_batches");

        // Compute log likelihoods
        _likelihood_batched(nObs, nCl, nDim,
                            llhd_handle.dInvdet_array,
                            llhd_handle.dBil_batches,
                            dLlhd, lddLlhd,
                            isLog);


        // print_matrix_device(nCl, nObs,
        //                     dLlhd, lddLlhd, "dLlhd");


        // batch_obs_offset += batch_nObs;
        // }
}

}
