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
#include "gmm/likelihood/likelihood_handles.h"
#include "magma/b_allocate.h"
#include "magma/b_split.h"

// #include "cuda_utils.h"

using namespace MLCommon::LinAlg;

namespace gmm {

// template <typename T>
// __global__
// void createBilinearBatchesKernel(int batchCount,
//                                  T **dX_batches, T *dX,
//                                  int nThreads_x){
//         int i_start = threadIdx.x + blockDim.x * blockIdx.x;
//
//         for (size_t bId = i_start; bId < batchCount; bId+=nThreads_x) {
//                 dX_batches[bId] = dX;
//         }
// }
//
// template <typename T>
// void create_bilinear_batches(int batchCount, T **&dX_batches, T *dX){
//         dim3 block(32);
//         dim3 grid(ceildiv((int) batchCount, (int) block.x));
//
//         int nThreads_x = grid.x * block.x;
//
//         createBilinearBatchesKernel<T> <<< grid, block >>>(batchCount,
//                                                            dX_batches, dX,
//                                                            nThreads_x);
//         CUDA_CHECK(cudaPeekAtLastError());
// }

template <typename T>
__device__
T dot(int n, T* x, T* y){
        T res = 0;
        for (size_t i = 0; i < n; i++) {
                res += x[i] * y[i];
        }
        return res;
}

template <typename T>
__global__
void dot_batched_kernel(int n,
                        T **dX_array, T **dY_array, T *dO,
                        magma_int_t batchCount,
                        int numThreads){
        int idxThread = threadIdx.x + blockDim.x * blockIdx.x;
        for (size_t i = idxThread; i < batchCount; i+=numThreads) {
                dO[i] = dot(n, dX_array[i], dY_array[i]);
        }
}

template <typename T>
void dot_batched(int n, T **dX_array, T **dY_array, T *dO,
                 magma_int_t batchCount){
        dim3 block(32);
        dim3 grid(ceildiv(batchCount, (int)block.x));
        int numThreads = grid.x * block.x;
        dot_batched_kernel<T> <<< grid, block >>>(n,
                                                  dX_array, dY_array,
                                                  dO,
                                                  batchCount,
                                                  numThreads);
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void createBilinearHandle_t_new(llhd_bilinearHandle_t<T>& handle, void* workspace){
        handle.dT_vec_array = (T **)((size_t)handle.dT_vec_array + (size_t)workspace);
        handle.dT_mat_array = (T **)((size_t)handle.dT_mat_array + (size_t)workspace);
        handle.dT = (T *)((size_t)handle.dT + (size_t)workspace);

        split_to_batches(handle.nObs * handle.nCl, handle.dT_vec_array,
                         handle.dT, handle.lddt);
        split_to_batches(handle.nCl, handle.dT_mat_array,
                         handle.dT, handle.lddt * handle.nObs);
}

template <typename T>
void bilinear_bufferSize(llhd_bilinearHandle_t<T>& handle,
                         int nDim, int nCl, int nObs, int lddt,
                         size_t& workspaceSize){
        workspaceSize = 0;
        const size_t granularity = 256;

        handle.dT_vec_array = (T **)workspaceSize;
        workspaceSize += alignTo(nObs * nCl * sizeof(T*), granularity);
        handle.dT_mat_array = (T **)workspaceSize;
        workspaceSize += alignTo(nCl * sizeof(T*), granularity);
        handle.dT = (T *)workspaceSize;
        workspaceSize += alignTo(nObs * nCl * lddt * sizeof(T), granularity);

        handle.lddt = lddt;
        handle.batchCount = nObs * nCl;
        handle.nCl = nCl;
        handle.nObs = nObs;
}

template <typename T>
void bilinear_batched(magma_int_t nDim, magma_int_t nObs, magma_int_t nCl,
                      T** dInvSigma_array, magma_int_t lddinv_sigma,
                      T **dDiff_mat_array, magma_int_t lddDiff,
                      T** dDiff_vec_array,
                      T *dO,
                      magma_int_t batchCount,
                      magma_queue_t queue, llhd_bilinearHandle_t<T> handle)
{
        T alpha = 1, beta = 0;

        magmablas_gemm_batched(
                MagmaNoTrans, MagmaNoTrans,
                nDim, nObs, nDim,
                alpha,
                dInvSigma_array, lddinv_sigma,
                dDiff_mat_array, lddDiff,
                beta,
                handle.dT_mat_array, handle.lddt,
                nCl, queue );

        // print_matrix_device(nDim, nObs * nCl, handle.dT, handle.lddt, "Dt matrix");
        //
        // print_matrix_batched(nDim, nObs, nCl,
        //                      dDiff_mat_array, lddDiff,
        //                      "dDiff_array");
        // print_matrix_batched(nDim, nDim, nCl,
        //                      dInvSigma_array, lddinv_sigma,
        //                      "dInvSigma_array");
        //
        // print_matrix_batched(nDim, 1, nObs * nCl, handle.dT_vec_array, handle.lddt, "dT_vec_array");

        // Batched dot
        // printf("nCl * nObs %d\n", nCl * nObs );
        dot_batched(nDim, handle.dT_vec_array, dDiff_vec_array, dO, nCl * nObs);
}
}
