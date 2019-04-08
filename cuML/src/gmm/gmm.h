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

#include <gmm/backend/gmm_backend.h>
#include <gmm/backend/sigmas_backend.h>
#include <gmm/backend/weights_backend.h>
#include <gmm/likelihood/b_likelihood.h>

using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace gmm {

template <typename T>
void _print_gmm_data(T* dX, GMM<T> &gmm, const std::string& msg) {
        printf("\n*************** .....\n");
        printf("%s\n", msg.c_str());
        print_matrix_device(gmm.nDim, gmm.nObs, dX, gmm.lddx, "dx matrix");
        print_matrix_device(gmm.nDim, gmm.nCl, gmm.dmu, gmm.lddmu, "dmu matrix");
        print_matrix_device(gmm.nDim, gmm.nDim * gmm.nCl, gmm.dsigma, gmm.lddsigma, "dSigma matrix");
        print_matrix_device(gmm.nCl, 1, gmm.dPis, gmm.lddPis, "dPis matrix");
        print_matrix_device(gmm.nCl, 1, gmm.dPis_inv, gmm.lddPis, "dPis inv matrix");
        print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");
        print_matrix_device(1, gmm.nObs, gmm.handle.dProbNorm, gmm.handle.lddprobnorm, "_prob_norm matrix");
        printf("\n..... ***************\n");
}

template <typename T>
void _print_gmm_data_bis(GMM<T> &gmm, const std::string& msg) {
        printf("\n*************** .....\n");
        printf("%s\n", msg.c_str());
        print_matrix_device(gmm.nCl, 1, gmm.dPis, gmm.lddPis, "dPis matrix");
        print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");
        printf("\n..... ***************\n");
}


template <typename T>
__global__
void fillKernel(T* dX){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        if (i_start == 0) {
                dX[0] = 112;
        }
}

template <typename T>
void create_GMMHandle_new(GMM<T> &gmm, void* workspace){
        gmm.handle.llhdWs = (T *)((size_t)gmm.handle.llhdWs + (size_t)workspace);
        createLlhdHandle_t_new(gmm.handle.llhd_handle, gmm.handle.llhdWs);

        gmm.handle.dX_array = (T **)((size_t)gmm.handle.dX_array + (size_t)workspace);
        gmm.handle.dmu_array = (T **)((size_t)gmm.handle.dmu_array + (size_t)workspace);
        gmm.handle.dsigma_array = (T **)((size_t)gmm.handle.dsigma_array + (size_t)workspace);

        gmm.handle.dX_batches = (T **)((size_t)gmm.handle.dX_batches + (size_t)workspace);
        gmm.handle.dmu_batches = (T **)((size_t)gmm.handle.dmu_batches + (size_t)workspace);
        gmm.handle.dsigma_batches = (T **)((size_t)gmm.handle.dsigma_batches + (size_t)workspace);
        // gmm.handle.dDiff = (T *)((size_t)gmm.handle.dDiff + (size_t)workspace);
        gmm.handle.dDiff_batches = (T **)((size_t)gmm.handle.dDiff_batches + (size_t)workspace);
        gmm.handle.dProbNorm = (T *)((size_t)gmm.handle.dProbNorm + (size_t)workspace);

        split_to_batches(gmm.handle.batchCount,
                         gmm.handle.dDiff_batches,
                         gmm.handle.llhd_handle.dDiff,
                         gmm.lddx * gmm.nObs);

        // create_GMMHandle(gmm);

        magma_init();
}

template <typename T>
size_t gmm_bufferSize(GMM<T> &gmm){

        size_t workspaceSize = 0;
        const size_t granularity = 256;
        size_t tempWsSize;

        gmm.handle.lddprobnorm = gmm.nObs;
        gmm.handle.batchCount = gmm.nCl;

        gmm.handle.dX_array = (T **)workspaceSize;
        workspaceSize += alignTo(gmm.nObs * sizeof(T*), granularity);

        gmm.handle.dmu_array = (T **)workspaceSize;
        workspaceSize += alignTo(gmm.nCl * sizeof(T*), granularity);

        gmm.handle.dsigma_array = (T **)workspaceSize;
        workspaceSize += alignTo(gmm.nCl * sizeof(T*), granularity);

        gmm.handle.dX_batches = (T **)workspaceSize;
        workspaceSize += alignTo(gmm.handle.batchCount * sizeof(T*), granularity);

        gmm.handle.dmu_batches = (T **)workspaceSize;
        workspaceSize += alignTo(gmm.handle.batchCount * sizeof(T*), granularity);

        gmm.handle.dsigma_batches = (T **)workspaceSize;
        workspaceSize += alignTo(gmm.handle.batchCount * sizeof(T*), granularity);

        gmm.handle.dDiff_batches = (T **)workspaceSize;
        workspaceSize += alignTo(gmm.handle.batchCount * sizeof(T*), granularity);

        gmm.handle.dProbNorm = (T *)workspaceSize;
        workspaceSize += alignTo(gmm.handle.lddprobnorm * sizeof(T), granularity);

        tempWsSize = alignTo(gmm.handle.batchCount * sizeof(T*), granularity);

        gmm.handle.llhdWs = (T *)workspaceSize;
        llhd_bufferSize(gmm.handle.llhd_handle,
                        gmm.nCl, gmm.nObs, gmm.nDim,
                        gmm.lddx, gmm.lddsigma, gmm.lddsigma_full,
                        tempWsSize);
        workspaceSize += alignTo(tempWsSize, granularity);
        return workspaceSize;
}

template <typename T>
void create_GMMHandle(GMM<T> &gmm){
        createllhdHandle_t(gmm.handle.llhd_handle,
                           gmm.nCl, gmm.nObs, gmm.nDim,
                           gmm.lddx, gmm.lddsigma, gmm.lddsigma_full);
}

template <typename T>
void setup(GMM<T> &gmm) {
        allocate(gmm.handle.dX_array, gmm.nObs);
        allocate(gmm.handle.dmu_array, gmm.nCl);
        allocate(gmm.handle.dsigma_array, gmm.nCl);

        int batchCount=gmm.nCl;

        allocate(gmm.handle.dX_batches, batchCount);
        allocate(gmm.handle.dmu_batches, batchCount);
        allocate(gmm.handle.dsigma_batches, batchCount);
        allocate_pointer_array(gmm.handle.dDiff_batches,
                               gmm.lddx * gmm.nObs, batchCount);

        gmm.handle.lddprobnorm = gmm.nObs;
        allocate(gmm.handle.dProbNorm, gmm.handle.lddprobnorm);

        create_GMMHandle(gmm);

        magma_init();
}

template <typename T>
void init(GMM<T> &gmm,
          T *dmu, T *dsigma, T *dPis, T *dPis_inv, T *dLlhd,
          int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
          T *cur_llhd, T reg_covar,
          int nCl, int nDim, int nObs) {
        gmm.dmu = dmu;
        gmm.dsigma = dsigma;
        gmm.dPis = dPis;
        gmm.dPis_inv = dPis_inv;
        gmm.dLlhd=dLlhd;

        gmm.cur_llhd = cur_llhd;

        gmm.lddx=lddx;
        gmm.lddmu=lddmu;
        gmm.lddsigma=lddsigma;
        gmm.lddsigma_full=lddsigma_full;
        gmm.lddPis=lddPis;
        gmm.lddLlhd=lddLlhd;

        gmm.nCl=nCl;
        gmm.nDim=nDim;
        gmm.nObs=nObs;

        gmm.reg_covar = reg_covar;

        // create_GMMHandle_new(gmm, Ws);
}


template <typename T>
void compute_lbow(GMM<T>& gmm){
        log(gmm.handle.dProbNorm, gmm.handle.dProbNorm, gmm.handle.lddprobnorm);
        MLCommon::Stats::sum(gmm.cur_llhd, gmm.handle.dProbNorm, 1, gmm.handle.lddprobnorm, true);
}

template <typename T>
void update_llhd(T* dX, GMM<T>& gmm,
                 cublasHandle_t cublasHandle,
                 magma_queue_t queue ){
        split_to_batches(gmm.nObs, gmm.handle.dX_array, dX, gmm.lddx);
        split_to_batches(gmm.nCl, gmm.handle.dmu_array, gmm.dmu, gmm.lddmu);
        split_to_batches(gmm.nCl, gmm.handle.dsigma_array, gmm.dsigma, gmm.lddsigma_full);

        likelihood_batched(gmm.nCl, gmm.nDim, gmm.nObs,
                           gmm.handle.dX_array, gmm.lddx,
                           gmm.handle.dmu_array, gmm.lddmu,
                           gmm.handle.dsigma_array, gmm.lddsigma_full, gmm.lddsigma,
                           gmm.dLlhd, gmm.lddLlhd,
                           false,
                           queue,
                           gmm.handle.llhd_handle);

        cublasdgmm(cublasHandle, CUBLAS_SIDE_LEFT, gmm.nCl, gmm.nObs,
                   gmm.dLlhd, gmm.lddLlhd, gmm.dPis, 1, gmm.dLlhd, gmm.lddLlhd);

        MLCommon::Stats::sum(gmm.handle.dProbNorm, gmm.dLlhd, gmm.nObs, gmm.lddLlhd, false);
}

template <typename T>
void update_rhos(T* dX, GMM<T>& gmm,
                 cublasHandle_t cublasHandle, magma_queue_t queue){
        normalize_matrix(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, true);
}

template <typename T>
void update_mus(T* dX, GMM<T>& gmm,
                cublasHandle_t cublasHandle, magma_queue_t queue){
        T alpha = (T)1.0 / gmm.nObs, beta = (T)0.0;
        CUBLAS_CHECK(cublasgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, gmm.nDim, gmm.nCl, gmm.nObs, &alpha, dX, gmm.lddx, gmm.dLlhd, gmm.lddLlhd, &beta, gmm.dmu, gmm.lddmu));
        inverse(gmm.dPis_inv, gmm.dPis, gmm.nCl);
        CUBLAS_CHECK(cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT,
                                gmm.nDim, gmm.nCl,
                                gmm.dmu, gmm.lddmu,
                                gmm.dPis_inv, 1,
                                gmm.dmu, gmm.lddmu));
}

template <typename T>
void update_sigmas(T* dX, GMM<T>& gmm,
                   cublasHandle_t cublasHandle, magma_queue_t queue){
        // int batchCount=gmm.nCl;
        int ldDiff= gmm.lddx;
        int batch_nObs, batch_obs_offset;
        int nBatches;

        CUDA_CHECK(cudaMemset(gmm.dsigma, 0, gmm.nCl * gmm.lddsigma_full));
        sqrt(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);

        nBatches = ceil((float) (gmm.nObs * gmm.nCl) /
                        (float) gmm.handle.llhd_handle.dDiff_size);
        batch_obs_offset = 0;
        batch_nObs = gmm.nObs;

        for (size_t batchId = 0; batchId < nBatches; batchId++) {
                if (batchId == nBatches - 1) {
                        batch_nObs = (gmm.nObs * gmm.nCl) % gmm.handle.llhd_handle.dDiff_size;
                        if (batch_nObs == 0) {
                                batch_nObs = gmm.handle.llhd_handle.dDiff_size;
                        }
                }
                else
                {
                        batch_nObs = gmm.handle.llhd_handle.dDiff_size;
                }
                batch_nObs /= gmm.nCl;

                create_sigmas_batches(gmm.nCl,
                                      gmm.handle.dX_batches,
                                      gmm.handle.dmu_batches,
                                      gmm.handle.dsigma_batches,
                                      // dX + IDX(0, batch_obs_offset, gmm.lddx), gmm.lddx,
                                      dX, gmm.lddx,
                                      gmm.dmu, gmm.lddmu,
                                      gmm.dsigma, gmm.lddsigma, gmm.lddsigma_full);

                // Compute diffs
                subtract_batched(gmm.nDim, batch_nObs, gmm.nCl,
                                 gmm.handle.dDiff_batches, ldDiff,
                                 gmm.handle.dX_batches, gmm.lddx,
                                 gmm.handle.dmu_batches, gmm.lddmu);

                dgmm_batched(gmm.nDim, batch_nObs, gmm.nCl,
                             gmm.handle.dDiff_batches, ldDiff,
                             gmm.handle.dDiff_batches, ldDiff,
                             gmm.dLlhd + IDX(0, batch_obs_offset, gmm.lddLlhd),
                             gmm.lddLlhd);

                // get the sum of all the covs
                T alpha = (T) 1.0 / gmm.nObs;
                T beta = (T)0.0;
                // T beta = (T)1.0;
                magmablas_gemm_batched(MagmaNoTrans, MagmaTrans,
                                       gmm.nDim, gmm.nDim, gmm.nObs,
                                       alpha, gmm.handle.dDiff_batches, ldDiff,
                                       gmm.handle.dDiff_batches, ldDiff, beta,
                                       gmm.handle.dsigma_batches, gmm.lddsigma, gmm.nCl,
                                       queue);

                batch_obs_offset += batch_nObs;
        }

// Normalize with respect to N_k
        inverse(gmm.dPis_inv, gmm.dPis, gmm.nCl);

        CUBLAS_CHECK(cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT,
                                gmm.lddsigma_full, gmm.nCl,
                                gmm.dsigma, gmm.lddsigma_full,
                                gmm.dPis_inv, 1,
                                gmm.dsigma, gmm.lddsigma_full));

        regularize_sigmas(gmm.nDim, gmm.nCl,
                          gmm.handle.dsigma_batches, gmm.lddsigma,
                          gmm.reg_covar);

        square(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);
}



// template <typename T>
// void update_sigmas(T* dX, GMM<T>& gmm,
//                    cublasHandle_t cublasHandle, magma_queue_t queue){
//         int batchCount=gmm.nCl;
//         int ldDiff= gmm.lddx;
//         // int batch_nObs, batch_obs_offset, nBatches;
//
//         sqrt(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);
//         create_sigmas_batches(gmm.nCl,
//                               gmm.handle.dX_batches, gmm.handle.dmu_batches, gmm.handle.dsigma_batches,
//                               dX, gmm.lddx, gmm.dmu, gmm.lddmu, gmm.dsigma, gmm.lddsigma, gmm.lddsigma_full);
//
//         // Compute sigmas
//
//         // nBatches = ceil((float) (gmm.nObs * gmm.nCl) /
//         //                 (float) gmm.handle.llhd_handle.dDiff_size);
//         // batch_obs_offset = 0;
//         // for (size_t batchId = 0; batchId < nBatches; batchId++) {
//         //         if (batchId == nBatches - 1) {
//         //                 batch_nObs = gmm.nObs % gmm.handle.llhd_handle.dDiff_size;
//         //                 if (batch_nObs == 0) {
//         //                         batch_nObs = gmm.handle.llhd_handle.dDiff_size;
//         //                 }
//         //         }
//         //         else
//         //         {
//         //                 batch_nObs = gmm.handle.llhd_handle.dDiff_size;
//         //         }
//         //         batchCount = batch_nObs * gmm.nCl;
//
//         // Compute diffs
//         subtract_batched(gmm.nDim, gmm.nObs, batchCount,
//                          gmm.handle.dDiff_batches, ldDiff,
//                          gmm.handle.dX_batches, gmm.lddx,
//                          gmm.handle.dmu_batches, gmm.lddmu);
//         // subtract_batched(gmm.nDim, gmm.nObs, batchCount,
//         //                  gmm.handle.dDiff_batches, ldDiff,
//         //                  gmm.handle.dX_batches + batch_obs_offset, gmm.lddx,
//         //                  gmm.handle.dmu_batches, gmm.lddmu);
//
//         dgmm_batched(gmm.nDim, gmm.nObs, gmm.nCl,
//                      gmm.handle.dDiff_batches, ldDiff,
//                      gmm.handle.dDiff_batches, ldDiff,
//                      gmm.dLlhd, gmm.lddLlhd);
//         // dgmm_batched(gmm.nDim, gmm.nObs, gmm.nCl,
//         //              gmm.handle.dDiff_batches, ldDiff,
//         //              gmm.handle.dDiff_batches, ldDiff,
//         //              gmm.dLlhd + IDX(0, batch_obs_offset, gmm.lddLlhd), gmm.lddLlhd);
//
//         // get the sum of all the covs
//         T alpha = (T) 1.0 / gmm.nObs;
//         T beta = (T)0.0;
//         // T beta = (T)1.0;
//         magmablas_gemm_batched(MagmaNoTrans, MagmaTrans,
//                                gmm.nDim, gmm.nDim, gmm.nObs,
//                                alpha, gmm.handle.dDiff_batches, ldDiff,
//                                gmm.handle.dDiff_batches, ldDiff, beta,
//                                gmm.handle.dsigma_batches, gmm.lddsigma, gmm.nCl,
//                                queue);
//         // magmablas_gemm_batched(MagmaNoTrans, MagmaTrans,
//         //                        gmm.nDim, gmm.nDim, gmm.nObs,
//         //                        alpha, gmm.handle.dDiff_batches, ldDiff,
//         //                        gmm.handle.dDiff_batches, ldDiff, beta,
//         //                        gmm.handle.dsigma_batches, gmm.lddsigma, gmm.nCl,
//         //                        queue);
//         // }
//         // Normalize with respect to N_k
//         inverse(gmm.dPis_inv, gmm.dPis, gmm.nCl);
//
//         CUBLAS_CHECK(cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT,
//                                 gmm.lddsigma_full, gmm.nCl,
//                                 gmm.dsigma, gmm.lddsigma_full,
//                                 gmm.dPis_inv, 1,
//                                 gmm.dsigma, gmm.lddsigma_full));
//
//         regularize_sigmas(gmm.nDim, gmm.nCl,
//                           gmm.handle.dsigma_batches, gmm.lddsigma,
//                           gmm.reg_covar);
//
//         square(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);
// }

template <typename T>
void update_pis(GMM<T>& gmm){
        // print_matrix_device(gmm.nCl, gmm.nObs,
        //                     gmm.dLlhd, gmm.lddLlhd, "dllhd matrix cpp");

        _update_pis(gmm.nObs,  gmm.nCl,
                    gmm.dPis, gmm.lddPis,
                    gmm.dLlhd, gmm.lddLlhd);
}


template <typename T>
void em_step(T* dX, GMM<T>& gmm,
             cublasHandle_t cublasHandle, magma_queue_t queue){

        // E step
        update_rhos(dX, gmm, cublasHandle, queue);

        // M step
        update_pis(gmm);
        update_mus(dX, gmm, cublasHandle, queue);
        update_sigmas(dX, gmm, cublasHandle, queue);

        // Likelihood estimate
        update_llhd(dX, gmm, cublasHandle, queue);
        compute_lbow(gmm);
}

template <typename T>
void fit(T* dX, int n_iter, GMM<T>& gmm,
         cublasHandle_t cublasHandle, magma_queue_t queue) {
        for (size_t i = 0; i < n_iter; i++) {
                em_step(dX, gmm, cublasHandle, queue);
        }
}

}
