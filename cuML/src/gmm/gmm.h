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

        // Update _prob_norm
        // _print_gmm_data_bis(gmm, "start of _prob_norm");
        MLCommon::Stats::sum(gmm.handle.dProbNorm, gmm.dLlhd, gmm.nObs, gmm.lddLlhd, false);
        // _print_gmm_data_bis(gmm, "start of _prob_norm");

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
        int batchCount=gmm.nCl;
        int ldDiff= gmm.lddx;

        create_sigmas_batches(gmm.nCl,
                              gmm.handle.dX_batches, gmm.handle.dmu_batches, gmm.handle.dsigma_batches,
                              dX, gmm.lddx, gmm.dmu, gmm.lddmu, gmm.dsigma, gmm.lddsigma, gmm.lddsigma_full);

        // Compute diffs
        subtract_batched(gmm.nDim, gmm.nObs, batchCount,
                         gmm.handle.dDiff_batches, ldDiff,
                         gmm.handle.dX_batches, gmm.lddx,
                         gmm.handle.dmu_batches, gmm.lddmu);

        // Compute sigmas
        sqrt(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);

        dgmm_batched(gmm.nDim, gmm.nObs, gmm.nCl,
                     gmm.handle.dDiff_batches, ldDiff,
                     gmm.handle.dDiff_batches, ldDiff,
                     gmm.dLlhd, gmm.lddLlhd);

        // get the sum of all the covs
        T alpha = (T) 1.0 / gmm.nObs, beta = (T)0.0;
        magmablas_gemm_batched(MagmaNoTrans, MagmaTrans,
                               gmm.nDim, gmm.nDim, gmm.nObs,
                               alpha, gmm.handle.dDiff_batches, ldDiff,
                               gmm.handle.dDiff_batches, ldDiff, beta,
                               gmm.handle.dsigma_batches, gmm.lddsigma, gmm.nCl,
                               queue);

        // Normalize with respect to N_k
        inverse(gmm.dPis_inv, gmm.dPis, gmm.nCl);

        CUBLAS_CHECK(cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT,
                                gmm.lddsigma_full, gmm.nCl,
                                gmm.dsigma, gmm.lddsigma_full,
                                gmm.dPis_inv, 1,
                                gmm.dsigma, gmm.lddsigma_full));

        // regularize_sigmas(gmm.nDim, dsigma_batches, gmm.lddsigma, gmm.reg_covar);

        square(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);

}

template <typename T>
void update_pis(GMM<T>& gmm){
        // _print_gmm_data_bis(gmm, "start of pis");

        MLCommon::Stats::sum(gmm.dPis, gmm.dLlhd, gmm.lddLlhd, gmm.nObs, true);
        // _print_gmm_data_bis(gmm, "\n\nend of pis");
        T epsilon;
        if(std::is_same<T,float>::value) {
                epsilon = 1.1920928955078125e-06;
        }
        else if(std::is_same<T,double>::value) {
                epsilon = 2.220446049250313e-15;
        }
        naiveAddElem(gmm.dPis, gmm.dPis, epsilon, gmm.nCl);

        normalize_matrix(gmm.lddPis, 1, gmm.dPis, gmm.lddPis, true);
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

template <typename T>
void free(GMM<T>& gmm){
        CUDA_CHECK(cudaFree(gmm.handle.dX_array));
        CUDA_CHECK(cudaFree(gmm.handle.dsigma_array));
        CUDA_CHECK(cudaFree(gmm.handle.dmu_array));
}


}
