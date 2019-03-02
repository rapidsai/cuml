#pragma once

// #include <hmm/structs.h>
#include <hmm/magma/b_likelihood.h>
#include <hmm/cuda/stats.h>
#include <hmm/cuda/random.h>

#include <stats/sum.h>
#include <ml_utils.h>

using namespace MLCommon::HMM;
using namespace MLCommon;

template <typename T>
void generate_trans_matrix(magma_int_t m, magma_int_t n, T* dA, magma_int_t lda, bool colwise){
        fill_matrix_gpu(m, n, dA, lda);
        normalize_matrix(m, n, dA, lda, colwise);
}

template <typename T>
struct GMM {
        T *dX, *dmu, *dsigma, *ps;
        T *dmu_array, *dsigma_array;
        T *dLlhd;

        magma_int_t ldx, ldmu, ldsigma, ldsigma_full, ldps, ldLlhd;

        int nCl, nDim, nObs;

        // // Random parameters
        paramsRandom<T> paramsRd = paramsRandom<T>((T) 0, (T) 1,
                                                   (unsigned long long) 1234ULL);

        // all the workspace related pointers
        int *info;
        bool initialized=false;

        cublasHandle_t cublasHandle;

        GMM(int _nCl, int _nDim, int _nObs) {
                nDim = _nDim;
                nCl = _nCl;
                nObs = _nObs;

                ldx = magma_roundup(nDim, RUP_SIZE);
                ldmu = magma_roundup(nDim, RUP_SIZE);
                ldsigma = magma_roundup(nDim, RUP_SIZE);
                ldsigma_full = nDim * ldsigma;
                ldLlhd = magma_roundup(nCl, RUP_SIZE);

                allocate(dX, ldx * nObs);
                allocate(dmu, ldmu * nCl);
                allocate(dsigma, ldsigma_full * nCl);
                allocate(dLlhd, ldLlhd * nObs);

                // CUBLAS_CHECK(cublasCreate(&cublasHandle));

        }

        void initialize() {
                random_matrix_batched(nDim, 1, nCl, dmu, ldmu, false);
                random_matrix_batched(nDim, nDim, nCl, dsigma, ldsigma, true);
                generate_trans_matrix(nCl, nObs, dLlhd, ldLlhd, false);
                generate_trans_matrix(nObs, 1, ps, nObs, false);
                initialized = true;
        }

        void free(){
                CUDA_CHECK(cudaFree(dmu_array));
                CUDA_CHECK(cudaFree(dsigma_array));
                CUDA_CHECK(cudaFree(dLlhd));
                CUDA_CHECK(cudaFree(ps));
                // CUBLAS_CHECK(cublasDestroy(cublasHandle));
        }

        // void fit(T* dX) {
        //         _em(dX);
        // }
        //
        //
        // void compute_ps(T* ps, T* dLlhd, int m, int n){
        //         sum(ps, dLlhd, n, m, true);
        // }

        // template <typename T>
        // void apply_ps( int m, int n, T* ps, T* dLlhd,){
        //         sum(ps, dLlhd, n, m, true);
        // }
        //
        // void _em(T* dX, int n_iter){
        //         assert(initialized);
        //
        //         MLCommon::HMM::gen_trans_matrix(1, nCl, ps,  paramsRd, true);
        //         // paramsRd->start = 1;
        //         // paramsRd->end = 5;
        //         MLCommon::HMM::gen_matrix(nDim, nCl, dmu_array, ldmu, paramsRd)
        //         MLCommon::HMM::gen_matrix(ldsigma_full, nCl, dsigma_array, ldsigma_full, paramsRd)
        //
        //         // Run the EM algorithm
        //         for (int it = 0; it < paramsEm->n_iter; it++) {
        //                 printf("\n -------------------------- \n");
        //                 printf(" iteration %d\n", it);
        //
        //                 // E step
        //                 likelihood_batched(nCl, nDim, nObs,
        //                                    dX, ldx,
        //                                    dmu, ldmu,
        //                                    dsigma, ldsigma_full, ldsigma,
        //                                    dLlhd, ldLlhd, false);
        //                 // apply_ps();
        //                 normalize_matrix(nCl, nObs, dLlhd, true);
        //
        //                 // M step
        //                 weighted_dmu_array(nDim, nObs, nCl,
        //                                    dLlhd, ldLlhd, dX, ldx,
        //                                    dmu_array, ldmu, ps, ldp,
        //                                    cublasHandle);
        //
        //                 weighted_covs(x, dLlhd, dmu_array, dsigma_array, ps,
        //                               nDim, nObs, nCl, cublasHandle);
        //                 compute_ps(dLlhd, ps, nObs, nCl, cublasHandle);
        //
        //         }
        // }
        //
        // void predict(T* dX, T* doutRho, int nObs){
        //         likelihood_batched(nCl, nDim, nObs,
        //                            dX, ldx,
        //                            dmu, ldmu,
        //                            dsigma, ldsigma_full, ldsigma,
        //                            doutRho, ldLlhd, false);
        // }

};
