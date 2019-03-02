#pragma once

#include <hmm/structs.h>
#include <hmm/magma/b_likelihood.h>
#include <hmm/cuda/stats.h>

#include <stats/sum.h>
#include <ml_utils.h>


using namespace ML::HMM;

template <typename T>
struct GMM {
        T *dmu_array, *dsigma_array, *ps;
        T *dLlhd;

        magma_int_t lddx, ldddmu_array, lddsigma, lddsigma_full, lddps, ldddLlhd;

        int nCl, nDim, nObs;

        // EM parameters
        paramsEM *paramsEm;

        // // Random parameters
        paramsRandom<T> *paramsRd;

        // all the workspace related pointers
        int *info;
        bool initialized=false;

        cublasHandle_t *cublasHandle;

        void set_gmm(int _nCl, int _nDim, int _nObs,
                     paramsRandom<T>* _paramsRd, paramsEM* _paramsEm) {
                nDim = _nDim;
                nCl = _nCl;
                nObs = _nObs;

                lddx = magma_roundup(nDim, RUP_SIZE);
                lddmu = magma_roundup(nDim, RUP_SIZE);
                lddsigma = magma_roundup(nDim, RUP_SIZE);
                lddsigma_full = nDim * lddsigma;
                ldddLlhd = magma_roundup(nCl, RUP_SIZE);

                paramsRd = _paramsRd;
                paramsEm = _paramsEm;

                allocate(dX, lddx * nObs);
                allocate(dmu, lddmu * nCl);
                allocate(dsigma, lddsigma_full * nCl);
                allocate(dLlhd, lddLlhd * nObs);
        }

        // void initialize() {
        //         MLCommon::HMM::gen_trans_matrix(dLlhd, nCl, nDim,
        //                                         paramsRd);
        //         MLCommon::HMM::gen_trans_matrix(ps, (int) 1, nCl, paramsRd);
        //
        //         MLCommon::HMM::gen_array(dmu_array, nDim * nCl, paramsRd);
        //         MLCommon::HMM::gen_array(dsigma_array, nDim * nDim * nCl,
        //                                  paramsRd);
        //
        //         initialized = true;
        // }

        void free(){
                CUDA_CHECK(cudaFree(dmu_array));
                CUDA_CHECK(cudaFree(dsigma_array));
                CUDA_CHECK(cudaFree(dLlhd));
                CUDA_CHECK(cudaFree(ps));
        }

        void fit(T* dX) {
                _em(dX);
        }


        template <typename T>
        void compute_ps(T* ps, T* dLlhd, int m, int n){
                sum(ps, dLlhd, n, m, true);
        }

        // template <typename T>
        // void apply_ps( int m, int n, T* ps, T* dLlhd,){
        //         sum(ps, dLlhd, n, m, true);
        // }

        template <typename T>
        void _em(T* dX){
                assert(initialized);

                MLCommon::HMM::gen_trans_matrix(1, nCl, ps,  paramsRd, true);
                // paramsRd->start = 1;
                // paramsRd->end = 5;
                MLCommon::HMM::gen_matrix(nDim, nCl, dmu_array, lddmu, paramsRd)
                MLCommon::HMM::gen_matrix(lddsigma_full, nCl, dsigma_array, lddsigma_full, paramsRd)

                // Run the EM algorithm
                for (int it = 0; it < paramsEm->n_iter; it++) {
                        printf("\n -------------------------- \n");
                        printf(" iteration %d\n", it);

                        // E step
                        likelihood_batched(nCl, nDim, nObs,
                                           dX, lddx,
                                           dmu, lddmu,
                                           dsigma, lddsigma_full, lddsigma,
                                           dLlhd, ldddLlhd, false);
                        // apply_ps();
                        normalize_matrix(nCl, nObs, dLlhd, true);

                        // M step
                        weighted_dmu_array(nDim, nObs, nCl,
                                           dLlhd, ldddLlhd, dX, lddx,
                                           dmu_array, lddmu, ps, lddp,
                                           cublasHandle);

                        weighted_covs(x, dLlhd, dmu_array, dsigma_array, ps,
                                      nDim, nObs, nCl, cublasHandle);
                        compute_ps(dLlhd, ps, nObs, nCl, cublasHandle);

                }
        }

        template <typename T>
        void predict(T* dX, T* doutRho, int nObs){
                likelihood_batched(nCl, nDim, nObs,
                                   dX, lddx,
                                   dmu, lddmu,
                                   dsigma, lddsigma_full, lddsigma,
                                   doutRho, ldddLlhd, false);
        }

};

}
}
