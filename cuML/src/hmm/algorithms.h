#pragma once

#include <hmm/structs.h>
#include <hmm/magma/likelihood.h>
#include <hmm/magma/stats.h>

#include <hmm/utils.h>
#include <stats/sum.h>

namespace ML {
namespace HMM {

template <typename T>
void compute_ps(T* ps, T* rhos, int m, int n){
        sum(ps, rhos, n, m, true);
}

template <typename T>
void weighted_mus(int nDim, int nObs, int nCl,
                  T* dW, int lddw, T* dX, int lddx,
                  T* mus, int lddmu, T* ps, int lddp,
                  cublasHandle_t handle){
        T alfa = (T)1.0 / nObs, beta = (T)0.0;
        CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nDim,
                                nCl, nObs, &alfa, dX, lddx, dW, lddw, &beta, mus, lddmu));

// TODO : Check the 1 in the next call !!!!!!!!!!!!!!!!

        CUBLAS_CHECK(cublasdgmm(handle, CUBLAS_SIDE_RIGHT, nDim, nCl, mus, lddmu, ps, 1, mus, lddmu));
}

template <typename T>
void _em(GMM<T>& gmm){
        assert(gmm.initialized);
        bool isLog = false;

        GMMLikelihood<T> GMMLhd(gmm.x, gmm.mus, gmm.sigmas, gmm.rhos,
                                gmm.nCl, gmm.nDim, gmm.nObs, isLog,
                                gmm.cublasHandle, gmm.cusolverHandle);

        MLCommon::HMM::gen_trans_matrix(gmm.ps, 1, gmm.nCl,
                                        gmm.paramsRd, true);

        MLCommon::HMM::gen_array(gmm.mus, gmm.nDim * gmm.nCl, gmm.paramsRd);

        gmm.paramsRd->start = 1;
        gmm.paramsRd->end = 5;
        MLCommon::HMM::gen_array(gmm.sigmas, gmm.nDim * gmm.nDim * gmm.nCl,
                                 gmm.paramsRd);

        // Run the EM algorithm
        for (int it = 0; it < gmm.paramsEm->n_iter; it++) {
                printf("\n -------------------------- \n");
                printf(" iteration %d\n", it);

                // printf("line number %d in file %s\n", __LINE__, __FILE__);

                // E step
                GMMLhd.fill_rhos(gmm.rhos, gmm.ps);

                // printf("line number %d in file %s\n", __LINE__, __FILE__);

                normalize_matrix(gmm.rhos, gmm.nObs, gmm.nCl, true);

                // printf("line number %d in file %s\n", __LINE__, __FILE__);

                print_matrix(gmm.x, gmm.nDim, gmm.nObs, "gmm.data");
                print_matrix(gmm.mus, gmm.nDim, gmm.nCl, "gmm.mus");
                print_matrix(gmm.sigmas, gmm.nDim * gmm.nDim, gmm.nCl, "gmm.sigmas");
                print_matrix(gmm.rhos, gmm.nObs, gmm.nCl, "rhos alg");
                print_matrix(gmm.ps, 1, gmm.nCl, "gmm.ps");


                // M step

                weighted_means(gmm.rhos, gmm.x, gmm.mus, gmm.ps,
                               gmm.nDim, gmm.nObs, gmm.nCl, *gmm.cublasHandle);
                // printf("line number %d in file %s\n", __LINE__, __FILE__);

                weighted_covs(gmm.x, gmm.rhos, gmm.mus, gmm.sigmas, gmm.ps,
                              gmm.nDim, gmm.nObs, gmm.nCl, gmm.cublasHandle);
                compute_ps(gmm.rhos, gmm.ps, gmm.nObs, gmm.nCl, gmm.cublasHandle);
                // printf("line number %d in file %s\n", __LINE__, __FILE__);

        }

        GMMLhd.TearDown();
}

//
// template <typename T>
// void _predict(GMM<T>& gmm, T* data, T* out_rhos, int nObs){
//         _gmm_likelihood_functor<T> gmm_likelihood(data, gmm.mus,
//                                                   gmm.sigmas, out_rhos,
//                                                   gmm.nCl, gmm.nDim);
//
//         thrust::device_vector<int>  samples_v(nObs);
//         thrust::device_vector<int> classes_v(gmm.nCl);
//         first = thrust::make_zip_iterator(thrust::make_tuple(samples_v.begin(),
//                                                              classes_v.begin()));
//         last  = thrust::make_zip_iterator(thrust::make_tuple(samples_v.end(),
//                                                              classes_v.end()));
//
//         thrust::for_each(thrust::device, first, last, gmm_likelihood);
//         MLCommon::HMM::normalize_matrix(out_rhos, gmm.nDim, gmm.nCl);
// }

}
}
