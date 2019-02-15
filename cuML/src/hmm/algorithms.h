#pragma once

#include <hmm/structs.h>
#include <hmm/likelihood.h>
#include <hmm/stats.h>

#include <hmm/utils.h>

namespace ML {
namespace HMM {

template <typename T>
void _em(GMM<T>& gmm){
        assert(gmm.initialized);
        bool isLog = true;

        GMMLikelihood<T> GMMLhd(gmm.x, gmm.mus, gmm.sigmas, gmm.rhos,
                                gmm.nCl, gmm.nDim, gmm.nObs, isLog, gmm.cublasHandle);

        printf("iterations %d\n", gmm.paramsEm->n_iter);
        // Run the EM algorithm
        for (int it = 0; it < gmm.paramsEm->n_iter; it++) {
                printf("\n -------------------------- \n");
                printf(" iteration %d\n", gmm.paramsEm->n_iter);
                print_matrix(gmm.mus, gmm.nDim,gmm.nCl, "gmm.mus");
                print_matrix(gmm.sigmas, gmm.nDim * gmm.nDim, gmm.nCl, "gmm.sigmas");
                print_matrix(gmm.rhos, gmm.nCl, gmm.nObs, "gmm.rhos");

                // E step
                GMMLhd.fill_rhos(gmm.rhos);
                normalize_matrix(gmm.rhos, gmm.nDim, gmm.nCl);

                // M step
                weighted_means(gmm.rhos, gmm.x, gmm.mus,
                               gmm.nDim, gmm.nObs, gmm.nCl, *gmm.cublasHandle);
                weighted_covs(gmm.x, gmm.rhos, gmm.mus, gmm.sigmas,
                              gmm.nDim, gmm.nObs, gmm.nCl, gmm.cublasHandle);
        }
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
