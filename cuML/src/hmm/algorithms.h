#include <structs.h>
#include <hmm/likelihood.h>


using namespace ML::HMM;

namespace ML {
namespace GMM {

template <typename T>
void _em(GMM<T>& gmm, T* data, int n_samples){
        assert(gmm.initialized);

        _gmm_likelihood_functor<T> gmm_likelihood(data, gmm->mus,
                                                  gmm->sigmas, gmm->rhos,
                                                  gmm.nCl, gmm.nDim);

        thrust::device_vector<int>  samples_v(n_samples);
        thrust::device_vector<int> classes_v(gmm.nCl);
        first = thrust::make_zip_iterator(thrust::make_tuple(samples_v.begin(),
                                                             classes_v.begin()));
        last  = thrust::make_zip_iterator(thrust::make_tuple(samples_v.end(),
                                                             classes_v.end()));

        // Run the EM algorithm
        for (size_t it = 0; it < gmm.paramsEm.n_iter; it++) {
                // E step
                thrust::for_each(thrust::device, first, last, gmm_likelihood);
                MLCommon::HMM::normalize_matrix(gmm->rhos, gmm.nDim, gmm.nCl);

                // M step
                MLCommon::HMM::weighted_mean(data, gmm->rhos, gmm->mus,
                                             gmm.nDim, gmm.nCl);
                MLCommon::HMM::weighted_cov(data, gmm->rhos, gmm->sigmas,
                                            gmm.nDim, gmm.nCl);
        }
}


template <typename T>
void _predict(GMM<T>& gmm, T* data, T* out_rhos, int n_samples){
        _gmm_likelihood_functor<T> gmm_likelihood(data, gmm->mus,
                                                  gmm->sigmas, out_rhos,
                                                  gmm.nCl, gmm.nDim);

        thrust::device_vector<int>  samples_v(n_samples);
        thrust::device_vector<int> classes_v(gmm.nCl);
        first = thrust::make_zip_iterator(thrust::make_tuple(samples_v.begin(),
                                                             classes_v.begin()));
        last  = thrust::make_zip_iterator(thrust::make_tuple(samples_v.end(),
                                                             classes_v.end()));

        thrust::for_each(thrust::device, first, last, gmm_likelihood);
        MLCommon::HMM::normalize_matrix(out_rhos, gmm.nDim, gmm.nCl);
}

}
}
