#pragma once

#include <hmm/random.h>
#include <hmm/structs.h>
#include <hmm/algorithms.h>
#include <ml_utils.h>


using namespace ML::HMM;

namespace ML {
namespace GMM {

template <typename T>
struct GMM {
        // these device pointers are NOT owned by this class
        T *z, *x;
        T *mus, *sigmas, *rhos;

        int n_classes;
        int dim_x;

        // EM parameters
        paramsEM paramsEm;

        // Random parameters
        HMM::Random::paramsRandom paramsRd;

        // all the workspace related pointers
        int *info;
        bool initialized=false;
        cublasHandle_t handle;
};

template <typename T>
void set(GMM<T> gmm, int n_classes, int dim_x, paramsRandom paramsRd,
         paramsEM paramsEm) {
        gmm.dim_x = dim_x;
        gmm.n_classes = n_classes;
        gmm.paramsRd = paramsRd;
        gmm.paramsEm = paramsEm;
}

template <typename T>
void initialize(GMM<T>& gmm) {
        MLCommon::HMM::gen_trans_matrix(gmm.rhos, gmm.n_classes, gmm.dim_x,
                                        gmm.paramsRd);
        MLCommon::HMM::gen_array(gmm.mus, gmm.dim_x * gmm.n_classes, gmm.paramsRd);
        MLCommon::HMM::gen_array(gmm.sigmas, gmm.dim_x * gmm.dim_x * gmm.n_classes,
                                 gmm.paramsRd);

        gmm.initialized = true;
}

template <typename T>
void fit(GMM<T>& gmm) {
        _em(gmm);
}

template <typename T>
void predict(GMM<T>& gmm, T* data, T* out_rhos, int n_samples) {
        _predict(gmm, data, out_rhos, n_samples)
}

}
}
