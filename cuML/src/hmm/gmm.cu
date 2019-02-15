#include <hmm/algorithms.h>
#include <ml_utils.h>


using namespace ML::HMM;

namespace ML {
namespace HMM {

template <typename T>
void allocate_gmm(GMM<T>& gmm){
        allocate(gmm.mus, gmm.nDim * gmm.nCl);
        CUDA_CHECK(cudaMemset(gmm.mus, (T) 0., gmm.nDim * gmm.nCl));

        allocate(gmm.sigmas, gmm.nDim * gmm.nDim * gmm.nCl);
        CUDA_CHECK(cudaMemset(gmm.sigmas, (T)0.,  gmm.nDim * gmm.nDim * gmm.nCl));

        allocate(gmm.rhos, gmm.nObs * gmm.nCl);
        CUDA_CHECK(cudaMemset(gmm.rhos, (T)0., gmm.nObs * gmm.nCl ));

        allocate(gmm.ps, gmm.nDim);
        CUDA_CHECK(cudaMemset(gmm.ps, (T)0., gmm.nDim));
}

template <typename T>
void free_gmm(GMM<T>& gmm){
        CUDA_CHECK(cudaFree(gmm.mus));
        CUDA_CHECK(cudaFree(gmm.sigmas));
        CUDA_CHECK(cudaFree(gmm.rhos));
        CUDA_CHECK(cudaFree(gmm.ps));
}

template <typename T>
void set_gmm(GMM<T>& gmm, int nCl, int nDim, int nObs,
             paramsRandom<T>* paramsRd, paramsEM* paramsEm) {
        gmm.nDim = nDim;
        gmm.nCl = nCl;
        gmm.nObs = nObs;

        // printf("iterations %d\n", gmm.paramsEm->n_iter);


        gmm.paramsRd = paramsRd;
        gmm.paramsEm = paramsEm;

        printf(" after alloc iterations %d\n", gmm.paramsEm->n_iter);
        allocate_gmm(gmm);
        printf(" after alloc iterations %d\n", gmm.paramsEm->n_iter);
}


template <typename T>
void initialize(GMM<T>& gmm) {
        MLCommon::HMM::gen_trans_matrix(gmm.rhos, gmm.nCl, gmm.nDim,
                                        gmm.paramsRd);
        MLCommon::HMM::gen_trans_matrix(gmm.ps, (int) 1, gmm.nCl, gmm.paramsRd);

        MLCommon::HMM::gen_array(gmm.mus, gmm.nDim * gmm.nCl, gmm.paramsRd);
        MLCommon::HMM::gen_array(gmm.sigmas, gmm.nDim * gmm.nDim * gmm.nCl,
                                 gmm.paramsRd);

        gmm.initialized = true;
}

template <typename T>
void fit(GMM<T>& gmm, T* data) {
        printf("iterations %d\n", gmm.paramsEm->n_iter);

        gmm.x = data;
        _em(gmm);
}

// template <typename T>
// void predict(GMM<T>& gmm, T* data, T* out_rhos, int nObs) {
//         _predict(gmm, data, out_rhos, nObs)
// }
//
}
}
