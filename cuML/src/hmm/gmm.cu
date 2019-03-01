#include <hmm/algorithms.h>
#include <ml_utils.h>


using namespace ML::HMM;

namespace ML {
namespace HMM {

template <typename T>
void allocate_gmm(GMM<T>& gmm){
        allocate(gmm.mus, gmm.nDim * gmm.nCl);
        allocate(gmm.sigmas, gmm.nDim * gmm.nDim * gmm.nCl);
        allocate(gmm.rhos, gmm.nObs * gmm.nCl);
        allocate(gmm.ps, gmm.nCl);
        allocate(gmm.info, 1);
}

template <typename T>
void TearDownGMM(GMM<T>& gmm){
        CUDA_CHECK(cudaFree(gmm.mus));
        CUDA_CHECK(cudaFree(gmm.sigmas));
        CUDA_CHECK(cudaFree(gmm.rhos));
        CUDA_CHECK(cudaFree(gmm.ps));
        CUDA_CHECK(cudaFree(gmm.info));
}

template <typename T>
void set_gmm(GMM<T>& gmm, int nCl, int nDim, int nObs,
             paramsRandom<T>* paramsRd, paramsEM* paramsEm,
             cusolverDnHandle_t *_cusolverHandle, cublasHandle_t *_cublasHandle) {

        gmm.nDim = nDim;
        gmm.nCl = nCl;
        gmm.nObs = nObs;

        gmm.paramsRd = paramsRd;
        gmm.paramsEm = paramsEm;

        gmm.cusolverHandle = _cusolverHandle;
        gmm.cublasHandle = _cublasHandle;

        allocate_gmm(gmm);
}



template <typename T>
void initialize(GMM<T>& gmm) {
        // MLCommon::HMM::gen_trans_matrix(gmm.rhos, gmm.nObs, gmm.nCl,
        //                                 gmm.paramsRd, true);
        MLCommon::HMM::gen_trans_matrix(gmm.ps, 1, gmm.nCl,
                                        gmm.paramsRd, true);
        // compute_ps(gmm.rhos, gmm.ps, gmm.nObs, gmm.nCl, gmm.cublasHandle);


        MLCommon::HMM::gen_array(gmm.mus, gmm.nDim * gmm.nCl, gmm.paramsRd);
        MLCommon::HMM::gen_array(gmm.sigmas, gmm.nDim * gmm.nDim * gmm.nCl,
                                 gmm.paramsRd);

        gmm.initialized = true;
}

template <typename T>
void fit(GMM<T>& gmm, T* data) {
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
