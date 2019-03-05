#include "hmm/gmm.h"
#include "hmm/gmm_py.h"

template <typename T>
void init_f32(GMM<T> &gmm,
              T *dmu, T *dsigma, T *dPis, T *dPis_inv, T *dLlhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
              int nCl, int nDim, int nObs){
        init(gmm,
             dmu, dsigma, dPis, dPis_inv, dLlhd,
             lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
             nCl, nDim, nObs);
}

template <typename T>
void update_rhos_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        update_rhos(dX, gmm, cublasHandle);
}

template <typename T>
void update_mus_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        update_mus(dX, gmm, cublasHandle);
}

template <typename T>
void update_sigmas_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        update_sigmas(dX, gmm, cublasHandle);
}

template <typename T>
void update_pis_f32(GMM<float>& gmm){
        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        update_sigmas(gmm, cublasHandle);
}
