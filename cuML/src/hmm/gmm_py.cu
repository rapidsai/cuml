#include "hmm/gmm.h"
#include "hmm/gmm_py.h"

void init_test(){
        // nCl = 1;
}


void init_f32(GMM<float> &gmm,
              float *dmu, float *dsigma, float *dPis, float *dPis_inv, float *dLlhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
              int nCl, int nDim, int nObs){
        init(gmm,
             dmu, dsigma, dPis, dPis_inv, dLlhd,
             lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
             nCl, nDim, nObs);
}

void update_rhos_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        update_rhos(dX, gmm, cublasHandle, queue);
}

void update_mus_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        update_mus(dX, gmm, cublasHandle, queue);
}

void update_sigmas_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        update_sigmas(dX, gmm, cublasHandle, queue);
}

void update_pis_f32(GMM<float>& gmm){
        update_pis(gmm);
}
