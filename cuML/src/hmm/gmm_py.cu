#include "hmm/gmm.h"
#include "hmm/gmm_py.h"

namespace gmm {

void init_f32(GMM<float> &gmm,
              float *dmu, float *dsigma, float *dPis, float *dPis_inv, float *dLlhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
              float *cur_llhd, float reg_covar,
              int nCl, int nDim, int nObs){
        init(gmm,
             dmu, dsigma, dPis, dPis_inv, dLlhd,
             lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
             cur_llhd, reg_covar,
             nCl, nDim, nObs);
}


void update_llhd_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        update_llhd(dX, gmm, cublasHandle);

        CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void update_rhos_f32(GMM<float>& gmm, float* dX){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        update_rhos(dX, gmm, cublasHandle, queue);

        CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void update_mus_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        update_mus(dX, gmm, cublasHandle, queue);

        CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void update_sigmas_f32(float* dX, GMM<float>& gmm){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        update_sigmas(dX, gmm, cublasHandle, queue);

        CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void update_pis_f32(GMM<float>& gmm){
        update_pis(gmm);
}

void setup_f32(GMM<float> &gmm) {
        setup(gmm);
}

void free_f32(GMM<float> &gmm) {
        free(gmm);
}

}
