#include "hmm/hmm.h"
#include "hmm/hmm_py.h"

void init_f32(){
        init_hmm();
}

void setup_f32(HMM<float> &hmm) {
        setup_hmm(hmm);
}

void forward_f32(HMM<float>& hmm,
                 float* dX,
                 int* len_array,
                 int nObs){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        workspaceCreate(hmm);

        forward(dX, len_array, hmm, nObs, cublasHandle, queue);

        workspaceFree(hmm);

}

void backward_f32(HMM<float>& hmm,
                  float* dX,
                  int* len_array,
                  int nObs){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        workspaceCreate(hmm);

        backward(dX, len_array, hmm, nObs, cublasHandle, queue);

        workspaceFree(hmm);
}

void viterbi_f32(HMM<float>& hmm,
                 float* dX,
                 int* len_array,
                 int nObs){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        workspaceCreate(hmm);

        viterbi(dX, len_array, hmm, nObs, cublasHandle, queue);

        workspaceFree(hmm);
}

void em_f32(HMM<float>& hmm,
            float* dX,
            int* len_array,
            int nObs){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        workspaceCreate(hmm);

        em(dX, len_array, hmm, nObs, cublasHandle, queue);

        workspaceFree(hmm);
}

void free_f32(HMM<float> &hmm) {
        free_hmm(hmm);
}
