#include "hmm/hmm.h"
#include "hmm/hmm_py.h"

namespace hmm {

void init_f64(HMM<double> &hmm,
              std::vector<gmm::GMM<double> > &gmms,
              int nStates,
              double* dT, int lddt,
              double* dB, int lddb,
              double* dGamma, int lddgamma){
        init(hmm, gmms, nStates,
             dT, lddt, dB, lddb, dGamma, lddgamma);
}

void setup_f64(HMM<double> &hmm) {
        setup(hmm);
}

void forward_backward_f64(HMM<double> &hmm,
                          double* dX, int* dlenghts, int nSeq,
                          bool doForward, bool doBackward){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        // workspaceCreate(hmm);

        _forward_backward(hmm, dX, dlenghts, nSeq, doForward, doBackward);

        // workspaceFree(hmm);

}

void viterbi_f64(HMM<double>& hmm,
                 int* dStates, int* dlenghts, int nSeq){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        // workspaceCreate(hmm);

        viterbi(hmm, dStates, dlenghts, nSeq);

        // workspaceFree(hmm);
}

// void em_f32(HMM<float>& hmm,
//             float* dX,
//             int* len_array,
//             int nObs){
//
//         cublasHandle_t cublasHandle;
//         CUBLAS_CHECK(cublasCreate(&cublasHandle));
//
//         int device = 0;
//         magma_queue_t queue;
//         magma_queue_create(device, &queue);
//         workspaceCreate(hmm);
//
//         em(dX, len_array, hmm, nObs, cublasHandle, queue);
//
//         workspaceFree(hmm);
// }
//
// void free_f32(HMM<float> &hmm) {
//         free_hmm(hmm);
// }
}
