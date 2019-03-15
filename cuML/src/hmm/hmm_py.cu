#include "hmm/hmm.h"
#include "hmm/hmm_py.h"

namespace hmm {

void init_f64(HMM<double> &hmm,
              std::vector<double*> dmu, std::vector<double*> dsigma, std::vector<double*> dPis, std::vector<double*> dPis_inv, double* dLlhd, double* cur_llhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
              int nCl, int nDim, int nObs,
              double reg_covar,
              int nStates,
              double* dT,
              int lddt){
        init(hmm,
             dmu, dsigma, dPis, dPis_inv, dLlhd, cur_llhd,
             lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
             nCl, nDim, nObs,
             reg_covar,
             nStates,
             dT,
             lddt);
}

// void setup_f32(HMM<float> &hmm) {
//         setup_hmm(hmm);
// }
//
// void forward_f32(HMM<float>& hmm,
//                  float* dX,
//                  int* len_array,
//                  int nObs){
//
//         cublasHandle_t cublasHandle;
//         CUBLAS_CHECK(cublasCreate(&cublasHandle));
//
//         int device = 0;
//         magma_queue_t queue;
//         magma_queue_create(device, &queue);
//         workspaceCreate(hmm);
//
//         forward(dX, len_array, hmm, nObs, cublasHandle, queue);
//
//         workspaceFree(hmm);
//
// }
//
// void backward_f32(HMM<float>& hmm,
//                   float* dX,
//                   int* len_array,
//                   int nObs){
//
//         cublasHandle_t cublasHandle;
//         CUBLAS_CHECK(cublasCreate(&cublasHandle));
//
//         int device = 0;
//         magma_queue_t queue;
//         magma_queue_create(device, &queue);
//         workspaceCreate(hmm);
//
//         backward(dX, len_array, hmm, nObs, cublasHandle, queue);
//
//         workspaceFree(hmm);
// }
//
// void viterbi_f32(HMM<float>& hmm,
//                  float* dX,
//                  int* len_array,
//                  int nObs){
//
//         cublasHandle_t cublasHandle;
//         CUBLAS_CHECK(cublasCreate(&cublasHandle));
//
//         int device = 0;
//         magma_queue_t queue;
//         magma_queue_create(device, &queue);
//         workspaceCreate(hmm);
//
//         viterbi(dX, len_array, hmm, nObs, cublasHandle, queue);
//
//         workspaceFree(hmm);
// }
//
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
