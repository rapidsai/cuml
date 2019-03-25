#include "hmm/hmm.h"
#include "hmm/hmm_py.h"

namespace hmm {

void init_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                     std::vector<gmm::GMM<double> > &gmms,
                     int nStates,
                     double* dStartProb, int lddsp,
                     double* dT, int lddt,
                     double* dB, int lddb,
                     double* dGamma, int lddgamma){
        init(hmm, gmms, nStates, dStartProb, lddsp,
             dT, lddt, dB, lddb, dGamma, lddgamma);
}

void setup_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm, int nObs, int nSeq) {
        setup(hmm, nObs, nSeq);
}

void forward_backward_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                                 double* dX, int* dlenghts, int nSeq,
                                 bool doForward, bool doBackward){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        // workspaceCreate(hmm);

        forward_backward(hmm, dX, dlenghts, nSeq,
                         cublasHandle, queue,
                         doForward, doBackward);

        // workspaceFree(hmm);

}

void init_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                   std::vector<multinomial::Multinomial<double> > &gmms,
                   int nStates,
                   double* dStartProb, int lddsp,
                   double* dT, int lddt,
                   double* dB, int lddb,
                   double* dGamma, int lddgamma){
        init(hmm, gmms, nStates, dStartProb, lddsp,
             dT, lddt, dB, lddb, dGamma, lddgamma);
}

void setup_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                    int nObs, int nSeq) {
        setup(hmm, nObs, nSeq);
}

void forward_backward_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                               int* dX, int* dlenghts, int nSeq,
                               bool doForward, bool doBackward){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        // workspaceCreate(hmm);

        forward_backward(hmm, dX, dlenghts, nSeq,
                         cublasHandle, queue,
                         doForward, doBackward);

        // workspaceFree(hmm);

}

}

namespace multinomial {
void init_multinomial_f64(multinomial::Multinomial<double> &multinomial,
                          double* dPis, int nCl){
        init_multinomial(multinomial, dPis, nCl);
}

}

// void viterbi_f64(HMM<double>& hmm,
//                  int* dStates, int* dlenghts, int nSeq){
//
//         cublasHandle_t cublasHandle;
//         CUBLAS_CHECK(cublasCreate(&cublasHandle));
//
//         int device = 0;
//         magma_queue_t queue;
//         magma_queue_create(device, &queue);
//         // workspaceCreate(hmm);
//
//         viterbi(hmm, dStates, dlenghts, nSeq);
//
//         // workspaceFree(hmm);
// }

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
