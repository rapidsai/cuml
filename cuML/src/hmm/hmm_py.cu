/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hmm/hmm.h"
#include "hmm/hmm_py.h"

namespace hmm {

void init_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                     std::vector<gmm::GMM<double> > &gmms,
                     int nStates,
                     double* dStartProb, int lddsp,
                     double* dT, int lddt,
                     double* dB, int lddb,
                     double* dGamma, int lddgamma,
                     double* logllhd){
        init(hmm, gmms, nStates, dStartProb, lddsp,
             dT, lddt, dB, lddb, dGamma, lddgamma,
             logllhd);
}

void setup_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm, int nObs, int nSeq, double* dLlhd) {
        setup(hmm, nObs, nSeq, dLlhd);
}

void forward_backward_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                                 double* dX, unsigned short int* dlenghts, int nSeq,
                                 bool doForward, bool doBackward, bool doGamma){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        // workspaceCreate(hmm);

        forward_backward(hmm, dX, dlenghts, nSeq,
                         cublasHandle, queue,
                         doForward, doBackward, doGamma);

        // workspaceFree(hmm);

}

void init_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                   std::vector<multinomial::Multinomial<double> > &gmms,
                   int nStates,
                   double* dStartProb, int lddsp,
                   double* dT, int lddt,
                   double* dB, int lddb,
                   double* dGamma, int lddgamma,
                   double* logllhd){
        init(hmm, gmms, nStates, dStartProb, lddsp,
             dT, lddt, dB, lddb, dGamma, lddgamma, logllhd);
}

void setup_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                    int nObs, int nSeq, double* dLlhd) {
        setup(hmm, nObs, nSeq, dLlhd);
}

void forward_backward_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                               unsigned short int* dX, unsigned short int* dlenghts, int nSeq,
                               bool doForward, bool doBackward, bool doGamma){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);
        // workspaceCreate(hmm);

        forward_backward(hmm, dX, dlenghts, nSeq,
                         cublasHandle, queue,
                         doForward, doBackward, doGamma);
}

void viterbi_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                      unsigned short int* dVStates, unsigned short int* dX, unsigned short int* dlenghts, int nSeq){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));
        viterbi(hmm, dVStates,
                dX, dlenghts, nSeq,
                cublasHandle);
}

void m_step_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                     unsigned short int* dX, unsigned short int* dlenghts, int nSeq){

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        m_step(hmm,
               dX, dlenghts, nSeq,
               cublasHandle, queue);
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
