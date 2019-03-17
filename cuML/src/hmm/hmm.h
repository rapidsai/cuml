#pragma once

#include <stdlib.h>

#include "hmm/gmm.h"
#include "hmm/hmm_backend.h"

namespace hmm {

template <typename T>
void init(HMM<T> &hmm,
          std::vector<gmm::GMM<T> > &gmms,
          int nStates,
          T* dT, int lddt,
          T* dB, int lddb,
          T* dGamma, int lddgamma
          ) {

        hmm.dT = dT;
        hmm.dB = dB;
        hmm.dGamma = dGamma;

        hmm.lddt = lddt;
        hmm.lddb = lddb;
        hmm.lddgamma = lddgamma;

        hmm.nStates = nStates;
        hmm.gmms = gmms;

        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                gmm.gmms[stateId] = hmm.dB + nObs * stateId;
        }
}

// template <typename T>
// void setup(){
//
// }

// template <typename T>
// void createPassWs(HMM<T> &hmm) {
//         allocate(hmm.dGamma, hmm.lddgamma * hmm.nStates);
// }
//
// template <typename T>
// void freePassWs(HMM<T> &hmm) {
//
// }

template <typename T>
void forward_backward(HMM<T> &hmm,
                      T* dX, int* dlenghts, int nSeq,
                      cublasHandle_t cublasHandle, magma_queue_t queue,
                      bool doForward, bool doBackward){

        _compute_emissions(dX, hmm, cublasHandle);
        _matrix_powers(hmm.nStates, hmm.dT_pows, gmm.max_len,
                       hmm.dT, hmm.lddt, queue);
        _forward_backward(hmm, dlenghts, nSeq, doForward, doBackward);
}

// template <typename T>
// void createViterbiWs(){
//         allocate(hmm.dV_ptr_array, hmm.lddv * hmm.max_len, hmm.nObs);
// }
//
// template <typename T>
// void freeViterbiWs(){
//
// }
//
// template <typename T>
// void viterbi(HMM<T>& hmm,
//              invector<T*> dV_idx_array, int lddv,
//              T* dV_array, int lddv,
//              T* dAlpha_array, int lddalpha,
//              T* dT, int lddt,
//              int* dMaxPath_array, int len_array){
//
//         // TODO : Fix the block grid sizes projectwise
//
//         dim3 block(32,32);
//         dim3 grid(ceildiv(n, (int)block.x),
//                   ceildiv(n, (int)block.y),
//                   1);
//         int nThreads_x = grid.x * block.x;
//         int nThreads_y = grid.y * block.y;
//         int nThreads_z = grid.z * block.z;
//
//         viterbiKernel<T> <<< grid, block >>>(dV_idx_array, lddv,
//                                              dV_array, lddv,
//                                              dAlpha_array, lddalpha,
//                                              dT, lddt,
//                                              dMaxPath_array, len_array,
//                                              nStates
//                                              nThreads_x, nThreads_y);
//         cudaDeviceSynchronize();
//         CUDA_CHECK(cudaPeekAtLastError());
// }
//
//
// template <typename T>
// void createEMWs(){
//         createViterbiWs();
//         createPassWs();
// }
//
// template <typename T>
// void freeViterbiWs(){
//         freeViterbiWs();
//         freePassWs();
// }
//
// template <typename T>
// void em(T* dX, int* len_array, HMM<T>& hmm,
//         cublasHandle_t cublasHandle, magma_queue_t queue){
//
//         // E step
//         forward();
//
//         if (hmm.train == Viterbi) {
//                 viterbi(dMaxPath_array, dX, len_array, hmm);
//                 _one_hot(hmm.dGamma, hmm.lddgamma, dMaxPath_array, nObs);
//         }
//         if (hmm.train == ForwardBackward) {
//                 _compute_gammas();
//         }
//
//         // M step
//         // TODO : Parallelize over several GPUs
//         for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
//                 // Train using gmm toolkit
//                 // _update_pis(hmm.gmms[i]);
//                 // _update_mus(dX, hmm.gmms[i], cublasHandle, queue);
//                 // _update_sigmas(dX, hmm.gmms[i], cublasHandle, queue);
//                 _m_step(dX, hmm.gmms[i], cublasHandle, queue);
//         }
// }
//
//
}
