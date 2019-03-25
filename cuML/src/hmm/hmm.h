#pragma once

#include <stdlib.h>

#include "gmm/gmm.h"
#include "hmm/algorithms/forward_backward.h"
#include "hmm/algorithms/viterbi.h"

#include <hmm/magma/magma_test_utils.h>

namespace hmm {

template <typename T, typename D>
void init(HMM<T, D> &hmm,
          std::vector<D> &gmms,
          int nStates,
          T* dStartProb, int lddsp,
          T* dT, int lddt,
          T* dB, int lddb
          ) {

        hmm.dT = dT;
        hmm.dB = dB;
        hmm.dStartProb = dStartProb;

        hmm.lddt = lddt;
        hmm.lddb = lddb;
        // TODO : align
        hmm.lddalpha = nStates;
        hmm.lddbeta = nStates;
        hmm.lddgamma = nStates;
        hmm.lddsp = lddsp;

        hmm.nStates = nStates;
        hmm.dists = gmms;

        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                hmm.dists[stateId].dLlhd = hmm.dB + hmm.nObs * stateId;
        }

        if(std::is_same<D, gmm::GMM<T> >::value) {
                hmm.distOption = GaussianMixture;
        }
        if(std::is_same<D, multinomial::Multinomial<T> >::value) {
                hmm.distOption = MultinomialDist;
        }
}




template <typename T, typename D>
void setup(HMM<T, D> &hmm, int nObs, int nSeq){
        hmm.nObs = nObs;
        hmm.nSeq = nSeq;

        allocate(hmm.dAlpha, hmm.lddalpha * nObs);
        allocate(hmm.dBeta, hmm.lddbeta * nObs);
        allocate(hmm.dGamma, hmm.lddgamma * nObs);
        allocate(hmm.dcumlenghts_exc, nSeq);
        allocate(hmm.dcumlenghts_inc, nSeq);

        // Create Pi array
        allocate(hmm.dPi_array, hmm.nStates);
        T **Pi_array;
        Pi_array = (T **)malloc(sizeof(T*) * nSeq);
        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                Pi_array[stateId] = hmm.dists[stateId].dPis;
        }
        updateDevice(hmm.dPi_array, Pi_array, hmm.nStates);
        free(Pi_array);


        //
        // allocate(hmm.dAlpha_array, nSeq);
        // allocate(hmm.dBeta_array, nSeq);
        // for (size_t seqId = 0; seqId < nSeq; seqId++) {
        //         hmm.dAlpha_array[seqId] = hmm.dAlpha + seqId * hmm.lddalpha;
        //         hmm.dBeta_array[seqId] = hmm.dBeta + seqId * hmm.lddalpha;
        // }
}

// template <typename T>
// void createPassWs(HMM<T> &hmm) {
//         allocate(hmm.dGamma, hmm.lddgamma * hmm.nStates);
// }
//
// template <typename T>
// void freePassWs(HMM<T> &hmm) {
//
// }

template <typename Tx, typename T, typename D>
void forward_backward(HMM<T, D> &hmm,
                      Tx* dX, int* dlenghts, int nSeq,
                      cublasHandle_t cublasHandle, magma_queue_t queue,
                      bool doForward, bool doBackward){

        _compute_emissions(dX, hmm, cublasHandle);
        _compute_cumlengths(hmm.dcumlenghts_inc, hmm.dcumlenghts_exc,
                            dlenghts, nSeq);
        _forward_backward(hmm, dlenghts, nSeq, doForward, doBackward);
        // if (doGamma) {
        _update_gammas(hmm);
        // }
        // print_matrix_device(1, nSeq, dlenghts, 1, "dlenghts");
        // print_matrix_device(1, nSeq, hmm.dcumlenghts_exc, 1, "dcumlenghts_exc");
        // print_matrix_device(1, nSeq, hmm.dcumlenghts_inc, 1, "dcumlenghts_inc");
        print_matrix_device(hmm.nStates, hmm.nObs, hmm.dGamma, hmm.lddgamma, "dGamma");
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

// template <typename T>
// void viterbi(HMM<T>& hmm,
//              int* dStates, int* dlenghts, int nSeq){
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
//         viterbiKernel<T> <<< grid, block >>>();
//         cudaDeviceSynchronize();
//         CUDA_CHECK(cudaPeekAtLastError());
// }

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
