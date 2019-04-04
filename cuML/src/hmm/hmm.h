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

#pragma once

#include <stdlib.h>

#include "gmm/gmm.h"
#include "hmm/algorithms/forward_backward.h"
#include "hmm/algorithms/viterbi.h"
#include "hmm/algorithms/em.h"

#include <magma/magma_utils.h>

namespace hmm {

template <typename T, typename D>
void init(HMM<T, D> &hmm,
          std::vector<D> &gmms,
          int nStates,
          T* dStartProb, int lddsp,
          T* dT, int lddt,
          T* dB, int lddb,
          T* dGamma, int lddgamma,
          T* logllhd
          ) {

        hmm.dT = dT;
        hmm.dB = dB;
        hmm.dStartProb = dStartProb;
        hmm.dGamma = dGamma;

        hmm.lddt = lddt;
        hmm.lddb = lddb;
        // TODO : align
        hmm.lddalpha = nStates;
        hmm.lddbeta = nStates;
        hmm.lddgamma = lddgamma;
        hmm.lddsp = lddsp;
        hmm.lddv = nStates;

        hmm.nStates = nStates;
        hmm.dists = gmms;

        hmm.logllhd = logllhd;

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
void setup(HMM<T, D> &hmm, int nObs, int nSeq, T* dLlhd){
        hmm.nObs = nObs;
        hmm.nSeq = nSeq;

        hmm.dLlhd = dLlhd;

        allocate(hmm.dAlpha, hmm.lddalpha * nObs);
        allocate(hmm.dBeta, hmm.lddbeta * nObs);
        allocate(hmm.dV, hmm.lddv * nObs);
        allocate(hmm.dcumlenghts_exc, nSeq);
        allocate(hmm.dcumlenghts_inc, nSeq);

        // allocate(hmm.logllhd, 1);

        // Create Pi array
        allocate(hmm.dPi_array, hmm.nStates);
        T **Pi_array;
        Pi_array = (T **)malloc(sizeof(T*) * hmm.nStates);
        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                Pi_array[stateId] = hmm.dists[stateId].dPis;
        }
        updateDevice(hmm.dPi_array, Pi_array, hmm.nStates);
        free(Pi_array);
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
                      Tx* dX, unsigned short int* dlenghts, int nSeq,
                      cublasHandle_t cublasHandle, magma_queue_t queue,
                      bool doForward, bool doBackward, bool doGamma){
        // print_matrix_device(1, nSeq, dlenghts, 1, "dlenghts");

        _compute_emissions(dX, hmm, cublasHandle, queue);
        _compute_cumlengths(hmm.dcumlenghts_inc, hmm.dcumlenghts_exc,
                            dlenghts, nSeq);
        _forward_backward(hmm, dlenghts, nSeq, doForward, doBackward);
        if (doGamma) {
                _update_gammas(hmm);
        }
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dAlpha, hmm.lddalpha, "dAlpha");
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dBeta, hmm.lddbeta, "dBeta");
        // // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dB, hmm.lddb, "dB");
        // print_matrix_device(1, nSeq, dlenghts, 1, "dlenghts");
        // print_matrix_device(1, nSeq, hmm.dcumlenghts_exc, 1, "dcumlenghts_exc");
        // print_matrix_device(1, nSeq, hmm.dcumlenghts_inc, 1, "dcumlenghts_inc");
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dGamma, hmm.lddgamma, "dGamma");
        // print_matrix_device(1, nSeq, hmm.dLlhd, 1, "dLlhd");
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

template <typename Tx, typename T, typename D>
void viterbi(HMM<T, D> &hmm, unsigned short int* dVStates,
             Tx* dX, unsigned short int* dlenghts, int nSeq,
             cublasHandle_t cublasHandle, magma_queue_t queue){

        _compute_emissions(dX, hmm, cublasHandle, queue);
        _compute_cumlengths(hmm.dcumlenghts_inc, hmm.dcumlenghts_exc,
                            dlenghts, nSeq);
        // print_matrix_device(1, nSeq, dlenghts, 1, "dlenghts");
        // print_matrix_device(1, nSeq, hmm.dcumlenghts_exc, 1, "dcumlenghts_exc");
        _viterbi(hmm, dVStates, dlenghts, nSeq);
        // print_matrix_device(1, nSeq, hmm.dcumlenghts_inc, 1, "dcumlenghts_inc");
}

template <typename Tx, typename T, typename D>
void m_step(HMM<T, D> &hmm,
            Tx* dX, unsigned short int* dlenghts, int nSeq,
            cublasHandle_t cublasHandle, magma_queue_t queue
            ){
        forward_backward(hmm, dX, dlenghts, nSeq,
                         cublasHandle, queue,
                         true, true, true);
        print_matrix_device(1, nSeq, hmm.dcumlenghts_exc, 1, "dcumlenghts_exc");
        print_matrix_device(hmm.nStates, hmm.nStates, hmm.dT, hmm.lddt, "dT");

        _m_step(hmm, dX, dlenghts, nSeq);
}

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
