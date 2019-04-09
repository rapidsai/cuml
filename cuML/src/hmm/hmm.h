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
// #include <hmm/utils/init_utils.h>

#include <magma/magma_utils.h>

namespace hmm {

template <typename T, typename D>
void init(HMM<T, D> &hmm,
          std::vector<D> &dists,
          int nStates,
          T* dStartProb, int lddsp,
          T* dT, int lddt,
          T* dB, int lddb,
          T* dGamma, int lddgamma,
          T* logllhd,
          int nObs, int nSeq,
          T* dLlhd
          ) {

        hmm.nObs = nObs;
        hmm.nSeq = nSeq;
        hmm.dLlhd = dLlhd;

        hmm.dT = dT;
        hmm.dB = dB;
        hmm.dStartProb = dStartProb;
        hmm.dGamma = dGamma;

        hmm.lddt = lddt;
        hmm.lddb = lddb;

        hmm.handle.lddalpha = nStates;
        hmm.handle.lddbeta = nStates;
        hmm.lddgamma = lddgamma;
        hmm.lddsp = lddsp;
        hmm.handle.lddv = nStates;

        hmm.nStates = nStates;
        hmm.dists = dists;

        hmm.logllhd = logllhd;

        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                hmm.dists[stateId].dLlhd = hmm.dB + hmm.nObs * stateId;
        }
}

template <typename T, typename D>
size_t hmm_bufferSize(HMM<T, D> &hmm){

        size_t workspaceSize = 0;
        const size_t granularity = 256;

        hmm.handle.dPi_array = (T **)workspaceSize;
        workspaceSize += alignTo(hmm.nStates * sizeof(T*), granularity);

        hmm.handle.dcumlenghts_inc = (unsigned short int *)workspaceSize;
        workspaceSize += alignTo(hmm.nSeq * sizeof(unsigned short int), granularity);

        hmm.handle.dcumlenghts_exc = (unsigned short int *)workspaceSize;
        workspaceSize += alignTo(hmm.nSeq * sizeof(unsigned short int), granularity);

        hmm.handle.dAlpha = (T *)workspaceSize;
        workspaceSize += alignTo(hmm.handle.lddalpha * hmm.nObs * sizeof(T), granularity);

        hmm.handle.dBeta = (T *)workspaceSize;
        workspaceSize += alignTo(hmm.handle.lddbeta * hmm.nObs * sizeof(T), granularity);

        hmm.handle.dV = (T *)workspaceSize;
        workspaceSize += alignTo(hmm.handle.lddv * hmm.nObs * sizeof(T), granularity);

        return workspaceSize;
}

template <typename T, typename D>
void create_HMMHandle(HMM<T, D> &hmm, void* workspace){
        hmm.handle.dPi_array = (T **)((size_t)hmm.handle.dPi_array + (size_t)workspace);
        hmm.handle.dcumlenghts_inc = (unsigned short int *)((size_t)hmm.handle.dcumlenghts_inc + (size_t)workspace);
        hmm.handle.dcumlenghts_exc = (unsigned short int *)((size_t)hmm.handle.dcumlenghts_exc + (size_t)workspace);
        hmm.handle.dAlpha = (T *)((size_t)hmm.handle.dAlpha + (size_t)workspace);
        hmm.handle.dBeta = (T *)((size_t)hmm.handle.dBeta + (size_t)workspace);
        hmm.handle.dV = (T *)((size_t)hmm.handle.dV + (size_t)workspace);

        // print_matrix_device(1, hmm.nSeq, hmm.handle.dcumlenghts_inc, 1, "hmm.handle.dcumlenghts_inc");
        // print_matrix_device(1, hmm.nSeq, hmm.handle.dcumlenghts_exc, 1, "hmm.handle.dcumlenghts_inc");
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.handle.dAlpha, hmm.handle.lddalpha, "hmm.handle.dAlpha");
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.handle.dBeta, hmm.handle.lddalpha, "hmm.handle.dBeta");
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.handle.dV, hmm.handle.lddv, "hmm.handle.dV");

        T **Pi_array;
        Pi_array = (T **)malloc(sizeof(T*) * hmm.nStates);
        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                Pi_array[stateId] = hmm.dists[stateId].dPis;
        }
        updateDevice(hmm.handle.dPi_array, Pi_array, hmm.nStates);
        free(Pi_array);
}

template <typename T, typename D>
void setup(HMM<T, D> &hmm, int nObs, int nSeq, T* dLlhd){
        // hmm.nObs = nObs;
        // hmm.nSeq = nSeq;
        //
        // hmm.dLlhd = dLlhd;

        // allocate(hmm.handle.dAlpha, hmm.handle.lddalpha * nObs);
        // allocate(hmm.handle.dBeta, hmm.handle.lddbeta * nObs);
        // allocate(hmm.handle.dV, hmm.handle.lddv * nObs);
        // allocate(hmm.handle.dcumlenghts_exc, nSeq);
        // allocate(hmm.handle.dcumlenghts_inc, nSeq);
        //
        // // Create Pi array
        // allocate(hmm.handle.dPi_array, hmm.nStates);
        // T **Pi_array;
        // Pi_array = (T **)malloc(sizeof(T*) * hmm.nStates);
        // for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
        //         Pi_array[stateId] = hmm.dists[stateId].dPis;
        // }
        // updateDevice(hmm.handle.dPi_array, Pi_array, hmm.nStates);
        // free(Pi_array);
}

template <typename Tx, typename T, typename D>
void forward_backward(HMM<T, D> &hmm,
                      Tx* dX, unsigned short int* dlenghts, int nSeq,
                      cublasHandle_t cublasHandle, magma_queue_t queue,
                      bool doForward, bool doBackward, bool doGamma){
        _compute_emissions(dX, hmm, cublasHandle, queue);
        _compute_cumlengths(hmm.handle.dcumlenghts_inc, hmm.handle.dcumlenghts_exc,
                            dlenghts, nSeq);
        _forward_backward(hmm, dlenghts, nSeq, doForward, doBackward);
        if (doGamma) {
                _update_gammas(hmm);
        }
}

template <typename Tx, typename T, typename D>
void viterbi(HMM<T, D> &hmm, unsigned short int* dVStates,
             Tx* dX, unsigned short int* dlenghts, int nSeq,
             cublasHandle_t cublasHandle, magma_queue_t queue){

        _compute_emissions(dX, hmm, cublasHandle, queue);
        _compute_cumlengths(hmm.handle.dcumlenghts_inc, hmm.handle.dcumlenghts_exc,
                            dlenghts, nSeq);
        _viterbi(hmm, dVStates, dlenghts, nSeq);
}

template <typename Tx, typename T, typename D>
void m_step(HMM<T, D> &hmm,
            Tx* dX, unsigned short int* dlenghts, int nSeq,
            cublasHandle_t cublasHandle, magma_queue_t queue
            ){
        forward_backward(hmm, dX, dlenghts, nSeq,
                         cublasHandle, queue,
                         true, true, true);
        _m_step(hmm, dX, dlenghts, nSeq);
}
}
