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

# pragma once

#include "magma/magma_batched_wrappers.h"
#include "stats/sum.h"

using namespace MLCommon::LinAlg;

namespace hmm {

template <typename T>
__device__
int _backward_argmax(T* dT, int lddt, T *dV_t, int nStates, int stateId){
        T maxVal = std::log(dT[IDX(0, stateId, lddt)]) + dV_t[0];
        T maxIdx = 0;
        T val = 0;
        for (size_t sumIdx = 1; sumIdx < nStates; sumIdx++) {
                val = std::log(dT[IDX(sumIdx, stateId, lddt)]) + dV_t[sumIdx];
                if ( val > maxVal) {
                        maxVal = val;
                        maxIdx = sumIdx;
                }
        }
        return maxIdx;
}

template <typename T>
__device__
T _forward_max(T* dV_prev, T* dT, int lddt, int nStates, int stateId){
        T maxVal = std::log(dT[IDX(0, stateId, lddt)]) + dV_prev[0];
        T val = 0;
        for (size_t sumIdx = 1; sumIdx < nStates; sumIdx++) {
                val = std::log(dT[IDX(sumIdx, stateId, lddt)]) + dV_prev[sumIdx];
                if ( val > maxVal) {
                        maxVal = val;
                }
        }
        return maxVal;
}


template <typename T>
__global__
void _ViterbiValuesKernel(int nStates, int nSeq, int nObs,
                          int* dlenghts,
                          int* dcumlengths_inc,
                          int* dcumlengths_exc,
                          T* dLlhd,
                          T* dV, int lddv,
                          unsigned short int* dVStates,
                          T* dStartProb, int lddsp,
                          T* dT, int lddt,
                          T* dB, int lddb,
                          int numThreads_x, int numThreads_y, int numThreads_z){
        int seqId_start = threadIdx.y + blockDim.y * blockIdx.y;

        int obsId;
        // Forward Max
        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                for (size_t tau = 0; tau < dlenghts[seqId]; tau++) {
                        for (size_t stateId = 0; stateId < nStates; stateId++) {
                                // printf("Forward\n");
                                obsId = dcumlengths_exc[seqId] + tau;
                                if (tau == 0) {
                                        dV[IDX(stateId, obsId, lddv)] = std::log(dStartProb[stateId]) + dB[IDX(stateId, obsId, lddb)];
                                }
                                else {
                                        dV[IDX(stateId, obsId, lddv)] = dB[IDX(stateId, obsId, lddb)] +
                                                                        _forward_max(dV + IDX(0, obsId - 1, lddv), dT, lddt, nStates, stateId);
                                }
                        }
                }
        }
}

template <typename T>
__global__
void _ViterbiBacktraceKernel(int nStates, int nSeq, int nObs,
                             int* dlenghts,
                             int* dcumlengths_inc,
                             int* dcumlengths_exc,
                             T* dLlhd,
                             T* dV, int lddv,
                             unsigned short int* dVStates,
                             T* dStartProb, int lddsp,
                             T* dT, int lddt,
                             T* dB, int lddb,
                             int numThreads_x, int numThreads_y, int numThreads_z){

        int seqId_start = threadIdx.y + blockDim.y * blockIdx.y;
        int obsId;

        // Traceback
        int argmaxVal;

        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                for (size_t tau = 0; tau < dlenghts[seqId]; tau++) {

                        obsId = dcumlengths_inc[seqId] - tau - 1;
                        if (tau == 0) {
                                argmaxVal = arg_max(dV + IDX(0, obsId, lddv), nStates);
                                dVStates[obsId] = argmaxVal;
                                // Store Likelihoods
                                dLlhd[seqId] = dV[IDX(argmaxVal, obsId, lddv)];
                        }
                        else{
                                dVStates[obsId] = _backward_argmax(dT, lddt, dV + IDX(0, obsId, lddv), nStates, dVStates[obsId + 1]);
                        }

                }
        }
}


template <typename T, typename D>
void _viterbi(HMM<T, D> &hmm, unsigned short int* dVStates,
              int* dlenghts, int nSeq){
        dim3 block, grid;
        int numThreads_x, numThreads_y, numThreads_z;


        block.x = 1;
        block.y = 512;
        block.z = 1;

        grid.x = 1;
        grid.y = ceildiv(nSeq, (int)block.y);
        grid.z = 1;

        numThreads_x = grid.x * block.x;
        numThreads_y = grid.y * block.y;
        numThreads_z = grid.z * block.z;



        _ViterbiValuesKernel<T> <<< grid, block >>>(hmm.nStates,
                                                    nSeq, hmm.nObs,
                                                    dlenghts, hmm.handle.dcumlenghts_inc,
                                                    hmm.handle.dcumlenghts_exc,
                                                    hmm.dLlhd,
                                                    hmm.handle.dV, hmm.handle.lddv,
                                                    dVStates,
                                                    hmm.dStartProb, hmm.lddsp,
                                                    hmm.dT, hmm.lddt,
                                                    hmm.dB, hmm.lddb,
                                                    numThreads_x, numThreads_y, numThreads_z);

        block.x = 1;
        block.y = 512;
        block.z = 1;

        grid.x = 1;
        grid.y = ceildiv(nSeq, (int)block.y);
        grid.z = 1;

        numThreads_x = grid.x * block.x;
        numThreads_y = grid.y * block.y;
        numThreads_z = grid.z * block.z;


        _ViterbiBacktraceKernel<T> <<< grid, block >>>(hmm.nStates,
                                                       nSeq, hmm.nObs,
                                                       dlenghts, hmm.handle.dcumlenghts_inc,
                                                       hmm.handle.dcumlenghts_exc,
                                                       hmm.dLlhd,
                                                       hmm.handle.dV, hmm.handle.lddv,
                                                       dVStates,
                                                       hmm.dStartProb, hmm.lddsp,
                                                       hmm.dT, hmm.lddt,
                                                       hmm.dB, hmm.lddb,
                                                       numThreads_x, numThreads_y, numThreads_z);

        MLCommon::Stats::sum(hmm.logllhd, hmm.dLlhd, 1, nSeq, false);

        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dV, hmm.lddv, "dV");
        // print_matrix_device(1, hmm.nObs, dVStates, 1, "dVStates");
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dB, hmm.lddb, "dB matrix");
        // print_matrix_device(hmm.nStates, hmm.nStates, hmm.dT, hmm.lddt, "dT matrix");

        CUDA_CHECK(cudaPeekAtLastError());
}

}
