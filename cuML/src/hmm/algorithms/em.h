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

#include "hmm/dists/multinomial.h"
#include "hmm/dists/gmm.h"
#include "gmm/backend/gmm_backend.h"

using namespace multinomial;
using namespace gmm;

namespace hmm {

template <typename T>
__global__
void update_startprob_kernel(int nStates, int nSeq,
                             unsigned short int* dlenghts,
                             T* dStartProb,
                             T* dGamma, int lddgamma,
                             int nThreads_x){
        int start = threadIdx.x + blockDim.x * blockIdx.x;

        for (size_t stateId = start; stateId < nStates; stateId+=nThreads_x) {
                dStartProb[stateId] = 0;
        }

        int obsId = 0;
        for (size_t seqId = 0; seqId < nSeq; seqId++) {
                for (size_t stateId = start; stateId < nStates; stateId+=nThreads_x) {
                        dStartProb[stateId] += dGamma[IDX(stateId, obsId, lddgamma)];
                }
                obsId += dlenghts[seqId];
        }

        if (start == 0) {
                T sumVal = 0;
                for (size_t stateId = 0; stateId < nStates; stateId++) {
                        sumVal += dStartProb[stateId];
                }
                for (size_t stateId = 0; stateId < nStates; stateId++) {
                        dStartProb[stateId] /= sumVal;
                }
        }

}

template <typename T>
__global__
void update_transitions_kernel(int nStates, int nSeq, int nObs,
                               unsigned short int* dlenghts,
                               unsigned short int* dcumlengths_exc,
                               T* dLlhd,
                               T* dAlpha, int lddalpha,
                               T* dBeta, int lddbeta,
                               T* dT, int lddt,
                               T* dB, int lddb,
                               int nThreads_x, int nThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int obsId;
        T temp_val, temp_logsum, temp_t;
        bool initialized;

        for (size_t i = i_start; i < nStates; i+=nThreads_x) {
                for (size_t j = j_start; j < nStates; j+=nThreads_y) {
                        temp_t = 0;
                        for (size_t seqId = 0; seqId < nSeq; seqId++) {
                                initialized =false;
                                for (size_t tau = 0; tau < dlenghts[seqId] - 1; tau++) {
                                        // for (size_t obsId = 0; obsId < nObs - 1; obsId++) {
                                        obsId = dcumlengths_exc[seqId] + tau;
                                        temp_val = dAlpha[IDX(i, obsId, lddalpha)]
                                                   + std::log(dT[IDX(i, j, lddt)])
                                                   + dB[IDX(j, obsId + 1, lddb)]
                                                   + dBeta[IDX(j, obsId + 1, lddbeta)]
                                                   - dLlhd[seqId];
                                        if (initialized) {
                                                temp_logsum = std::log(std::exp(temp_val) + std::exp(temp_logsum));
                                        }
                                        else {
                                                temp_logsum = temp_val;
                                                initialized = true;
                                        }
                                }
                                temp_t += std::exp(temp_logsum);
                        }
                        dT[IDX(i, j, lddt)] = temp_t;
                }
        }


}

template <typename Tx, typename T, typename D>
void _m_step(HMM<T, D> &hmm,
             Tx* dX, unsigned short int* dlenghts, int nSeq) {

        dim3 block;
        dim3 grid;
        int nThreads_x, nThreads_y;

        update_emissions(hmm, dX);
        block.x = 32;
        grid.x = 1;
        nThreads_x = grid.x * block.x;
        update_startprob_kernel<T> <<< grid, block >>>(hmm.nStates,
                                                       nSeq,
                                                       dlenghts,
                                                       hmm.dStartProb,
                                                       hmm.dGamma,
                                                       hmm.lddgamma,
                                                       nThreads_x);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

        block.x = 32;
        block.y = 32;
        grid.x = ceildiv((int) hmm.nStates, (int) block.x);
        grid.y = ceildiv((int) hmm.nStates, (int) block.y);
        nThreads_x = grid.x * block.x;
        nThreads_y = grid.y * block.y;

        update_transitions_kernel<T> <<< grid, block >>>(hmm.nStates,
                                                         nSeq,
                                                         hmm.nObs,
                                                         dlenghts,
                                                         hmm.handle.dcumlenghts_exc,
                                                         hmm.dLlhd,
                                                         hmm.handle.dAlpha, hmm.handle.lddalpha,
                                                         hmm.handle.dBeta, hmm.handle.lddbeta,
                                                         hmm.dT, hmm.lddt,
                                                         hmm.dB, hmm.lddb,
                                                         nThreads_x, nThreads_y);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
        gmm::normalize_matrix(hmm.nStates, hmm.nStates, hmm.dT, hmm.lddt, false);
}
}
