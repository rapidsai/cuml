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

#include <gmm/likelihood/b_likelihood.h>
#include <hmm/dists/dists_variables.h>
#include <hmm/hmm_variables.h>

namespace multinomial {

template <typename T>
size_t get_workspace_size(Multinomial<T> &multinomial){
        size_t workspaceSize = 0;
        return workspaceSize;
}

template <typename T>
void create_handle(Multinomial<T> &multinomial, void* workspace){

}

// Multinomial
template <typename T>
__global__
void multinomial_likelihood_batched_kernel(int nObs, int batchCount,
                                           unsigned short int* dX,
                                           T** dPb_array,
                                           T* dO, int lddo,
                                           bool isLog,
                                           int nThreads_x, int nThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        T llhd;

        for (size_t obsId = j_start; obsId < nObs; obsId+=nThreads_y) {
                for (size_t bId = i_start; bId <  batchCount; bId+=nThreads_x) {
                        llhd = dPb_array[bId][dX[obsId]];
                        if (isLog) {
                                dO[IDX(bId, obsId, lddo)] = std::log(llhd);
                        }
                        else{
                                dO[IDX(bId, obsId, lddo)] = llhd;
                        }
                }
        }
}




template <typename T>
void multinomial_likelihood_batched(int nObs, int batchCount,
                                    unsigned short int* dX,
                                    T** dPb_array,
                                    T* dO, int lddo,
                                    bool isLog){
        dim3 block(32,32);
        dim3 grid(ceildiv(batchCount, (int)block.x),
                  ceildiv(nObs, (int)block.y));
        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        multinomial_likelihood_batched_kernel<T> <<< grid, block >>>(nObs,batchCount,
                                                                     dX,
                                                                     dPb_array,
                                                                     dO, lddo,
                                                                     isLog,
                                                                     nThreads_x, nThreads_y);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

}

template <typename T>
void update_llhd(unsigned short int* dX, hmm::HMM<T, Multinomial<T> >& hmm, bool isLog){
        multinomial_likelihood_batched(hmm.nObs,
                                       hmm.nStates,
                                       dX,
                                       hmm.handle.dPi_array,
                                       hmm.dB,
                                       hmm.lddb,
                                       isLog);
}

template <typename T>
void init_multinomial(Multinomial<T>& multinomial,
                      T* dPis, int nFeatures) {
        multinomial.dPis = dPis;
        multinomial.nFeatures = nFeatures;
}


template <typename T>
__global__
void update_emissions_kernel(int nDim, int nObs, int nStates,
                             unsigned short int* dX,
                             T** dPi_array, T* dGamma, int lddgamma,
                             int nThreads_x){
        int start = threadIdx.x + blockDim.x * blockIdx.x;
        T sumVal;
        for (size_t stateId = start; stateId < nStates; stateId+=nThreads_x) {
                // printf("thread idxx %d\n", threadIdx.x);
                sumVal = 0;
                for (size_t dimId = 0; dimId < nDim; dimId++) {
                        dPi_array[stateId][dimId]=0;
                }
                for (size_t obsId = 0; obsId < nObs; obsId++) {
                        dPi_array[stateId][dX[obsId]] += dGamma[IDX(stateId, obsId, lddgamma)];
                }

                for (size_t dimId = 0; dimId < nDim; dimId++) {
                        sumVal += dPi_array[stateId][dimId];
                }
                for (size_t dimId = 0; dimId < nDim; dimId++) {
                        dPi_array[stateId][dimId] /= sumVal;
                }
        }
}


template <typename T>
void update_emissions(hmm::HMM<T, Multinomial<T> > &hmm,
                      unsigned short int* dX){
        dim3 block;
        dim3 grid;

        int nThreads_x;

        block.x = 512;
        grid.x = ceildiv((int) hmm.nStates, (int) block.x);

        nThreads_x = grid.x * block.x;

        // // TODO : Run on different streams
        update_emissions_kernel<T> <<< grid, block>>>(
                hmm.dists[0].nFeatures,
                hmm.nObs,
                hmm.nStates,
                dX,
                hmm.handle.dPi_array,
                hmm.dGamma,
                hmm.lddgamma,
                nThreads_x);
        CUDA_CHECK(cudaPeekAtLastError());
}


}
