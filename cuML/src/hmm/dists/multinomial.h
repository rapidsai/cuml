#pragma once

#include <magma/b_likelihood.h>
#include <hmm/dists/dists_variables.h>
#include <hmm/hmm_variables.h>

namespace multinomial {


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
                  ceildiv(nObs, (int)block.y),
                  1);
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
                                       hmm.dPi_array,
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




}
