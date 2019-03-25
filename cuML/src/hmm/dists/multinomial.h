#pragma once

#include <hmm/magma/b_likelihood.h>
#include <hmm/dists/dists_variables.h>
#include <hmm/hmm_variables.h>

namespace multinomial {


// Multinomial
template <typename T>
__global__
void multinomial_likelihood_batched_kernel(int nObs, int batchCount,
                                           int* dX,
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
                                    int* dX,
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
void update_llhd(int* dX, hmm::HMM<T, Multinomial<T> >& hmm, bool isLog){
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
}
