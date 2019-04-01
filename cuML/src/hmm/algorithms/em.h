#pragma once

#include "hmm/dists/multinomial.h"
#include "gmm/gmm_backend.h"

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
                               T* logprob,
                               T* dAlpha, int lddalpha,
                               T* dBeta, int lddbeta,
                               T* dT, int lddt,
                               T* dB, int lddb,
                               int numThreads_x, int numThreads_y){
        int stateId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int seqId_start = threadIdx.y + blockDim.y * blockIdx.y;
        int obsId;
        T temp_val, temp_logsum;
        bool initialized;

        if (stateId_start == 0 & seqId_start == 0) {
                for (size_t i = 0; i < nStates; i+=1) {
                        for (size_t j = 0; j < nStates; j+=1) {
                                printf("i %d\n",i );
                                printf("j %d\n", j );
                                initialized =false;
                                for (size_t seqId = 0; seqId < nSeq; seqId++) {
                                        for (size_t tau = 0; tau < dlenghts[seqId] - 1; tau++) {
                                                obsId = dcumlengths_exc[seqId] + tau;
                                                temp_val = dAlpha[IDX(i, obsId, lddalpha)]
                                                           + std::log(dT[IDX(i, j, lddt)])
                                                           + dB[IDX(j, obsId + 1, lddb)]
                                                           + dBeta[IDX(j, obsId + 1, lddbeta)]
                                                           - *logprob;
                                                if (initialized) {
                                                        temp_logsum = std::log(std::exp(temp_val) + std::exp(temp_logsum));
                                                }
                                                else {
                                                        temp_logsum = temp_val;
                                                        initialized = true;
                                                }
                                        }
                                }
                                dT[IDX(i, j, lddt)] = std::exp(temp_logsum);
                        }
                }

        }
}

template <typename Tx, typename T, typename D>
void _m_step(HMM<T, D> &hmm,
             Tx* dX, unsigned short int* dlenghts, int nSeq) {

        dim3 block(32);
        dim3 grid(1);
        int nThreads_x = grid.x * block.x;

        // // TODO : Run on different streams
        multinomial::update_emissions_kernel<T> <<< grid, block>>>(
                hmm.dists[0].nFeatures,
                hmm.nObs,
                hmm.nStates,
                dX,
                hmm.dPi_array,
                hmm.dGamma,
                hmm.lddgamma,
                nThreads_x);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

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

        block.x = 8;
        block.y = 8;
        grid.x = 1;
        nThreads_x = grid.x * block.x;
        int nThreads_y = block.y;

        update_transitions_kernel<T> <<< grid, block >>>(hmm.nStates,
                                                         nSeq,
                                                         hmm.nObs,
                                                         dlenghts,
                                                         hmm.dcumlenghts_exc,
                                                         hmm.logllhd,
                                                         hmm.dAlpha, hmm.lddalpha,
                                                         hmm.dBeta, hmm.lddbeta,
                                                         hmm.dT, hmm.lddt,
                                                         hmm.dB, hmm.lddb,
                                                         nThreads_x, nThreads_y);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
        gmm::normalize_matrix(hmm.nStates, hmm.nStates, hmm.dT, hmm.lddt, false);

}
}
