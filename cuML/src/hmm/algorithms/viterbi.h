# pragma once

#include "hmm/magma/magma_batched_wrappers.h"

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
void _ViterbiKernel(int nStates, int nSeq, int nObs,
                    unsigned short int* dlenghts,
                    unsigned short int* dcumlengths_inc,
                    unsigned short int* dcumlengths_exc,
                    T* dLlhd,
                    T* dV, int lddv,
                    unsigned short int* dVStates,
                    T* dStartProb, int lddsp,
                    T* dT, int lddt,
                    T* dB, int lddb,
                    int numThreads_x, int numThreads_y, int numThreads_z){
        int stateId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int seqId_start = threadIdx.y + blockDim.y * blockIdx.y;

        int obsId;
        // Forward Max
        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                for (size_t tau = 0; tau < dcumlengths_inc[seqId]; tau++) {
                        for (size_t stateId = stateId_start; stateId < nStates; stateId+=numThreads_x) {
                                printf("Forward\n");
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


        // Traceback
        int argmaxVal;
        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                if (threadIdx.x == 0) {
                        printf("Traceback\n");
                        for (size_t tau = 0; tau < dcumlengths_inc[seqId]; tau++) {

                                obsId = dcumlengths_inc[seqId] - tau - 1;
                                if (tau == 0) {
                                        argmaxVal = arg_max(dV + IDX(0, obsId, lddv), nStates);
                                        dVStates[obsId] = argmaxVal;
                                        // Store Likelihoods
                                        printf("%f\n", dV[IDX(0, obsId, lddv)]);
                                        printf("%d\n", argmaxVal);
                                        dLlhd[seqId] = dV[IDX(argmaxVal, obsId, lddv)];
                                }
                                else{
                                        dVStates[obsId] = _backward_argmax(dT, lddt, dV + IDX(0, obsId, lddv), nStates, dVStates[obsId + 1]);
                                }
                        }

                }
        }
}

template <typename T, typename D>
void _viterbi(HMM<T, D> &hmm, unsigned short int* dVStates,
              unsigned short int* dlenghts, int nSeq){
        dim3 block(32, 1, 1);
        dim3 grid(1, 1, 1);
        // dim3 grid(ceildiv(hmm.nStates, (int)block.x),
        //           ceildiv(nSeq, (int)block.y),
        //           ceildiv(hmm.nObs, (int)block.z));

        int numThreads_x = grid.x * block.x;
        int numThreads_y = grid.y * block.y;
        int numThreads_z = grid.z * block.z;

        _ViterbiKernel<T> <<< grid, block >>>(hmm.nStates,
                                              nSeq, hmm.nObs,
                                              dlenghts, hmm.dcumlenghts_inc,
                                              hmm.dcumlenghts_exc,
                                              hmm.dLlhd,
                                              hmm.dV, hmm.lddv,
                                              dVStates,
                                              hmm.dStartProb, hmm.lddsp,
                                              hmm.dT, hmm.lddt,
                                              hmm.dB, hmm.lddb,
                                              numThreads_x, numThreads_y, numThreads_z);
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dV, hmm.lddv, "dV");
        // print_matrix_device(1, hmm.nObs, dVStates, 1, "dVStates");
        // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dB, hmm.lddb, "dB matrix");
        // print_matrix_device(hmm.nStates, hmm.nStates, hmm.dT, hmm.lddt, "dT matrix");

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

}
