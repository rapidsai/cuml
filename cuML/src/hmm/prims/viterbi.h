# pragma once

#include "hmm/magma/magma_batched_wrappers.h"

using namespace MLCommon::LinAlg;

namespace hmm {


template <typename T>
__device__
void compute_max_states(int *dV_idx, int lddv,
                        T* dV, int lddv,
                        T* dAlpha, int lddalpha,
                        T* dT, int lddt,
                        int curStateId, int curTime
                        ){

        T val, maxVal=0.;
        int maxValIdx;
        // Parallelize
        for (size_t prevStateId = 0; prevStateId < nStates; prevStateId++) {
                val = std::log(dT[IDX(prevStateId, curStateId, lddT)]) +
                      std::log(alphas[prevStateId]) +
                      std::log(B[curStateId]);
                if (val > maxVal) {
                        maxVal = val;
                        maxValIdx = prevStateId;
                }
        }

        dV[stateId] = maxVal;
        dV_idx[IDX(curStateId, curTime, lddv)] = maxValIdx;
}

template <typename T>
__device__
void compute_max_path(int *dV_idx, int lddv,
                      int* dMaxPath, int len,
                      T* dAlpha, int lddalpha){
        dMaxPath[len - 1] = arg_max(dAlpha + (len - 1) * lddalpha);
        for (i = len - 2; i >= 0; --i) {
                max_path[i] = dV_idx[IDX(max_path[i+1], i, lddv)];
        }
}


template <typename T>
__global__
void viterbiKernel(
        ){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;


        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                for (size_t timeId = 0; timeId < len_array[bId]; timeId++) {
                        for (size_t stateId = j_start; stateId < nStates; stateId+=nThreads_y) {
                                compute_max_states(dV_idx_array[bId], lddv,
                                                   dV_array[bId], lddv,
                                                   dAlpha_array[bId], lddalpha,
                                                   dT, lddt,
                                                   curStateId, timeId);
                                compute_max_path(dV_idx_array[bId], lddv,
                                                 dMaxPath_array[bId], len,
                                                 dAlpha_array[bId], lddalpha);
                        }
                }

        }


}


}
