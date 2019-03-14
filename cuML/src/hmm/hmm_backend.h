#pragma once

#include "hmm/hmm_utils.h"

// References :
// http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
//

template <typename T>
void _compute_emissions(T* dX, int* len_array, HMM<T>& hmm){
        // Compute the emissions likelihoods B
        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                // TODO : Change the API to compute the batches in the backend
                _likelihood_batched();
                _compute_likelihood();
        }
}



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
void viterbiKernel(int *dV_idx_array, int lddv,
                   T* dV_array, int lddv,
                   T* dAlpha_array, int lddalpha,
                   T* dT, int lddt,
                   int* dMaxPath_array, int len_array,
                   int nStates,
                   int nThreads_x, int nThreads_y
                   ){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t bId = i_start; bId < batchCount; bId+=nThreads_x) {
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

template <typename T>
_computeGammasKernel(){
        for (size_t obsId = 0; obsId < nObs; obsId++) {
                for (size_t stateId = 0; stateId < nStates; stateId++) {
                        dGamma[IDX(stateId, obsId, lddgamma)] = std::log(dAlpha[IDX(stateId, obsId, lddalpha)]) + std::log(dBeta[IDX(stateId, obsId, lddalpha)]) + std::log(dT[IDX(stateId, obsId, lddalpha)]);
                }

        }

}

template <typename T>
void compute_gammas(){
        forward();
        backward();
        // elementwise
        // Normalize
}
