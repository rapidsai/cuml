#pragma once

#include "hmm/prims/hmm_utils.h"
#include "gmm/gmm.h"
#include "hmm/hmm_variables.h"
#include "hmm/prims/dists.h"

#include "hmm/magma/magma_test_utils.h"
#include "hmm/magma/magma_batched_wrappers.h"

// References :
// http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
//

namespace hmm {
template <typename T, typename D>
void _compute_emissions(T* dX, HMM<T, D> &hmm, cublasHandle_t cublasHandle){
        // Compute the emissions likelihoods B

        if(hmm.type == GaussianMixture) {
                for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                        gmm::update_llhd(dX, hmm.dists[stateId], cublasHandle);
                        // Stores the likelihoods in dProbNorm which points to dB + offset
                }
        }
        else if(hmm.type == Multinomial) {
                multinomial::update_llhd(dX, hmm.dists[stateId]);
        }
}

template <typename T>
void _matrix_powers(magma_int_t n, T** dA_pow_array, magma_int_t max_p,
                    T* dA, magma_int_t ldda, magma_queue_t queue){

        T alpha =1, beta=0;
        dA_pow_array[0] = dA;
        for (size_t p = 1; p < max_p; p++) {
                MLCommon::LinAlg::magmablas_gemm ( MagmaNoTrans,
                                                   MagmaNoTrans,
                                                   n,
                                                   n,
                                                   n,
                                                   alpha,
                                                   dA_pow_array[p-1],
                                                   ldda,
                                                   dA,
                                                   ldda,
                                                   beta,
                                                   dA_pow_array[p],
                                                   ldda,
                                                   queue
                                                   );
        }
}

template <typename T>
__global__
void _ForwardBackwardKernel(magma_int_t nStates,
                            T** dO_array, magma_int_t lddo,
                            T** dT_pows, magma_int_t lddt,
                            T** dB_array, magma_int_t lddb,
                            magma_int_t *dlenghts, magma_int_t nSeq,
                            bool isForward,
                            int numThreads_x, int numThreads_y){
        int seqId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int stateId_start = threadIdx.y + blockDim.y * blockIdx.y;
        T cum_prod = 1;

        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                for (size_t stateId = stateId_start; stateId < nStates; stateId+=numThreads_x) {
                        if (isForward) {
                                for (size_t obsId = 0; obsId < dlenghts[seqId]; obsId++) {
                                        cum_prod = _prod(cum_prod, dB_array[seqId][IDX(stateId, obsId, lddb)]);
                                        dO_array[seqId][IDX(stateId, obsId, lddo)] = cum_prod * dT_pows[obsId][IDX(0, stateId, lddt)];
                                }
                        }

                }
        }
}

template <typename T, typename D>
void _forward_backward(HMM<T, D> &hmm,
                       int* dlenghts, int nSeq,
                       bool doForward, bool doBackward ){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv(hmm.nStates, (int)block.x),
                  ceildiv(nSeq, (int)block.y),
                  1);

        int numThreads_x = grid.x * block.x;
        int numThreads_y = grid.y * block.y;

        if (doForward) {
                _ForwardBackwardKernel<T> <<< grid, block >>>(hmm.nStates,
                                                              hmm.dAlpha_array, hmm.lddalpha,
                                                              hmm.dT_pows, hmm.lddt,
                                                              hmm.dB_array, hmm.lddb,
                                                              dlenghts, nSeq,
                                                              true,
                                                              numThreads_x, numThreads_y);
        }
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}


// template <typename T>
// _updateGammasKernel(){
//         for (size_t obsId = 0; obsId < nObs; obsId++) {
//                 for (size_t stateId = 0; stateId < nStates; stateId++) {
//                         dGamma[IDX(stateId, obsId, lddgamma)] = std::log(dAlpha[IDX(stateId, obsId, lddalpha)]) + std::log(dBeta[IDX(stateId, obsId, lddalpha)]) + std::log(dT[IDX(stateId, obsId, lddalpha)]);
//                 }
//
//         }
// }





}
