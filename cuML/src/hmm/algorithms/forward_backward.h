#pragma once

#include "hmm/algorithms/hmm_utils.h"
#include "gmm/gmm.h"
#include "hmm/hmm_variables.h"
#include "hmm/dists/multinomial.h"

#include "hmm/magma/magma_test_utils.h"
#include "hmm/magma/magma_batched_wrappers.h"

// References :
// http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
//

namespace hmm {

__global__
void _cumlenghtsKernel(int *dcumlenghts_inc, int *dcumlenghts_exc,
                       int *dlenghts, int nSeq){
        if (threadIdx.x == 0) {
                dcumlenghts_exc[0] = 0;
                for (size_t i = 1; i < nSeq; i++) {
                        dcumlenghts_exc[i] = dcumlenghts_exc[i - 1] + dlenghts[i - 1];
                }
        }
        if (threadIdx.x == 1) {
                dcumlenghts_inc[0] = dlenghts[0];
                for (size_t i = 1; i < nSeq; i++) {
                        dcumlenghts_inc[i] = dcumlenghts_inc[i - 1] + dlenghts[i];
                }
        }

}

void _compute_cumlengths(int *dcumlenghts_inc, int *dcumlenghts_exc,
                         int *dlenghts, int nSeq){
        dim3 block(32, 1, 1);
        dim3 grid(1, 1, 1);
        _cumlenghtsKernel<<< grid, block >>>(dcumlenghts_inc, dcumlenghts_exc,
                                             dlenghts, nSeq);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void _compute_emissions(T* dX,
                        HMM<T, gmm::GMM<T> > &hmm,
                        cublasHandle_t cublasHandle){
        // Compute the emissions likelihoods B
        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                gmm::update_llhd(dX, hmm.dists[stateId], cublasHandle);
        }
}


template <typename T>
void _compute_emissions(int *dX,
                        HMM<T, multinomial::Multinomial<T> > &hmm,
                        cublasHandle_t cublasHandle){
        // Compute the emissions likelihoods B
        multinomial::update_llhd(dX, hmm, true);
}


template <typename T>
__device__
T _forward_dot(T* dT, int lddt, T* prevdist, int nStates, int stateId){
        T res=0;
        T temp =0;
        for (size_t sumIdx = 0; sumIdx < nStates; sumIdx++) {
                temp = prevdist[sumIdx] + std::log(dT[IDX(sumIdx, stateId, lddt)]);
                res += std :: exp(temp);
        }
        return std::log(res);
}

template <typename T>
__device__
T _backward_dot(T* dT, int lddt, T* prevdist, T* nextdB, int nStates, int stateId){
        T res=0;
        T temp =0;
        for (size_t sumIdx = 0; sumIdx < nStates; sumIdx++) {
                temp = prevdist[sumIdx] + std::log(dT[IDX(stateId, sumIdx, lddt)]) +
                       nextdB[sumIdx];
                res += std :: exp(temp);
        }
        return std::log(res);
}

template <typename T>
__global__
void _ForwardBackwardKernel(int nStates, int nSeq, int nObs,
                            int* dlenghts, int* dcumlengths_inc,
                            int* dcumlengths_exc,
                            T* dO, int lddo,
                            T* dStartProb, int lddsp,
                            T* dT, int lddt,
                            T* dB, int lddb,
                            bool doForward, bool doBackward,
                            int numThreads_x, int numThreads_y, int numThreads_z){
        int stateId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int seqId_start = threadIdx.y + blockDim.y * blockIdx.y;

        int obsId;
        T temp;
        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                for (size_t tau = 0; tau < dcumlengths_inc[seqId]; tau++) {
                        for (size_t stateId = stateId_start; stateId < nStates; stateId+=numThreads_x) {
                                if (doForward) {
                                        obsId = dcumlengths_exc[seqId] + tau;
                                        if (tau == 0) {
                                                dO[IDX(stateId, obsId, lddo)] = std::log(dStartProb[stateId]) + dB[IDX(stateId, obsId, lddb)];
                                        }
                                        else {
                                                temp = _forward_dot(dT, lddt, dO + IDX(0, obsId - 1, lddo), nStates, stateId);
                                                dO[IDX(stateId, obsId, lddo)] = dB[IDX(stateId, obsId, lddb)] + temp;
                                        }
                                }

                                if (doBackward) {
                                        obsId = dcumlengths_inc[seqId] - tau - 1;
                                        if (tau == 0) {
                                                dO[IDX(stateId, obsId, lddo)] = 0;
                                        }
                                        else{
                                                dO[IDX(stateId, obsId, lddo)] = _backward_dot(dT, lddt, dO + IDX(0, obsId + 1, lddo), dB + IDX(0, obsId + 1, lddb), nStates, stateId);;
                                        }
                                }

                        }
                }
        }
}

template <typename T, typename D>
void _forward_backward(HMM<T, D> &hmm,
                       int* dlenghts, int nSeq,
                       bool doForward, bool doBackward ){
        dim3 block(32, 1, 1);
        dim3 grid(1, 1, 1);
        // dim3 grid(ceildiv(hmm.nStates, (int)block.x),
        //           ceildiv(nSeq, (int)block.y),
        //           ceildiv(hmm.nObs, (int)block.z));

        int numThreads_x = grid.x * block.x;
        int numThreads_y = grid.y * block.y;
        int numThreads_z = grid.z * block.z;

        _ForwardBackwardKernel<T> <<< grid, block >>>(hmm.nStates, nSeq, hmm.nObs,
                                                      dlenghts, hmm.dcumlenghts_inc,
                                                      hmm.dcumlenghts_exc,
                                                      hmm.dAlpha, hmm.lddalpha,
                                                      hmm.dStartProb, hmm.lddsp,
                                                      hmm.dT, hmm.lddt,
                                                      hmm.dB, hmm.lddb,
                                                      doForward, doBackward,
                                                      numThreads_x, numThreads_y, numThreads_z);
        print_matrix_device(hmm.nStates, hmm.nObs, hmm.dAlpha, hmm.lddalpha, "dAlpha matrix after");
        print_matrix_device(hmm.nStates, hmm.nObs, hmm.dB, hmm.lddb, "dB matrix");
        print_matrix_device(hmm.nStates, hmm.nStates, hmm.dT, hmm.lddt, "dT matrix");

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
