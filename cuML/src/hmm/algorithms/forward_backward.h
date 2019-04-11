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

#include "hmm/algorithms/hmm_utils.h"
#include "gmm/gmm.h"
#include "hmm/hmm_variables.h"
#include "hmm/dists/multinomial.h"

#include "magma/magma_utils.h"
#include "magma/magma_batched_wrappers.h"
#include "stats/sum.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// References :
// http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
//

namespace hmm {

// __global__
// void _cumlenghtsKernel(int *dcumlenghts_inc, int *dcumlenghts_exc,
//                        unsigned short int *dlenghts, int nSeq){
//         if (threadIdx.x == 0) {
//                 dcumlenghts_exc[0] = 0;
//                 for (size_t i = 1; i < nSeq; i++) {
//                         dcumlenghts_exc[i] = dcumlenghts_exc[i - 1] + dlenghts[i - 1];
//                 }
//         }
//         if (threadIdx.x == 1) {
//                 dcumlenghts_inc[0] = dlenghts[0];
//                 for (size_t i = 1; i < nSeq; i++) {
//                         dcumlenghts_inc[i] = dcumlenghts_inc[i - 1] + dlenghts[i];
//                 }
//         }
//
// }

void _compute_cumlengths(int *dcumlenghts_inc, int *dcumlenghts_exc,
                         int *dlenghts, int nSeq){
        // dim3 block(32, 1, 1);
        // dim3 grid(1, 1, 1);
        // _cumlenghtsKernel<<< grid, block >>>(dcumlenghts_inc, dcumlenghts_exc,
        //                                      dlenghts, nSeq);
        // cudaDeviceSynchronize();
        // CUDA_CHECK(cudaPeekAtLastError());

        thrust::exclusive_scan(thrust::device, dlenghts, dlenghts + nSeq, dcumlenghts_exc, 0); // in-place scan

        thrust::inclusive_scan(thrust::device, dlenghts, dlenghts + nSeq, dcumlenghts_inc);
}

template <typename T>
void _compute_emissions(T* dX,
                        HMM<T, gmm::GMM<T> > &hmm,
                        cublasHandle_t cublasHandle, magma_queue_t queue){
        // Compute the emissions likelihoods B
        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                gmm::update_llhd(dX, hmm.dists[stateId], cublasHandle, queue);
        }
        // TODO ; Transpose
}


template <typename T>
void _compute_emissions(unsigned short int *dX,
                        HMM<T, multinomial::Multinomial<T> > &hmm,
                        cublasHandle_t cublasHandle, magma_queue_t queue){
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
T _backward_dot(T* dT, int lddt,
                T* prevdist,
                T* nextdB,
                int nStates, int stateId){
        T res=0;
        T temp =0;
        for (size_t sumIdx = 0; sumIdx < nStates; sumIdx++) {
                temp = prevdist[sumIdx]
                       + std::log(dT[IDX(stateId, sumIdx, lddt)])
                       // ;
                       // + nextdB[1];
                       + nextdB[sumIdx];
                res += std :: exp(temp);
        }
        return std::log(res);
}

template <typename T>
__device__
void sum_llhd(T* dO, int len, T* darray){
        T sumVal=0;

        for (size_t i = 0; i < len; i++) {
                sumVal += std :: exp(darray[i]);
        }
        *dO = std::log(sumVal);
}

template <typename T>
__device__
void _sum(T* dO, int len, T* darray){
        T sumVal=0;

        for (size_t i = 0; i < len; i++) {
                sumVal += darray[i];
        }
        *dO = sumVal;
}

// template <typename T>
// __global__
// void _ForwardBackwardKernel(int nStates, int nSeq, int nObs,
//                             int* dlenghts,
//                             unsigned short int* dcumlengths_inc,
//                             unsigned short int* dcumlengths_exc,
//                             T* dLlhd, T* logllhd,
//                             T* dAlpha, int lddalpha,
//                             T* dBeta, int lddbeta,
//                             T* dStartProb, int lddsp,
//                             T* dT, int lddt,
//                             T* dB, int lddb,
//                             bool doForward, bool doBackward,
//                             int numThreads_x, int numThreads_y){
//         int stateId_start = threadIdx.x + blockDim.x * blockIdx.x;
//         int seqId_start = threadIdx.y + blockDim.y * blockIdx.y;
//
//         int obsId;
//         T temp;
//         if (doForward) {
//                 for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
//                         for (size_t tau = 0; tau < dlenghts[seqId]; tau++) {
//                                 for (size_t stateId = stateId_start; stateId < nStates; stateId+=numThreads_x) {
//                                         obsId = dcumlengths_exc[seqId] + tau;
//                                         if (tau == 0) {
//                                                 dAlpha[IDX(stateId, obsId, lddalpha)] = std::log(dStartProb[stateId]) + dB[IDX(stateId, obsId, lddb)];
//                                         }
//                                         else {
//                                                 temp = _forward_dot(dT, lddt, dAlpha + IDX(0, obsId - 1, lddalpha), nStates, stateId);
//                                                 dAlpha[IDX(stateId, obsId, lddalpha)] = dB[IDX(stateId, obsId, lddb)] + temp;
//                                         }
//                                 }
//                         }
//                         if (stateId_start == 0 && stateId_start == 0) {
//                                 // Compute the log likelihoods
//                                 sum_llhd(dLlhd + seqId, nStates,
//                                          dAlpha + IDX(0, obsId, lddalpha));
//                         }
//                 }
//
//                 __syncthreads();
//
//                 if (stateId_start == 0 && seqId_start == 0) {
//                         _sum(logllhd, nSeq, dLlhd);
//                 }
//         }
//         if (doBackward) {
//                 for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
//                         for (size_t tau = 0; tau < dlenghts[seqId]; tau++) {
//                                 for (size_t stateId = stateId_start; stateId < nStates; stateId+=numThreads_x) {
//                                         obsId = dcumlengths_inc[seqId] - tau - 1;
//                                         if (tau == 0) {
//                                                 dBeta[IDX(stateId, obsId, lddbeta)] = 0;
//                                         }
//                                         else{
//                                                 // dBeta[IDX(stateId, obsId, lddbeta)] =obsId;
//                                                 dBeta[IDX(stateId, obsId, lddbeta)] = _backward_dot(dT, lddt, dBeta + IDX(0, obsId + 1, lddbeta), dB + IDX(0, obsId + 1, lddb), nStates, stateId);
//                                         }
//                                 }
//
//                         }
//                 }
//         }
// }


// template <typename T, typename D>
// void _forward_backward(HMM<T, D> &hmm,
//                        int* dlenghts, int nSeq,
//                        bool doForward, bool doBackward ){
//         dim3 block, grid;
//         int numThreads_x, numThreads_y;
//
//         // block.x = 32;
//         // block.y = 32;
//         //
//         // grid.x =ceildiv(hmm.nStates, (int)block.x);
//         // grid.y =ceildiv(nSeq, (int)block.y);
//         //
//         // numThreads_x = grid.x * block.x;
//         // numThreads_y = grid.y * block.y;
//
//         // _ForwardBackwardKernel<T> <<< grid, block >>>(hmm.nStates, nSeq, hmm.nObs,
//         //                                               dlenghts, hmm.handle.dcumlenghts_inc,
//         //                                               hmm.handle.dcumlenghts_exc,
//         //                                               hmm.dLlhd, hmm.logllhd,
//         //                                               hmm.handle.dAlpha, hmm.handle.lddalpha,
//         //                                               hmm.handle.dBeta, hmm.handle.lddbeta,
//         //                                               hmm.dStartProb, hmm.lddsp,
//         //                                               hmm.dT, hmm.lddt,
//         //                                               hmm.dB, hmm.lddb,
//         //                                               doForward, doBackward,
//         //                                               numThreads_x, numThreads_y);
//         // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dAlpha, hmm.lddalpha, "dAlpha");
//         // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dBeta, hmm.lddbeta, "dBeta");
//         // print_matrix_device(hmm.nStates, hmm.nObs, hmm.dB, hmm.lddb, "dB matrix");
//         // print_matrix_device(hmm.nStates, hmm.nStates, hmm.dT, hmm.lddt, "dT matrix");
//
// }

template <typename T>
__global__
void _ForwardKernel(int nStates, int nSeq, int nObs,
                    int* dlenghts,
                    int* dcumlengths_inc,
                    int* dcumlengths_exc,
                    T* dLlhd, T* logllhd,
                    T* dAlpha, int lddalpha,
                    T* dBeta, int lddbeta,
                    T* dStartProb, int lddsp,
                    T* dT, int lddt,
                    T* dB, int lddb,
                    bool doForward, bool doBackward,
                    int numThreads_x, int numThreads_y){
        // int stateId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int seqId_start = threadIdx.x + blockDim.x * blockIdx.x;

        int obsId;
        T temp;
        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_x) {
                for (size_t tau = 0; tau < dlenghts[seqId]; tau++) {
                        // Ensure \alpha_{t-1} is ready before computing \alpha_t
                        for (size_t stateId = 0; stateId < nStates; stateId+=1) {
                                obsId = dcumlengths_exc[seqId] + tau;
                                if (tau == 0) {
                                        dAlpha[IDX(stateId, obsId, lddalpha)] = std::log(dStartProb[stateId]) + dB[IDX(stateId, obsId, lddb)];
                                }
                                else {
                                        temp = _forward_dot(dT, lddt, dAlpha + IDX(0, obsId - 1, lddalpha), nStates, stateId);
                                        dAlpha[IDX(stateId, obsId, lddalpha)] = dB[IDX(stateId, obsId, lddb)] + temp;
                                }
                        }
                }
        }
}

template <typename T>
__global__
void _ForwardLlhdKernel(int nStates, int nSeq, int nObs,
                        int* dlenghts,
                        int* dcumlengths_inc,
                        int* dcumlengths_exc,
                        T* dLlhd, T* logllhd,
                        T* dAlpha, int lddalpha,
                        T* dBeta, int lddbeta,
                        T* dStartProb, int lddsp,
                        T* dT, int lddt,
                        T* dB, int lddb,
                        bool doForward, bool doBackward,
                        int numThreads_x, int numThreads_y){
        int seqId_start = threadIdx.x + blockDim.x * blockIdx.x;

        int obsId;
        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_x) {
                obsId = dcumlengths_inc[seqId] - 1;
                sum_llhd(dLlhd + seqId, nStates,
                         dAlpha + IDX(0, obsId, lddalpha));
        }
}




template <typename T>
__global__
void _BackwardKernel(int nStates, int nSeq, int nObs,
                     int* dlenghts,
                     int* dcumlengths_inc,
                     int* dcumlengths_exc,
                     T* dLlhd, T* logllhd,
                     T* dAlpha, int lddalpha,
                     T* dBeta, int lddbeta,
                     T* dStartProb, int lddsp,
                     T* dT, int lddt,
                     T* dB, int lddb,
                     bool doForward, bool doBackward,
                     int numThreads_x, int numThreads_y){
        // int stateId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int seqId_start = threadIdx.x + blockDim.x * blockIdx.x;

        int obsId;

        // for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_x) {
        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_x) {
                for (size_t tau = 0; tau < dlenghts[seqId]; tau++) {
                        for (size_t stateId = 0; stateId < nStates; stateId+=1) {
                                obsId = dcumlengths_inc[seqId] - tau - 1;
                                if (tau == 0) {
                                        dBeta[IDX(stateId, obsId, lddbeta)] = 0;
                                }
                                else{
                                        dBeta[IDX(stateId, obsId, lddbeta)] = _backward_dot(dT, lddt,
                                                                                            dBeta + IDX(0, obsId + 1, lddbeta),
                                                                                            dB + IDX(0, obsId + 1, lddb),
                                                                                            nStates, stateId);

                                }
                        }

                }
        }
}



template <typename T, typename D>
void _forward_backward(HMM<T, D> &hmm,
                       int* dlenghts, int nSeq,
                       bool doForward, bool doBackward ){
        dim3 block, grid;
        int numThreads_x, numThreads_y;

        if (doForward) {
                block.x = 32;
                block.y = 1;

                grid.x =ceildiv(nSeq, (int)block.x);
                grid.y =1;

                numThreads_x = grid.x * block.x;
                numThreads_y = grid.y * block.y;

                _ForwardKernel<T> <<< grid, block >>>(hmm.nStates, nSeq, hmm.nObs,
                                                      dlenghts, hmm.handle.dcumlenghts_inc,
                                                      hmm.handle.dcumlenghts_exc,
                                                      hmm.dLlhd, hmm.logllhd,
                                                      hmm.handle.dAlpha, hmm.handle.lddalpha,
                                                      hmm.handle.dBeta, hmm.handle.lddbeta,
                                                      hmm.dStartProb, hmm.lddsp,
                                                      hmm.dT, hmm.lddt,
                                                      hmm.dB, hmm.lddb,
                                                      doForward, doBackward,
                                                      numThreads_x, numThreads_y);

                block.x = 512;
                block.y = 1;

                grid.x =ceildiv(nSeq, (int)block.x);
                grid.y =1;

                numThreads_x = grid.x * block.x;
                numThreads_y = grid.y * block.y;

                _ForwardLlhdKernel<T> <<< grid, block >>>(hmm.nStates, nSeq, hmm.nObs,
                                                          dlenghts, hmm.handle.dcumlenghts_inc,
                                                          hmm.handle.dcumlenghts_exc,
                                                          hmm.dLlhd, hmm.logllhd,
                                                          hmm.handle.dAlpha, hmm.handle.lddalpha,
                                                          hmm.handle.dBeta, hmm.handle.lddbeta,
                                                          hmm.dStartProb, hmm.lddsp,
                                                          hmm.dT, hmm.lddt,
                                                          hmm.dB, hmm.lddb,
                                                          doForward, doBackward,
                                                          numThreads_x, numThreads_y);

                MLCommon::Stats::sum(hmm.logllhd, hmm.dLlhd, 1, nSeq, false);
        }

        if (doBackward) {
                block.x = 512;
                block.y = 1;

                grid.x =ceildiv(nSeq, (int)block.x);
                grid.y =1;

                numThreads_x = grid.x * block.x;
                numThreads_y = grid.y * block.y;

                _BackwardKernel<T> <<< grid, block >>>(hmm.nStates, nSeq, hmm.nObs,
                                                       dlenghts, hmm.handle.dcumlenghts_inc,
                                                       hmm.handle.dcumlenghts_exc,
                                                       hmm.dLlhd, hmm.logllhd,
                                                       hmm.handle.dAlpha, hmm.handle.lddalpha,
                                                       hmm.handle.dBeta, hmm.handle.lddbeta,
                                                       hmm.dStartProb, hmm.lddsp,
                                                       hmm.dT, hmm.lddt,
                                                       hmm.dB, hmm.lddb,
                                                       doForward, doBackward,
                                                       numThreads_x, numThreads_y);
        }

        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
__global__
void _updateGammasKernel(int nObs, int nStates,
                         T* dGamma, int lddgamma,
                         T* dAlpha, int lddalpha,
                         T* dBeta, int lddbeta,
                         int numThreads_x){
        T _sum;
        int obsId_start = threadIdx.x + blockDim.x * blockIdx.x;
        for (size_t obsId = obsId_start; obsId < nObs; obsId+=numThreads_x) {
                _sum = 0;
                for (size_t stateId = 0; stateId < nStates; stateId++) {
                        dGamma[IDX(stateId, obsId, lddgamma)] = std::exp(dAlpha[IDX(stateId, obsId, lddalpha)] + dBeta[IDX(stateId, obsId, lddbeta)]);
                        _sum += dGamma[IDX(stateId, obsId, lddgamma)];
                }
                for (size_t stateId = 0; stateId < nStates; stateId++) {
                        dGamma[IDX(stateId, obsId, lddgamma)] /= _sum;
                }
        }
}


template <typename T, typename D>
void _update_gammas(HMM<T, D> &hmm){
        dim3 block(512);
        dim3 grid(ceildiv(hmm.nObs, (int)block.x));

        int numThreads_x = grid.x * block.x;
        // int numThreads_y = grid.y * block.y;

        _updateGammasKernel<T> <<< grid, block >>>(hmm.nObs, hmm.nStates,
                                                   hmm.dGamma, hmm.lddgamma,
                                                   hmm.handle.dAlpha, hmm.handle.lddalpha,
                                                   hmm.handle.dBeta, hmm.handle.lddbeta,
                                                   numThreads_x);
        // cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}


}
