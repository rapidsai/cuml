# pragma once

#include "hmm/magma/magma_batched_wrappers.h"

using namespace MLCommon::LinAlg;

namespace hmm {
template <typename T>
void _matrix_powers(magma_int_t n, T** dA_pow_array, magma_int_t max_p,
                    T* dA, magma_int_t ldda, magma_queue_t queue){

        T alpha =1, beta=0;
        dA_pow_array[0] = dA;
        for (size_t p = 1; p < max_p; p++) {
                magma_gemm ( MagmaNoTrans,
                             MagmaNoTrans,
                             n,
                             n,
                             n,
                             alpha,
                             dA_pow_array[i-1],
                             ldda,
                             dA,
                             ldda,
                             beta,
                             dA_pow_array[i],
                             ldda,
                             queue
                             )
        }
}


template <typename T>
__device__
T safe_prod(T x, T y){
        return std::exp(std::log(x) + std::log(y));
}

template <typename T>
__global__
void cumprodKernel(){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        T cum_prod = 1;

        for (size_t i = i_start; i < n_seq; i+=numThreads_x) {
                for (size_t j = 0; j < count; j+=numThreads_y) {
                        for (size_t j = 0; j < dlenghts[i]; j++) {

                                dO_array[k][IDX(i, j, lddo)] = safe_sum(dO_array[k][IDX(i, j-1, lddo)],
                                                                        dA[IDX(i, j-1, lddo)]);
                        }
                }

        }
}

template <typename T>
void _compute_cumprod(magma_int_t m,
                      T** dO_array, magma_int_t lddo,
                      T* dA, magma_int_t ldda,
                      magma_int_t *dlenghts, magma_int_t n_seq,
                      bool isForward){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)m, (int)block.x),
                  ceildiv((int)n_seq, (int)block.y),
                  1);

        int numThreads_x = grid.x * block.x;
        int numThreads_y = grid.y * block.y;

        cumprodKernel<T> <<< grid, block >>>();
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaFree(sums));


}

template <typename T>
void _compute_states_distributions(){
        T **dT_batches, **dSDists_batches;

        // Create the batches
        int curIdx = 0;
        for (size_t seqId = 0; seqId < nSeq; seqId++) {
                for (size_t obsId = 0; obsId < lenghts[seqId]; obsId++) {
                        dT_batches[curIdx] = dT_array[obsId];
                        dSDists_batches = dStateDists_array[seqId][IDX(0, obsId, lddsdist)];
                        curIdx+=1;
                }
        }

        // Compute the states distributions
        T alpha =1, beta=0;
        magmablas_cgemv_batched (MagmaNoTrans,
                                 hmm.nStates,
                                 hmm.nStates,
                                 alpha,
                                 dT_batches,
                                 lddt,
                                 dSDists_batches,
                                 1,
                                 beta,
                                 dSDists_batches,
                                 1,
                                 magma_int_t batchCount,
                                 magma_queue_t queue
                                 )
}

}

template <typename T>
__global__
void computeStateDistsKernel(){
        int seqId_start = threadIdx.x + blockDim.x * blockIdx.x;
        int dimId_start = threadIdx.y + blockDim.y * blockIdx.y;
        T cum_prod = 1;

        for (size_t seqId = seqId_start; seqId < nSeq; seqId+=numThreads_y) {
                for (size_t dimId = dimId_start; dimId < nDim; dimId+=numThreads_x) {
                        if (isForward) {
                                for (size_t obsId = 0; obsId < lenghts[seqId]; obsId++) {
                                        cum_prod = safe_prod(cum_prod, dA[IDX(i, j-1, lddo)]);
                                        dO_array[seqId][IDX(dimId, obsId, lddo)] = cum_prod * dB_array[seqId][IDX(dimId, obsId, lddo)];
                                }
                        }

                }
        }
}

template <typename T>
void _compute_state_dists(magma_int_t m,
                          T** dO_array, magma_int_t lddo,
                          T* dA, magma_int_t ldda,
                          magma_int_t *dlenghts, magma_int_t n_seq,
                          bool isForward){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)m, (int)block.x),
                  ceildiv((int)n_seq, (int)block.y),
                  1);

        int numThreads_x = grid.x * block.x;
        int numThreads_y = grid.y * block.y;

        cumprodKernel<T> <<< grid, block >>>();
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaFree(sums));


}

}
