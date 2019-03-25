template <typename Tx, typename T, typename D>
void forward_backward(HMM<T, D> &hmm,
                      Tx* dX, int* dlenghts, int nSeq,
                      cublasHandle_t cublasHandle, magma_queue_t queue,
                      bool doForward, bool doBackward){

        _compute_emissions(dX, hmm, cublasHandle);
        _matrix_powers(hmm.nStates, hmm.dT_pows, hmm.max_len,
                       hmm.dT, hmm.lddt, queue);
        _forward_backward(hmm, dlenghts, nSeq, doForward, doBackward);
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
