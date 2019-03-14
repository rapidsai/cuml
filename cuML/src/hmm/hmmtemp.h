template <typename T>
__global__
void _createForwardBatchesKernel(){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;

        *batchCount = 0;
        for (size_t batchId = 0; batchId < maxBatchCount; batchId+=1) {
                if (len_array[bId] > max_len) {
                        dBatchCount[tau] += 1;
                        dX_batches[bId][tau] = dX + cum_len[bId] + tau * lddx;
                        dAlpha_batches[bId][tau] = dAlpha + IDX(0, bId, lddalpha);
                }
        }

}

template <typename T>
void _create_batches(){

}

template <typename T>
__global__
void _forwardLikelihoodKernel(){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;
        llhd_array[bId] = 1;
        llhd_array[bId] *= std::exp(scaleCoefs[bId]);

}

template <typename T>
__global__
void _forwardKernel(){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        if (i_start == 0 && j_start == 0) {
                for (size_t bId = 0; bId < batchCount; bId++) {
                        // Multiply with Bs elementwise
                        elementwise(dAlpha_array[bId], dB_array[bId], dAlpha_array[bId], nStates);

                        // Update the scale factors
                        T sum = compute_sum(alphas + bId * lddalpha, lddalpha);
                        scaleCoefs[bId] += std::log(sum);

                        // Normalize alphas
                        for (size_t i = 0; i < nStates; i++) {
                                scaleCoefs[bId][i] /= sum;
                        }

                }
        }
}

template <typename T>
void _forward_step_bis(){
        dim3 block(32,32);
        dim3 grid(ceildiv(n, (int)block.x),
                  ceildiv(n, (int)block.y),
                  1);
        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        _forwardKernel<T> <<< grid, block >>>();
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
void _forward_likelihood(){
        // Rescale alphas and compute llhd

        dim3 block(32,32);
        dim3 grid(ceildiv(n, (int)block.x),
                  ceildiv(n, (int)block.y),
                  1);
        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        _forwardLikelihoodKernel<T> <<< grid, block >>>();
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}



void _step(){
        magmablas_dgemv_batched ( MagmaTrans,
                                  n,
                                  n,
                                  1,
                                  dT_batches,
                                  hmm.lddt,
                                  dAlpha[t],
                                  1,
                                  0,
                                  dAlpha_batches[t],
                                  1,
                                  batchCount_array[t],
                                  queue
                                  );
        _forward_step_bis();
}
