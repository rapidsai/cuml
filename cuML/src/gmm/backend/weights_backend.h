#pragma once

#include <cuda.h>

namespace gmm {

template <typename T>
__global__
void _update_pis_kernel(size_t nObs, size_t nCl,
                        T *dPis, size_t lddPis,
                        T *dLlhd, size_t lddLlhd,
                        T epsilon,
                        int numThreads_x){
        int start = threadIdx.x + blockDim.x * blockIdx.x;
        T sum;

        for (size_t k = start; k < nCl; k+=numThreads_x) {
                sum = 0;
                for (size_t j = 0; j < nObs; j++) {
                        // printf("%f\n", (float) dLlhd[IDX(k, j, lddLlhd)]);
                        sum += dLlhd[IDX(k, j, lddLlhd)];
                }
                dPis[k] = sum / nObs;
        }

}

template <typename T>
void _update_pis(size_t nObs, size_t nCl,
                 T *dPis, size_t lddPis,
                 T *dLlhd, size_t lddLlhd,
                 cudaStream_t stream = 0)
{
        dim3 block(32);
        dim3 grid = (ceildiv((int)nCl, (int)block.x));

        int numThreads_x = grid.x * block.x;

        T epsilon=0;
        // if(std::is_same<T,float>::value) {
        //         epsilon = 1.1920928955078125e-06;
        // }
        // else if(std::is_same<T,double>::value) {
        //         epsilon = 2.220446049250313e-15;
        // }


        _update_pis_kernel<T> <<< grid, block, 0, stream >>>(nObs, nCl,
                                                             dPis, lddPis,
                                                             dLlhd, lddLlhd,
                                                             epsilon,
                                                             numThreads_x);
        CUDA_CHECK(cudaPeekAtLastError());
}

}
