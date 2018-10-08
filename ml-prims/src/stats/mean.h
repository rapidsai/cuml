#pragma once

#include "cuda_utils.h"
#include "linalg/eltwise.h"
#include <cub/cub.cuh>


namespace MLCommon {
namespace Stats {


///@todo: ColsPerBlk has been tested only for 32!
template <typename Type, int TPB, int ColsPerBlk=32>
__global__ void meanKernelRowMajor(Type* mu, const Type* data, int D, int N) {
    const int RowsPerBlkPerIter = TPB / ColsPerBlk;
    int thisColId = threadIdx.x % ColsPerBlk;
    int thisRowId = threadIdx.x / ColsPerBlk;
    int colId = thisColId + (blockIdx.y * ColsPerBlk);
    int rowId = thisRowId + (blockIdx.x * RowsPerBlkPerIter);
    Type thread_data = Type(0);
    const int stride = RowsPerBlkPerIter * gridDim.x;
    for(int i=rowId;i<N;i+=stride)
        thread_data += (colId < D)? data[i*D+colId] : Type(0);
    __shared__ Type smu[ColsPerBlk];
    if(threadIdx.x < ColsPerBlk)
        smu[threadIdx.x] = Type(0);
    __syncthreads();
    myAtomicAdd(smu+thisColId, thread_data);
    __syncthreads();
    if(threadIdx.x < ColsPerBlk)
        myAtomicAdd(mu+colId, smu[thisColId]);
}

template <typename Type, int TPB>
__global__ void meanKernelColMajor(Type* mu, const Type* data, int D, int N) {
    typedef cub::BlockReduce<Type, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Type thread_data = Type(0);
    int colStart = blockIdx.x * N;
    for(int i=threadIdx.x;i<N;i+=TPB) {
        int idx = colStart + i;
        thread_data += data[idx];
    }
    Type acc = BlockReduce(temp_storage).Sum(thread_data);
    if(threadIdx.x == 0) {
        mu[blockIdx.x] = acc / N;
    }
}

/**
 * @brief Compute mean of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param mu the output mean vector
 * @param data the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample mean or not. In other words, whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 */
template <typename Type>
void mean(Type* mu, const Type* data, int D, int N, bool sample, bool rowMajor) {
    static const int TPB = 256;
    if(rowMajor) {
        static const int RowsPerThread = 4;
        static const int ColsPerBlk = 32;
        static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
        dim3 grid(ceildiv(N,RowsPerBlk), ceildiv(D,ColsPerBlk));
        CUDA_CHECK(cudaMemset(mu, 0, sizeof(Type)*D));
        meanKernelRowMajor<Type,TPB,ColsPerBlk><<<grid,TPB>>>(mu, data, D, N);
        CUDA_CHECK(cudaPeekAtLastError());
        Type ratio = Type(1) / (sample? Type(N-1) : Type(N));
        LinAlg::scalarMultiply(mu, mu, ratio, D);
    } else {
        meanKernelColMajor<Type,TPB><<<D,TPB>>>(mu, data, D, N);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

}; // end namespace Stats
}; // end namespace MLCommon
