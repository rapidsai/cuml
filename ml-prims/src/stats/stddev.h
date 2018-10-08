#pragma once

#include "cuda_utils.h"
#include <cub/cub.cuh>
#include "linalg/binary_op.h"


namespace MLCommon {
namespace Stats {


///@todo: ColPerBlk has been tested only for 32!
template <typename Type, int TPB, int ColsPerBlk=32>
__global__ void stddevKernelRowMajor(Type* std, const Type* data, int D, int N) {
    const int RowsPerBlkPerIter = TPB / ColsPerBlk;
    int thisColId = threadIdx.x % ColsPerBlk;
    int thisRowId = threadIdx.x / ColsPerBlk;
    int colId = thisColId + (blockIdx.y * ColsPerBlk);
    int rowId = thisRowId + (blockIdx.x * RowsPerBlkPerIter);
    Type thread_data = Type(0);
    const int stride = RowsPerBlkPerIter * gridDim.x;
    for(int i=rowId;i<N;i+=stride) {
        Type val = (colId < D)? data[i*D+colId] : Type(0);
        thread_data += val * val;
    }
    __shared__ Type sstd[ColsPerBlk];
    if(threadIdx.x < ColsPerBlk)
        sstd[threadIdx.x] = Type(0);
    __syncthreads();
    myAtomicAdd(sstd+thisColId, thread_data);
    __syncthreads();
    if(threadIdx.x < ColsPerBlk)
        myAtomicAdd(std+colId, sstd[thisColId]);
}

template <typename Type, int TPB>
__global__ void stddevKernelColMajor(Type* std, const Type* data, const Type* mu,
                                     int D, int N) {
    typedef cub::BlockReduce<Type, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Type thread_data = Type(0);
    int colStart = blockIdx.x * N;
    Type m = mu[blockIdx.x];
    for(int i=threadIdx.x;i<N;i+=TPB) {
        int idx = colStart + i;
        Type diff = data[idx] - m;
        thread_data += diff * diff;
    }
    Type acc = BlockReduce(temp_storage).Sum(thread_data);
    if(threadIdx.x == 0) {
        std[blockIdx.x] = mySqrt(acc / N);
    }
}

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param std the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words, whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 */
template <typename Type>
void stddev(Type* std, const Type* data, const Type* mu, int D, int N,
            bool sample, bool rowMajor) {
    static const int TPB = 256;
    if(rowMajor) {
        static const int RowsPerThread = 4;
        static const int ColsPerBlk = 32;
        static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
        dim3 grid(ceildiv(N,RowsPerBlk), ceildiv(D,ColsPerBlk));
        CUDA_CHECK(cudaMemset(std, 0, sizeof(Type)*D));
        stddevKernelRowMajor<Type,TPB,ColsPerBlk><<<grid,TPB>>>(std, data, D, N);
        Type ratio = Type(1) / (sample? Type(N-1) : Type(N));
        LinAlg::binaryOp(std, std, mu, D,
                         [ratio] __device__ (Type a, Type b) {
                             return mySqrt(a * ratio - b * b);
                         });
    } else {
        stddevKernelColMajor<Type,TPB><<<D,TPB>>>(std, data, mu, D, N);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

}; // end namespace Stats
}; // end namespace MLCommon
