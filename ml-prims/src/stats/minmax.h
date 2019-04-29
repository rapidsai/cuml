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

#include "cuda_utils.h"
#include <limits>


namespace MLCommon {
namespace Stats {

///@todo: implement a proper "fill" kernel
template <typename T>
__global__ void minmaxInitKernel(int ncols, T* globalmin, T* globalmax,
                                 T init_val) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= ncols)
        return;
    globalmin[tid] = init_val;
    globalmax[tid] = -init_val;
}

template <typename T>
__global__ void minmaxKernel(const T* data, const int* rowids,
                             const int* colids, int nrows, int ncols,
                             T* g_min, T* g_max, T* sampledcols,
                             T init_min_val) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ char shmem[];
    T *s_min = (T*)shmem;
    T *s_max = (T*)(shmem + sizeof(T) * ncols);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        s_min[i] = init_min_val;
        s_max[i] = -init_min_val;
    }
    __syncthreads();
    for (int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x) {
        int col = i / nrows;
        int row = i % nrows;
        if(colids != nullptr) {
            col = colids[col];
        }
        if(rowids != nullptr) {
            row = rowids[row];
        }
        if(row < 0 || col < 0) {
            continue;
        }
        int index = row + col * nrows;
        T coldata = data[index];
        myAtomicMin(&s_min[col], coldata);
        myAtomicMax(&s_max[col], coldata);
        if(sampledcols != nullptr) {
            sampledcols[i] = coldata;
        }
    }
    __syncthreads();
    // finally, perform global mem atomics
    for (int j = threadIdx.x; j < ncols; j+= blockDim.x) {
        myAtomicMin(&g_min[j], s_min[j]);
        myAtomicMax(&g_max[j], s_max[j]);
    }
}

/**
 * @brief Computes min/max across every column of the input matrix.
 *
 * @tparam T the data type
 * @tparam TPB number of threads per block
 * @param data input data
 * @param rowids actual row ID mappings. If you need to skip a row, pass a -1
 * at that location. If you want to skip this map lookup entirely, pass nullptr
 * @param colids actual col ID mappings. If you need to skip a col, pass a -1
 * at that location. If you want to skip this map lookup entirely, pass nullptr
 * @param nrows number of rows of data
 * @param ncols number of cols of data
 * @param globalmin final col-wise global minimum (size = ncols)
 * @param globalmax final col-wise global maximum (size = ncols)
 * @param sampledcols output sampled data. Pass nullptr if you don't need this
 * @param init_val initial minimum value to be 
 * @param stream: cuda stream
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
template <typename T, int TPB = 512>
void minmax(const T* data, const int* rowids, const int* colids, int nrows,
            int ncols, T* globalmin, T* globalmax, T* sampledcols,
            cudaStream_t stream) {
    int nblks = ceildiv(ncols, TPB);
    T init_val = std::numeric_limits<T>::max();
    minmaxInitKernel<T><<<nblks, TPB, 0, stream>>>(ncols, globalmin,
                                                   globalmax, init_val);
    CUDA_CHECK(cudaPeekAtLastError());
    nblks = ceildiv(nrows * ncols, TPB);
    nblks = max(nblks, 65536);
    size_t smemSize = sizeof(T) * 2 * ncols;
    minmaxKernel<T><<<nblks, TPB, smemSize, stream>>>(
        data, rowids, colids, nrows, ncols, globalmin, globalmax, sampledcols,
        init_val);
    CUDA_CHECK(cudaPeekAtLastError());
}

}; // end namespace Stats
}; // end namespace MLCommon
