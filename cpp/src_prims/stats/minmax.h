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
__global__ void minmaxKernel(const T* data, const unsigned int* rowids,
                             const unsigned int* colids, int nrows, int ncols,
                             int row_stride, T* g_min, T* g_max, T* sampledcols,
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
        int index = row + col * row_stride;
        T coldata = data[index];
	//Min max values are saved in shared memory and global memory as per the shuffled colids.
        myAtomicMin(&s_min[(int)(i / nrows)], coldata);
        myAtomicMax(&s_max[(int)(i / nrows)], coldata);
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
 * @brief Computes min/max across every column of the input matrix, as well as
 * optionally allow to subsample based on the given row/col ID mapping vectors
 *
 * @tparam T the data type
 * @tparam TPB number of threads per block
 * @param data input data
 * @param rowids actual row ID mappings. It is of length nrows. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param colids actual col ID mappings. It is of length ncols. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param nrows number of rows of data to be worked upon. The actual rows of the
 * input "data" can be bigger than this!
 * @param ncols number of cols of data to be worked upon. The actual cols of the
 * input "data" can be bigger than this!
 * @param row_stride stride (in number of elements) between 2 adjacent columns
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
void minmax(const T* data, const unsigned int* rowids, const unsigned int* colids, int nrows,
            int ncols, int row_stride, T* globalmin, T* globalmax,
            T* sampledcols, cudaStream_t stream) {
    int nblks = ceildiv(ncols, TPB);
    T init_val = std::numeric_limits<T>::max();
    minmaxInitKernel<T><<<nblks, TPB, 0, stream>>>(ncols, globalmin,
                                                   globalmax, init_val);
    CUDA_CHECK(cudaPeekAtLastError());
    nblks = ceildiv(nrows * ncols, TPB);
    nblks = min(nblks, 65536);
    size_t smemSize = sizeof(T) * 2 * ncols;
    minmaxKernel<T><<<nblks, TPB, smemSize, stream>>>(
        data, rowids, colids, nrows, ncols, row_stride, globalmin, globalmax,
        sampledcols, init_val);
    CUDA_CHECK(cudaPeekAtLastError());
}

}; // end namespace Stats
}; // end namespace MLCommon
