/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "linalg/eltwise.h"
#include <cub/cub.cuh>


namespace MLCommon {
namespace Stats {

using namespace MLCommon;

///@todo: ColsPerBlk has been tested only for 32!
template <typename Type, int TPB, int ColsPerBlk=32>
__global__ void sumKernelRowMajor(Type* mu, const Type* data, int D, int N) {
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
__global__ void sumKernelColMajor(Type* mu, const Type* data, int D, int N) {
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
        mu[blockIdx.x] = acc;
    }
}

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param output the output mean vector
 * @param input the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether the input data is row or col major
 */
template <typename Type>
void sum(Type* output, const Type* input, int D, int N, bool rowMajor, cudaStream_t stream = 0) {
    static const int TPB = 256;
    if(rowMajor) {
        static const int RowsPerThread = 4;
        static const int ColsPerBlk = 32;
        static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
        dim3 grid(ceildiv(N,RowsPerBlk), ceildiv(D,ColsPerBlk));
        CUDA_CHECK(cudaMemset(output, 0, sizeof(Type)*D));
        sumKernelRowMajor<Type,TPB,ColsPerBlk><<<grid,TPB, 0, stream>>>(output, input, D, N);
    } else {
        sumKernelColMajor<Type,TPB><<<D,TPB, 0, stream>>>(output, input, D, N);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param output the output mean vector
 * @param input the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param n_gpu: number of gpus.
 * @param rowMajor whether the input data is row or col major
 * @param row_split: true if the data is broken by row
 * @param sync: synch the streams if it's true
 */
template <typename Type>
void sumMG(TypeMG<Type>* output, const TypeMG<Type>* input, int D, int N,
		   int n_gpus, bool rowMajor, bool row_split = false,
			bool sync = false) {

	if (row_split) {
		ASSERT(false, "sumMG: row split is not supported");
	} else {
		for (int i = 0; i < n_gpus; i++) {
			CUDA_CHECK(cudaSetDevice(input[i].gpu_id));

			sum(output[i].d_data, input[i].d_data, input[i].n_cols, input[i].n_rows,
				rowMajor, input[i].stream);
		}
	}

	if (sync)
        streamSyncMG(input, n_gpus);
}

}; // end namespace Stats
}; // end namespace MLCommon
