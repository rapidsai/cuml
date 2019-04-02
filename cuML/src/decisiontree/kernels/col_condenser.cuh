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
#include <thrust/sort.h>
#include "atomic_minmax.h"

/* Merged kernel: gets sampled column and also produces min and max values. */
template<typename T>
__global__ void get_sampled_column_minmax_kernel(const T *column, T *outcolumn, const unsigned int* rowids, T * col_min_max, const int N) {

	__shared__ T shmem_min_max[2];
	T column_val;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {
		int index = rowids[tid];
		column_val = column[index];
		outcolumn[tid] = column_val;

		//  Initialize min max values in shared memory
		if (threadIdx.x == 0) { 
			shmem_min_max[0] = column_val;
			shmem_min_max[1] = column_val;
		}

		// Initialize min max values in global memory. 
		if (tid == 0) {
			col_min_max[0] = column_val;
			col_min_max[1] = column_val;
		}
	}

	__syncthreads();

	// Min - max reduction within each block.
	if (tid < N) {
		atomicMinFD(&shmem_min_max[0], column_val);
		atomicMaxFD(&shmem_min_max[1], column_val);
	}

	__syncthreads();

	// Min - max reduction across blocks.
	if (threadIdx.x == 0) {
		atomicMinFD(&col_min_max[0], shmem_min_max[0]);
		atomicMaxFD(&col_min_max[1], shmem_min_max[1]);
	}
	return;
}

template<typename T>
__global__ void allcolsampler_kernel(const T* __restrict__ data, const unsigned int* __restrict__ rowids, const int* __restrict__ colids, const int nrows, const int ncols, const int rowoffset, T* sampledcols)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (unsigned int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x) {
		int newcolid = (int)(i / nrows);
		int myrowstart;
		if( colids != NULL)
			myrowstart = colids[ newcolid ] * rowoffset;
		else
			myrowstart = newcolid * rowoffset;
		
		int index = rowids[ i % nrows] + myrowstart;
		sampledcols[i] = data[index];
	}
	return;
}

template<typename T>
void get_sampled_column(const T *column, T *outcolumn, unsigned int* rowids, const int n_sampled_rows,  TemporaryMemory<T> * tempmem, const int split_algo) {

	ASSERT(n_sampled_rows != 0, "Column sampling for empty column\n");
	if (split_algo == 0) { // Histograms
		get_sampled_column_minmax_kernel<<<(int)(n_sampled_rows / 128) + 1, 128, 0, tempmem->stream>>>(column, outcolumn, rowids, tempmem->d_min_max, n_sampled_rows);
	} else { //Global Quantile; split_algo should be 2
		get_sampled_column_kernel<<<(int)(n_sampled_rows / 128) + 1, 128, 0, tempmem->stream>>>(column, outcolumn, rowids, n_sampled_rows);
	}
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	return;
}


