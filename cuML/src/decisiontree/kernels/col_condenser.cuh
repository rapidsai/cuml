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
#include "cuda_utils.h"

template<typename T>
__global__ void get_sampled_column_kernel(const T* __restrict__ column, T *outcolumn, const unsigned int* __restrict__ rowids, const int N) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		int index = rowids[tid];
		outcolumn[tid] = column[index];
	}
	return;
}

void get_sampled_labels(const int *labels, int *outlabels, unsigned int* rowids, const int n_sampled_rows, const cudaStream_t stream) {
	int threads = 128;
	get_sampled_column_kernel<int><<<MLCommon::ceildiv(n_sampled_rows, threads), threads, 0, stream>>>(labels, outlabels, rowids, n_sampled_rows);
	CUDA_CHECK(cudaGetLastError());
	return;
}

template<typename T>
__global__ void allcolsampler_kernel(const T* __restrict__ data, const unsigned int* __restrict__ rowids, const int* __restrict__ colids, const int nrows, const int ncols, const int rowoffset, T* sampledcols)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (unsigned int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x) {
		int newcolid = (int)(i / nrows);
		int myrowstart;
		if( colids != nullptr)
			myrowstart = colids[ newcolid ] * rowoffset;
		else
			myrowstart = newcolid * rowoffset;

		int index = rowids[ i % nrows] + myrowstart;
		sampledcols[i] = data[index];
	}
	return;
}

template<typename T>
__global__ void allcolsampler_minmax_kernel(const T* __restrict__ data, const unsigned int* __restrict__ rowids, const int* __restrict__ colids, const int nrows, const int ncols, const int rowoffset, T* globalmin, T* globalmax, T* sampledcols, T init_min_val)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];
	T *minshared = (T*)shmem;
	T *maxshared = (T*)(shmem + sizeof(T) * ncols);

	for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
		minshared[i] = init_min_val;
		maxshared[i] = -init_min_val;
	}

	// Initialize min max in  global memory
	if (tid < ncols) {
		globalmin[tid] = init_min_val;
		globalmax[tid] = -init_min_val;
	}

	__syncthreads();

	for (unsigned int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x) {
		int newcolid = (int)(i / nrows);
		int myrowstart = colids[ newcolid ] * rowoffset;
		int index = rowids[ i % nrows] + myrowstart;
		T coldata = data[index];

		atomicMinFD(&minshared[newcolid], coldata);
		atomicMaxFD(&maxshared[newcolid], coldata);
		sampledcols[i] = coldata;
	}

	__syncthreads();

	for (int j = threadIdx.x; j < ncols; j+= blockDim.x) {
		atomicMinFD(&globalmin[j], minshared[j]);
		atomicMaxFD(&globalmax[j], maxshared[j]);
	}

	return;
}

