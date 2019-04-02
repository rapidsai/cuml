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
#include "cub/cub.cuh"
#include "col_condenser.cuh"

template<typename T>
__global__ void get_sampled_column_kernel(const T* __restrict__ column, T *outcolumn, const unsigned int* __restrict__ rowids, const int N) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		int index = rowids[tid];
		outcolumn[tid] = column[index];
	}
	return;
}

template<typename T>
void get_sampled_column(const T *column, T *outcolumn, unsigned int* rowids, const int n_sampled_rows, const cudaStream_t stream = 0) {
	get_sampled_column_kernel<T><<<(int)(n_sampled_rows / 128) + 1, 128, 0, stream>>>(column, outcolumn, rowids, n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}


void get_sampled_labels(const int *labels, int *outlabels, unsigned int* rowids, const int n_sampled_rows, const cudaStream_t stream = 0) {
	get_sampled_column_kernel<int><<<(int)(n_sampled_rows / 128) + 1, 128, 0, stream>>>(labels, outlabels, rowids, n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}

template<typename T>
__global__ void get_quantiles(const T* __restrict__ column, T* quantile, const int nrows, const int nbins) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nbins) {		
		int binoff = (int)(nrows/nbins);
		quantile[tid] = column[ (tid + 1) * binoff - 1];
	}	
	return;
}

template<typename T>
void get_sampled_column_quantile(const T *column, T *outcolumn, unsigned int* rowids, const int n_sampled_rows, const int nbins, TemporaryMemory<T> * tempmem) {

	ASSERT(n_sampled_rows != 0, "Column sampling for empty column\n");
	// To protect against an illegal memory access in get_quantiles kernel
	ASSERT(n_sampled_rows >= nbins, "n_sampled_rows %d needs to be >= nbins %d", n_sampled_rows, nbins);
	get_sampled_column(column, outcolumn, rowids, n_sampled_rows, tempmem->stream);
	
	T *temp_sampcol = tempmem->d_temp_sampledcolumn;
	CUDA_CHECK(cudaMemcpyAsync(temp_sampcol, outcolumn, n_sampled_rows*sizeof(T), cudaMemcpyDeviceToDevice, tempmem->stream));
	thrust::sort(thrust::cuda::par.on(tempmem->stream), temp_sampcol, temp_sampcol+n_sampled_rows);
	int threads = 128;
	get_quantiles<<< (int)(nbins/threads) + 1, threads, 0, tempmem->stream >>>(temp_sampcol, tempmem->d_quantile, n_sampled_rows, nbins);
	
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	return;
}

__global__ void set_sorting_offset(const int nrows, const int ncols, int* offsets) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid <= ncols)
		offsets[tid] = tid*nrows;

	return;
}

template<typename T>
__global__ void get_all_quantiles(const T* __restrict__ data, T* quantile, const int nrows, const int ncols, const int nbins) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nbins*ncols) {		
		int binoff = (int)(nrows/nbins);
		int coloff = (int)(tid/nbins) * nrows;
		quantile[tid] = data[ ( (tid%nbins) + 1 ) * binoff - 1 + coloff];
	}	
	return;
}
	
template<typename T>
void preprocess_quantile(const T* data, const unsigned int* rowids, const int n_sampled_rows, const int ncols, const int rowoffset, const int nbins, TemporaryMemory<T> * tempmem) {

	int threads = 128;
	int  num_items = n_sampled_rows * ncols; // number of items to sort across all segments (i.e., cols)
	int  num_segments = ncols;
	int  *d_offsets;         
	T  *d_keys_in = tempmem->temp_data;         
	T  *d_keys_out;        
	int *colids = NULL;

	CUDA_CHECK(cudaMalloc((void**)&d_offsets, (num_segments + 1) * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_keys_out, num_items * sizeof(T)));
	
	int blocks = (int) ( (ncols * n_sampled_rows) / threads) + 1;
	allcolsampler_kernel<<< blocks , threads, 0, tempmem->stream >>>( data, rowids, colids, n_sampled_rows, ncols, rowoffset, d_keys_in);
	blocks = (int)((ncols+1)/threads) + 1;
	set_sorting_offset<<< blocks, threads, 0, tempmem->stream >>>(n_sampled_rows, ncols, d_offsets);

	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
						num_items, num_segments, d_offsets, d_offsets + 1, 0, 8*sizeof(T), tempmem->stream));

	// Allocate temporary storage
	CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	// Run sorting operation
	CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
						num_items, num_segments, d_offsets, d_offsets + 1, 0, 8*sizeof(T), tempmem->stream));

	blocks = (int)( (ncols*nbins) / threads) + 1;
	get_all_quantiles<<< blocks, threads, 0, tempmem->stream >>>( d_keys_out, tempmem->d_quantile, n_sampled_rows, ncols, nbins);
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	CUDA_CHECK(cudaFree(d_keys_out));
	CUDA_CHECK(cudaFree(d_offsets));
	CUDA_CHECK(cudaFree(d_temp_storage));

	return;
}
