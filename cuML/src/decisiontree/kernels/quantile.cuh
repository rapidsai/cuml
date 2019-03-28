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
template <class type>
__global__ void get_sampled_column_kernel(const type* __restrict__ column, type *outcolumn, const unsigned int* __restrict__ rowids, const int N) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		int index = rowids[tid];
		outcolumn[tid] = column[index];
	}
	return;
}

void get_sampled_column(const float *column, float *outcolumn, unsigned int* rowids, const int n_sampled_rows, const cudaStream_t stream = 0) {
	get_sampled_column_kernel<float><<<(int)(n_sampled_rows / 128) + 1, 128, 0, stream>>>(column, outcolumn, rowids, n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}


void get_sampled_labels(const int *labels, int *outlabels, unsigned int* rowids, const int n_sampled_rows, const cudaStream_t stream = 0) {
	get_sampled_column_kernel<int><<<(int)(n_sampled_rows / 128) + 1, 128, 0, stream>>>(labels, outlabels, rowids, n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}

__global__ void get_quantiles(const float* __restrict__ column, float* quantile, const int nrows, const int nbins) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nbins) {		
		int myoff = (int)(nrows/nbins);
		quantile[tid] = column[(tid+1)*myoff - 1];
	}	
	return;
}
void get_sampled_column_quantile(const float *column, float *outcolumn, unsigned int* rowids, const int n_sampled_rows, const int nbins, TemporaryMemory* tempmem) {

	ASSERT(n_sampled_rows != 0, "Column sampling for empty column\n");
	// To protect against an illegal memory access in get_quantiles kernel
	ASSERT(n_sampled_rows >= nbins, "n_sampled_rows %d needs to be >= nbins %d", n_sampled_rows, nbins);
	get_sampled_column(column, outcolumn, rowids, n_sampled_rows, tempmem->stream);
	
	float *temp_sampcol = tempmem->d_temp_sampledcolumn;
	CUDA_CHECK(cudaMemcpyAsync(temp_sampcol, outcolumn, n_sampled_rows*sizeof(float), cudaMemcpyDeviceToDevice, tempmem->stream));
	thrust::sort(thrust::cuda::par.on(tempmem->stream), temp_sampcol, temp_sampcol+n_sampled_rows);
	int threads = 128;
	get_quantiles<<< (int)(nbins/threads) + 1, threads, 0, tempmem->stream >>>(temp_sampcol, tempmem->d_quantile, n_sampled_rows, nbins);
	
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
}
