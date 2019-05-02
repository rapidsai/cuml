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
#include "cub/cub.cuh"
#include "col_condenser.cuh"


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
void preprocess_quantile(const T* data, const unsigned int* rowids, const int n_sampled_rows, const int ncols, const int rowoffset, const int nbins, std::shared_ptr<TemporaryMemory<T>> tempmem) {

	int threads = 128;
	int  num_items = n_sampled_rows * ncols; // number of items to sort across all segments (i.e., cols)
	int  num_segments = ncols;
	MLCommon::device_buffer<int> *d_offsets;
	MLCommon::device_buffer<T> *d_keys_out;
	T  *d_keys_in = tempmem->temp_data->data();
	int *colids = nullptr;

	d_offsets = new MLCommon::device_buffer<int>(tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, num_segments + 1);
	d_keys_out = new MLCommon::device_buffer<T>(tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, num_items);

	int blocks = MLCommon::ceildiv(ncols * n_sampled_rows, threads);
	allcolsampler_kernel<<< blocks , threads, 0, tempmem->stream >>>( data, rowids, colids, n_sampled_rows, ncols, rowoffset, d_keys_in);
	CUDA_CHECK(cudaGetLastError());
	blocks = MLCommon::ceildiv(ncols + 1, threads);
	set_sorting_offset<<< blocks, threads, 0, tempmem->stream >>>(n_sampled_rows, ncols, d_offsets->data());
	CUDA_CHECK(cudaGetLastError());

	// Determine temporary device storage requirements
	MLCommon::device_buffer<char> *d_temp_storage = nullptr;
	size_t   temp_storage_bytes = 0;
	CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out->data(),
						num_items, num_segments, d_offsets->data(), d_offsets->data() + 1, 0, 8*sizeof(T), tempmem->stream));

	// Allocate temporary storage
	d_temp_storage = new MLCommon::device_buffer<char>(tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, temp_storage_bytes);

	// Run sorting operation
	CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys((void *)d_temp_storage->data(), temp_storage_bytes, d_keys_in, d_keys_out->data(),
						num_items, num_segments, d_offsets->data(), d_offsets->data() + 1, 0, 8*sizeof(T), tempmem->stream));

	blocks = MLCommon::ceildiv(ncols * nbins, threads);
	get_all_quantiles<<< blocks, threads, 0, tempmem->stream >>>(d_keys_out->data(), tempmem->d_quantile->data(), n_sampled_rows, ncols, nbins);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

	d_keys_out->release(tempmem->stream);
	d_offsets->release(tempmem->stream);
	d_temp_storage->release(tempmem->stream);
	delete d_keys_out;
	delete d_offsets;
	delete d_temp_storage;

	return;
}
