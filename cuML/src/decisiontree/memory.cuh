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
#include <utils.h>
#include "cub/cub.cuh"
#include <thrust/extrema.h>

template<class T>
struct TemporaryMemory
{
	// Labels after boostrapping
	int *sampledlabels;

	// Used for gini histograms (root tree node)
	int *d_hist, *h_hist;

	//Host/Device histograms and device minmaxs
	T *d_globalminmax;
	int *h_histout, *d_histout;
	int *d_colids;

	//Below pointers are shared for split functions
	char *d_flags_left;
	char *d_flags_right;
	void *d_split_temp_storage = nullptr;
	size_t split_temp_storage_bytes = 0;
	int *d_num_selected_out;
	int *temprowids;
	T *question_value;
	T *temp_data;

	//Total temp mem
	size_t totalmem = 0;

	//CUDA stream
	cudaStream_t stream;

	//For quantiles
	T *d_quantile = nullptr;
	T *d_temp_sampledcolumn = nullptr;

	TemporaryMemory(int N, int Ncols, int maxstr, int n_unique, int n_bins, const int split_algo)
	{

		int n_hist_bytes = n_unique * n_bins * sizeof(int);

		CUDA_CHECK(cudaMallocHost((void**)&h_hist, n_hist_bytes));
		CUDA_CHECK(cudaMalloc((void**)&d_hist, n_hist_bytes));

		int extra_bytes = Ncols * sizeof(T);
		int quantile_bytes = (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) ? extra_bytes : sizeof(T);

		CUDA_CHECK(cudaMalloc((void**)&temp_data, N * extra_bytes));
		totalmem += n_hist_bytes + N * extra_bytes;

		if (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
			CUDA_CHECK(cudaMalloc((void**)&d_quantile, n_bins * quantile_bytes));
			CUDA_CHECK(cudaMalloc((void**)&d_temp_sampledcolumn, N * extra_bytes));
			totalmem += (n_bins + N) * extra_bytes;
		}

		CUDA_CHECK(cudaMalloc((void**)&sampledlabels, N*sizeof(int)));
		totalmem += N*sizeof(int);

		//Allocate Temporary for split functions
		cub::DeviceSelect::Flagged(d_split_temp_storage, split_temp_storage_bytes, temprowids, d_flags_left, temprowids, d_num_selected_out, N);

		CUDA_CHECK(cudaMalloc((void**)&d_split_temp_storage, split_temp_storage_bytes));
		CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&d_flags_left, N*sizeof(char)));
		CUDA_CHECK(cudaMalloc((void**)&d_flags_right, N*sizeof(char)));
		CUDA_CHECK(cudaMalloc((void**)&temprowids, N*sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&question_value, sizeof(T)));

		totalmem += split_temp_storage_bytes + (N + 1)*sizeof(int) + 2*N*sizeof(char) + sizeof(T);

		CUDA_CHECK(cudaMallocHost((void**)&h_histout, n_hist_bytes * Ncols));

		CUDA_CHECK(cudaMalloc((void**)&d_globalminmax, sizeof(T)*Ncols*2));
		CUDA_CHECK(cudaMalloc((void**)&d_histout, n_hist_bytes * Ncols));
		CUDA_CHECK(cudaMalloc((void**)&d_colids, sizeof(int)*Ncols));
		totalmem += (n_hist_bytes + sizeof(int) + 2*sizeof(T))* Ncols;

		//Create Streams
		if (maxstr == 1)
			stream = 0;
		else
			CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	}

	void print_info()
	{
		std::cout << " Total temporary memory usage--> "<< ((double)totalmem/ (1024*1024)) << "  MB" << std::endl;
		return;
	}

	~TemporaryMemory()
	{
		cudaFreeHost(h_hist);
		cudaFree(d_hist);
		cudaFree(temp_data);

		if (d_quantile != nullptr)
			cudaFree(d_quantile);
		if (d_temp_sampledcolumn != nullptr)
			cudaFree(d_temp_sampledcolumn);

		cudaFree(sampledlabels);
		cudaFree(d_split_temp_storage);
		cudaFree(d_num_selected_out);
		cudaFree(d_flags_left);
		cudaFree(d_flags_right);
		cudaFree(temprowids);
		cudaFree(question_value);
		cudaFreeHost(h_histout);

		cudaFree(d_globalminmax);
		cudaFree(d_histout);
		cudaFree(d_colids);

		if (stream != 0)
			cudaStreamDestroy(stream);
	}

};
