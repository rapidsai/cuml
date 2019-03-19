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
#include <utils.h>
#include "cub/cub.cuh"

struct TemporaryMemory
{
	//Below four are for tree building
	int *sampledlabels;
	float *sampledcolumns;

	//Below are for gini & get_class functions
	int *d_hist, *h_hist; // for histograms in gini

	// FIXME none of these mem allocs for gini are currently used.
	int *ginilabels;
	int *d_unique_out, *h_unique_out;      
	int *d_counts_out, *h_counts_out;      
	int *d_num_runs_out, *h_num_runs_out;    
  	void *d_gini_temp_storage = NULL;
	size_t gini_temp_storage_bytes = 0;
	
	//Below pointers are shared for split functions
	char *d_flags_left;
	char *d_flags_right;
	void *d_split_temp_storage = NULL;
	size_t split_temp_storage_bytes = 0;
	int *d_num_selected_out;
	int *temprowids;
	int *h_left_rows, *h_right_rows;
	
	//Total temp mem
	size_t totalmem = 0;

	//CUDA stream
	cudaStream_t stream;

	//For min max;
	float *h_min, *h_max;
	
	TemporaryMemory(int N, int maxstr, int n_unique, int n_bins)
	{

		int n_hist_bytes = n_unique * n_bins * sizeof(int);

		CUDA_CHECK(cudaMallocHost((void**)&h_hist, n_hist_bytes));
		CUDA_CHECK(cudaMalloc((void**)&d_hist, n_hist_bytes));
		CUDA_CHECK(cudaMalloc((void**)&sampledcolumns, N*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void**)&sampledlabels, N*sizeof(int)));
		
		totalmem += N*sizeof(int) + N*sizeof(float) + n_hist_bytes;

		// Allocate temporary storage for gini		
		CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_gini_temp_storage, gini_temp_storage_bytes, ginilabels, d_unique_out, d_counts_out, d_num_runs_out, N));
		
		CUDA_CHECK(cudaMalloc((void**)(&ginilabels), N*sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)(&d_unique_out), N*sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)(&d_counts_out), N*sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)(&d_num_runs_out), sizeof(int)));
		CUDA_CHECK(cudaMalloc(&d_gini_temp_storage, gini_temp_storage_bytes));

		CUDA_CHECK(cudaMallocHost((void**)(&h_unique_out), N*sizeof(int)));
		CUDA_CHECK(cudaMallocHost((void**)(&h_counts_out), N*sizeof(int)));
		CUDA_CHECK(cudaMallocHost((void**)(&h_num_runs_out), sizeof(int)));
		
		totalmem += gini_temp_storage_bytes + sizeof(int) + 3*N*sizeof(int);
		
		//Allocate Temporary for split functions
		cub::DeviceSelect::Flagged(d_split_temp_storage, split_temp_storage_bytes, ginilabels, d_flags_left, ginilabels, d_num_selected_out, N);
		
		CUDA_CHECK(cudaMalloc((void**)&d_split_temp_storage, split_temp_storage_bytes));
		CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&d_flags_left, N*sizeof(char)));
		CUDA_CHECK(cudaMalloc((void**)&d_flags_right, N*sizeof(char)));
		CUDA_CHECK(cudaMalloc((void**)&temprowids, N*sizeof(int)));

		CUDA_CHECK(cudaMallocHost((void**)&h_left_rows, sizeof(int)));
		CUDA_CHECK(cudaMallocHost((void**)&h_right_rows, sizeof(int)));
		
		totalmem += split_temp_storage_bytes + sizeof(int) + N*sizeof(int) + 2*N*sizeof(char);

		//for min max
		CUDA_CHECK(cudaMallocHost((void**)&h_min, sizeof(float)));
		CUDA_CHECK(cudaMallocHost((void**)&h_max, sizeof(float)));
		
		//Create Streams
		if(maxstr == 1)
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
		
		cudaFree(d_hist);
		cudaFreeHost(h_hist);
		cudaFree(ginilabels);
		cudaFree(d_unique_out);
		cudaFree(d_counts_out);
		cudaFree(d_num_runs_out);
		cudaFreeHost(h_unique_out);
		cudaFreeHost(h_counts_out);
		cudaFreeHost(h_num_runs_out);
		cudaFree(d_gini_temp_storage);
		cudaFree(sampledcolumns);
		cudaFree(sampledlabels);
		cudaFree(d_split_temp_storage);
		cudaFree(d_num_selected_out);
		cudaFree(d_flags_left);
		cudaFree(d_flags_right);
		cudaFree(temprowids);
		cudaFreeHost(h_left_rows);
		cudaFreeHost(h_right_rows);
		cudaFreeHost(h_min);
		cudaFreeHost(h_max);
		if(stream != 0)
			cudaStreamDestroy(stream);
	}
	
};
