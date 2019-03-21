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
#include <thrust/extrema.h>

struct TemporaryMemory
{
	//Below four are for tree building
	int *sampledlabels;
	float *sampledcolumns;

	//Below are for gini & get_class functions
	int *d_hist, *h_hist; // for histograms in gini
	
	//Below pointers are shared for split functions
	char *d_flags_left;
	char *d_flags_right;
	void *d_split_temp_storage = NULL;
	size_t split_temp_storage_bytes = 0;
	int *d_num_selected_out;
	int *temprowids;
	int *h_left_rows, *h_right_rows;
	float * question_value;
	float *temp_data;
	
	//Total temp mem
	size_t totalmem = 0;

	//CUDA stream
	cudaStream_t stream;

	//For min max;
	thrust::pair<float *, float *> d_min_max; 
	float *d_ques_info, *h_ques_info; //holds delta and base_val
	
	TemporaryMemory(int N, int Ncols, int maxstr, int n_unique, int n_bins)
	{

		int n_hist_bytes = n_unique * n_bins * sizeof(int);

		CUDA_CHECK(cudaMallocHost((void**)&h_hist, n_hist_bytes));
		CUDA_CHECK(cudaMalloc((void**)&d_hist, n_hist_bytes));
		CUDA_CHECK(cudaMalloc((void**)&sampledcolumns, N*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void**)&sampledlabels, N*sizeof(int)));
		
		totalmem += N*sizeof(int) + N*sizeof(float) + n_hist_bytes;

				
		//Allocate Temporary for split functions
		cub::DeviceSelect::Flagged(d_split_temp_storage, split_temp_storage_bytes, temprowids, d_flags_left, temprowids, d_num_selected_out, N);
		
		CUDA_CHECK(cudaMalloc((void**)&d_split_temp_storage, split_temp_storage_bytes));
		CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&d_flags_left, N*sizeof(char)));
		CUDA_CHECK(cudaMalloc((void**)&d_flags_right, N*sizeof(char)));
		CUDA_CHECK(cudaMalloc((void**)&temprowids, N*sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**) &question_value, sizeof(float)));
		CUDA_CHECK(cudaMalloc((void**) &d_ques_info, 2*sizeof(float)));

		CUDA_CHECK(cudaMallocHost((void**)&h_left_rows, sizeof(int)));
		CUDA_CHECK(cudaMallocHost((void**)&h_right_rows, sizeof(int)));
		CUDA_CHECK(cudaMallocHost((void**)&h_ques_info, 2*sizeof(float)));
		
		totalmem += split_temp_storage_bytes + sizeof(int) + N*sizeof(int) + 2*N*sizeof(char) + 3*sizeof(float);

		//For lot of temp data
		CUDA_CHECK(cudaMalloc((void**)&temp_data,sizeof(float)*Ncols*N));
		totalmem += sizeof(float)*Ncols*N;
		
		//for min max
		
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
		cudaFree(temp_data);
		cudaFree(d_hist);
		cudaFreeHost(h_hist);
		cudaFree(sampledcolumns);
		cudaFree(sampledlabels);
		cudaFree(d_split_temp_storage);
		cudaFree(d_num_selected_out);
		cudaFree(d_flags_left);
		cudaFree(d_flags_right);
		cudaFree(temprowids);
		cudaFree(question_value);
		cudaFree(d_ques_info);
		cudaFreeHost(h_left_rows);
		cudaFreeHost(h_right_rows);
		cudaFreeHost(h_ques_info);
		if(stream != 0)
			cudaStreamDestroy(stream);
	}
	
};
