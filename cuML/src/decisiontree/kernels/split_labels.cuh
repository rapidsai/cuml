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
#include <thrust/sort.h>
#include <algorithm>
#include "gini.cuh"

__global__ void flag_kernel(float* column, char* leftflag, char* rightflag, const int nrows,
			    const float ques_min, const float ques_max, const int ques_nbins, const int ques_batch_id,
			    float * ques_val)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nrows)
		{
			char lflag, rflag;
			float data = column[tid];
			float delta = (ques_max - ques_min) / ques_nbins;
			float ques_base_val = ques_min + delta;
			float local_ques_val = ques_base_val + ques_batch_id * delta;
			
			if (data <= local_ques_val)
				{
					lflag = 1;
					rflag = 0;
				}
			else
				{
					lflag = 0;
					rflag = 1;
				}
			leftflag[tid] = lflag;
			rightflag[tid] = rflag;
			
			if(tid == 0)
				ques_val[0] = local_ques_val;
			
		}
	return;
}
/* node_hist[i] holds the # times label i appear in current data. The vector is computed during gini
   computation. */
int get_class_hist(std::vector<int> & node_hist) {

	int classval =  std::max_element(node_hist.begin(), node_hist.end()) - node_hist.begin();
	return classval;
}

void make_split(float *column, GiniQuestion & ques, const int nrows, int& nrowsleft, int& nrowsright, unsigned int* rowids, const TemporaryMemory* tempmem)
{

	int *temprowids = tempmem->temprowids;
	char *d_flags_left = tempmem->d_flags_left;
	char *d_flags_right = tempmem->d_flags_right;
	float *question_value = tempmem->question_value;
	
	flag_kernel<<< (int)(nrows/128) + 1, 128>>>(column, d_flags_left, d_flags_right, nrows, ques.min, ques.max, ques.nbins, ques.batch_id, question_value);
	CUDA_CHECK(cudaGetLastError());

	void *d_temp_storage = tempmem->d_split_temp_storage;
	size_t temp_storage_bytes = tempmem->split_temp_storage_bytes;
	
	int *d_num_selected_out = tempmem->d_num_selected_out;

	
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, rowids, d_flags_left, temprowids, d_num_selected_out, nrows);
	
	CUDA_CHECK(cudaMemcpy(&nrowsleft, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost));
	
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, rowids, d_flags_right, &temprowids[nrowsleft], d_num_selected_out, nrows);
	
	CUDA_CHECK(cudaMemcpy(&nrowsright, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(rowids, temprowids, nrows*sizeof(int), cudaMemcpyDeviceToDevice));

	// Copy GPU-computed question value to tree node.
	CUDA_CHECK(cudaMemcpy(&(ques.value), question_value, sizeof(float), cudaMemcpyDeviceToHost));

	return;
}
