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

__global__ void flag_kernel(float* column,char* leftflag,char* rightflag,float quesval,const int nrows)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < nrows)
		{
			char lflag,rflag;
			float data = column[tid];
			if(data <= quesval)
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
			
		}
	return;
}
/* node_hist[i] holds the # times label i appear in current data. The vector is computed during gini
   computation. */
int get_class_hist(std::vector<int> & node_hist) {

	int classval =  std::max_element(node_hist.begin(), node_hist.end()) - node_hist.begin();
	return classval;
}

void make_split(float *column,const float quesval,const int nrows,int& nrowsleft,int& nrowsright,unsigned int* rowids, const TemporaryMemory* tempmem)
{

	int *temprowids = tempmem->temprowids;
	char *d_flags_left = tempmem->d_flags_left;
	char *d_flags_right = tempmem->d_flags_right;
	
	flag_kernel<<< (int)(nrows/128) + 1,128>>>(column,d_flags_left,d_flags_right,quesval,nrows);
	CUDA_CHECK(cudaGetLastError());

	void *d_temp_storage = tempmem->d_split_temp_storage;
	size_t temp_storage_bytes = tempmem->split_temp_storage_bytes;
	
	int *d_num_selected_out = tempmem->d_num_selected_out;

	
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, rowids, d_flags_left, temprowids,d_num_selected_out, nrows);
	
	CUDA_CHECK(cudaMemcpy(&nrowsleft,d_num_selected_out,sizeof(int),cudaMemcpyDeviceToHost));
	
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, rowids, d_flags_right, &temprowids[nrowsleft],d_num_selected_out, nrows);
	
	CUDA_CHECK(cudaMemcpy(&nrowsright,d_num_selected_out,sizeof(int),cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(rowids,temprowids,nrows*sizeof(int),cudaMemcpyDeviceToDevice));

	return;
}
