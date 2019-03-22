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
#include "../memory.cuh"
//#include <map>
#include <vector>

struct GiniQuestion {
	int column;
	float value;

	/* 
	   delta = (max - min) /nbins
	   base_ques_val = min + delta
	   value = base_ques_val + batch_id * delta.

	Due to fp computation differences between GPU and CPU, we need to ensure
	the question value is always computed on the GPU. Otherwise, the flag_kernel
	called via make_split would make a split that'd be inconsistent with the one that
	produced the histograms during the gini computation. This issue arises when there is
	a data value close to the question that gets split differently in gini than in
	flag_kernel.
	*/
	int batch_id;
	float min, max;
	int nbins;
	
	void set_question_fields(int cfg_column, int cfg_batch_id, float cfg_min, float cfg_max, int cfg_nbins) {
		column = cfg_column;
		batch_id = cfg_batch_id;
		min = cfg_min;
		max = cfg_max;
		nbins = cfg_nbins;
		value = 0.0f; // Will be udpate in make_split
	};
};

struct GiniInfo {
	float best_gini = -1.0f;
	//std::map<int, int> hist; 
	std::vector<int> hist; //Element hist[i] stores # labels with label i for a given node.

};
__global__ void gini_kernel(const int* __restrict__ labels,const int nrows, const int nmax, int* histout)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ unsigned int shmemhist[];
	if(threadIdx.x < nmax)
		shmemhist[threadIdx.x] = 0;
	
	__syncthreads();
	
	if(tid < nrows)
		{
			int label = labels[tid];
			atomicAdd(&shmemhist[label],1);			  
		}
	
	__syncthreads();
	
	if(threadIdx.x < nmax)
		atomicAdd(&histout[threadIdx.x],shmemhist[threadIdx.x]);
	
	return;
}
void gini(int *labels_in, const int nrows, const TemporaryMemory* tempmem, GiniInfo & split_info, int & unique_labels, const cudaStream_t stream = 0)
{
	int *dhist = tempmem->d_hist;
	int *hhist = tempmem->h_hist;
	float gval =1.0;
	
	CUDA_CHECK(cudaMemsetAsync(dhist,0,sizeof(int)*unique_labels,tempmem->stream));
	gini_kernel<<< (int)(nrows/128) + 1, 128,sizeof(int)*unique_labels,tempmem->stream>>>(labels_in,nrows,unique_labels,dhist);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpyAsync(hhist,dhist,sizeof(int)*unique_labels,cudaMemcpyDeviceToHost,tempmem->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));	   
	
	split_info.hist.resize(unique_labels, 0);
	for(int i=0; i < unique_labels; i++) {
		split_info.hist[i] = hhist[i]; //update_gini_hist
		float prob = ((float)hhist[i]) / nrows;
		gval -= prob*prob; 
	}
	
	split_info.best_gini = gval; //Update gini val

	return;
}

/* Compute gini info from parent and left node histograms. On CPU for now. */ 
void gini_right_node(const int nrows, GiniInfo & parent_info, GiniInfo & left_node_info, GiniInfo & right_node_info, int & unique_labels, const cudaStream_t stream = 0) {

	float gval = 1.0;
	right_node_info.hist.resize(unique_labels, 0);
	for (int i = 0; i < unique_labels; i++) {
	    right_node_info.hist[i] = parent_info.hist[i] - left_node_info.hist[i];	
		float prob = ((float) right_node_info.hist[i]) / nrows;
		gval -= prob*prob; 
	}
	right_node_info.best_gini = gval;
	return;
}

