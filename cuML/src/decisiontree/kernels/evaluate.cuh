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
#include "gini.cuh"
#include "../memory.cuh"

__global__ void evaluate_kernel(const float* __restrict__ column,const int* __restrict__ labels,const float quesval,const int nrows, const int nmax, int* histout)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ unsigned int shmemhist[];
	if(threadIdx.x < nmax)
		shmemhist[threadIdx.x] = 0;
	
	__syncthreads();
	
	if(tid < nrows)
		{
			float data = column[tid];
			int label = labels[tid];
			if(data <= quesval)
				{
					atomicAdd(&shmemhist[label],1);
				}
						  
		}
	
	__syncthreads();
	
	if(threadIdx.x < nmax)
		atomicAdd(&histout[threadIdx.x],shmemhist[threadIdx.x]);
	
	return;
}

void evaluate_and_leftgini(const float *column,const int *labels,const float quesval,const int nrows,const int n_unique_labels,const float ginibefore,GiniInfo& split_info,int& lnrows,int& rnrows,TemporaryMemory* tempmem)	
{
	int *dhist = tempmem->d_hist;
	int *hhist = tempmem->h_hist;

	CUDA_CHECK(cudaMemsetAsync(dhist,0,sizeof(int)*n_unique_labels,tempmem->stream));
	evaluate_kernel<<< (int)(nrows/128) + 1, 128,sizeof(int)*n_unique_labels,tempmem->stream>>>(column,labels,quesval,nrows,n_unique_labels,dhist);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpyAsync(hhist,dhist,sizeof(int)*n_unique_labels,cudaMemcpyDeviceToHost,tempmem->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));	   
	
	float gval = 1.0;
	lnrows = 0;
	split_info.hist.resize(n_unique_labels, 0);
	
	for(int i=0; i < n_unique_labels; i++) {
		split_info.hist[i] = hhist[i];
		float prob = (float)(split_info.hist[i])  / nrows;
		lnrows += split_info.hist[i];
		gval -= prob*prob;
	}

	rnrows = nrows - lnrows;
	split_info.best_gini = gval; //Update gini val
       
	return;
}

