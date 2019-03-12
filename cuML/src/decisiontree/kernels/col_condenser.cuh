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
__global__ void get_sampled_column_kernel(const type *column,type *outcolumn,unsigned int* rowids,const int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		{
			int index = rowids[tid];
			outcolumn[tid] = column[index];
		}
	return;
}
void get_sampled_column(const float *column,float *outcolumn,unsigned int* rowids,const int n_sampled_rows,const cudaStream_t stream = 0)
{
	get_sampled_column_kernel<float><<<(int)(n_sampled_rows / 128) + 1,128,0,stream>>>(column,outcolumn,rowids,n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}
void get_sampled_labels(const int *labels,int *outlabels,unsigned int* rowids,const int n_sampled_rows,const cudaStream_t stream = 0)
{
	get_sampled_column_kernel<int><<<(int)(n_sampled_rows / 128) + 1,128,0,stream>>>(labels,outlabels,rowids,n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}
