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


/* 

Assumptions: 
	- There will be a 64-bit mask (row_mask)  associated with each row of a dataset per tree. This value can be changed atomically (as different nodes in that same tree are built in //).
    - A 64-bit mask will allow us to go up to 64-levels deep which seems to be sufficient (good enough)  according to our SKL-rf-def experiments.

Inputs:
	- data points to entire dataset in col major format.
	- cur_tree_depth: an int that tells us to look at the least significant (cur_tree_depth - 1) bits of the row_mask. 
		=> So, if we're about to build a child of the root, we'll consider all rows in the bootstrapped sample.
		=> If we're about to build a child of the left child of the root, we'll consider all rows with the least significant bit set to 0. 
	- node_mask: the identifier of the parent node. We only care for rows where the first (cur_tree_depth - 1) bits of the node match with this node_mask identifier
		
	- n_rows the original number of rows in the dataset.
	- col: the column we care about


Output:
	- condensed column to look up that includes only the relevant rows. 
	- condensed labels 

*/


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
void get_sampled_column(const float *column,float *outcolumn,unsigned int* rowids,const int n_sampled_rows)
{
	get_sampled_column_kernel<float><<<(int)(n_sampled_rows / 128) + 1,128>>>(column,outcolumn,rowids,n_sampled_rows);
	CUDA_CHECK(cudaDeviceSynchronize());
	return;
}
void get_sampled_labels(const int *labels,int *outlabels,unsigned int* rowids,const int n_sampled_rows)
{
	get_sampled_column_kernel<int><<<(int)(n_sampled_rows / 128) + 1,128>>>(labels,outlabels,rowids,n_sampled_rows);
	CUDA_CHECK(cudaDeviceSynchronize());
	return;
}
