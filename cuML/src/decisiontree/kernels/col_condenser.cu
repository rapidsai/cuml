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

using namespace cub;

//Dummy Operator. Always return true. It's the transformation we care about.
struct Match
{
    __host__ __device__ __forceinline__ Match() {}

    __host__ __device__ __forceinline__
    bool operator()(const unsigned long long &a) const {
		return true;
	}
};


//Return true when the mask of a row matches the node mask for a given depth. 
struct RowMatch {
	unsigned long long node_mask;
	int depth;
	
    __host__ __device__ __forceinline__
    RowMatch(unsigned long long node_mask, int depth) : node_mask(node_mask), depth(depth) {}

	__host__ __device__ __forceinline__
	bool operator()(const unsigned long long & row_mask) const {
		return (((row_mask ^ node_mask) & ((1 << depth)-1)) == 0);
	}	
};


void col_condenser(float * input_data, int * labels, unsigned long long * row_masks, int col_id, 
				   const int n_rows, const int n_cols, int cur_tree_depth, 
				   unsigned long long node_mask, float * condensed_col, int * condensed_labels) {

	/* First step: generate flags memory array using a TransformInputIterator.
	   All rows will be selected, and the flags elements will only be set for the rows that matter. 
	*/

	bool * flags;
	int * n_selected_rows;
	cudaMalloc((void**)&flags, n_rows * sizeof(bool));
	cudaMalloc((void**)&condensed_col, n_rows * sizeof(float));
	cudaMalloc((void**)&condensed_labels, n_rows * sizeof(int));
	cudaMalloc((void**)&n_selected_rows, sizeof(int));
	
	Match select_op;
	RowMatch conversion_op(node_mask, cur_tree_depth); 

	// Assumption: row_masks is a device pointer. 
	cub::TransformInputIterator<bool, RowMatch, unsigned long long *> itr(row_masks, conversion_op);

    void * tmp_storage = NULL;
    size_t tmp_storage_bytes = 0;

    CubDebugExit(DeviceSelect::If(tmp_storage, tmp_storage_bytes, itr, flags, n_selected_rows, n_rows, select_op)); //n_selected_rows will be n_rows

	cudaMalloc(&tmp_storage, tmp_storage_bytes);

    CubDebugExit(DeviceSelect::If(tmp_storage, tmp_storage_bytes, itr,
                 flags, n_selected_rows, n_rows, select_op));
	cudaFree(tmp_storage);


	/* Second step: use the previously generated flags array to condense the col_id column.
	   of input_data.
	*/

	tmp_storage = NULL;
	tmp_storage_bytes = 0;


    CubDebugExit(DeviceSelect::Flagged(tmp_storage, tmp_storage_bytes, &input_data[col_id * n_rows], 
				flags, condensed_col, n_selected_rows, n_rows));

	cudaMalloc(&tmp_storage, tmp_storage_bytes);

    CubDebugExit(DeviceSelect::Flagged(tmp_storage, tmp_storage_bytes, &input_data[col_id * n_rows], 
				flags, condensed_col, n_selected_rows, n_rows));




	/* Final step: Select labels */	

	tmp_storage = NULL;
	tmp_storage_bytes = 0;

    CubDebugExit(DeviceSelect::Flagged(tmp_storage, tmp_storage_bytes, labels,
				flags, condensed_labels, n_selected_rows, n_rows));

	cudaMalloc(&tmp_storage, tmp_storage_bytes);

    CubDebugExit(DeviceSelect::Flagged(tmp_storage, tmp_storage_bytes, labels,
				flags,  condensed_labels, n_selected_rows, n_rows));



	// Cleanup
	cudaFree(tmp_storage);
	cudaFree(flags);
	cudaFree(n_selected_rows);


}

