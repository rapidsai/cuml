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
#include <algorithm>
#include "gini.cuh"
#include "../algo_helper.h"

template<typename T>
__global__ void flag_kernel(T* column, char* leftflag, char* rightflag, const int nrows,
			    T * d_ques_min, T * d_ques_max, const int ques_nbins, const int ques_batch_id,
			    T * ques_val) {

	T ques_max = *d_ques_max;
	T ques_min = *d_ques_min;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nrows) {

		char lflag, rflag;
		T data = column[tid];
		T delta = (ques_max - ques_min) / ques_nbins;
		T ques_base_val = ques_min + delta;
		T local_ques_val = ques_base_val + ques_batch_id * delta;

		if (data <= local_ques_val) {
			lflag = 1;
			rflag = 0;
		}
		else {
			lflag = 0;
			rflag = 1;
		}
		leftflag[tid] = lflag;
		rightflag[tid] = rflag;

		if (tid == 0)
			ques_val[0] = local_ques_val;

	}
	return;
}

template<typename T>
__global__ void flag_kernel_quantile(T* column, char* leftflag, char* rightflag, const int nrows,
				     const T local_ques_val)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nrows) {

		char lflag, rflag;
		T data = column[tid];
		if (data <= local_ques_val) {

			lflag = 1;
			rflag = 0;
		}
		else {

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

template<typename T, typename L>
void make_split(T *column, MetricQuestion<T> & ques, const int nrows, int& nrowsleft, int& nrowsright, unsigned int* rowids, int split_algo, const std::shared_ptr<TemporaryMemory<T, L>> tempmem)
{

	unsigned int *temprowids = tempmem->temprowids->data();
	char *d_flags_left = tempmem->d_flags_left->data();
	char *d_flags_right = tempmem->d_flags_right->data();
	T *question_value = tempmem->question_value->data();

	if (split_algo != ML::SPLIT_ALGO::HIST) {
		flag_kernel_quantile<<< MLCommon::ceildiv(nrows, 128), 128, 0, tempmem->stream >>>(column, d_flags_left, d_flags_right, nrows, ques.value);
	} else {
		flag_kernel<<< MLCommon::ceildiv(nrows, 128), 128, 0, tempmem->stream >>>(column, d_flags_left, d_flags_right, nrows, &tempmem->d_globalminmax->data()[ques.bootstrapped_column], &tempmem->d_globalminmax->data()[ques.bootstrapped_column + ques.ncols], ques.nbins, ques.batch_id, question_value);
	}
	CUDA_CHECK(cudaGetLastError());

	void *d_temp_storage = tempmem->d_split_temp_storage->data();
	size_t temp_storage_bytes = tempmem->split_temp_storage_bytes;
	int *nrowsleftright = tempmem->nrowsleftright->data();
	int *d_num_selected_out = tempmem->d_num_selected_out->data();


	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, rowids, d_flags_left, temprowids, d_num_selected_out, nrows, tempmem->stream);
	MLCommon::updateHost(&nrowsleftright[0], d_num_selected_out, 1, tempmem->stream);
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

	nrowsleft = nrowsleftright[0];
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, rowids, d_flags_right, &temprowids[nrowsleft], d_num_selected_out, nrows, tempmem->stream);
	MLCommon::updateHost(&nrowsleftright[1], d_num_selected_out, 1, tempmem->stream);
	MLCommon::copyAsync(rowids, temprowids, nrows, tempmem->stream);
	
	// Copy GPU-computed question value to tree node.
	if (split_algo == ML::SPLIT_ALGO::HIST) {
		MLCommon::updateHost(&(ques.value), question_value, 1, tempmem->stream);
	}

	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	nrowsright = nrowsleftright[1];
	return;
}
