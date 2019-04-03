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
#include "atomic_minmax.h"
#include "col_condenser.cuh"
#include <float.h>

/* 
   The output of the function is a histogram array, of size ncols * nbins * n_unique_lables
   column order is as per colids (bootstrapped random cols) for each col there are nbins histograms
 */
template<typename T>
__global__ void all_cols_histograms_kernel(const T* __restrict__ data, const int* __restrict__ labels, const unsigned int* __restrict__ rowids, const int* __restrict__ colids, const int nbins, const int nrows, const int ncols, const int rowoffset, const int n_unique_labels, const T* __restrict__ globalminmax, int* histout) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];
	T *minmaxshared = (T*)shmem;
	int *shmemhist = (int*)(shmem + 2*ncols*sizeof(T));

	for (int i=threadIdx.x; i < 2*ncols; i += blockDim.x) {
		minmaxshared[i] = globalminmax[i];
	}
	
	for (int i = threadIdx.x; i < n_unique_labels*nbins*ncols; i += blockDim.x) {
		shmemhist[i] = 0;
	}
	
	__syncthreads();

	for (unsigned int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x) {
		int mycolid = (int)( i / nrows);
		int coloffset = mycolid*n_unique_labels*nbins;

		// nbins is # batched bins. Use (batched bins + 1) for delta computation.
		T delta = (minmaxshared[mycolid + ncols] - minmaxshared[mycolid]) / (nbins);
		T base_quesval = minmaxshared[mycolid] + delta;

		T localdata = data[i];
		int label = labels[ rowids[ i % nrows ] ];
		for (int j=0; j < nbins; j++) {
			T quesval = base_quesval + j * delta;
			
			if (localdata <= quesval) {
				atomicAdd(&shmemhist[label + n_unique_labels * j + coloffset], 1);
			}
		}

	}

	__syncthreads();

	for (int i = threadIdx.x; i < ncols*n_unique_labels*nbins; i += blockDim.x) {
		atomicAdd(&histout[i], shmemhist[i]);
	}
}

template<typename T>
__global__ void all_cols_histograms_global_quantile_kernel(const T* __restrict__ data, const int* __restrict__ labels, const unsigned int* __restrict__ rowids, const int* __restrict__ colids, const int nbins, const int nrows, const int ncols, const int rowoffset, const int n_unique_labels, int* histout, const T* __restrict__ quantile) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];
	int *shmemhist = (int*)(shmem);

	for (int i = threadIdx.x; i < n_unique_labels*nbins*ncols; i += blockDim.x) {
		shmemhist[i] = 0;
	}

	__syncthreads();

	for (unsigned int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x) {
		int mycolid = (int)( i / nrows);
		int coloffset = mycolid*n_unique_labels*nbins;

		// nbins is # batched bins.
		T localdata = data[i];
		int label = labels[ rowids[ i % nrows ] ];
		for (int j=0; j < nbins; j++) {
			int quantile_index = colids[mycolid] * nbins + j; //TODO FIXME Is this valid? Confirm if there's any issue w/ bins vs. batch bins
			T quesval = quantile[quantile_index];
			if (localdata <= quesval) {
				atomicAdd(&shmemhist[label + n_unique_labels * j + coloffset], 1);
			}
		}

	}

	__syncthreads();

	for (int i = threadIdx.x; i < ncols*n_unique_labels*nbins; i += blockDim.x) {
		atomicAdd(&histout[i], shmemhist[i]);
	}
}

template<typename T>
void find_best_split(const TemporaryMemory<T> * tempmem, const int nbins, const int n_unique_labels, const std::vector<int>& col_selector, GiniInfo split_info[3], const int nrows, GiniQuestion<T> & ques, float & gain, const int split_algo) {

	gain = 0.0f;
	int best_col_id = -1;
	int best_bin_id = -1;

	int n_cols = col_selector.size();
	for (int col_id = 0; col_id < n_cols; col_id++) {

		int col_hist_base_index = col_id * nbins * n_unique_labels;			
		// tempmem->h_histout holds n_cols histograms of nbins of n_unique_labels each.
		for (int i = 0; i < nbins; i++) {

			// if tmp_lnrows or tmp_rnrows is 0, the corresponding gini will be 1 but that doesn't
			// matter as it won't count in the info_gain computation.
			float tmp_gini_left = 1.0f;
			float tmp_gini_right = 1.0f;
			int tmp_lnrows = 0;

			//separate loop for now to avoid overflow.
			for (int j = 0; j < n_unique_labels; j++) {
				int hist_index = i * n_unique_labels + j;
				tmp_lnrows += tempmem->h_histout[col_hist_base_index + hist_index];
			}
			int tmp_rnrows = nrows - tmp_lnrows;

			if (tmp_lnrows == 0 || tmp_rnrows == 0)
				continue;

			// Compute gini right and gini left value for each bin.
			for (int j = 0; j < n_unique_labels; j++) {
				int hist_index = i * n_unique_labels + j;

				float prob_left = (float) (tempmem->h_histout[col_hist_base_index + hist_index]) / tmp_lnrows;
				tmp_gini_left -= prob_left * prob_left;

				float prob_right = (float) (split_info[0].hist[j] - tempmem->h_histout[col_hist_base_index + hist_index]) / tmp_rnrows;
				tmp_gini_right -=  prob_right * prob_right;
			}

			ASSERT((tmp_gini_left >= 0.0f) && (tmp_gini_left <= 1.0f), "gini left value %f not in [0.0, 1.0]", tmp_gini_left);
			ASSERT((tmp_gini_right >= 0.0f) && (tmp_gini_right <= 1.0f), "gini right value %f not in [0.0, 1.0]", tmp_gini_right);

			float impurity = (tmp_lnrows * 1.0f/nrows) * tmp_gini_left + (tmp_rnrows * 1.0f/nrows) * tmp_gini_right;
			float info_gain = split_info[0].best_gini - impurity;


			// Compute best information col_gain so far
			if (info_gain > gain) {
				gain = info_gain;
				best_bin_id = i;
				best_col_id = col_id;
				split_info[1].best_gini = tmp_gini_left;
				split_info[2].best_gini = tmp_gini_right;
			}
		}
	}
	
	if (best_col_id == -1 || best_bin_id == -1)
		return;
	
	split_info[1].hist.resize(n_unique_labels);
	split_info[2].hist.resize(n_unique_labels);
	for (int j = 0; j < n_unique_labels; j++) {
		split_info[1].hist[j] = tempmem->h_histout[best_col_id * n_unique_labels * nbins + best_bin_id * n_unique_labels + j];
		split_info[2].hist[j] = split_info[0].hist[j] - split_info[1].hist[j];
	}

	if (split_algo == 0) { // HIST
		ques.set_question_fields(best_col_id, col_selector[best_col_id], best_bin_id, nbins, n_cols, set_min_val<T>(), -set_min_val<T>(), (T) 0);
	} else if (split_algo == 2) { // Global quantile
		T ques_val;
		T *d_quantile = tempmem->d_quantile;
		int q_index = col_selector[best_col_id] * nbins  + best_bin_id;
		CUDA_CHECK(cudaMemcpyAsync(&ques_val, &d_quantile[q_index], sizeof(T), cudaMemcpyDeviceToHost, tempmem->stream));
		CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
		ques.set_question_fields(best_col_id, col_selector[best_col_id], best_bin_id, nbins, n_cols, set_min_val<T>(), -set_min_val<T>(), ques_val);
	}
	return;
}


template<typename T>
void best_split_all_cols(const T *data, const unsigned int* rowids, const int *labels, const int nbins, const int nrows, const int n_unique_labels, const int rowoffset, const std::vector<int>& colselector, const TemporaryMemory<T> * tempmem, GiniInfo split_info[3], GiniQuestion<T> & ques, float & gain, const int split_algo)
{
	int* d_colids = tempmem->d_colids;
	T* d_globalminmax = tempmem->d_globalminmax;
	int *d_histout = tempmem->d_histout;
	int *h_histout = tempmem->h_histout;
	
	int ncols = colselector.size();
	int col_minmax_bytes = sizeof(T) * 2 * ncols;
	int n_hist_bytes = n_unique_labels * nbins * sizeof(int) * ncols;

	CUDA_CHECK(cudaMemsetAsync((void*)d_histout, 0, n_hist_bytes, tempmem->stream));
	
	unsigned int threads = 512;
	unsigned int blocks  = (int)((nrows * ncols) / threads) + 1;
	if (blocks > 65536)
		blocks = 65536;
	
	/* Kernel allcolsampler_*_kernel:
		- populates tempmem->tempdata with the sampled column data,
		- and computes min max histograms in tempmem->d_globalminmax *if minmax in name
	   across all columns.
	*/
	size_t shmemsize = col_minmax_bytes;
	if (split_algo == 0) { // Histograms (min, max)
		allcolsampler_minmax_kernel<<<blocks, threads, shmemsize, tempmem->stream>>>(data, rowids, d_colids, nrows, ncols, rowoffset, &d_globalminmax[0], &d_globalminmax[colselector.size()], tempmem->temp_data, set_min_val<T>());
	} else if (split_algo == 2) { // Global quantiles; just col condenser
		allcolsampler_kernel<<<blocks, threads, 0, tempmem->stream>>>(data, rowids, d_colids, nrows, ncols, rowoffset, tempmem->temp_data);
	}
	CUDA_CHECK(cudaGetLastError());
	
	shmemsize = n_hist_bytes;
	
	if (split_algo == 0) {
		shmemsize += col_minmax_bytes;
		all_cols_histograms_kernel<<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data, labels, rowids, d_colids, nbins, nrows, ncols, rowoffset, n_unique_labels, d_globalminmax, d_histout);
	} else if (split_algo == 2) {
		all_cols_histograms_global_quantile_kernel<<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data, labels, rowids, d_colids, nbins, nrows, ncols, rowoffset, n_unique_labels,  d_histout, tempmem->d_quantile);
	}
	CUDA_CHECK(cudaGetLastError());
	
	CUDA_CHECK(cudaMemcpyAsync(h_histout, d_histout, n_hist_bytes, cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream)); //added
	
	find_best_split(tempmem, nbins, n_unique_labels, colselector, &split_info[0], nrows, ques, gain, split_algo);
	return;
}

