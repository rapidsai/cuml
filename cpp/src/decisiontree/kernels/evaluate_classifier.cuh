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
#include <limits>
#include "metric.cuh"
#include "../memory.cuh"
#include "col_condenser.cuh"
#include <float.h>
#include "../algo_helper.h"
#include "stats/minmax.h"

/*
   The output of the function is a histogram array, of size ncols * nbins * n_unique_labels
   column order is as per colids (bootstrapped random cols) for each col there are nbins histograms
 */
template<typename T>
__global__ void all_cols_histograms_kernel_class(const T* __restrict__ data, const int* __restrict__ labels, const int nbins, const int nrows, const int ncols, const int n_unique_labels, const T* __restrict__ globalminmax, int* histout) {

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

		T delta = (minmaxshared[mycolid + ncols] - minmaxshared[mycolid]) / (nbins);
		T base_quesval = minmaxshared[mycolid] + delta;

		T localdata = data[i];
		int label = labels[ i % nrows ];
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
__global__ void all_cols_histograms_global_quantile_kernel_class(const T* __restrict__ data, const int* __restrict__ labels, const unsigned int* __restrict__ colids, const int nbins, const int nrows, const int ncols, const int n_unique_labels, int* histout, const T* __restrict__ quantile) {

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

		T localdata = data[i];
		int label = labels[ i % nrows ];
		for (int j=0; j < nbins; j++) {
			int quantile_index = colids[mycolid] * nbins + j;
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

template<typename T, typename L, typename F>
void find_best_split_classifier(const std::shared_ptr<TemporaryMemory<T, L>> tempmem, const int nbins, const int n_unique_labels, const std::vector<unsigned int>& col_selector, MetricInfo<T> split_info[3], const int nrows, MetricQuestion<T> & ques, float & gain, const int split_algo) {

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
			int tmp_lnrows = 0;
			
			//separate loop for now to avoid overflow.
			for (int j = 0; j < n_unique_labels; j++) {
				int hist_index = i * n_unique_labels + j;
				tmp_lnrows += tempmem->h_histout->data()[col_hist_base_index + hist_index];
			}
			int tmp_rnrows = nrows - tmp_lnrows;
			
			if (tmp_lnrows == 0 || tmp_rnrows == 0)
				continue;

			std::vector<int> tmp_histleft(n_unique_labels);
			std::vector<int> tmp_histright(n_unique_labels);
			
			// Compute gini right and gini left value for each bin.
			for (int j = 0; j < n_unique_labels; j++) {
				int hist_index = i * n_unique_labels + j;
				tmp_histleft[j] = tempmem->h_histout->data()[col_hist_base_index + hist_index];
				tmp_histright[j] = split_info[0].hist[j] - tmp_histleft[j];
			}

			float tmp_gini_left = F::exec(tmp_histleft, tmp_lnrows);
			float tmp_gini_right = F::exec(tmp_histright, tmp_rnrows);
			
			ASSERT((tmp_gini_left >= 0.0f) && (tmp_gini_left <= 1.0f), "gini left value %f not in [0.0, 1.0]", tmp_gini_left);
			ASSERT((tmp_gini_right >= 0.0f) && (tmp_gini_right <= 1.0f), "gini right value %f not in [0.0, 1.0]", tmp_gini_right);

			float impurity = (tmp_lnrows * 1.0f/nrows) * tmp_gini_left + (tmp_rnrows * 1.0f/nrows) * tmp_gini_right;
			float info_gain = split_info[0].best_metric - impurity;


			// Compute best information col_gain so far
			if (info_gain > gain) {
				gain = info_gain;
				best_bin_id = i;
				best_col_id = col_id;
				split_info[1].best_metric = tmp_gini_left;
				split_info[2].best_metric = tmp_gini_right;
			}
		}
	}

	if (best_col_id == -1 || best_bin_id == -1)
		return;

	split_info[1].hist.resize(n_unique_labels);
	split_info[2].hist.resize(n_unique_labels);
	for (int j = 0; j < n_unique_labels; j++) {
		split_info[1].hist[j] = tempmem->h_histout->data()[best_col_id * n_unique_labels * nbins + best_bin_id * n_unique_labels + j];
		split_info[2].hist[j] = split_info[0].hist[j] - split_info[1].hist[j];
	}

	if (split_algo == ML::SPLIT_ALGO::HIST) {
		ques.set_question_fields(best_col_id, col_selector[best_col_id], best_bin_id, nbins, n_cols, std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(), (T) 0);
	} else if (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
		T ques_val;
		T *d_quantile = tempmem->d_quantile->data();
		int q_index = col_selector[best_col_id] * nbins  + best_bin_id;
		MLCommon::updateHost(&ques_val, &d_quantile[q_index], 1, tempmem->stream);
		CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
		ques.set_question_fields(best_col_id, col_selector[best_col_id], best_bin_id, nbins, n_cols, std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(), ques_val);
	}
	return;
}


template<typename T, typename L, typename F>
void best_split_all_cols_classifier(const T *data, const unsigned int* rowids, const L *labels, const int nbins, const int nrows, const int n_unique_labels, const int rowoffset, const std::vector<unsigned int>& colselector, const std::shared_ptr<TemporaryMemory<T, L>> tempmem, MetricInfo<T> split_info[3], MetricQuestion<T> & ques, float & gain, const int split_algo)
{
	unsigned int* d_colids = tempmem->d_colids->data();
	T* d_globalminmax = tempmem->d_globalminmax->data();
	int *d_histout = tempmem->d_histout->data();
	int *h_histout = tempmem->h_histout->data();

	int ncols = colselector.size();
	int col_minmax_bytes = sizeof(T) * 2 * ncols;
	int n_hist_elements = n_unique_labels * nbins * ncols;
	int n_hist_bytes = n_hist_elements * sizeof(int);

	CUDA_CHECK(cudaMemsetAsync((void*)d_histout, 0, n_hist_bytes, tempmem->stream));

	const int threads = 512;
	int blocks = MLCommon::ceildiv(nrows * ncols, threads);
	if (blocks > 65536)
		blocks = 65536;

	/* Kernel allcolsampler_*_kernel:
		- populates tempmem->tempdata with the sampled column data,
		- and computes min max histograms in tempmem->d_globalminmax *if minmax in name
	   across all columns.
	*/
	size_t shmemsize = col_minmax_bytes;
	if (split_algo == ML::SPLIT_ALGO::HIST) { // Histograms (min, max)
		MLCommon::Stats::minmax<T, threads>(data, rowids, d_colids, nrows, ncols, rowoffset, &d_globalminmax[0], &d_globalminmax[colselector.size()], tempmem->temp_data->data(), tempmem->stream);
	} else if (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) { // Global quantiles; just col condenser
		allcolsampler_kernel<<<blocks, threads, 0, tempmem->stream>>>(data, rowids, d_colids, nrows, ncols, rowoffset, tempmem->temp_data->data());
	}
	CUDA_CHECK(cudaGetLastError());
	L *labelptr = tempmem->sampledlabels->data();
	get_sampled_labels<L>(labels, labelptr, rowids, nrows, tempmem->stream);

	shmemsize = n_hist_bytes;

	if (split_algo == ML::SPLIT_ALGO::HIST) {
		shmemsize += col_minmax_bytes;
		all_cols_histograms_kernel_class<<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data->data(), labelptr, nbins, nrows, ncols, n_unique_labels, d_globalminmax, d_histout);
	} else if (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
		all_cols_histograms_global_quantile_kernel_class<<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data->data(), labelptr, d_colids, nbins, nrows, ncols, n_unique_labels,  d_histout, tempmem->d_quantile->data());
	}
	CUDA_CHECK(cudaGetLastError());

	MLCommon::updateHost(h_histout, d_histout, n_hist_elements, tempmem->stream);
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	
	find_best_split_classifier<T, L, F>(tempmem, nbins, n_unique_labels, colselector, &split_info[0], nrows, ques, gain, split_algo);
	return;
}

