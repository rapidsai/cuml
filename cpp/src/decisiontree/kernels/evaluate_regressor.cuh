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
#include "../memory.h"
#include "batch_cal.cuh"
#include "col_condenser.cuh"
#include <float.h>
#include "../algo_helper.h"
#include "stats/minmax.h"

template<typename T, typename F>
__global__ void compute_mse_minmax_kernel_reg(const T* __restrict__ data, const T* __restrict__ labels, const int nbins, const int nrows, const int ncols, const int batch_ncols, const T* __restrict__ globalminmax, T* mseout, const T* __restrict__ predout, const int* __restrict__ countout, const T pred_parent) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];
	
	int colstep = (int)(ncols/batch_ncols);
	if((ncols % batch_ncols) != 0)
		colstep++;
	
	int batchsz = batch_ncols;	
	for(int k = 0; k < colstep; k++) {
		
		if(k == (colstep-1) && ( (ncols % batch_ncols) != 0) ) {
			batchsz = ncols % batch_ncols;
		}

		T *minmaxshared = (T*)shmem;
		T *shmem_pred = (T*)(shmem + 2*batchsz*sizeof(T));
		T *shmem_mse = (T*)(shmem + 2*batchsz*sizeof(T) + nbins*batchsz*sizeof(T));	
		int *shmem_count = (int*)(shmem  + 2*batchsz*sizeof(T) + 3*nbins*batchsz*sizeof(T));
				
		for (int i=threadIdx.x; i < 2*batchsz; i += blockDim.x) {
			(i < batchsz) ? (minmaxshared[i] = globalminmax[k*batch_ncols + i] ) : (minmaxshared[i] = globalminmax[k*batch_ncols + (i-batchsz) + ncols]);
		}
		
		for (int i = threadIdx.x; i < nbins*batchsz; i += blockDim.x) {
			shmem_count[i] = countout[i + k*nbins*batch_ncols];
			shmem_pred[i] = predout[i + k*nbins*batch_ncols];
			shmem_mse[i] = 0.0;
			shmem_mse[i + batchsz*nbins] = 0.0;
		}
	
		__syncthreads();
		
		for (unsigned int i = tid; i < nrows*batchsz; i += blockDim.x*gridDim.x) {
			int mycolid = (int)( i / nrows);
			int coloffset = mycolid*nbins;

			T delta = (minmaxshared[mycolid + batchsz] - minmaxshared[mycolid]) / (nbins);
			T base_quesval = minmaxshared[mycolid] + delta;

			T localdata = data[i + k*batch_ncols*nrows];
			T label = labels[ i % nrows];
			for (int j=0; j < nbins; j++) {
				T quesval = base_quesval + j * delta;
			
				if (localdata <= quesval) {
					T temp = shmem_pred[coloffset +j] / shmem_count[coloffset + j] ;
					temp = label - temp;
					atomicAdd(&shmem_mse[j + coloffset], F::exec(temp));
				} else {
					T temp = ( pred_parent*nrows - shmem_pred[coloffset +j] ) / (nrows - shmem_count[coloffset + j] );
					temp = label - temp;
					atomicAdd(&shmem_mse[j + coloffset + batchsz*nbins], F::exec(temp));
				}
		       
			}
		}
	
		__syncthreads();

		for (int i = threadIdx.x; i < batchsz*nbins; i += blockDim.x) {
			atomicAdd(&mseout[i + k*batch_ncols*nbins], shmem_mse[i]);
			atomicAdd(&mseout[i + k*batch_ncols*nbins + ncols*nbins], shmem_mse[i + batchsz*nbins]);
		}

		__syncthreads();
	}
}

/*
   The output of the function is a histogram array, of size ncols * nbins * n_unique_lables
   column order is as per colids (bootstrapped random cols) for each col there are nbins histograms
 */
template<typename T>
__global__ void all_cols_histograms_minmax_kernel_reg(const T* __restrict__ data, const T* __restrict__ labels, const int nbins, const int nrows, const int ncols, const int batch_ncols, const T* __restrict__ globalminmax, T* predout, int* countout) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];

	int colstep = (int)(ncols/batch_ncols);
	if((ncols % batch_ncols) != 0)
		colstep++;
	
	int batchsz = batch_ncols;	
	for(int k = 0; k < colstep; k++) {
		
		if(k == (colstep-1) && ( (ncols % batch_ncols) != 0) ) {
			batchsz = ncols % batch_ncols;			
		}

		T *minmaxshared = (T*)shmem;
		T *shmem_pred = (T*)(shmem + 2*batchsz*sizeof(T));
		int *shmem_count = (int*)(shmem  + 2*batchsz*sizeof(T) + nbins*batchsz*sizeof(T));
	
		for (int i=threadIdx.x; i < 2*batchsz; i += blockDim.x) {
			(i < batchsz) ? (minmaxshared[i] = globalminmax[k*batch_ncols + i] ) : (minmaxshared[i] = globalminmax[k*batch_ncols + (i-batchsz) + ncols]);
		}

		for (int i = threadIdx.x; i < nbins*batchsz; i += blockDim.x) {
			shmem_pred[i] = 0;
			shmem_count[i] = 0;
		}

		__syncthreads();

		for (unsigned int i = tid; i < nrows*batchsz; i += blockDim.x*gridDim.x) {
			int mycolid = (int)( i / nrows);
			int coloffset = mycolid*nbins;

			T delta = (minmaxshared[mycolid + batchsz] - minmaxshared[mycolid]) / (nbins);
			T base_quesval = minmaxshared[mycolid] + delta;

			T localdata = data[i + k*batch_ncols*nrows];
			T label = labels[ i % nrows ];
			for (int j=0; j < nbins; j++) {
				T quesval = base_quesval + j * delta;

				if (localdata <= quesval) {
					atomicAdd(&shmem_count[j + coloffset], 1);
					atomicAdd(&shmem_pred[j + coloffset], label);
				}
			}
			
		}
		
		__syncthreads();

		for (int i = threadIdx.x; i < batchsz*nbins; i += blockDim.x) {
			atomicAdd(&predout[i + k*batch_ncols*nbins], shmem_pred[i]);
			atomicAdd(&countout[i + k*batch_ncols*nbins], shmem_count[i]);
		}

		__syncthreads();
	}
}

template<typename T>
__global__ void all_cols_histograms_global_quantile_kernel_reg(const T* __restrict__ data, const T* __restrict__ labels, const unsigned int* __restrict__ colids, const int nbins, const int nrows, const int ncols, const int batch_ncols, T* predout, int* countout, const T* __restrict__ quantile) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];
	
	int colstep = (int)(ncols/batch_ncols);
	if((ncols % batch_ncols) != 0)
		colstep++;
	
	int batchsz = batch_ncols;	
	for(int k = 0; k < colstep; k++) {
		
		if(k == (colstep-1) && ( (ncols % batch_ncols) != 0) ) {
			batchsz = ncols % batch_ncols;			
		}

		T *shmem_pred = (T*) (shmem);
		int *shmem_count = (int*)(shmem + nbins*batchsz*sizeof(T));

		for (int i = threadIdx.x; i < nbins*batchsz; i += blockDim.x) {
			shmem_pred[i] = 0;
			shmem_count[i] = 0;
		}

		__syncthreads();

		for (unsigned int i = tid; i < nrows*batchsz; i += blockDim.x*gridDim.x) {
			int mycolid = (int)( i / nrows);
			int coloffset = mycolid*nbins;

			T localdata = data[i + k*batch_ncols*nrows];
			T label = labels[ i % nrows ];
			for (int j=0; j < nbins; j++) {
				int quantile_index = colids[mycolid + k*batch_ncols] * nbins + j;
				T quesval = quantile[quantile_index];
				if (localdata <= quesval) {
					atomicAdd(&shmem_count[j + coloffset], 1);
					atomicAdd(&shmem_pred[j + coloffset], label);
				}
			}
			
		}

		__syncthreads();
	
		for (int i = threadIdx.x; i < batchsz*nbins; i += blockDim.x) {
			atomicAdd(&predout[i + k*batch_ncols*nbins], shmem_pred[i]);
			atomicAdd(&countout[i + k*batch_ncols*nbins], shmem_count[i]);
		}
		__syncthreads();

	}
}

template<typename T, typename F>
__global__ void compute_mse_global_quantile_kernel_reg(const T* __restrict__ data, const T* __restrict__ labels, const unsigned int* __restrict__ colids, const int nbins, const int nrows, const int ncols, const int batch_ncols, T* mseout, const T* __restrict__ predout, const int* __restrict__ countout, const T* __restrict__ quantile, const T pred_parent) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];

	int colstep = (int)(ncols/batch_ncols);
	if((ncols % batch_ncols) != 0)
		colstep++;
	
	int batchsz = batch_ncols;	
	for(int k = 0; k < colstep; k++) {
		
		if(k == (colstep-1) && ( (ncols % batch_ncols) != 0) ) {
			batchsz = ncols % batch_ncols;			
		}
		
		T *shmem_pred = (T*)(shmem);
		T *shmem_mse = (T*)(shmem + nbins*batchsz*sizeof(T));	
		int *shmem_count = (int*)(shmem + 3*nbins*batchsz*sizeof(T));
		
		for (int i = threadIdx.x; i < nbins*batchsz; i += blockDim.x) {
			shmem_count[i] = countout[i + k*nbins*batch_ncols];
			shmem_pred[i] = predout[i + k*nbins*batch_ncols];
			shmem_mse[i] = 0.0;
			shmem_mse[i + batchsz*nbins] = 0.0;
		}
		
		__syncthreads();
	
		for (unsigned int i = tid; i < nrows*batchsz; i += blockDim.x*gridDim.x) {
			int mycolid = (int)( i / nrows);
			int coloffset = mycolid*nbins;
		
			T localdata = data[i + k*batch_ncols*nrows];
			T label = labels[ i % nrows ];
			for (int j=0; j < nbins; j++) {
				int quantile_index = colids[mycolid + k*batch_ncols] * nbins + j;
				T quesval = quantile[quantile_index];
				
				if (localdata <= quesval) {
					T temp = shmem_pred[coloffset +j] / shmem_count[coloffset + j] ;
					temp = label - temp;
					atomicAdd(&shmem_mse[j + coloffset], F::exec(temp));
				} else {
					T temp = ( pred_parent*nrows - shmem_pred[coloffset +j] ) / (nrows - shmem_count[coloffset + j] );
					temp = label - temp;
					atomicAdd(&shmem_mse[j + coloffset + batchsz*nbins], F::exec(temp));
				}
				
			}
			
		}
	
		__syncthreads();
	
		for (int i = threadIdx.x; i < batchsz*nbins; i += blockDim.x) {
			atomicAdd(&mseout[i + k*batch_ncols*nbins], shmem_mse[i]);
			atomicAdd(&mseout[i + k*batch_ncols*nbins + ncols*nbins], shmem_mse[i + batchsz*nbins]);			
		}
		__syncthreads();

	}
			
}

template<typename T>
void find_best_split_regressor(const std::shared_ptr<TemporaryMemory<T,T>> tempmem, const int nbins, const std::vector<unsigned int>& col_selector, MetricInfo<T> split_info[3], const int nrows, MetricQuestion<T> & ques, float & gain, const int split_algo) {
	
	gain = 0.0f;
	int best_col_id = -1;
	int best_bin_id = -1;
	
	int n_cols = col_selector.size();
	for (int col_id = 0; col_id < n_cols; col_id++) {
		
		int col_count_base_index = col_id * nbins;
		// tempmem->h_histout holds n_cols histograms of nbins of n_unique_labels each.
		for (int i = 0; i < nbins; i++) {
			
			int tmp_lnrows = tempmem->h_histout->data()[col_count_base_index + i];
			int tmp_rnrows = nrows - tmp_lnrows;
			
			if (tmp_lnrows == 0 || tmp_rnrows == 0)
				continue;
			
			float tmp_pred_left = tempmem->h_predout->data()[col_count_base_index + i];
			float tmp_pred_right = (nrows * split_info[0].predict) - tmp_pred_left;
			tmp_pred_left /= tmp_lnrows;
			tmp_pred_right /= tmp_rnrows;
			
			// Compute MSE right and MSE left value for each bin.
			float tmp_mse_left  = tempmem->h_mseout->data()[col_count_base_index + i];
			float tmp_mse_right = tempmem->h_mseout->data()[col_count_base_index + i + n_cols*nbins];
			tmp_mse_left /= tmp_lnrows;
			tmp_mse_right /= tmp_rnrows;
			
			float impurity = (tmp_lnrows * 1.0f/nrows) * tmp_mse_left + (tmp_rnrows * 1.0f/nrows) * tmp_mse_right;
			float info_gain = split_info[0].best_metric - impurity;
			
			// Compute best information col_gain so far
			if (info_gain > gain) {
				gain = info_gain;
				best_bin_id = i;
				best_col_id = col_id;
				split_info[1].best_metric = tmp_mse_left;
				split_info[2].best_metric = tmp_mse_right;
				split_info[1].predict = tmp_pred_left;
				split_info[2].predict = tmp_pred_right;
			}
		}
	}

	if (best_col_id == -1 || best_bin_id == -1)
		return;

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


template<typename T, typename F>
void best_split_all_cols_regressor(const T *data, const unsigned int* rowids, const T *labels, const int nbins, const int nrows, const int rowoffset, const std::vector<unsigned int>& colselector, const std::shared_ptr<TemporaryMemory<T,T>> tempmem, MetricInfo<T> split_info[3], MetricQuestion<T> & ques, float & gain, const int split_algo, const size_t max_shared_mem)
{
	unsigned int* d_colids = tempmem->d_colids->data();
	T* d_globalminmax = tempmem->d_globalminmax->data();
	int *d_histout = tempmem->d_histout->data();
	int *h_histout = tempmem->h_histout->data();
	T* d_mseout = tempmem->d_mseout->data();
	T* h_mseout = tempmem->h_mseout->data();
	T* d_predout = tempmem->d_predout->data();
	T* h_predout = tempmem->h_predout->data();
	
	int ncols = colselector.size();
	int col_minmax_bytes = sizeof(T) * 2 * ncols;
	int n_pred_bytes = nbins * sizeof(T) * ncols;
	int n_count_bytes = nbins * ncols * sizeof(int);
	int n_mse_bytes = 2 * nbins * sizeof(T) * ncols;
	
	CUDA_CHECK(cudaMemsetAsync((void*)d_mseout, 0, n_mse_bytes, tempmem->stream));
	CUDA_CHECK(cudaMemsetAsync((void*)d_predout, 0, n_pred_bytes, tempmem->stream));
	CUDA_CHECK(cudaMemsetAsync((void*)d_histout, 0, n_count_bytes, tempmem->stream));
	
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

	int batch_ncols;
	size_t shmem_needed;

	T *labelptr = tempmem->sampledlabels->data();
	get_sampled_labels<T>(labels, labelptr, rowids, nrows, tempmem->stream);

	if (split_algo == ML::SPLIT_ALGO::HIST) {
		shmem_needed = n_pred_bytes + n_count_bytes + col_minmax_bytes;
		update_kernel_config(max_shared_mem, shmem_needed, ncols, nrows, threads, batch_ncols, blocks, shmemsize);
		all_cols_histograms_minmax_kernel_reg<<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data->data(), labelptr, nbins, nrows, ncols, batch_ncols, d_globalminmax, d_predout, d_histout);
		
		shmem_needed += n_mse_bytes;
		update_kernel_config(max_shared_mem, shmem_needed, ncols, nrows, threads, batch_ncols, blocks, shmemsize);
		compute_mse_minmax_kernel_reg<T, F><<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data->data(), labelptr, nbins, nrows, ncols, batch_ncols, d_globalminmax, d_mseout, d_predout, d_histout, split_info[0].predict);
	} else if (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
		shmem_needed = n_pred_bytes + n_count_bytes;
		update_kernel_config(max_shared_mem, shmem_needed, ncols, nrows, threads, batch_ncols, blocks, shmemsize);
		all_cols_histograms_global_quantile_kernel_reg<<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data->data(), labelptr, d_colids, nbins, nrows, ncols, batch_ncols, d_predout, d_histout, tempmem->d_quantile->data());

		shmem_needed += n_mse_bytes;
		update_kernel_config(max_shared_mem, shmem_needed, ncols, nrows, threads, batch_ncols, blocks, shmemsize);

		compute_mse_global_quantile_kernel_reg<T, F><<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data->data(), labelptr, d_colids, nbins, nrows, ncols, batch_ncols, d_mseout, d_predout, d_histout, tempmem->d_quantile->data(), split_info[0].predict);
	}
	CUDA_CHECK(cudaGetLastError());
	
	MLCommon::updateHost(h_mseout, d_mseout, n_mse_bytes / sizeof(T), tempmem->stream);
	MLCommon::updateHost(h_histout, d_histout, n_count_bytes / sizeof(int), tempmem->stream);
	MLCommon::updateHost(h_predout, d_predout, n_pred_bytes / sizeof(T), tempmem->stream);
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	
	find_best_split_regressor(tempmem, nbins, colselector, &split_info[0], nrows, ques, gain, split_algo);
	return;
}

