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
#include <float.h>
#include <cooperative_groups.h>

/* Each kernel invocation produces left gini hists (histout) for batch_bins questions for specified column. */
__global__ void batch_evaluate_kernel(const float* __restrict__ column, const int* __restrict__ labels, const int nbins, const int batch_bins, const int nrows, const int n_unique_labels, int* histout, float * col_min, float * col_max, float * ques_info) {

	// Reset shared memory histograms
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ unsigned int shmemhist[];
	for (int i = threadIdx.x; i < n_unique_labels*batch_bins; i += blockDim.x) {
		shmemhist[i] = 0;
	}
	
	__syncthreads();

	float delta = (*col_max - *col_min) / nbins;
	float base_quesval = *col_min + delta;
	
	if (tid < nrows) {
		float data = column[tid];
		int label = labels[tid];
		// Each thread evaluates batch_bins questions and populates respective buckets.
		for (int i = 0; i < batch_bins; i++) {
			float quesval = base_quesval + i * delta;
			if (data <= quesval) {
				atomicAdd(&shmemhist[label + n_unique_labels * i], 1);
			}
		}
		
	}
	
	__syncthreads();
	
	// Merge shared mem histograms to the global memory hist
	for(int i = threadIdx.x; i < n_unique_labels*batch_bins; i += blockDim.x) {
		atomicAdd(&histout[i], shmemhist[i]);
	}

	if (tid == 0) {
		ques_info[0] = *col_min;
		ques_info[1] = *col_max;
	}
	
}

/* Compute best information gain for this batch. This code merges  gini_left and gini_right computation in  a single function.
   Outputs: split_info[1] and split_info[2] are updated with the correct info for the best split among the considered batch.
   batch_id specifies which question (bin) within the batch  gave the best split.
*/
float batch_evaluate_gini(const float *column, const int *labels, const int nbins, 
							const int batch_bins, int & batch_id, const int nrows, const int n_unique_labels, 
							GiniInfo split_info[3], TemporaryMemory* tempmem) {

	int *dhist = tempmem->d_hist;
	int *hhist = tempmem->h_hist;
	int n_hists_bytes = sizeof(int) * n_unique_labels * batch_bins;
	
	CUDA_CHECK(cudaMemsetAsync(dhist, 0, n_hists_bytes, tempmem->stream));
	// Each thread does more work: it answers batch_bins questions for the same column data. Could change this in the future.
	ASSERT((n_unique_labels <= 128), "Error! Kernel cannot support %d labels. Current limit is 128", n_unique_labels);
	
	//FIXME TODO: if delta is 0 just go through one batch_bin.

	//Kernel launch
	batch_evaluate_kernel<<< (int)(nrows /128) + 1, 128, n_hists_bytes, tempmem->stream>>>(column, labels, 
		nbins, batch_bins,  nrows, n_unique_labels, dhist, &tempmem->d_min_max[0], &tempmem->d_min_max[1], tempmem->d_ques_info);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpyAsync(hhist, dhist, n_hists_bytes, cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

	float gain = 0.0f;
	int best_batch_id = 0;

	// hhist holds batch_bins of n_unique_labels each.
	// Todo note: we could do some of these computations on the gpu side too.
	for (int i = 0; i < batch_bins; i++) {

		// if tmp_lnrows or tmp_rnrows is 0, the corresponding gini will be 1 but that doesn't
		// matter as it won't count in the info_gain computation.
		float tmp_gini_left = 1.0f;
		float tmp_gini_right = 1.0f;
		int tmp_lnrows = 0;

		//separate loop for now to avoid overflow.
		for (int j = 0; j < n_unique_labels; j++) {
			int hist_index = i * n_unique_labels + j;
			tmp_lnrows += hhist[hist_index];
		}
		int tmp_rnrows = nrows - tmp_lnrows;

		// Compute gini right and gini left value for each bin.
		for (int j = 0; j < n_unique_labels; j++) {
			int hist_index = i * n_unique_labels + j;

			if (tmp_lnrows != 0) {
				float prob_left = (float) (hhist[hist_index]) / tmp_lnrows;
				tmp_gini_left -= prob_left * prob_left;
			}

			if (tmp_rnrows != 0) {
				float prob_right = (float) (split_info[0].hist[j] - hhist[hist_index]) / tmp_rnrows;
				tmp_gini_right -=  prob_right * prob_right;
			}
		}

		/*std::cout << "\nBatch id is " << i <<  ":\n";
		std::cout << "nrows/lnrows/rnrows " << nrows << ", " << tmp_lnrows << ", " << tmp_rnrows << std::endl;
		std::cout << "Gini parent/left/right " << split_info[0].best_gini << ", " << tmp_gini_left << ", " << tmp_gini_right << std::endl;*/

		ASSERT((tmp_gini_left >= 0.0f) && (tmp_gini_left <= 1.0f), "gini left value %f not in [0.0, 1.0]", tmp_gini_left);
		ASSERT((tmp_gini_right >= 0.0f) && (tmp_gini_right <= 1.0f), "gini right value %f not in [0.0, 1.0]", tmp_gini_right);

		float impurity = (tmp_lnrows * 1.0f/nrows) * tmp_gini_left + (tmp_rnrows * 1.0f/nrows) * tmp_gini_right;
		float info_gain = split_info[0].best_gini - impurity;

		/*std::cout << "Impurity is " << impurity << " info gain is " << info_gain << " gain so far is " << gain << std::endl;
		ASSERT(info_gain + FLT_EPSILON >= 0.0, "Cannot have negative info_gain %f", info_gain);

		// Note: It is possible to get negative (a bit below <0) information gain. By default this will result in no gain update due to its
		// initialization to zero.
		*/


		// Compute best information gain so far in the batch.
		if (info_gain > gain) {
			gain = info_gain;
			best_batch_id = i;
			split_info[1].best_gini = tmp_gini_left;
			split_info[2].best_gini = tmp_gini_right;
		}
	}


	// The batch id best_batch_id, within the batch, resulted in the best split. Update split_info accordingly.
	// This code is to avoid the hist copy every time within above loop.

	// The best_batch_id and rest info is dummy if we didn't go through the if-statement above. But that's OK because this will be treated as a leaf?
	// FIXME What should best_gini vals be in that case?
	split_info[1].hist.resize(n_unique_labels);
	split_info[2].hist.resize(n_unique_labels);
	for (int j = 0; j < n_unique_labels; j++) {
		split_info[1].hist[j] = hhist[ best_batch_id * n_unique_labels + j];
		split_info[2].hist[j] = split_info[0].hist[j] - hhist[ best_batch_id * n_unique_labels + j];
	}
	batch_id = best_batch_id;

	return gain;

}

__global__ void allcolsampler_kernel(const float* __restrict__ data, const unsigned int* __restrict__ rowids, const int* __restrict__ colids, const int nrows, const int ncols, const int rowoffset, float* globalmin, float* globalmax, float* sampledcols)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];
	float *minshared = (float*)shmem;
	float *maxshared = (float*)(shmem + sizeof(float) * ncols);

	for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
		minshared[i] = FLT_MAX;
		maxshared[i] = -FLT_MAX;
	}

	// Initialize min max in  global memory
	if (tid < ncols) {
		globalmin[tid] = FLT_MAX;
		globalmax[tid] = -FLT_MAX;
	}

	__syncthreads();
	
	for(unsigned int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x)
		{
			int newcolid = (int)(i / nrows);
			int myrowstart = colids[ newcolid ] * rowoffset;
			int index = rowids[ i % nrows] + myrowstart;
			float coldata = data[index];

			atomicMinFloat(&minshared[newcolid], coldata);
			atomicMaxFloat(&maxshared[newcolid], coldata);
			sampledcols[i] = coldata;
		}

	__syncthreads();
	
	for(int j = threadIdx.x; j < ncols; j+= blockDim.x)
		{
			atomicMinFloat(&globalmin[j], minshared[j]);
			atomicMaxFloat(&globalmax[j], maxshared[j]);
		}

	return;
}

__global__ void letsdoitall_kernel(const float* __restrict__ data, const int* __restrict__ labels, const unsigned int* __restrict__ rowids, const int* __restrict__ colids, const int nbins, const int nrows, const int ncols, const int rowoffset, const int n_unique_labels, const float* __restrict__ globalminmax, int* histout)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ char shmem[];
	float *minmaxshared = (float*)shmem;
	int *shmemhist = (int*)(shmem + 2*ncols*sizeof(float));

	for(int i=threadIdx.x;i<2*ncols;i += blockDim.x)
		{
			minmaxshared[i] = globalminmax[i];
		}
	
	for (int i = threadIdx.x; i < n_unique_labels*nbins*ncols; i += blockDim.x)
		{
			shmemhist[i] = 0;
		}

	__syncthreads();

	for(unsigned int i = tid;i < nrows*ncols; i += blockDim.x*gridDim.x)
		{
			int mycolid = (int)( i / nrows);
			int coloffset = mycolid*n_unique_labels*nbins;
			
			// nbins is # batched bins. Use (batched bins + 1) for delta computation.
			float delta = (minmaxshared[mycolid + ncols] - minmaxshared[mycolid]) / (nbins + 1);
			float base_quesval = minmaxshared[mycolid] + delta;
			
			float localdata = data[i];
			int label = labels[ rowids[ i % nrows ] ];
			for(int j=0;j<nbins;j++)
				{
					float quesval = base_quesval + j * delta;
					if (localdata <= quesval) {
						atomicAdd(&shmemhist[label + n_unique_labels * j + coloffset], 1);
					}
				}
			
		}
	
	__syncthreads();
	
	for(int i = threadIdx.x; i < ncols*n_unique_labels*nbins; i += blockDim.x)
		{
			atomicAdd(&histout[i], shmemhist[i]);
		}
	return;	
}

void find_best_split(const TemporaryMemory * tempmem, const int nbins, const int n_unique_labels, const std::vector<int>& col_selector, GiniInfo split_info[3], const int nrows, GiniQuestion & ques, float & gain) {

	gain = 0.0f;
	int best_col_id = -1;
	int best_bin_id = -1;
	
	int n_cols = col_selector.size();
	for (int col_id = 0; col_id < n_cols; col_id++)
		{
			int col_hist_base_index = col_id * nbins * n_unique_labels;			
			// tempmem->h_histout holds n_cols histograms of nbins of n_unique_labels each.
			for (int i = 0; i < nbins; i++)
				{
					
					// if tmp_lnrows or tmp_rnrows is 0, the corresponding gini will be 1 but that doesn't
					// matter as it won't count in the info_gain computation.
					float tmp_gini_left = 1.0f;
					float tmp_gini_right = 1.0f;
					int tmp_lnrows = 0;
					
					//separate loop for now to avoid overflow.
					for (int j = 0; j < n_unique_labels; j++)
						{
							int hist_index = i * n_unique_labels + j;
							tmp_lnrows += tempmem->h_histout[col_hist_base_index + hist_index];
						}
					int tmp_rnrows = nrows - tmp_lnrows;

					if(tmp_lnrows == 0 || tmp_rnrows == 0)
						continue;
					
					// Compute gini right and gini left value for each bin.
					for (int j = 0; j < n_unique_labels; j++)
						{
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
					if (info_gain > gain)
						{
							gain = info_gain;
							best_bin_id = i;
							best_col_id = col_id;
							split_info[1].best_gini = tmp_gini_left;
							split_info[2].best_gini = tmp_gini_right;
						}
				}
		}
	
	if(best_col_id == -1 || best_bin_id == -1)
		return;
	
	split_info[1].hist.resize(n_unique_labels);
	split_info[2].hist.resize(n_unique_labels);
	for (int j = 0; j < n_unique_labels; j++) {
		split_info[1].hist[j] = tempmem->h_histout[  best_col_id * n_unique_labels * nbins + best_bin_id * n_unique_labels + j];
		split_info[2].hist[j] = split_info[0].hist[j] - split_info[1].hist[j];
	}

	ques.set_question_fields( best_col_id, col_selector[best_col_id], best_bin_id, nbins+1, n_cols);
	return;
}

//rowoffset appears to be NLocalrows which is nrows. Why?
void lets_doit_all(const float *data, const unsigned int* rowids, const int *labels, const int nbins, const int nrows, const int n_unique_labels, const int rowoffset, const std::vector<int>& colselector, const TemporaryMemory* tempmem, GiniInfo split_info[3], GiniQuestion & ques, float & gain)
{
	int* d_colids = tempmem->d_colids;
	float* d_globalminmax = tempmem->d_globalminmax;
	int *d_histout = tempmem->d_histout;
	int *h_histout = tempmem->h_histout;
	
	int ncols = colselector.size();
	int col_minmax_bytes = sizeof(float) * 2 * ncols;
	int n_hist_bytes = n_unique_labels * nbins * sizeof(int) * ncols;

	CUDA_CHECK(cudaMemsetAsync((void*)d_histout, 0, n_hist_bytes, tempmem->stream));
	
	unsigned int threads = 512;
	unsigned int blocks  = (int)((nrows * ncols) / threads) + 1;
	if(blocks > 65536)
		blocks = 65536;
	
	/* Kernel allcolsampler_kernel:
		- populates tempmem->tempdata with the sampled column data,
		- and computes min max histograms in tempmem->d_globalminmax
	   across all columns.
	*/
	size_t shmemsize = col_minmax_bytes;
	allcolsampler_kernel<<<blocks, threads, shmemsize, tempmem->stream>>>(data, rowids, d_colids, nrows, ncols, rowoffset, &d_globalminmax[0], &d_globalminmax[colselector.size()], tempmem->temp_data);
	CUDA_CHECK(cudaGetLastError());

	shmemsize = col_minmax_bytes + n_hist_bytes;
	
	letsdoitall_kernel<<<blocks, threads, shmemsize, tempmem->stream>>>(tempmem->temp_data, labels, rowids, d_colids, nbins, nrows, ncols, rowoffset, n_unique_labels, d_globalminmax, d_histout);
	CUDA_CHECK(cudaGetLastError());
	
	CUDA_CHECK(cudaMemcpyAsync(h_histout, d_histout, n_hist_bytes, cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream)); //added
	
	find_best_split(tempmem, nbins, n_unique_labels, colselector, &split_info[0], nrows, ques, gain);
	
}

