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

/* Each kernel invocation produces left gini hists (histout) for batch_bins questions for specified column. */
__global__ void batch_evaluate_kernel(const float* __restrict__ column, const int* __restrict__ labels, const float base_quesval, const int batch_bins, const float delta, const int nrows, const int n_unique_labels, int* histout) {
	
	// Reset shared memory histograms
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ unsigned int shmemhist[];
	for (int i = threadIdx.x; i < n_unique_labels*batch_bins; i += blockDim.x) {
		shmemhist[i] = 0;
	}
	
	__syncthreads();
	
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
	
}


__global__ void evaluate_kernel(const float* __restrict__ column, const int* __restrict__ labels, const float quesval, const int nrows, const int nmax, int* histout)
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
					atomicAdd(&shmemhist[label], 1);
				}

		}

	__syncthreads();

	if(threadIdx.x < nmax)
		atomicAdd(&histout[threadIdx.x], shmemhist[threadIdx.x]);

	return;
}

void evaluate_and_leftgini(const float *column, const int *labels, const float quesval, const int nrows, const int n_unique_labels, GiniInfo& split_info, int& lnrows, int& rnrows, TemporaryMemory* tempmem)
{
	int *dhist = tempmem->d_hist;
	int *hhist = tempmem->h_hist;

	CUDA_CHECK(cudaMemsetAsync(dhist, 0, sizeof(int)*n_unique_labels, tempmem->stream));
	evaluate_kernel<<< (int)(nrows/128) + 1, 128, sizeof(int)*n_unique_labels, tempmem->stream>>>(column, labels, quesval, nrows, n_unique_labels, dhist);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpyAsync(hhist, dhist, sizeof(int)*n_unique_labels, cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

	float gval = 1.0f;
	lnrows = 0;
	split_info.hist.resize(n_unique_labels, 0);

	for(int i=0; i < n_unique_labels; i++) {
		split_info.hist[i] = hhist[i];
		lnrows += split_info.hist[i];
	}

	for(int i=0; i < n_unique_labels; i++) {
		float prob_left = (float) (split_info.hist[i]) / lnrows; // gini left has to be divided by lnrows and not nrows
		gval -= prob_left * prob_left;
	}

	rnrows = nrows - lnrows;
	split_info.best_gini = gval; //Update gini val

	return;
}

/* Compute best information gain for this batch. This code merges  gini_left and gini_right computation in  a single function.
   Outputs: split_info[1] and split_info[2] are updated with the correct info for the best split among the considered batch.
   batch_id specifies which question (bin) within the batch  gave the best split.
*/
float batch_evaluate_gini(const float *column, const int *labels, const float base_quesval, const float delta,
							const int batch_bins, int & batch_id, const int nrows, const int n_unique_labels,
							GiniInfo split_info[3], TemporaryMemory* tempmem) {

	int *dhist = tempmem->d_hist;
	int *hhist = tempmem->h_hist;
	int n_hists_bytes = sizeof(int) * n_unique_labels * batch_bins;

	CUDA_CHECK(cudaMemsetAsync(dhist, 0, n_hists_bytes, tempmem->stream));
	// Each thread does more work: it answers batch_bins questions for the same column data. Could change this in the future.
	ASSERT((n_unique_labels <= 128), "Error! Kernel cannot support %d labels. Current limit is 128", n_unique_labels);

	//Kernel launch
	batch_evaluate_kernel<<< (int)(nrows /128) + 1, 128, n_hists_bytes, tempmem->stream>>>(column, labels,
		base_quesval, batch_bins, delta, nrows, n_unique_labels, dhist);

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
