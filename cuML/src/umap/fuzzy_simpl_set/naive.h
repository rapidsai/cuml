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

#include "knn/knn.h"
#include "umap/umap.h"

#include "cuda_utils.h"

#include "stats/mean.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>

#include <limits>
#include <math.h>

namespace UMAP {
namespace FuzzySimplSet {
namespace Naive {


	using namespace ML;

	/** number of threads in a CTA along X dim */
	static const int TPB_X = 32;
	/** number of threads in a CTA along Y dim */
	static const int TPB_Y = 8;

	static const float SMOOTH_K_TOLERANCE = 1.0;

	static const float MIN_K_DIST_SCALE = 1.0;


	/**
	 * Computes a continuous version of the distance to the kth nearest neighbor.
	 * That is, this is similar to knn-distance but allows continuous k values
	 * rather than requiring an integral k. In essence, we are simply computing
	 * the distance such that the cardinality of fuzzy set we generate is k.
	 *
	 * TODO: Optimize for coalesced reads
	 *
	 *
	 * @param knn_dists: Distances to nearest neighbors for each sample. Each row should
	 * 					 be a sorted list of distances to a given sample's nearest neighbors.
	 *
	 * @param n: The number of samples
	 * @param k: The number of neighbors
	 *
	 * @param local_connectivity: The local connectivity required -- i.e. the number of nearest
	 * 							  neighbors that should be assumed to be connected at a local
	 * 							  level. The higher this value the more connecte the manifold
	 * 							  becomes locally. In practice, this should not be more than the
	 * 							  local intrinsic dimension of the manifold.
	 *
	 * @param sigmas: An array of size n representing the distance to the kth nearest neighbor,
	 * 				  as suitably approximated.
	 * @parasm rhos:  An array of size n representing the distance to the 1st nearest neighbor
	 * 				  for each point.
	 *
	 * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
	 *
	 */
	template<typename T>
	__global__ void smooth_knn_dist(const T *knn_dists, int n,
								    float mean_dist,
									T *sigmas, T *rhos,			// Size of n, iniitalized to zeros
									UMAPParams *params,
									int n_iter = 64, float bandwidth = 1.0) {

		float target = log2(params->n_neighbors) * bandwidth;

		// row-based matrix is best
		int row = (blockIdx.y * TPB_Y) + threadIdx.y;
		int col = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = (row+col)*params->n_neighbors; // each thread processes one row of the dist matrix

		float lo = 0.0;
		float hi = std::numeric_limits<float>::max();
		float mid = 1.0;

		float ith_distances[params->n_neighbors];

		std::vector<float> non_zero_dists;

		int total_nonzero = 0;
		int max_nonzero = -1;
		float sum = 0;
		for(int idx = 0; idx < params->n_neighbors; idx++) {
			ith_distances[idx] = knn_dists[i+idx];

			sum += ith_distances[idx];

			if(ith_distances[idx] > 0.0) {
				total_nonzero+= 1;
				non_zero_dists.push_back(ith_distances[idx]);
			}

			if(ith_distances[idx] > max_nonzero)
				max_nonzero = ith_distances[idx];
		}

		float ith_distances_mean = sum / params->n_neighbors;

		non_zero_dists.resize(total_nonzero);
		if(total_nonzero > params->local_connectivity) {
			int index = floor(params->local_connectivity);
			float interpolation = params->local_connectivity - index;

			if(index > 0) {
				rhos[i] = non_zero_dists[index-1];
				if(interpolation > SMOOTH_K_TOLERANCE)
					rhos[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index-1]);
				else
					rhos[i] = interpolation * non_zero_dists[0];

			} else if(total_nonzero > 0)
				rhos[i] = max_nonzero;
		}

		for(int iter = 0; iter < n_iter; iter++) {
			float psum = 0.0;
			for(int j = 0; j < params->n_neighbors; j++) {
				float d = knn_dists[i + j] - rhos[i];
				if(d > 0)
					psum += exp(-(d/mid));
				else
					psum += 1.0;
			}

			if((psum - target) < SMOOTH_K_TOLERANCE)
				break;

			if(psum > target) {
				hi = mid;
				mid = (lo + hi) / 2.0;
			} else {
				lo = mid;
				if(hi == std::numeric_limits<float>::max())
					mid *= 2;
				else
					mid = (lo + hi) / 2.0;
			}
		}

		sigmas[i] = mid;

		if(rhos[i] > 0.0) {
			if(sigmas[i] < MIN_K_DIST_SCALE * ith_distances_mean)
				sigmas[i] =MIN_K_DIST_SCALE * ith_distances_mean;
		} else {
			if(sigmas[i] < MIN_K_DIST_SCALE * mean_dist)
				sigmas[i] = MIN_K_DIST_SCALE * mean_dist;
		}
	}


	/**
	 * Construct the membership strength data for the 1-skeleton of each local
	 * fuzzy simplicial set -- this is formed as a sparse matrix where each row is
	 * a local fuzzy simplicial set, with a membership strength for the 1-simplex
	 * to each other data point.
	 *
	 * @param knn_indices: the knn index matrix of size (n, k)
	 * @param knn_dists: the knn distance matrix of size (n, k)
	 * @param sigmas: array of size n representing distance to kth nearest neighbor
	 * @param rhos: array of size n representing distance to the first nearest neighbor
	 *
	 * @return rows: long array of size n
	 * 		   cols: long array of size k
	 * 		   vals: T array of size n*k
	 *
	 * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
	 */
	template<typename T>
	__global__ void compute_membership_strength(const long *knn_indices, const float *knn_dists,  // nn outputs
									 const T *sigmas, const T *rhos, // continuous dists to nearest neighbors
									 T *vals, int *rows, int *cols,  // result coo
									 T *tvals, int *trows, int *tcols, // result coo transposed rows
									 int n, UMAPParams *params) {	 // model params

		// row-based matrix is best
		int row = (blockIdx.y * TPB_Y) + threadIdx.y;
		int col = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = (row+col)*params->n_neighbors; // each thread processes one row of the dist matrix

		T cur_rho = rhos[i];
		T cur_sigma = sigmas[i];

		for(int j = 0; j < params->n_neighbors; j++) {

			int idx = i*params->n_neighbors+j;
			int t_idx = j*params->n_neighbors+i;

			long cur_knn_ind = knn_indices[idx];
			T cur_knn_dist = knn_dists[idx];

			T val = 0.0;
			if(cur_knn_ind == -1)
				continue;

			if(cur_knn_ind == i)
				val = 0.0;
			else if(cur_knn_dist - cur_rho <= 0.0)
				val = 1.0;
			else
				val = exp(-((cur_knn_dist - cur_rho) / (cur_sigma)));

			// TODO: Make both of these lower-triangular
			rows[idx] = i;
			cols[idx] = cur_knn_ind;
			vals[idx] = val;
			tcols[t_idx] = i;
			trows[t_idx] = cur_knn_ind;
			tvals[t_idx] = val;
		}
	}

	template<typename T>
	__global__ void compute_result(int *rows, int *cols, int *vals,
								   int *trows, int *tcols, int *tvals,
								   int *orows, int *ocols, int *ovals,
								   int *rnnz, int n, UMAPParams *params) {

		int row = (blockIdx.y * TPB_Y) + threadIdx.y;
		int col = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = (row+col)*params->n_neighbors; // each thread processes one row
		// Grab the n_neighbors from our transposed matrix,

		int nnz = 0;
		for(int j = 0; j < params->n_neighbors; j++) {

			int idx = i*params->n_neighbors+j;

			T result = vals[idx];
			T transpose = tvals[idx];
			T prod_matrix = vals[idx] * tvals[idx];

			T res = params->set_op_mix_ratio * (result - transpose - prod_matrix)
							+ (1.0 - params->set_op_mix_ratio) + prod_matrix;

			orows[idx] = rows[idx];
			ocols[idx] = cols[idx];
			ovals[idx] = res;

			if(res != 0)
				++nnz;
		}

		rnnz[i] = nnz;
		atomicAdd(rnnz+n, nnz);
	}

	template <typename T>
	__global__ void compress_coo(int *rows, int *cols, int *vals,
								 int *crows, int *ccols, int *cvals,
								 int *ex_scan, UMAPParams *params) {

		int row = (blockIdx.y * TPB_Y) + threadIdx.y;
		int col = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = (row+col)*params->n_neighbors; // each thread processes one row

		int start = ex_scan[i];
		int cur_out_idx = start;

		for(int j = 0; j < params->n_neighbors; j++) {
			int idx = i*params->n_neighbors+j;
			if(vals[idx] != 0.0) {
				crows[cur_out_idx] = rows[idx];
				ccols[cur_out_idx] = cols[idx];
				cvals[cur_out_idx] = vals[idx];

				++cur_out_idx;
			}
		}
	}

	/**
	 * Given a set of X, a neighborhood size, and a measure of distance, compute
	 * the fuzzy simplicial set (here represented as a fuzzy graph in the form of
	 * a sparse coo matrix) associated to the data. This is done by locally
	 * approximating geodesic (manifold surface) distance at each point, creating
	 * a fuzzy simplicial set for each such point, and then combining all the local
	 * fuzzy simplicial sets into a global one via a fuzzy union.
	 */
	template<typename T>
	void launcher(const long *knn_indices, const float *knn_dists,
			      int n, int *rows, int *cols, T *vals,
				  UMAPParams *params, int algorithm) {

		/**
		 * Calculate mean distance through a parallel reduction
		 */
		T *dist_means_dev;
		MLCommon::allocate(dist_means_dev, params->n_neighbors);
		MLCommon::Stats::mean(dist_means_dev, knn_dists, params->n_neighbors, n, false, true);

		T *dist_means_host = malloc(params->n_neighbors*sizeof(T));
		MLCommon::updateHost(dist_means_host, dist_means_dev, params->n_neighbors);

		// Might make sense to do this on device
		float sum = 0.0;
		for(int i = 0; i < params->n_neighbors; i++)
			sum += dist_means_host[i];

		float mean_dist = sum / params->n_neighbors;

		/**
		 * Immediately free up memory for subsequent algorithms
		 */
		delete dist_means_host;
		CUDA_CHECK(cudaFree(dist_means_dev));

		T *sigmas;
		T *rhos;

	    dim3 grid(MLCommon::ceildiv(n, TPB_X), MLCommon::ceildiv(n, TPB_Y), 1);
	    dim3 blk(TPB_X, TPB_Y, 1);

		/**
		 * Call smooth_knn_dist to get sigmas and rhos
		 */
		smooth_knn_dist<<<grid,blk>>>(knn_dists, n, mean_dist, sigmas, rhos, params);

		int k = params->n_neighbors;

		// We will keep arrays, with space O(n*k*9) representing 3 O(n^2) matrices.
		int *trows, *tcols;
		T *tvals;
		MLCommon::allocate(trows, n*k, true);
		MLCommon::allocate(tcols, n*k, true);
		MLCommon::allocate(tvals, n*k, true);

		/**
		 * Call compute_membership_strength
		 */
		compute_membership_strength<<<grid,blk>>>(knn_indices, knn_dists,
												  sigmas, rhos,
												  rows, cols, vals,
												  trows, tcols, tvals,
												  n, params);

		int *orows, *ocols, *rnnz;
		T *ovals;
		MLCommon::allocate(orows, n*k, true);
		MLCommon::allocate(ocols, n*k, true);
		MLCommon::allocate(ovals, n*k, true);
		MLCommon::allocate(rnnz, n+1, true);


		/**
		 * Finish computation of matrix sums, Hadamard products, weighting, etc...
		 */
		compute_result<<<grid, blk>>>(rows, cols, vals,
					   trows, tcols, tvals,
					   orows, ocols, ovals,
					   n, params);

		CUDA_CHECK(cudaFree(rows));
		CUDA_CHECK(cudaFree(cols));
		CUDA_CHECK(cudaFree(vals));
		CUDA_CHECK(cudaFree(trows));
		CUDA_CHECK(cudaFree(trows));
		CUDA_CHECK(cudaFree(trows));


		/**
		 * Compress resulting COO matrix
		 */

		int *ex_scan;
		MLCommon::allocate(ex_scan, n+1);

	    thrust::device_ptr<int> dev_rnnz = thrust::device_pointer_cast(rnnz);
	    thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
	    thrust::exclusive_scan(dev_rnnz, dev_rnnz+n, dev_ex_scan);

	    int cur_coo_len = 0;
		MLCommon::updateHost(&cur_coo_len, rnnz+n, 1);


		int *crows, *ccols;
		T *cvals;
		MLCommon::allocate(crows, cur_coo_len, true);
		MLCommon::allocate(ccols, cur_coo_len, true);
		MLCommon::allocate(cvals, cur_coo_len, true);

	    compress_coo<<<grid, blk>>>(orows, ocols, ovals,
	    						    crows, ccols, cvals,
	    						    rnnz, params);
	}


}
}
};
