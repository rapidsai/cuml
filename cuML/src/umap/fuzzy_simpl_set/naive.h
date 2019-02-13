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

#include <cusparse_v2.h>

#include <thrust/device_vector.h>

#include <stdio.h>
#include <string>

namespace UMAP {
namespace FuzzySimplSet {
namespace Naive {


	using namespace ML;

	static const float MAX_FLOAT = std::numeric_limits<float>::max();

	/** number of threads in a CTA along X dim */
	static const int TPB_X = 32;

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
									int n_neighbors, float local_connectivity,
									int n_iter = 64, float bandwidth = 1.0) {

		float target = __log2f(n_neighbors) * bandwidth;

		// row-based matrix 1 thread per row
		int row = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = row*n_neighbors; // each thread processes one row of the dist matrix

		if(row < n) {
			printf("row=%d, i=%d\n", row, i);

			float lo = 0.0;
			float hi = MAX_FLOAT;
			float mid = 1.0;

			float *ith_distances = new float[n_neighbors];
			float *non_zero_dists = new float[n_neighbors];

			int total_nonzero = 0;
			int max_nonzero = -1;
			float sum = 0;

			for(int idx = 0; idx < n_neighbors; idx++) {
				ith_distances[idx] = knn_dists[i+idx];
				printf("i=%d, idx=%d, knn_dists=%f\n", i, idx, knn_dists[i+idx]);

				sum += ith_distances[idx];

				if(ith_distances[idx] > 0.0) {
					non_zero_dists[total_nonzero] = ith_distances[idx];
					total_nonzero+= 1;
					printf("i=%d, total_nonzero=%d\n", i, total_nonzero);
				}

				if(ith_distances[idx] > max_nonzero)
					max_nonzero = ith_distances[idx];
			}

			float ith_distances_mean = sum / n_neighbors;

			printf("i=%d, ith_distances_mean=%f\n", i, ith_distances_mean);

			if(total_nonzero > local_connectivity) {
				int index = int(local_connectivity);
				float interpolation = local_connectivity - index;

				printf("i=%d, index=%d, interpolation=%f\n", i, index, interpolation);

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
				for(int j = 0; j < n_neighbors; j++) {
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
					if(hi == MAX_FLOAT)
						mid *= 2;
					else
						mid = (lo + hi) / 2.0;
				}
			}

			sigmas[i] = mid;

			if(rhos[i] > 0.0) {
				if(sigmas[i] < MIN_K_DIST_SCALE * ith_distances_mean)
					sigmas[i] = MIN_K_DIST_SCALE * ith_distances_mean;
			} else {
				if(sigmas[i] < MIN_K_DIST_SCALE * mean_dist)
					sigmas[i] = MIN_K_DIST_SCALE * mean_dist;
			}

		}

	}


	/**
	 * Construct the membership strength data for the 1-skeleton of each local
	 * fuzzy simplicial set -- this is formed as a sparse matrix where each row is
	 * a local fuzzy simplicial set, with a membership strength for the 1-simplex
	 * to each other data point.
	 *
	 * TODO: Optimize for coalesced reads.
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
									 int n, int n_neighbors) {	 // model params

		// row-based matrix is best
		int row = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = row*n_neighbors; // each thread processes one row of the dist matrix

		if(row < n) {
			printf("ROW=%d\n", row);
			printf("compute_membership_strength(i=%d)\n", i);

			T cur_rho = rhos[i];
			T cur_sigma = sigmas[i];

			for(int j = 0; j < n_neighbors; j++) {

				int idx = i+j;

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
			}
		}
	}

	template<typename T>
	__global__ void compute_result(int *rows, int *cols, T *vals,
								   int *orows, int *ocols, T *ovals,
								   int *rnnz, int n,
								   int n_neighbors, float set_op_mix_ratio) {

		int row = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = row*n_neighbors; // each thread processes one row
		// Grab the n_neighbors from our transposed matrix,

		if(row < n) {

			int nnz = 0;
			for(int j = 0; j < n_neighbors; j++) {

				int idx = i+j;
				int out_idx = i*2;

				/**
				 * In order to do the Hadamard with the transposed
				 * matrix, look up the row corresponding to the current
				 * column and iterate n_neighbors times max to find
				 * if the value we are looking for is in here.
				 *
				 * Since a metric is symmetric, we can expect symmetry in
				 * the knn_indices matrix and, thus, we only need a single
				 * pass through it.
				 */

				int row_lookup = cols[idx];
				int t_start = row_lookup*n_neighbors; // Start at

				T transpose = 0.0;
				bool found_match = false;
				for(int t_idx = 0; t_idx < n_neighbors; t_idx++) {

					int f_idx = t_idx + t_start;
					// If we find a match, let's get out of the loop
					if(	cols[f_idx] == rows[idx] && rows[f_idx] == cols[idx] && vals[f_idx] != 0.0) {
						transpose = vals[f_idx];
						printf("Found transpose!\n");
						found_match = true;
						break;
					}
					printf("End loop.\n");
				}

				// if we didn't find an exact match, we still need to add
				// the transposed value into our current matrix.
				if(!found_match && vals[idx] != 0.0) {
					orows[out_idx+nnz] = cols[idx];
					ocols[out_idx+nnz] = rows[idx];
					ovals[out_idx+nnz] = vals[idx];
					++nnz;
				}

				T result = vals[idx];
				T prod_matrix = result * transpose;

				T res = set_op_mix_ratio * (result - transpose - prod_matrix)
								+ (1.0 - set_op_mix_ratio) + prod_matrix;

				if(res != 0.0) {
					orows[out_idx+nnz] = rows[idx];
					ocols[out_idx+nnz] = cols[idx];
					ovals[out_idx+nnz] = res;
					++nnz;
				}
			}
			printf("rnnz[n]=%d\n", rnnz[n]);

			rnnz[row] = nnz;
			printf("rnnz[%d]=%d\n", row, nnz);

			printf("Adding %d\n", nnz);
			atomicAdd(rnnz+n, nnz);


		}
	}

	template <typename T>
	__global__ void compress_coo(int n,
								 int *rows, int *cols, T*vals,
								 int *crows, int *ccols, T *cvals,
								 int *ex_scan, int n_neighbors) {

		int row = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = row*n_neighbors; // each thread processes one row

		if(row < n) {
			int start = ex_scan[row];

			printf("row=%d, start=%d\n", row, start);
			int cur_out_idx = start;
			for(int j = 0; j < 4; j++) {
				int idx = i+j;

				printf("row=%d, idx=%d\n", row,idx);

				if(vals[idx] != 0.0) {
					crows[cur_out_idx] = rows[idx];
					ccols[cur_out_idx] = cols[idx];
					cvals[cur_out_idx] = vals[idx];
					++cur_out_idx;
				}
			}
		}

	}

	void print(std::string msg) {
	    std::cout << msg << std::endl;
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
				  UMAPParams *params) {

		/**
		 * Calculate mean distance through a parallel reduction
		 */

		print("About to call mean");
		std::cout << "n_neighbors: " << params->n_neighbors << std::endl;

		T *dist_means_dev;
		MLCommon::allocate(dist_means_dev, params->n_neighbors);

		MLCommon::Stats::mean(dist_means_dev, knn_dists, params->n_neighbors, n, false, false);

		cudaDeviceSynchronize();
		print("Done calling mean.");

	    CUDA_CHECK(cudaPeekAtLastError());

		T *dist_means_host = (T*)malloc(params->n_neighbors*sizeof(T));
		MLCommon::updateHost(dist_means_host, dist_means_dev, params->n_neighbors);

		// TODO: In the case of high number of features, it might make more sense to
		// find this on the device.
		float sum = 0.0;
		for(int i = 0; i < params->n_neighbors; i++)
			sum += dist_means_host[i];

		float mean_dist = sum / params->n_neighbors;

	    std::cout << "Mean: " << mean_dist << std::endl;


		/**
		 * Immediately free up memory for subsequent algorithms
		 */
		delete dist_means_host;
		CUDA_CHECK(cudaFree(dist_means_dev));

		T *sigmas;
		T *rhos;

		MLCommon::allocate(sigmas, n);
		MLCommon::allocate(rhos, n);

	    dim3 grid(MLCommon::ceildiv(n, TPB_X), 1, 1);
	    dim3 blk(TPB_X, 1, 1);

		/**
		 * Call smooth_knn_dist to get sigmas and rhos
		 */
		print("About to call smooth_knn_dist");
		smooth_knn_dist<<<grid,blk>>>(knn_dists, n, mean_dist,
				sigmas, rhos,
				params->n_neighbors, params->local_connectivity);
		cudaDeviceSynchronize();

	    CUDA_CHECK(cudaPeekAtLastError());
	    print("Done.");

	    T* sigmas_h = (T*)malloc(n * sizeof(T));
	    T* rhos_h = (T*)malloc(n * sizeof(T));
	    MLCommon::updateHost(sigmas_h, sigmas, n);
	    MLCommon::updateHost(rhos_h, rhos, n);

	    std::cout << "Sigmas: ";
	    for(int i = 0; i < n; i++) {
	    	std::cout << sigmas_h[i] << ", ";
	    }
	    std::cout << std::endl;

	    std::cout << "Rhos: ";
	    for(int i = 0; i < n; i++) {
	    	std::cout << rhos_h[i] << ", ";
	    }
	    std::cout << std::endl;

	    int k = params->n_neighbors;

		/**
		 * Call compute_membership_strength
		 */

		print("About to call compute_membership_strength");
		compute_membership_strength<<<grid,blk>>>(knn_indices, knn_dists,
												  sigmas, rhos,
												  vals, rows, cols,
												  n, params->n_neighbors);
		cudaDeviceSynchronize();

	    CUDA_CHECK(cudaPeekAtLastError());
	    print("Done.");

	    int *orows, *ocols, *rnnz;
		T *ovals;
		MLCommon::allocate(orows, n*k*2, true);
		MLCommon::allocate(ocols, n*k*2, true);
		MLCommon::allocate(ovals, n*k*2, true);
		MLCommon::allocate(rnnz, n+1, true);

		/**
		 * Finish computation of matrix sums, Hadamard products, weighting, etc...
		 */
		print("About to call compute_result");
		compute_result<<<grid, blk>>>(rows, cols, vals,
					   orows, ocols, ovals,
					   rnnz, n, params->n_neighbors, params->set_op_mix_ratio);
		cudaDeviceSynchronize();
		print("Done.");

	    CUDA_CHECK(cudaPeekAtLastError());

	    int *rows_h1 = (int*)malloc(n*k*sizeof(int));
	    int *cols_h1 = (int*)malloc(n*k*sizeof(int));
	    T *vals_h1 = (T*)malloc(n*k*sizeof(T));

	    MLCommon::updateHost(rows_h1, rows, n*k);
	    MLCommon::updateHost(cols_h1, cols, n*k);
	    MLCommon::updateHost(vals_h1, vals, n*k);

	    printf("After Compute Result\n");
	    for(int i = 0; i < n*k; i++) {
	    	printf("row=%d, col=%d, val=%f\n", rows_h1[i], cols_h1[i], vals_h1[i]);
	    }
	    print("Done.");

	    int *rows_h = (int*)malloc(n*k*2*sizeof(int));
	    int *cols_h = (int*)malloc(n*k*2*sizeof(int));
	    T *vals_h = (T*)malloc(n*k*2*sizeof(T));

	    MLCommon::updateHost(rows_h, orows, n*k*2);
	    MLCommon::updateHost(cols_h, ocols, n*k*2);
	    MLCommon::updateHost(vals_h, ovals, n*k*2);

	    printf("After Compute Result\n");
	    for(int i = 0; i < n*k*2; i++) {
	    	printf("row=%d, col=%d, val=%f\n", rows_h[i], cols_h[i], vals_h[i]);
	    }
	    print("Done.");

	    int cur_coo_len = 0;
		MLCommon::updateHost(&cur_coo_len, rnnz+n, 1);
		std::cout << "cur_coo_len=" << cur_coo_len << std::endl;


	    /**
		 * Compress resulting COO matrix
		 */
		int *ex_scan;
		MLCommon::allocate(ex_scan, n+1);

	    thrust::device_ptr<int> dev_rnnz = thrust::device_pointer_cast(rnnz);
	    thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
	    thrust::exclusive_scan(dev_rnnz, dev_rnnz+n, dev_ex_scan);

		cudaDeviceSynchronize();

		int *crows, *ccols;
		T *cvals;
		MLCommon::allocate(crows, cur_coo_len, true);
		MLCommon::allocate(ccols, cur_coo_len, true);
		MLCommon::allocate(cvals, cur_coo_len, true);

	    compress_coo<<<grid, blk>>>(n, orows, ocols, ovals,
	    						    crows, ccols, cvals,
	    						    dev_ex_scan.get(), params->n_neighbors);
		cudaDeviceSynchronize();


	    std::cout << MLCommon::arr2Str(crows, cur_coo_len, "compressed rows") << std::endl;
	    std::cout << MLCommon::arr2Str(ccols, cur_coo_len, "compressed cols") << std::endl;
	    std::cout << MLCommon::arr2Str(cvals, cur_coo_len, "compressed vals") << std::endl;

	    CUDA_CHECK(cudaPeekAtLastError());
	}
}
}
};
