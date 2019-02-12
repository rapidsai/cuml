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
									 T *tvals, int *trows, int *tcols, // result coo transposed rows
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

				// Transposed- swap rows/cols. Will sort by row.
				tcols[idx] = i;
				trows[idx] = cur_knn_ind;
				tvals[idx] = val;
			}
		}
	}

	template<typename T>
	__global__ void compute_result(int *rows, int *cols, T *vals,
								   int *trows, int *tcols, T *tvals,
								   int *orows, int *ocols, T *ovals,
								   int *rnnz, int n,
								   int n_neighbors, float set_op_mix_ratio) {

		int row = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = row*n_neighbors; // each thread processes one row
		// Grab the n_neighbors from our transposed matrix,

		int nnz = 0;
		for(int j = 0; j < n_neighbors; j++) {

			int idx = i+j;

			T result = vals[idx];
			T transpose = tvals[idx];
			T prod_matrix = vals[idx] * tvals[idx];

			T res = set_op_mix_ratio * (result - transpose - prod_matrix)
							+ (1.0 - set_op_mix_ratio) + prod_matrix;

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
	__global__ void compress_coo(int *rows, int *cols, T*vals,
								 int *crows, int *ccols, T *cvals,
								 int *ex_scan, int n_neighbors) {

		int row = (blockIdx.x * TPB_X) + threadIdx.x;
		int i = row*n_neighbors; // each thread processes one row

		int start = ex_scan[i];
		int cur_out_idx = start;

		for(int j = 0; j < n_neighbors; j++) {
			int idx = i*n_neighbors+j;
			if(vals[idx] != 0.0) {
				crows[cur_out_idx] = rows[idx];
				ccols[cur_out_idx] = cols[idx];
				cvals[cur_out_idx] = vals[idx];

				++cur_out_idx;
			}
		}
	}


	void print(char* msg) {
	    std::cout << msg << std::endl;
	}

	template<typename T>
	void coo_sort(const int m, const int n, const int nnz,
				  int *rows, int *cols, T *vals) {

		cusparseHandle_t handle = NULL;
	    cudaStream_t stream = NULL;

	    size_t pBufferSizeInBytes = 0;
	    void *pBuffer = NULL;
	    int *d_P = NULL;

	    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	    cusparseCreate(&handle);

	    cusparseSetStream(handle, stream);

	    cusparseXcoosort_bufferSizeExt(
	        handle,
	        m,
	        n,
	        nnz,
	        rows,
	        cols,
	        &pBufferSizeInBytes
	    );

	    cudaMalloc(&d_P, sizeof(int)*nnz);
	    cudaMalloc(&pBuffer, sizeof(char)* pBufferSizeInBytes);

	    cusparseCreateIdentityPermutation(
	        handle,
	        nnz,
	        d_P);

	    cusparseXcoosortByRow(
	        handle,
	        m,
	        n,
	        nnz,
	        rows,
	        cols,
	        d_P,
	        pBuffer
	    );

	    T* vals_sorted;
	    MLCommon::allocate(vals_sorted, m*n);

	    cusparseSgthr(
	        handle,
	        nnz,
	        vals,
	        vals_sorted,
	        d_P,
	        CUSPARSE_INDEX_BASE_ZERO
	    );
	    cudaDeviceSynchronize(); /* wait until the computation is done */

	    MLCommon::copy(vals, vals_sorted, m*n);

	    cudaFree(d_P);
	    cudaFree(vals_sorted);
	    cudaFree(pBuffer);
	    cusparseDestroy(handle);
	    cudaStreamDestroy(stream);
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

		// We will keep arrays, with space O(n*k*9) representing 3 O(n^2) matrices.
		int *trows, *tcols;
		T *tvals;
		MLCommon::allocate(trows, n*k, true);
		MLCommon::allocate(tcols, n*k, true);
		MLCommon::allocate(tvals, n*k, true);

		/**
		 * Call compute_membership_strength
		 */

		print("About to call compute_membership_strength");
		compute_membership_strength<<<grid,blk>>>(knn_indices, knn_dists,
												  sigmas, rhos,
												  vals, rows, cols,
												  tvals, trows, tcols,
												  n, params->n_neighbors);
		cudaDeviceSynchronize();

		/**
		 * We use a strategy here to minimize the amount of memory for
		 * storage while simultaneously minimizing the amount of
		 * time spent scanning through indices.
		 *
		 * In order to accomplish this in the first naive computation,
		 * I am storing the transposed matrix in its own set of
		 * coo arrays. This transposed form of A then gets sorted by
		 * row using the following function.
		 *
		 * The original A's lookup is deterministic since the hadamard
		 * product and additional element-wise operations that follow
		 * are done in parallel for each row. Since the lookup for A
		 * transpose is not deterministic. We avoid having to store
		 * an additional array, with a size n upper bound, by making
		 * a log(n) lookup for the items by row in the transposed matrix.
		 */
		coo_sort(n, params->n_neighbors, n*k, trows, tcols, tvals);
		cudaDeviceSynchronize();

		// Need to do array sort here.

	    int *rows_h = (int*) malloc(n*k*sizeof(int));
	    int *cols_h = (int*) malloc(n*k*sizeof(int));
	    float *vals_h = (float*) malloc(n*k*sizeof(float));

	    MLCommon::updateHost(rows_h, trows, n*k);
	    MLCommon::updateHost(cols_h, tcols, n*k);
	    MLCommon::updateHost(vals_h, tvals, n*k);

	    printf("Transposed\n");
	    for(int i = 0; i < n*k; i++) {
	    	printf("row=%d, col=%d, val=%f\n", rows_h[i], cols_h[i], vals_h[i]);
	    }

	    print("Normal\n");
	    MLCommon::updateHost(rows_h, rows, n*k);
	    MLCommon::updateHost(cols_h, cols, n*k);
	    MLCommon::updateHost(vals_h, vals, n*k);

	    for(int i = 0; i < n*k; i++) {
	    	printf("row=%d, col=%d, val=%f\n", rows_h[i], cols_h[i], vals_h[i]);
	    }



	    CUDA_CHECK(cudaPeekAtLastError());
	    print("Done.");

	    int *orows, *ocols, *rnnz;
		T *ovals;
		MLCommon::allocate(orows, n*k, true);
		MLCommon::allocate(ocols, n*k, true);
		MLCommon::allocate(ovals, n*k, true);
		MLCommon::allocate(rnnz, n+1, true);


		/**
		 * Finish computation of matrix sums, Hadamard products, weighting, etc...
		 */
		print("About to call compute_result");
		compute_result<<<grid, blk>>>(rows, cols, vals,
					   trows, tcols, tvals,
					   orows, ocols, ovals,
					   rnnz, n, params->n_neighbors, params->set_op_mix_ratio);
		cudaDeviceSynchronize();
		print("Done.");

	    CUDA_CHECK(cudaPeekAtLastError());


		CUDA_CHECK(cudaFree(trows));
		CUDA_CHECK(cudaFree(tcols));
		CUDA_CHECK(cudaFree(tvals));



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
		cudaDeviceSynchronize();

		int *crows, *ccols;
		T *cvals;
		MLCommon::allocate(crows, cur_coo_len, true);
		MLCommon::allocate(ccols, cur_coo_len, true);
		MLCommon::allocate(cvals, cur_coo_len, true);

	    compress_coo<<<grid, blk>>>(orows, ocols, ovals,
	    						    crows, ccols, cvals,
	    						    rnnz, params->n_neighbors);
		cudaDeviceSynchronize();

	    CUDA_CHECK(cudaPeekAtLastError());
	}
}
}
};
