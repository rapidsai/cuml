/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/manifold/umapparams.h>
#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn.hpp>

#include <common/cudart_utils.h>
#include "cuda_utils.h"


#include "sparse/coo.h"
#include "stats/mean.h"
#include "linalg/reduce.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <string>

namespace UMAPAlgo {
namespace FuzzySimplSet {
namespace Naive {

using namespace ML;

static const float MAX_FLOAT = std::numeric_limits<float>::max();
static const float MIN_FLOAT = std::numeric_limits<float>::min();

static const float SMOOTH_K_TOLERANCE = 1e-5;
static const float MIN_K_DIST_SCALE = 1e-3;


/**
 * Computes a continuous version of the distance to the kth nearest neighbor.
 * That is, this is similar to knn-distance but allows continuous k values
 * rather than requiring an integral k. In essence, we are simply computing
 * the distance such that the cardinality of fuzzy set we generate is k.
 *
 * TODO: The data needs to be in column-major format (and the indices
 * of knn_dists and knn_inds transposed) so that we can take advantage
 * of read-coalescing within each block where possible.
 *
 * @param knn_dists: Distances to nearest neighbors for each sample. Each row should
 *                   be a sorted list of distances to a given sample's nearest neighbors.
 *
 * @param n: The number of samples
 * @param mean_dist: The mean distance
 * @param sigmas: An array of size n representing the distance to the kth nearest neighbor,
 *                as suitably approximated.
 * @param rhos:  An array of size n representing the distance to the 1st nearest neighbor
 *               for each point.
 * @param n_neighbors: The number of neighbors
 *
 * @param local_connectivity: The local connectivity required -- i.e. the number of nearest
 *                            neighbors that should be assumed to be connected at a local
 *                            level. The higher this value the more connecte the manifold
 *                            becomes locally. In practice, this should not be more than the
 *                            local intrinsic dimension of the manifold.
 *
 * @param n_iter The number of smoothing iterations to run
 * @param bandwidth Scale factor for log of neighbors
 *
 * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
 *
 */

template <int TPB_X, typename T>
__global__ void set_sigmas_kernel(int n, T *sigmas,
		T *rhos, T *mid, T *mean_dists, T mean_dist, T target) {

	int row = (blockIdx.x * TPB_X) + threadIdx.x;
	if(row >= n) return;

	T cur_mid = mid[row];

	// depends only on the row
	if (rhos[row] > 0.0) {
	  T ith_distances_mean = mean_dists[row];
	  if (cur_mid < MIN_K_DIST_SCALE * ith_distances_mean)
		sigmas[row] = MIN_K_DIST_SCALE * ith_distances_mean;
	} else {
	  if (cur_mid < MIN_K_DIST_SCALE * mean_dist)
		sigmas[row] = MIN_K_DIST_SCALE * mean_dist;
	}
}


template <int TPB_X, typename T>
__global__ void find_sigma_single_iter2(
	const T *knn_dists,
	int n,
	T *psum,
	T *hi,
	T *mid,
	T *lo,
	T *sigmas,
	int n_neighbors,
	T target) {

	// row-based dense matrix- single thread per row
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row >= n) return;

	T cur_psum = psum[row];

	if(fabsf(cur_psum - target) < SMOOTH_K_TOLERANCE) {
		return;
	}

	T cur_hi = hi[row];
	T cur_mid = mid[row];
	T cur_lo = lo[row];

  if (cur_psum > target) {
	cur_hi = cur_mid;
	cur_mid = (cur_lo + cur_hi) / 2.0;
  } else {
	cur_lo = cur_mid;
	if (cur_hi == MAX_FLOAT)
	  cur_mid *= 2;
	else
	  cur_mid = (cur_lo + cur_hi) / 2.0;
  }

  hi[row] = cur_hi;
  mid[row] = cur_mid;
  lo[row] = cur_lo;
  sigmas[row] = cur_mid;
}


template <int TPB_X, typename T>
__global__ void find_sigma_single_iter(
	const T *knn_dists,
	int n,
	T *psum,
	T *mid,
	T *sigmas,
	T *rhos,
	int n_neighbors,
	T target) {

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(idx >= n * n_neighbors) return;

	int row = idx / n_neighbors;

	if(fabsf(psum[row] - target) < SMOOTH_K_TOLERANCE)
		return;

	T d = knn_dists[idx] - rhos[row];
	atomicAdd(psum + row, d > 0.0 ? exp(-(d/mid[row])) : 1.0);
}

template <int TPB_X, typename T>
__global__ void smooth_knn_dist_kernel(
	const T *knn_dists,
	int n,
	T *means,
	T *total_nonzero_arr,
	T *start_nonzero_arr,
	T mean_dist,
	T *sigmas,
	T *rhos,  // Size of n, iniitalized to zeros
	int n_neighbors, T local_connectivity = 1.0) {

	// row-based dense matrix- single thread per row
	int row = (blockIdx.x * TPB_X) + threadIdx.x;
	int i =
	row * n_neighbors;  // each thread processes one row of the dist matrix

	if (row >= n) return;

	int total_nonzero = total_nonzero_arr[row];

	// depends only on the row
	if (total_nonzero >= local_connectivity) {
		int start_nonzero = start_nonzero_arr[row];
		int index = floor(local_connectivity);
		T interpolation = local_connectivity - index;

	  if (index > 0) {
		T penult = knn_dists[i + start_nonzero + (index - 1)];
		rhos[row] = penult;

		if (interpolation > SMOOTH_K_TOLERANCE) {
		  T ult = knn_dists[i + start_nonzero + index];
		  rhos[row] +=
			interpolation * (ult - penult);
		}
	  } else
		rhos[row] = interpolation * knn_dists[i + start_nonzero];
	} else if (total_nonzero > 0)
	  // we assume columns are sorted such that max nonzero will
      // be the last column
	  rhos[row] = knn_dists[i + (n_neighbors-1)];

}

/**
 * Construct the membership strength data for the 1-skeleton of each local
 * fuzzy simplicial set -- this is formed as a sparse matrix (COO) where each
 * row is a local fuzzy simplicial set, with a membership strength for the
 * 1-simplex to each other data point.
 *
 * @param knn_indices: the knn index matrix of size (n, k)
 * @param knn_dists: the knn distance matrix of size (n, k)
 * @param sigmas: array of size n representing distance to kth nearest neighbor
 * @param rhos: array of size n representing distance to the first nearest neighbor
 * @param vals: T array of size n*k
 * @param rows: int64_t array of size n
 * @param cols: int64_t array of size k
 * @param n Number of samples (rows in knn indices/distances)
 * @param n_neighbors number of columns in knn indices/distances
 *
 * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
 */
template <int TPB_X, typename T>
__global__ void compute_membership_strength_kernel(
  const int64_t *knn_indices,
  const float *knn_dists,          // nn outputs
  const T *sigmas, const T *rhos,  // continuous dists to nearest neighbors
  T *vals, int *rows, int *cols,   // result coo
  int n, int n_neighbors) {        // model params

  // row-based matrix is best
  int idx = (blockIdx.x * TPB_X) + threadIdx.x;

  if (idx < n * n_neighbors) {
    int row = idx / n_neighbors;  // one neighbor per thread

    int64_t cur_knn_ind = knn_indices[idx];

    if (cur_knn_ind != -1) {

      T cur_rho = rhos[row];
      T cur_sigma = sigmas[row];

      T cur_knn_dist = knn_dists[idx];

      T val = 0.0;
      if (cur_knn_ind == row)
        val = 0.0;
      else if (cur_knn_dist - cur_rho <= 0.0 || cur_sigma == 0.0)
        val = 1.0;
      else {
        val = exp(-((cur_knn_dist - cur_rho) / (cur_sigma)));

        if (val < MIN_FLOAT) val = MIN_FLOAT;
      }

      rows[idx] = row;
      cols[idx] = cur_knn_ind;
      vals[idx] = val;
    }
  }
}

/*
 * Sets up and runs the knn dist smoothing
 */
template <int TPB_X, typename T>
void smooth_knn_dist(int n, const int64_t *knn_indices, const T *knn_dists,
                     T *rhos, T *sigmas, UMAPParams *params, int n_neighbors,
                     T local_connectivity,
                     std::shared_ptr<deviceAllocator> d_alloc,
                     cudaStream_t stream, T bandwidth = 1.0, int n_iter = 64) {

  MLCommon::device_buffer<T> dist_means_dev(d_alloc, stream, n_neighbors);
  MLCommon::device_buffer<T> full_dist_means(d_alloc, stream, n);
  MLCommon::device_buffer<T> total_nonzero(d_alloc, stream, n);
  MLCommon::device_buffer<T> start_nonzero(d_alloc, stream, n);

  // perform column-wise mean
  MLCommon::LinAlg::reduce<T>(full_dist_means.data(), knn_dists, n_neighbors, n, 0.0f,
              true, true, stream,
              false, MLCommon::Nop<T, int>(), MLCommon::Sum<T>(),
              [n_neighbors] __device__ (T in) { return in / n_neighbors; });

  MLCommon::LinAlg::reduce<T>(total_nonzero.data(), knn_dists, n_neighbors, n, 0.0f,
              true, true, stream,
              false, [] __device__ (T val, int idx) { return val > 0.0 ? 1.0 : 0.0; },
              MLCommon::Sum<T>(),
              MLCommon::Nop<T>());

  MLCommon::LinAlg::reduce<T>(start_nonzero.data(), knn_dists, n_neighbors, n, MAX_FLOAT,
              true, true, stream,
              false, [] __device__ (T val, int idx) {
	  	  	  	  return val > 0.0 ? T(idx) : MAX_FLOAT; },
              [] __device__ (T a, T b) {
	  	  	  		  return a < b ? a : b; },
              MLCommon::Nop<T>());

  MLCommon::Stats::mean(dist_means_dev.data(), knn_dists, 1, n_neighbors * n,
                        false, false, stream);

  T mean_dist = 0.0;
  MLCommon::updateHost(&mean_dist, dist_means_dev.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /**
   * Smooth kNN distances to be continuous
   */
  int rows = MLCommon::ceildiv(n, TPB_X);

  smooth_knn_dist_kernel<TPB_X><<<rows, TPB_X, 0, stream>>>(
    knn_dists, n, full_dist_means.data(), total_nonzero.data(), start_nonzero.data(),
    mean_dist, sigmas, rhos, n_neighbors, local_connectivity);
	CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::device_buffer<T> psums(d_alloc, stream, n);
  MLCommon::device_buffer<T> hi_buffer(d_alloc, stream, n);

  T *hi = hi_buffer.data();
  T *mid = total_nonzero.data();
  T *lo = start_nonzero.data();

  MLCommon::LinAlg::unaryOp<T>(hi, hi, n, [] __device__(T input) { return MAX_FLOAT; }, stream);
  MLCommon::LinAlg::unaryOp<T>(mid, mid, n, [] __device__(T input) { return 1.0; }, stream);
  CUDA_CHECK(cudaMemsetAsync(lo, 0, n * sizeof(T), stream));

  T target = log2f(n_neighbors) * bandwidth;
	for (int iter = 0; iter < n_iter; iter++) {

		CUDA_CHECK(cudaMemsetAsync(psums.data(), 0, n * sizeof(T), stream));

		int neighbors = MLCommon::ceildiv(n*n_neighbors, TPB_X);
		find_sigma_single_iter<TPB_X, T><<<neighbors, TPB_X, 0, stream>>>(
			knn_dists, n,
			psums.data(), mid, sigmas, rhos,
			n_neighbors, target);

		find_sigma_single_iter2<TPB_X, T><<<rows, TPB_X, 0, stream>>>(
			knn_dists, n,
			psums.data(), hi, mid, lo, sigmas,
			n_neighbors, target);
	}

	set_sigmas_kernel<TPB_X, T><<<rows, TPB_X, 0, stream>>>(n, sigmas, rhos, mid,
			full_dist_means.data(), mean_dist, target);
	CUDA_CHECK(cudaPeekAtLastError());
//
//	std::cout << MLCommon::arr2Str(rhos, n, "rhos", stream) << std::endl;
//	std::cout << MLCommon::arr2Str(sigmas, n, "sigmas", stream) << std::endl;


}

/**
 * Given a set of X, a neighborhood size, and a measure of distance, compute
 * the fuzzy simplicial set (here represented as a fuzzy graph in the form of
 * a sparse coo matrix) associated to the data. This is done by locally
 * approximating geodesic (manifold surface) distance at each point, creating
 * a fuzzy simplicial set for each such point, and then combining all the local
 * fuzzy simplicial sets into a global one via a fuzzy union.
 *
 * @param n the number of rows/elements in X
 * @param knn_indices indexes of knn search
 * @param knn_dists distances of knn search
 * @param n_neighbors number of neighbors in knn search arrays
 * @param out The output COO sparse matrix
 * @param params UMAPParams config object
 * @param d_alloc the device allocator to use for temp memory
 * @param stream cuda stream to use for device operations
 */
template <int TPB_X, typename T>
void launcher(int n, const int64_t *knn_indices, const float *knn_dists,
              int n_neighbors, MLCommon::Sparse::COO<T> *out,
              UMAPParams *params, std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream) {
  /**
   * Calculate mean distance through a parallel reduction
   */
  MLCommon::device_buffer<T> sigmas(d_alloc, stream, n);
  MLCommon::device_buffer<T> rhos(d_alloc, stream, n);
  CUDA_CHECK(cudaMemsetAsync(sigmas.data(), 0, n * sizeof(T), stream));
  CUDA_CHECK(cudaMemsetAsync(rhos.data(), 0, n * sizeof(T), stream));

  smooth_knn_dist<TPB_X, T>(n, knn_indices, knn_dists, rhos.data(),
                            sigmas.data(), params, n_neighbors,
                            params->local_connectivity, d_alloc, stream);

  MLCommon::Sparse::COO<T> in(d_alloc, stream, n * n_neighbors, n, n);

  CUDA_CHECK(cudaPeekAtLastError());

  /**
   * Compute graph of membership strengths
   */

  dim3 grid_elm(MLCommon::ceildiv(n * n_neighbors, TPB_X), 1, 1);
  dim3 blk_elm(TPB_X, 1, 1);

  compute_membership_strength_kernel<TPB_X><<<grid_elm, blk_elm, 0, stream>>>(
    knn_indices, knn_dists, sigmas.data(), rhos.data(), in.vals(), in.rows(),
    in.cols(), in.n_rows, n_neighbors);
  CUDA_CHECK(cudaPeekAtLastError());

  /**
   * Combines all the fuzzy simplicial sets into a global
   * one via a fuzzy union. (Symmetrize knn graph).
   */

  float set_op_mix_ratio = params->set_op_mix_ratio;

  MLCommon::Sparse::coo_symmetrize<TPB_X, T>(&in, out, d_alloc, stream);

  MLCommon::device_buffer<int> out_row_ind(d_alloc, stream, out->n_rows);
  sorted_coo_to_csr(out, out_row_ind.data(), d_alloc, stream);

  // Could compute degree, maybe?

  int *cols = out->cols();
  int *rows = out->rows();
  T *vals = out->vals();

  /**
   * Performing a rolling (w*X+X.T) + ((1-w)*X*X.T)
   */
  MLCommon::Sparse::csr_row_op(
		  out_row_ind.data(), out->n_rows, out->nnz,
		  [cols, rows, vals, set_op_mix_ratio] __device__(int row, int start_idx, int stop_idx) {
	  int last_col = -1;

	  T prod = 1.0;
	  T sum = 0.0;
	  int n = 0.0;

	  for(int cur_idx = start_idx; cur_idx < stop_idx; cur_idx++) {
		  int cur_col = cols[cur_idx];
		  int cur_val = vals[cur_idx];

		  bool write_idx = 0.0;
		  if(start_idx == stop_idx-1 && n == 0.0) {
			  write_idx = cur_idx;
		  } else{
			  write_idx = cur_idx-1;
		  }

		  if((cur_col != last_col && last_col != -1) || start_idx == stop_idx-1) {
			  prod = prod * n > 1.0; // simulate transpose being zero
			  vals[write_idx] = set_op_mix_ratio *
					  (sum - prod) + (1.0 - set_op_mix_ratio) * prod;
			  prod = cur_val;
			  sum = cur_val;
			  n = 1;
		  } else {
			  vals[write_idx] = 0.0;
			  prod *= cur_val;
			  sum *= cur_val;
			  n += 1;
		  }
	  }
  }, stream);
}

}  // namespace Naive
}  // namespace FuzzySimplSet
};  // namespace UMAPAlgo
