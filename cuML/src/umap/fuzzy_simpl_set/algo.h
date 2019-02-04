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

namespace UMAP {
namespace FuzzySimplSet {
namespace Algo {


	/**
	 * Computes a continuous version of the distance to the kth nearest neighbor.
	 * That is, this is similar to knn-distance but allows continuous k values
	 * rather than requiring an integral k. In essence, we are simply computing
	 * the distance such that the cardinality of fuzzy set we generate is k.
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
	void smooth_knn_dist(const T *knn_dists, int n,
						 T *sigmas, T *rhos,			// Size of n, iniitalized to zeros
						 UMAPParams *params,
						 int n_iter = 64, float bandwidth = 1) {

		/**
		 *
		 * target = log_2(k) * bandwidth
		 *
		 *
		 * Loop through rows, i,  of distance matrix:
		 * 		lo = 0.0
		 * 		hi = MAX
		 * 		mid = 1.0
		 *
		 * 		ith_distances = distances[i]   - pull all items for row i into local memory for each thread
		 * 		non_zero_dists = ith_distances[ith_distances > 0.0] - filter >0, count, calc max
		 * 		if len(non_zero_dists) >= local_connectivity:    -
		 * 		    index = floor(local_connectivity)
		 * 		    interpolation = local_connectivity - index
		 * 		    if index > 0:
		 * 		        rhos[i] = non_zero_dists[index-1]
		 * 		        if interpolation > SMOOTH_K_TOLERANCE:
		 * 		            rhos[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index-1])
		 * 		    else:
		 * 		        rhos[i] = interpolation * non_zero_dists[0]
		 * 		elif len(non_zero_dists) > 0:
		 * 		    rhos[i] = max(non_zero_dists)
		 *
		 *
		 * 		for n in range(n_iter):
		 *
		 * 		    psum = 0.0
		 * 		    loop through columns j of dist matrix:        // this could be slow
		 * 		        d = distances[i, j] - rhos[i]
		 * 		        if d > 0:
		 * 		            psum += np.exp(-(d/mid))
		 * 		        else:
		 * 		            psum += 1.0
		 *
		 *
		 * 		   if element_wise_abs(psum-target) < SMOOTH_K_TOLERANCE:
		 * 		   	   break;
		 *
		 * 		   if psum > target:
		 * 		       hi = mid
		 * 		       mid = (lo + hi) / 2.0
		 * 		   else:
		 * 		       lo = mid
		 * 		       if hi = MAX:
		 * 		           mid *= 2
		 * 		       else:
		 * 		           mid = (lo + hi) / 2.0
		 *
		 * 	sigmas[i] = mid
		 *
		 * 	if rhos[i] > 0.0:
		 * 	    if sigmas[i] < MIN_K_DIST_SCALE * mean(ith_distances):
		 * 	    	sigmas[i] = MIN_K_DIST_SCALE * mean(ith_distances)
		 *
		 * 	    else:
		 * 	        if sigmas[i] < MIN_K_DIST_SCALE * mean(distances):
		 * 	            sigmas[i] = MIN_K_DIST_SCALE * mean(distances)
		 */

	}


	/**
	 * Construct the membership stength data for the 1-skeleton of each local
	 * fuzzy simplicial set -- this is formed as a sparse matrix where each row is
	 * a local fuzzy simplicial set, with a membership strength for the 1-simplex
	 * to each other data point.
	 *
	 * @param knn_indices: the knn index matrix of size (n, k)
	 * @param knn_dists: the knn distance matrix of size (n, k)
	 * @param sigmas: array of size n representing distance to kth nearest neighbor
	 * @param rhos: array of size n representing distance to the first nearest neighbor
	 *
	 * @return rows: long matrix of size (n, k)
	 * 		   cols: long matrix of size (n, k)
	 * 		   vals: T matrix of size (n, k)
	 *
	 * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
	 */
	template<typename T>
	void compute_membership_strength(const T *knn_indices, const T *knn_dists,
									 const T *sigmas, const T *rhos,
									 long *rows, long *cols, T *vals,
									 int n, UMAPParams *params) {

		/**
		 * Initialize rows, cols, vals
		 */

		/**
		 * Nested loop: samples (i) then neighbors (j)
		 *
		 * if knn_indices[i, j] == -1: continue
		 *
		 * if knn_indices[i, j] == i:
		 *     val = 0.0
		 * else if knn_dists[i, j] - rhos[i] <= 0.0:
		 *     val = 1.0
		 * else:
		 *     val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
		 *
		 * rows[i*n_neighbors + j] = i
		 * cols[i*n_neighbors + j] = knn_indices[i, j]
		 * vals[i*n_neighbors + j] = val
		 *
		 * return rows, cols, vals
	     */
	}


	/**
	 * Given a set of X, a neighborhoos size, and a measure of distance, compute
	 * the fuzzy simplicial set (here represented as a fuzzy graph in the form of
	 * a sparse coo matrix) associated to the data. This is done by locally
	 * approximating geodesic (manifold surface) distance at each point, creating
	 * a fuzzy simplicial set for each such point, and then combining all the local
	 * fuzzy simplicial sets into a global one via a fuzzy union.
	 */
	void launcher(const long *knn_indices, const T *knn_dists,
			      int n, int *rows, int *cols, T *vals,
				  UMAPParams *params, int algorithm) {

		T *sigmas;
		T *rhos;

		/**
		 * Call smooth_knn_dist to get sigmas and rhos
		 */
		smooth_knn_dist(knn_dists, n, sigmas, rhos, params);

		/**
		 * Call compute_membership_strength
		 */
		compute_membership_strength(knn_indices, knn_dists,
								    sigmas, rhos,
								    rows, cols, vals,
								    n, params);


		/**
		 * - result = <Eliminate zeros from coo>
		 * - transpose = <Transpose result>
		 * - prod_matrix = <Multiply coo matrix by its transpose> (would cuSparse help here?)
		 *
		 * - result = set_op_mix_ratio * (result + transpose - prod_matrix) +
		 * 			   (1.0 - set_op_mix_ratio) * prod_matrix
		 *
		 * - result = eliminate zeros
		 *
		 * - return result
		 */




	}


}
}
};
