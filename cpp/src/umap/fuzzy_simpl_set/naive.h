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

#include "knn/knn.h"
#include "umap/umapparams.h"

#include "cuda_utils.h"

#include "stats/mean.h"
#include "sparse/coo.h"

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
             * @param k: The number of neighbors
             *
             * @param local_connectivity: The local connectivity required -- i.e. the number of nearest
             *                            neighbors that should be assumed to be connected at a local
             *                            level. The higher this value the more connecte the manifold
             *                            becomes locally. In practice, this should not be more than the
             *                            local intrinsic dimension of the manifold.
             *
             * @param sigmas: An array of size n representing the distance to the kth nearest neighbor,
             *                as suitably approximated.
             * @parasm rhos:  An array of size n representing the distance to the 1st nearest neighbor
             *                for each point.
             *
             * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
             *
             */
            template<int TPB_X, typename T>
            __global__ void smooth_knn_dist_kernel(
                    const T *knn_dists, int n,
                    float mean_dist, T *sigmas,
                    T *rhos,            // Size of n, iniitalized to zeros
                    int n_neighbors,
                    float local_connectivity = 1.0,
                    int n_iter = 64,
                    float bandwidth = 1.0) {

                // row-based matrix 1 thread per row
                int row = (blockIdx.x * TPB_X) + threadIdx.x;
                int i = row * n_neighbors; // each thread processes one row of the dist matrix

                if (row < n) {

                    float target = __log2f(n_neighbors) * bandwidth;

                    float lo = 0.0;
                    float hi = MAX_FLOAT;
                    float mid = 1.0;

                    int total_nonzero = 0;
                    int max_nonzero = -1;

                    int start_nonzero = -1;
                    float sum = 0.0;

                    for (int idx = 0; idx < n_neighbors; idx++) {

                        float cur_dist = knn_dists[i+idx];
                        sum += cur_dist;

                        if (cur_dist > 0.0) {
                            if (start_nonzero == -1)
                                start_nonzero = idx;
                            total_nonzero++;
                        }

                        if (cur_dist > max_nonzero)
                            max_nonzero = cur_dist;
                    }

                    float ith_distances_mean = sum / float(n_neighbors);
                    if (total_nonzero >= local_connectivity) {
                        int index = int(floor(local_connectivity));
                        float interpolation = local_connectivity - index;

                        if (index > 0) {
                            rhos[row] = knn_dists[i+start_nonzero+(index-1)];

                            if (interpolation > SMOOTH_K_TOLERANCE) {
                                rhos[row] += interpolation
                                        * (knn_dists[i+start_nonzero+index]
                                                  - knn_dists[i+start_nonzero+(index-1)]);
                            }
                        } else
                            rhos[row] = interpolation * knn_dists[i+start_nonzero];
                    } else if (total_nonzero > 0)
                        rhos[row] = max_nonzero;

                    for (int iter = 0; iter < n_iter; iter++) {

                        float psum = 0.0;

                        for (int j = 1; j < n_neighbors; j++) {
                            float d = knn_dists[i + j] - rhos[row];
                            if (d > 0)
                                psum += exp(-(d / mid));
                            else
                                psum += 1.0;
                        }

                        if (fabsf(psum - target) < SMOOTH_K_TOLERANCE) {
                            break;
                        }

                        if (psum > target) {
                            hi = mid;
                            mid = (lo + hi) / 2.0;
                        } else {
                            lo = mid;
                            if (hi == MAX_FLOAT)
                                mid *= 2;
                            else
                                mid = (lo + hi) / 2.0;
                        }
                    }

                    sigmas[row] = mid;

                    if (rhos[row] > 0.0) {
                        if (sigmas[row] < MIN_K_DIST_SCALE * ith_distances_mean)
                            sigmas[row] = MIN_K_DIST_SCALE * ith_distances_mean;
                    } else {
                        if (sigmas[row] < MIN_K_DIST_SCALE * mean_dist)
                            sigmas[row] = MIN_K_DIST_SCALE * mean_dist;
                    }
                }
            }

            /**
             * Construct the membership strength data for the 1-skeleton of each local
             * fuzzy simplicial set -- this is formed as a sparse matrix (COO) where each
             * row is a local fuzzy simplicial set, with a membership strength for the
             * 1-simplex to each other data point.
             *
             * TODO: Optimize for coalesced reads (use col-major inputs).
             *
             * @param knn_indices: the knn index matrix of size (n, k)
             * @param knn_dists: the knn distance matrix of size (n, k)
             * @param sigmas: array of size n representing distance to kth nearest neighbor
             * @param rhos: array of size n representing distance to the first nearest neighbor
             *
             * @return rows: long array of size n
             *         cols: long array of size k
             *         vals: T array of size n*k
             *
             * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
             */
            template<int TPB_X, typename T>
            __global__ void compute_membership_strength_kernel(
                    const long *knn_indices,
                    const float *knn_dists,  // nn outputs
                    const T *sigmas, const T *rhos, // continuous dists to nearest neighbors
                    T *vals, int *rows, int *cols,  // result coo
                    int n, int n_neighbors) {    // model params

                // row-based matrix is best
                int row = (blockIdx.x * TPB_X) + threadIdx.x;
                int i = row * n_neighbors; //   one row per thread

                if (row < n) {

                    T cur_rho = rhos[row];
                    T cur_sigma = sigmas[row];

                    for (int j = 0; j < n_neighbors; j++) {

                        int idx = i + j;

                        long cur_knn_ind = knn_indices[idx];
                        T cur_knn_dist = knn_dists[idx];

                        if (cur_knn_ind == -1)
                            continue;

                        double val = 0.0;
                        if (cur_knn_ind == row)
                            val = 0.0;
                        else if (cur_knn_dist - cur_rho <= 0.0)
                            val = 1.0;
                        else {
                            val = exp(
                                    -((double(cur_knn_dist) - double(cur_rho)) / (double(cur_sigma))));

                            if(val < MIN_FLOAT)
                                val = MIN_FLOAT;
                        }

                        rows[idx] = row;
                        cols[idx] = cur_knn_ind;
                        vals[idx] = float(val);
                    }
                }
            }

            /*
             * Sets up and runs the knn dist smoothing
             */
            template< int TPB_X, typename T>
            void smooth_knn_dist(int n, const long *knn_indices, const float *knn_dists,
                    T *rhos, T *sigmas, UMAPParams *params, int n_neighbors, float local_connectivity,
                    cudaStream_t stream) {

                int blks = MLCommon::ceildiv(n, TPB_X);

                dim3 grid(blks, 1, 1);
                dim3 blk(TPB_X, 1, 1);

                T *dist_means_dev;
                MLCommon::allocate(dist_means_dev, n_neighbors);

                MLCommon::Stats::mean(dist_means_dev, knn_dists,
                        n_neighbors, n, false, false, stream);
                CUDA_CHECK(cudaPeekAtLastError());

                T *dist_means_host = (T*) malloc(n_neighbors * sizeof(T));
                MLCommon::updateHost(dist_means_host, dist_means_dev,n_neighbors, stream);

                float sum = 0.0;
                for (int i = 0; i < n_neighbors; i++)
                    sum += dist_means_host[i];

                T mean_dist = sum / float(n_neighbors);

                /**
                 * Clean up memory for subsequent algorithms
                 */
                free(dist_means_host);
                CUDA_CHECK(cudaFree(dist_means_dev));

                /**
                 * Smooth kNN distances to be continuous
                 */
                smooth_knn_dist_kernel<TPB_X><<<grid, blk, 0, stream>>>(knn_dists, n, mean_dist, sigmas,
                        rhos, n_neighbors, local_connectivity);
                CUDA_CHECK(cudaPeekAtLastError());
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
             * @param rrows output COO rows array
             * @param rcols output COO cols array
             * @param rvals output COO vals array
             * @param params UMAPParams config object
             * @param stream cuda stream to use for device operations
             */
            template<int TPB_X, typename T>
            void launcher(int n, const
                    long *knn_indices, const float *knn_dists,
                    int n_neighbors,
                   MLCommon::Sparse::COO<T> *out,
                   UMAPParams *params, cudaStream_t stream) {

                /**
                 * All of the kernels in this algorithm are row-based and
                 * upper-bounded by k. Prefer 1-row per thread, scheduled
                 * as a single dimension.
                 */
                dim3 grid(MLCommon::ceildiv(n, TPB_X), 1, 1);
                dim3 blk(TPB_X, 1, 1);

                /**
                 * Calculate mean distance through a parallel reduction
                 */
                T *sigmas;
                T *rhos;
                MLCommon::allocate(sigmas, n, true);
                MLCommon::allocate(rhos, n, true);

                smooth_knn_dist<TPB_X, T>(n, knn_indices, knn_dists,
                        rhos, sigmas, params, n_neighbors, params->local_connectivity, stream
                );

                MLCommon::Sparse::COO<T> in(n*n_neighbors, n, n);

                if(params->verbose) {
                    std::cout << "Smooth kNN Distances" << std::endl;
                    std::cout << MLCommon::arr2Str(sigmas, n, "sigmas", stream) << std::endl;
                    std::cout << MLCommon::arr2Str(rhos, n, "rhos", stream) << std::endl;
                }

                CUDA_CHECK(cudaPeekAtLastError());

                /**
                 * Compute graph of membership strengths
                 */
                compute_membership_strength_kernel<TPB_X><<<grid, blk, 0, stream>>>(knn_indices,
                        knn_dists, sigmas, rhos, in.vals, in.rows, in.cols, in.n_rows,
                        n_neighbors);
                CUDA_CHECK(cudaPeekAtLastError());

                if(params->verbose) {
                    std::cout << "Compute Membership Strength" << std::endl;
                    std::cout << in << std::endl;
                }


                /**
                 * Combines all the fuzzy simplicial sets into a global
                 * one via a fuzzy union. (Symmetrize knn graph and weight
                 * based on directionality).
                 */
                float set_op_mix_ratio = params->set_op_mix_ratio;
                MLCommon::Sparse::coo_symmetrize<TPB_X, T>(&in, out,
                    [set_op_mix_ratio] __device__(int row, int col, T result, T transpose) {
                        T prod_matrix = result * transpose;
                        T res = set_op_mix_ratio
                                * (result + transpose - prod_matrix)
                                + (1.0 - set_op_mix_ratio) * prod_matrix;
                        return T(res);
                    },
                    stream);

                MLCommon::Sparse::coo_sort<T>(out, stream);

                CUDA_CHECK(cudaFree(rhos));
                CUDA_CHECK(cudaFree(sigmas));
            }
        }
    }
}
;
