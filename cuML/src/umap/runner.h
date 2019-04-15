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

#include "umapparams.h"
#include "optimize.h"

#include "fuzzy_simpl_set/runner.h"
#include "knn_graph/runner.h"
#include "simpl_set_embed/runner.h"
#include "init_embed/runner.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include "sparse/csr.h"
#include "sparse/coo.h"

#include "knn/knn.h"

#include "cuda_utils.h"

#include <iostream>
#include <cuda_runtime.h>



namespace UMAPAlgo {

    // Swap this as impls change for now.
    namespace FuzzySimplSetImpl = FuzzySimplSet::Naive;
    namespace SimplSetEmbedImpl = SimplSetEmbed::Algo;

	using namespace ML;

    template<int TPB_X, typename T>
	__global__ void init_transform(int *indices, T *weights, int n,
	                    const T *embeddings, int embeddings_n, int n_components,
	                    T *result, int n_neighbors) {

        // row-based matrix 1 thread per row
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        int i = row * n_neighbors; // each thread processes one row of the dist matrix

        if(row < n) {
            for(int j = 0; j < n_neighbors; j++) {
                for(int d = 0; d < n_components; d++) {
                    result[row*n_components+d] += weights[i+j] * embeddings[indices[i+j]*n_components+d];
                }
            }
        }
	}


    /**
     * Simple helper function to set the values of an array to zero.
     *
     * @param vals array of values
     * @param nnz size of array of values
     */
    template<int TPB_X>
    __global__ void reset_vals(int *vals, int nnz) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < nnz)
            vals[row] = 0.0;
    }

    template<int TPB_X, typename T>
    __global__ void fast_intersection_kernel(
        int * rows, int *cols, T *vals, int nnz,
        T *target,
        float unknown_dist = 1.0,
        float far_dist = 5.0
    ) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < nnz) {
            int i = rows[row];
            int j = cols[row];
            if(target[i] == -1 || target[j] == -1)
                vals[row] *= exp(-unknown_dist);
            else
                vals[row] *= exp(-far_dist);
        }
    }

    template< typename T, int TPB_X>
    __global__ void reset_membership_strengths_kernel(
            int *rows, int *cols, T *vals,
            int *orows, int *ocols, T *ovals, int *rnnz,
            int n, int n_neighbors) {

        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        int i = row * n_neighbors; // each thread processes one row

        if (row < n) {

            int nnz = 0;
            for (int j = 0; j < n_neighbors; j++) {

                int idx = i + j;
                int out_idx = i * 2;

                int row_lookup = cols[idx];
                int t_start = row_lookup * n_neighbors; // Start at

                T transpose = 0.0;
                bool found_match = false;
                for (int t_idx = 0; t_idx < n_neighbors; t_idx++) {

                    int f_idx = t_idx + t_start;
                    // If we find a match, let's get out of the loop
                    if (cols[f_idx] == rows[idx]
                            && rows[f_idx] == cols[idx]
                            && vals[f_idx] != 0.0) {
                        transpose = vals[f_idx];
                        found_match = true;
                        break;
                    }
                }

                // if we didn't find an exact match, we need to add
                // the transposed value into our current matrix.
                if (!found_match && vals[idx] != 0.0) {
                    orows[out_idx + nnz] = cols[idx];
                    ocols[out_idx + nnz] = rows[idx];
                    ovals[out_idx + nnz] = vals[idx];
                    ++nnz;
                }

                T result = vals[idx];
                T prod_matrix = result * transpose;

                // @todo: This line is the only difference between
                // this function and compute_result from the fuzzy
                // simplicial set. Should combine these to do
                // transposed eltwise ops.
                T res = result + transpose - prod_matrix;

                if (res != 0.0) {
                    orows[out_idx + nnz] = rows[idx];
                    ocols[out_idx + nnz] = cols[idx];
                    ovals[out_idx + nnz] = T(res);
                    ++nnz;
                }
            }

            rnnz[row] = nnz;
            atomicAdd(rnnz + n, nnz); // fused operation to count number of nonzeros
        }
    }

    template<typename T, int TPB_X>
    void reset_local_connectivity(
        int *row_ind, int *graph_rows, int *graph_cols, T *graph_vals, int nnz,
        int *orows, int *ocols, T *ovals, int *rnnz, // size = nnz*2
        int m, int n_neighbors
    ) {
        // Perform l_inf normalization
        MLCommon::Sparse::csr_row_normalize_max(row_ind, graph_vals,nnz, m, graph_vals);

        // reset membership strengths
        dim3 grid_n(MLCommon::ceildiv(m, TPB_X), 1, 1);
        dim3 blk_n(TPB_X, 1, 1);

        reset_membership_strengths_kernel<T, TPB_X><<<grid_n, blk_n>>>(
            graph_rows, graph_cols, graph_vals,
            orows, ocols, ovals, rnnz, m,
            n_neighbors
        );

        CUDA_CHECK(cudaFree(row_ind));
    }

    template<typename T, int TPB_X>
    void categorical_simplicial_set_intersection(
         int *graph_rows, int *graph_cols, T *graph_vals,
         int nnz, T *target,
         float far_dist = 5.0, float unknown_dist = 1.0) {

        dim3 grid(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);
        fast_intersection_kernel<<<grid,blk>>>(
                graph_rows,
                graph_cols,
                graph_vals,
                nnz,
                target,
                unknown_dist,
                far_dist
        );
    }


    template<typename T, int TPB_X>
    __global__ void sset_intersection_kernel(
        int *row_ind1, int *cols1, T *vals1, int nnz1,
        int *row_ind2, int *cols2, T *vals2, int nnz2,
        int *result_ind, int *result_cols, T *result_vals, int nnz,
        T left_min, T right_min,
        int m, float mix_weight = 0.5
    ) {

        int row = (blockIdx.x * TPB_X) + threadIdx.x;

        if(row < m) {
            int i = row;
            int start_idx_res = result_ind[row];
            int stop_idx_res = MLCommon::Sparse::get_stop_idx(row, m, nnz, result_ind);

            int start_idx1 = row_ind1[row];
            int stop_idx1 = MLCommon::Sparse::get_stop_idx(row, m, nnz1, row_ind1);

            int start_idx2 = row_ind2[row];
            int stop_idx2 = MLCommon::Sparse::get_stop_idx(row, m, nnz2, row_ind2);

            for(int j = start_idx_res; j < stop_idx_res; j++) {
                int col = result_cols[j];

                T left_val = left_min;
                for(int k = start_idx1; k < stop_idx2; k++) {
                    if(cols1[k] == col) {
                        left_val = vals1[k];
                    }
                }

                T right_val = right_min;
                for(int k = start_idx2; k < stop_idx2; k++) {
                    if(cols2[k] == col) {
                        right_val = vals2[k];
                    }
                }

                if(left_val > left_min || right_val > right_min) {
                    if(mix_weight < 0.5) {
                        result_vals[row] = left_val *
                                powf(right_val, mix_weight / (1.0 - mix_weight));
                    } else {
                        result_vals[row] = powf(left_val, (1.0 - mix_weight) / mix_weight)
                                * right_val;
                    }
                }
            }
        }
    }




    template<typename T, int TPB_X>
    void general_simplicial_set_intersection_row_ind(
        int *row1_ind, int *cols1, T *vals1, int nnz1,
        int *row2_ind, int *cols2, T *vals2, int nnz2,
        int m, int *result_ind, int *nnz
    ) {
        MLCommon::Sparse::csr_add_calc_inds<float, 32>(
            row1_ind, cols1, vals1, nnz1,
            row2_ind, cols2, vals2, nnz2,
            m, nnz, result_ind
        );
    }

    template<typename T, int TPB_X>
    void general_simplicial_set_intersection(
        int *row1_ind, int *cols1, T *vals1, int nnz1,
        int *row2_ind, int *cols2, T *vals2, int nnz2,
        int *result_ind, int *result_ind_ptr, T *result_val,
        int nnz, int m, float weight
    ) {
        MLCommon::Sparse::csr_add_finalize<float, 32>(
            row1_ind, cols1, vals1, nnz1,
            row2_ind, cols2, vals2, nnz2,
            m, result_ind, result_ind_ptr, result_val
        );

        thrust::device_ptr<const T> d_ptr1 = thrust::device_pointer_cast(vals1);
        T min1 = *(thrust::min_element(d_ptr1, d_ptr1+nnz1));

        thrust::device_ptr<const T> d_ptr2 = thrust::device_pointer_cast(vals2);
        T min2 = *(thrust::min_element(d_ptr2, d_ptr2+nnz2));

        T left_min = min(min1 / 2.0, 1e-8);
        T right_min = min(min2 / 2.0, 1e-8);

        dim3 grid(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);

        sset_intersection_kernel<<<grid, blk>>>(
            row1_ind, cols1, vals1, nnz1,
            row2_ind, cols2, vals2, nnz2,
            result_ind, result_ind_ptr, result_val, nnz,
            left_min, right_min,
            m, weight
        );
    }


    /**
     * Firt exponential decay curve to find the parameters
     * a and b, which are based on min_dist and spread
     * parameters.
     */
    void find_ab(UMAPParams *params) {
        Optimize::find_params_ab(params);
    }

    /**
     * Fit
     */
	template<typename T, int TPB_X>
	size_t _fit(T *X,       // input matrix
	            int n,      // rows
	            int d,      // cols
	            kNN *knn,
	            UMAPParams *params,
	            T *embeddings) {

	    find_ab(params);

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors);
		MLCommon::allocate(knn_dists, n*params->n_neighbors);

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params);
		CUDA_CHECK(cudaPeekAtLastError());

        int *rgraph_rows, *rgraph_cols;
        T *rgraph_vals;

		/**
		 * Allocate workspace for fuzzy simplicial set.
		 */
        MLCommon::allocate(rgraph_rows, n*params->n_neighbors*2);
        MLCommon::allocate(rgraph_cols, n*params->n_neighbors*2);
        MLCommon::allocate(rgraph_vals, n*params->n_neighbors*2);


		/**
		 * Run Fuzzy simplicial set
		 */
		int nnz = 0;

		FuzzySimplSet::run<TPB_X, T>(n, knn_indices, knn_dists,
                           rgraph_rows,
                           rgraph_cols,
                           rgraph_vals,
						   params, &nnz,0);
		CUDA_CHECK(cudaPeekAtLastError());

		InitEmbed::run(X, n, d,
		        knn_indices, knn_dists,
		        rgraph_rows, rgraph_cols, rgraph_vals,
		        nnz,
		        params, embeddings, params->init);

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
		SimplSetEmbed::run<TPB_X, T>(
		        X, n, d,
		        rgraph_rows, rgraph_cols, rgraph_vals, nnz,
		        params, embeddings);

        CUDA_CHECK(cudaPeekAtLastError());

		CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));
        CUDA_CHECK(cudaFree(rgraph_rows));
        CUDA_CHECK(cudaFree(rgraph_cols));
        CUDA_CHECK(cudaFree(rgraph_vals));

		return 0;
	}

    template<typename T, int TPB_X>
	size_t _fit(T *X,    // input matrix
	            T *y,    // labels
                int n,
                int d,
                kNN *knn,
                UMAPParams *params,
                T *embeddings) {

	    if(params->target_n_neighbors == -1)
	        params->target_n_neighbors = params->n_neighbors;

        find_ab(params);

        /**
         * Allocate workspace for kNN graph
         */
        long *knn_indices;
        T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors);
        MLCommon::allocate(knn_dists, n*params->n_neighbors);

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params);
        CUDA_CHECK(cudaPeekAtLastError());

        int *rgraph_rows, *rgraph_cols;
        T *rgraph_vals;

        /**
         * Allocate workspace for fuzzy simplicial set.
         */
        MLCommon::allocate(rgraph_rows, n*params->n_neighbors*2);
        MLCommon::allocate(rgraph_cols, n*params->n_neighbors*2);
        MLCommon::allocate(rgraph_vals, n*params->n_neighbors*2);

        /**
         * Run Fuzzy simplicial set
         */
        int nnz = 0;
        FuzzySimplSet::run<TPB_X, T>(n, knn_indices, knn_dists,
                           rgraph_rows,
                           rgraph_cols,
                           rgraph_vals,
                           params, &nnz,0);  // @todo: Break this up to stop overallocation
        CUDA_CHECK(cudaPeekAtLastError());

        int *final_rows, *final_cols, *final_nnz, nnz_before_compress;
        T *final_vals;

        /**
         * If target metric is 'categorical',
         * run categorical_simplicial_set_intersection() on
         * the current graph and target labels.
         */
        if(params->target_metric == ML::UMAPParams::MetricType::CATEGORICAL) {

            float far_dist = 1.0e12;  // target weight
            if(params->target_weights < 1.0)
                far_dist = 2.5 * (1.0 / (1.0 - params->target_weights));

            categorical_simplicial_set_intersection(
                    rgraph_rows, rgraph_cols, rgraph_vals, nnz,
                    y, far_dist);

            int *result_ind, result_nnz = 0;
            MLCommon::allocate(result_ind, n);

            MLCommon::Sparse::sorted_coo_to_csr(rgraph_rows, nnz, result_ind, n);

            nnz_before_compress = nnz*2;

            // reset local connectivity
            MLCommon::allocate(final_nnz, n);
            MLCommon::allocate(final_rows, nnz_before_compress*2);
            MLCommon::allocate(final_cols, nnz_before_compress*2);
            MLCommon::allocate(final_vals, nnz_before_compress*2);

            reset_local_connectivity(
                result_ind, rgraph_rows, rgraph_cols,
                rgraph_vals, nnz,
                final_rows, final_cols, final_vals, final_nnz,
                n, params->n_neighbors
            );

            CUDA_CHECK(cudaFree(result_ind));

        /**
         * Else, knn query labels & create fuzzy simplicial set w/ them:
         * self.target_graph = fuzzy_simplicial_set(y, target_n_neighbors)
         * general_simplicial_set_intersection(self.graph_, target_graph, self.target_weight)
         * self.graph_ = reset_local_connectivity(self.graph_)
         *
         */
        } else {

            int *ygraph_rows, *ygraph_cols;
            T *ygraph_vals;

            /**
             * Allocate workspace for fuzzy simplicial set.
             */
            MLCommon::allocate(ygraph_rows, n*params->target_n_neighbors*2);
            MLCommon::allocate(ygraph_cols, n*params->target_n_neighbors*2);
            MLCommon::allocate(ygraph_vals, n*params->target_n_neighbors*2);

            kNN y_knn(1);
            long *y_knn_indices;
            T *y_knn_dists;

            MLCommon::allocate(y_knn_indices, n*params->target_n_neighbors);
            MLCommon::allocate(y_knn_dists, n*params->target_n_neighbors);

            kNNGraph::run(y, n, 1, y_knn_indices, y_knn_dists, &y_knn, params);
            CUDA_CHECK(cudaPeekAtLastError());

            int &ynnz = 0;
            FuzzySimplSet::run<TPB_X, T>(n, y_knn_indices, y_knn_dists,
                               ygraph_rows,
                               ygraph_cols,
                               ygraph_vals,
                               params, &ynnz, 0);  // TODO: explicitly pass in n_neighbors so that target_n_neighbors can be used
            CUDA_CHECK(cudaPeekAtLastError());

            // perform general simplicial set intersection
            int *result_ind, *result_rows, *xrow_ind, *yrow_ind, *result_cols;
            T *result_vals;
            MLCommon::allocate(result_ind, n);
            MLCommon::allocate(xrow_ind, n);
            MLCommon::allocate(yrow_ind, n);

            MLCommon::Sparse::sorted_coo_to_csr(ygraph_rows, ynnz, yrow_ind, n);
            MLCommon::Sparse::sorted_coo_to_csr(rgraph_rows, nnz, xrow_ind, n);

            int result_nnz = 0;

            general_simplicial_set_intersection_row_ind(
                xrow_ind, rgraph_cols, rgraph_vals, nnz,
                yrow_ind, ygraph_cols, ygraph_vals, ynnz,
                n, result_ind, &result_nnz
            );

            MLCommon::allocate(result_cols, result_nnz);
            MLCommon::allocate(result_vals, result_nnz);
            MLCommon::allocate(result_rows, result_nnz);

            general_simplical_set_intersection(
                xrow_ind, rgraph_cols, rgraph_vals, nnz,
                yrow_ind, ygraph_cols, ygraph_vals, ynnz,
                result_ind, result_cols, result_vals,
                result_nnz, n, params->target_weights
            );

            MLCommon::Sparse::csr_to_coo<TPB_X>(result_ind, n, result_rows, result_nnz);

            nnz_before_compress = result_nnz*2;

            MLCommon::allocate(final_nnz, n);
            MLCommon::allocate(final_rows, nnz_before_compress);
            MLCommon::allocate(final_cols, nnz_before_compress);
            MLCommon::allocate(final_vals, nnz_before_compress);

            reset_local_connectivity(
                result_ind, result_rows, result_cols,
                result_vals, result_nnz,
                final_rows, final_cols, final_vals, final_nnz,
                n, params->n_neighbors
            );

        }

        CUDA_CHECK(cudaFree(rgraph_rows));
        CUDA_CHECK(cudaFree(rgraph_cols));
        CUDA_CHECK(cudaFree(rgraph_vals));

        /**
         * Remove zeros
         */

        int *orows, *ocols;
        T *ovals;

        int onnz = *(final_nnz+n); // actual number of nonzero values
        MLCommon::allocate(ocols, onnz);
        MLCommon::allocate(ovals, onnz);
        MLCommon::allocate(orows, onnz);

        coo_remove_zeros(nnz_before_compress,
            final_rows, final_cols, final_vals,
            orows, ocols, ovals,
            final_nnz, onnz);


        /**
         * Initialize embeddings
         */

        InitEmbed::run(X, n, d,
                knn_indices, knn_dists,
                orows, ocols, ovals, onnz,
                params, embeddings, params->init);

        /**
         * Run simplicial set embedding to approximate low-dimensional representation
         */

        SimplSetEmbed::run<TPB_X, T>(
                X, n, d,
                orows, ocols, ovals, onnz,
                params, embeddings);

        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));
        CUDA_CHECK(cudaFree(orows));
        CUDA_CHECK(cudaFree(ocols));
        CUDA_CHECK(cudaFree(ovals));

	    return 0;
	}

	/**
	 *
	 */
	template<typename T, int TPB_X>
	size_t _transform(const float *X,
	                  int n,
	                  int d,
	                  T *embedding,
	                  int embedding_n,
                      kNN *knn,
	                  UMAPParams *params,
	                  T *transformed) {

	    /**
	     * Perform kNN of X
	     */
        long *knn_indices;
        float *knn_dists;
        MLCommon::allocate(knn_indices, n*params->n_neighbors);
        MLCommon::allocate(knn_dists, n*params->n_neighbors);

        knn->search(X, n, knn_indices, knn_dists, params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        MLCommon::LinAlg::unaryOp<T>(knn_dists, knn_dists, n*params->n_neighbors,
            [] __device__(T input) { return sqrt(input); }
        );

	    float adjusted_local_connectivity = max(0.0, params->local_connectivity - 1.0);

	    if(params->verbose)
	        std::cout << "adjusted_local_connectivity=" << adjusted_local_connectivity << std::endl;

	    /**
	     * Perform smooth_knn_dist
	     */
        T *sigmas;
        T *rhos;
        MLCommon::allocate(sigmas, n, true);
        MLCommon::allocate(rhos, n, true);

        dim3 grid_n(MLCommon::ceildiv(n, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);

        FuzzySimplSetImpl::smooth_knn_dist<TPB_X, T>(n, knn_indices, knn_dists,
                rhos, sigmas, params, adjusted_local_connectivity
        );

        /**
         * Compute graph of membership strengths
         */

        int nnz = n*params->n_neighbors;

        dim3 grid_nnz(MLCommon::ceildiv(nnz, TPB_X), 1, 1);

        /**
         * Allocate workspace for fuzzy simplicial set.
         */
        int *graph_rows, *graph_cols;
        T *graph_vals;
        MLCommon::allocate(graph_rows, nnz);
        MLCommon::allocate(graph_cols, nnz);
        MLCommon::allocate(graph_vals, nnz);

        if(params->verbose)
            std::cout << "n_neighbors=" << params->n_neighbors << std::endl;

        FuzzySimplSetImpl::compute_membership_strength_kernel<TPB_X><<<grid_n, blk>>>(
                knn_indices, knn_dists,
                sigmas, rhos,
                graph_vals, graph_rows, graph_cols, n,
                params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose) {
            std::cout << MLCommon::arr2Str(sigmas, n, "sigmas") << std::endl;
            std::cout << MLCommon::arr2Str(rhos, n, "rhos") << std::endl;
        }

        int *ia, *ex_scan;
        MLCommon::allocate(ia, n, true);
        MLCommon::allocate(ex_scan, n, true);

        // COO should be sorted by row at this point- we get the counts and then normalize
        MLCommon::Sparse::coo_row_count<TPB_X><<<grid_nnz, blk>>>(graph_rows, nnz, ia, n);

        thrust::device_ptr<int> dev_ia = thrust::device_pointer_cast(ia);
        thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
        exclusive_scan(dev_ia, dev_ia + n, dev_ex_scan);

        T *vals_normed;
        MLCommon::allocate(vals_normed, nnz, true);

         MLCommon::Sparse::csr_row_normalize_l1<TPB_X, T><<<grid_n, blk>>>(dev_ex_scan.get(), graph_vals, nnz,
                 n, vals_normed);

        init_transform<TPB_X, T><<<grid_n,blk>>>(graph_cols, vals_normed, n,
                embedding, embedding_n, params->n_components,
                transformed, params->n_neighbors);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaFree(vals_normed));

        reset_vals<TPB_X><<<grid_n,blk>>>(ia, n);

        MLCommon::Sparse::coo_row_count_nz<TPB_X, T><<<grid_nnz,blk>>>(graph_rows, graph_vals, nnz, ia, n);

        /**
         * Go through COO values and set everything that's less than
         * vals.max() / params->n_epochs to 0.0
         */
        thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(graph_vals);
        T max = *(thrust::max_element(d_ptr, d_ptr+nnz));

        int n_epochs = 1000;//params->n_epochs;

        MLCommon::LinAlg::unaryOp<T>(graph_vals, graph_vals, nnz,
            [=] __device__(T input) {
                if (input < (max / float(n_epochs)))
                    return 0.0f;
                else
                    return input;
            }
        );

        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose)
            std::cout << "nnz=" << nnz << std::endl;

        thrust::device_ptr<T> dev_gvals = thrust::device_pointer_cast(graph_vals);
        int non_zero_vals = nnz-thrust::count(dev_gvals, dev_gvals+nnz, T(0));
        CUDA_CHECK(cudaPeekAtLastError());


        if(params->verbose)
            std::cout << "non_zero_vals=" << non_zero_vals << std::endl;

        /**
         * Remove zeros
         */
        int *crows, *ccols;
        T *cvals;
        MLCommon::allocate(crows, non_zero_vals, true);
        MLCommon::allocate(ccols, non_zero_vals, true);
        MLCommon::allocate(cvals, non_zero_vals, true);

        MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(nnz,
                graph_rows, graph_cols, graph_vals,
                crows, ccols, cvals,
                ia, n);
        CUDA_CHECK(cudaPeekAtLastError());

        T *epochs_per_sample;
        MLCommon::allocate(epochs_per_sample, nnz);

        SimplSetEmbedImpl::make_epochs_per_sample(cvals, non_zero_vals, params->n_epochs, epochs_per_sample);
        CUDA_CHECK(cudaPeekAtLastError());

        SimplSetEmbedImpl::optimize_layout<TPB_X, T>(
            transformed, n,
            embedding, embedding_n,
            crows, ccols, non_zero_vals,
            epochs_per_sample,
            n,
            params->repulsion_strength,
            params,
            n_epochs
        );
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));


        CUDA_CHECK(cudaFree(sigmas));
        CUDA_CHECK(cudaFree(rhos));

        CUDA_CHECK(cudaFree(graph_rows));
        CUDA_CHECK(cudaFree(graph_cols));
        CUDA_CHECK(cudaFree(graph_vals));

        CUDA_CHECK(cudaFree(ia));
        CUDA_CHECK(cudaFree(ex_scan));

        CUDA_CHECK(cudaFree(epochs_per_sample));
        CUDA_CHECK(cudaFree(crows));
        CUDA_CHECK(cudaFree(ccols));
        CUDA_CHECK(cudaFree(cvals));

        return 0;

	}

}
