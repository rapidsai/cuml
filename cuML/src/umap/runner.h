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
            if(target[i] == -1.0 || target[j] == -1.0)
                vals[row] *= exp(-unknown_dist);
            else
                vals[row] *= exp(-far_dist);
        }
    }

    /**
     * In order to reset membership strengths, we need to perform a
     * P+P.T, which can be done with the COO matrices by creating
     * the transposed elements only if they are not already in P,
     * and adding them.
     */
    template< typename T, int TPB_X>
    __global__ void reset_membership_strengths_kernel(
            int *row_ind,
            int *rows, int *cols, T *vals,
            int *orows, int *ocols, T *ovals, int *rnnz,
            int n, int cnnz) {

        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        int start_idx = row_ind[row]; // each thread processes one row
        int stop_idx = MLCommon::Sparse::get_stop_idx(row, n, cnnz, row_ind);

        if (row < n) {

            int nnz = 0;
            for (int idx = start_idx; idx < stop_idx; idx++) {

                int out_idx = start_idx*2+idx;
                int row_lookup = cols[idx];
                int t_start = row_ind[row_lookup]; // Start at
                int t_stop = MLCommon::Sparse::get_stop_idx(row_lookup, n, cnnz, row_ind);

                T transpose = 0.0;
                bool found_match = false;
                for (int t_idx = t_start; t_idx < t_stop; t_idx++) {

                    // If we find a match, let's get out of the loop
                    if (cols[t_idx] == rows[idx]
                            && rows[t_idx] == cols[idx]
                            && vals[t_idx] != 0.0) {
                        transpose = vals[t_idx];
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

            printf("nnz=%d\n", nnz);

            rnnz[row] = nnz;
            atomicAdd(rnnz + n, nnz); // fused operation to count number of nonzeros
        }
    }

    template<typename T, int TPB_X>
    void reset_local_connectivity(
        int *row_ind,
        int *graph_rows, int *graph_cols, T *graph_vals, int nnz,
        int *orows, int *ocols, T *ovals, int *rnnz, // size = nnz*2
        int m
    ) {

        dim3 grid_n(MLCommon::ceildiv(m, TPB_X), 1, 1);
        dim3 blk_n(TPB_X, 1, 1);

        // Perform l_inf normalization
        MLCommon::Sparse::csr_row_normalize_max<TPB_X, T><<<grid_n, blk_n>>>(
                row_ind,
                graph_vals,
                nnz,
                m,
                graph_vals
        );

        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << MLCommon::arr2Str(graph_rows, nnz, "graph_rows") << std::endl;
        std::cout << MLCommon::arr2Str(graph_cols, nnz, "graph_cols") << std::endl;
        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_vals") << std::endl;


        // reset membership strengths
        reset_membership_strengths_kernel<T, TPB_X><<<grid_n, blk_n>>>(
            row_ind,
            graph_rows, graph_cols, graph_vals,
            orows, ocols, ovals, rnnz,
            m, nnz
        );

        std::cout << MLCommon::arr2Str(orows, nnz, "orows") << std::endl;
        std::cout << MLCommon::arr2Str(ocols, nnz, "ocols") << std::endl;
        std::cout << MLCommon::arr2Str(ovals, nnz, "ovals") << std::endl;

        CUDA_CHECK(cudaPeekAtLastError());
    }

    template<typename T, int TPB_X>
    void categorical_simplicial_set_intersection(
         int *graph_rows, int *graph_cols, T *graph_vals,
         int nnz, T *target,
         float far_dist = 5.0, float unknown_dist = 1.0) {

        dim3 grid(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);
        fast_intersection_kernel<TPB_X, T><<<grid,blk>>>(
                graph_rows,
                graph_cols,
                graph_vals,
                nnz,
                target,
                unknown_dist,
                far_dist
        );

//        std::cout << MLCommon::arr2Str(graph_rows, nnz, "graph_rows") << std::endl;
//        std::cout << MLCommon::arr2Str(graph_cols, nnz, "graph_cols") << std::endl;
//        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_vals") << std::endl;
//
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


    /**
     * Computes the output CSR row index array and nnz for the
     * general simplicial set intersection.
     */
    template<typename T, int TPB_X>
    size_t general_simplicial_set_intersection_row_ind(
        int *row1_ind, int *cols1, T *vals1, int nnz1,
        int *row2_ind, int *cols2, T *vals2, int nnz2,
        int m, int *result_ind
    ) {
        return MLCommon::Sparse::csr_add_calc_inds<float, 32>(
            row1_ind, cols1, vals1, nnz1,
            row2_ind, cols2, vals2, nnz2,
            m, result_ind
        );
    }

    /**
     * Computes the CSR column index pointer and values
     * for the general simplicial set intersecftion.
     */
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

        sset_intersection_kernel<T, TPB_X><<<grid, blk>>>(
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
    void find_ab(UMAPParams *params, cudaStream_t stream) {
        Optimize::find_params_ab(params, stream);
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
	            T *embeddings,
              cudaStream_t stream) {

        int k = params->n_neighbors;

	    find_ab(params, stream);

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

        MLCommon::allocate(knn_indices, n*k);
		MLCommon::allocate(knn_dists, n*k);

        std::cout << "Running knn graph" << std::endl;

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params, stream);
		CUDA_CHECK(cudaPeekAtLastError());

        int *rgraph_rows, *rgraph_cols;
        T *rgraph_vals;

		/**
		 * Allocate workspace for fuzzy simplicial set.
		 */
        MLCommon::allocate(rgraph_rows, n*k*2);
        MLCommon::allocate(rgraph_cols, n*k*2);
        MLCommon::allocate(rgraph_vals, n*k*2);

        std::cout << "Running fuzzy simpl set" << std::endl;

		FuzzySimplSet::run<TPB_X, T>(n,
		                   knn_indices, knn_dists,
		                   k,
                           rgraph_rows,
                           rgraph_cols,
                           rgraph_vals,
						   params, 0);
		CUDA_CHECK(cudaPeekAtLastError());

		/**
		 * Remove zeros from simplicial set
		 */
        int *row_count_nz, *row_count;
        MLCommon::allocate(row_count_nz, n, true);
        MLCommon::allocate(row_count, n, true);

        std::cout << "Running row counts" << std::endl;

        MLCommon::Sparse::coo_row_count_nz<TPB_X, T>(rgraph_rows, rgraph_vals, n*k*2, row_count_nz, n);
        MLCommon::Sparse::coo_row_count<TPB_X>(rgraph_rows, n*k*2, row_count, n);

        std::cout << "Done." << MLCommon::arr2Str(row_count_nz, n, "row_count_nz") << std::endl;

        std::cout << "Running reduction." << std::endl;

        thrust::device_ptr<int> d_row_count_nz = thrust::device_pointer_cast(row_count_nz);
        int compressed_nnz = thrust::reduce(thrust::cuda::par.on(stream), d_row_count_nz, d_row_count_nz+n);

        std::cout << "Done." << std::endl;

        int *crows, *ccols;
        T *cvals;
        MLCommon::allocate(crows, compressed_nnz);
        MLCommon::allocate(ccols, compressed_nnz);
        MLCommon::allocate(cvals, compressed_nnz);

        std::cout << "Removing zeros" << std::endl;

        MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(
                rgraph_rows, rgraph_cols, rgraph_vals, n*k*2,
                crows, ccols, cvals,
                row_count_nz, row_count, n);

        /**
         * Run initialization method
         */
		InitEmbed::run(X, n, d,
		        knn_indices, knn_dists,
		        crows, ccols, cvals,
		        compressed_nnz,
		        params, embeddings, stream,
		        params->init);

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
		SimplSetEmbed::run<TPB_X, T>(
		        X, n, d,
		        crows, ccols, cvals, compressed_nnz,
		        params, embeddings, stream);

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
                T *embeddings, cudaStream_t stream) {

        int k = params->n_neighbors;

        std::cout << params->verbose << std::endl;

	    if(params->target_n_neighbors == -1)
	        params->target_n_neighbors = params->n_neighbors;

        find_ab(params, stream);

        if(params->verbose)
            std::cout << "Running KNN on X" << std::endl;

        /**
         * Allocate workspace for kNN graph
         */
        long *knn_indices;
        T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors, true);
        MLCommon::allocate(knn_dists, n*params->n_neighbors, true);

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params, stream);
        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose)
            std::cout << "Done." << std::endl;

        int *rgraph_rows, *rgraph_cols;
        T *rgraph_vals;

        /**
         * Allocate workspace for fuzzy simplicial set.
         */
        MLCommon::allocate(rgraph_rows, n*params->n_neighbors*2, true);
        MLCommon::allocate(rgraph_cols, n*params->n_neighbors*2, true);
        MLCommon::allocate(rgraph_vals, n*params->n_neighbors*2, true);

        if(params->verbose)
            std::cout << "Running fuzzy simpl set" << std::endl;


        /**
         * Run Fuzzy simplicial set
         */
        int nnz = n*k*2;
        FuzzySimplSet::run<TPB_X, T>(n,
                           knn_indices, knn_dists,
                           params->n_neighbors,
                           rgraph_rows,
                           rgraph_cols,
                           rgraph_vals,
                           params, 0);
        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose)
            std::cout << "Done." << std::endl;

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


            if(params->verbose)
                std::cout << "Runnning categorical simpl set intersection" << std::endl;

            categorical_simplicial_set_intersection<T, TPB_X>(
                    rgraph_rows, rgraph_cols, rgraph_vals,
                    n*k*2, y, far_dist);

            if(params->verbose)
                std::cout << "Done." << std::endl;

            if(params->verbose)
                std::cout << "Converting to CSR" << std::endl;

            int *result_ind;
            MLCommon::allocate(result_ind, n, true);

            MLCommon::Sparse::sorted_coo_to_csr(rgraph_rows, nnz, result_ind, n);

            if(params->verbose) {
                std::cout << MLCommon::arr2Str(rgraph_rows, nnz, "rgraph_rows") << std::endl;
                std::cout << MLCommon::arr2Str(rgraph_cols, nnz, "rgraph_cols") << std::endl;
                std::cout << MLCommon::arr2Str(rgraph_vals, nnz, "rgraph_vals") << std::endl;
                std::cout << MLCommon::arr2Str(result_ind, n, "result_ind") << std::endl;
            }

            if(params->verbose)
                std::cout << "Done." << std::endl;

            nnz_before_compress = nnz*2;

            if(params->verbose)
                std::cout << "Resetting local connectivity" << std::endl;

            // reset local connectivity
            MLCommon::allocate(final_nnz, n+1, true);
            MLCommon::allocate(final_rows, nnz_before_compress, true);
            MLCommon::allocate(final_cols, nnz_before_compress, true);
            MLCommon::allocate(final_vals, nnz_before_compress, true);

            reset_local_connectivity<T, TPB_X>(
                result_ind,
                rgraph_rows, rgraph_cols, rgraph_vals, nnz,
                final_rows, final_cols, final_vals, final_nnz,
                n
            );

            CUDA_CHECK(cudaPeekAtLastError());

            if(params->verbose)
                std::cout << "Done." << std::endl;

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
            MLCommon::allocate(ygraph_rows, n*params->target_n_neighbors*2, true);
            MLCommon::allocate(ygraph_cols, n*params->target_n_neighbors*2, true);
            MLCommon::allocate(ygraph_vals, n*params->target_n_neighbors*2, true);

            if(params->verbose)
                std::cout << "Runnning knn_Graph on Y" << std::endl;

            kNN y_knn(1);
            long *y_knn_indices;
            T *y_knn_dists;

            MLCommon::allocate(y_knn_indices, n*params->target_n_neighbors, true);
            MLCommon::allocate(y_knn_dists, n*params->target_n_neighbors, true);

            kNNGraph::run(y, n, 1, y_knn_indices, y_knn_dists, &y_knn, params, stream);
            CUDA_CHECK(cudaPeekAtLastError());


            if(params->verbose)
                std::cout << "Done." << std::endl;

            if(params->verbose)
                std::cout << "Runnning Fuzzy simpl set on Y" << std::endl;

            FuzzySimplSet::run<TPB_X, T>(n,
                               y_knn_indices, y_knn_dists,
                               params->target_n_neighbors,
                               ygraph_rows,
                               ygraph_cols,
                               ygraph_vals,
                               params, 0);
            CUDA_CHECK(cudaPeekAtLastError());

            if(params->verbose)
                std::cout << "Done." << std::endl;

            if(params->verbose)
                std::cout << "Converting to CSR" << std::endl;

            // perform general simplicial set intersection
            int *result_ind, *result_rows, *xrow_ind, *yrow_ind, *result_cols;
            T *result_vals;
            MLCommon::allocate(result_ind, n, true);
            MLCommon::allocate(xrow_ind, n, true);
            MLCommon::allocate(yrow_ind, n, true);

            MLCommon::Sparse::sorted_coo_to_csr(ygraph_rows, n*2, yrow_ind, n);
            MLCommon::Sparse::sorted_coo_to_csr(rgraph_rows, n*k*2, xrow_ind, n);

            if(params->verbose)
                std::cout << "Done." << std::endl;

            if(params->verbose)
                std::cout << "Running general simpl set intersection" << std::endl;


            /**
             * Compute general simplicial set intersection.
             * This is broken into two separate calls so that the first
             * call can determine how much memory to allocate for the
             * second call.
             */
            int result_nnz = general_simplicial_set_intersection_row_ind<T, TPB_X>(
                xrow_ind, rgraph_cols, rgraph_vals, nnz,
                yrow_ind, ygraph_cols, ygraph_vals, n*2,
                n, result_ind
            );

            MLCommon::allocate(result_cols, result_nnz, true);
            MLCommon::allocate(result_vals, result_nnz, true);
            MLCommon::allocate(result_rows, result_nnz, true);

            general_simplicial_set_intersection<T, TPB_X>(
                xrow_ind, rgraph_cols, rgraph_vals, nnz,
                yrow_ind, ygraph_cols, ygraph_vals, n*2,
                result_ind, result_cols, result_vals,
                result_nnz, n, params->target_weights
            );

            if(params->verbose)
                std::cout << "Done." << std::endl;

            if(params->verbose)
                std::cout << "Convert result to COO" << std::endl;


            dim3 grid_n(MLCommon::ceildiv(result_nnz, TPB_X), 1, 1);
            dim3 blk(TPB_X, 1, 1);

            MLCommon::Sparse::csr_to_coo<TPB_X><<<grid_n, blk>>>(result_ind, n, result_rows, result_nnz);

            if(params->verbose)
                std::cout << "Done." << std::endl;


            if(params->verbose) {
                std::cout << "Reset Local connectivity" << std::endl;
                std::cout << "result_nnz=" << result_nnz << std::endl;

            }

            nnz_before_compress = result_nnz;

            MLCommon::allocate(final_nnz, n+1, true);
            MLCommon::allocate(final_rows, nnz_before_compress, true);
            MLCommon::allocate(final_cols, nnz_before_compress, true);
            MLCommon::allocate(final_vals, nnz_before_compress, true);

            reset_local_connectivity<T, TPB_X>(
                result_ind,
                result_rows, result_cols, result_vals, result_nnz,
                final_rows, final_cols, final_vals, final_nnz,
                n
            );
            CUDA_CHECK(cudaPeekAtLastError());
        }

        if(params->verbose)
            std::cout << "Done." << std::endl;

        CUDA_CHECK(cudaFree(rgraph_rows));
        CUDA_CHECK(cudaFree(rgraph_cols));
        CUDA_CHECK(cudaFree(rgraph_vals));

        /**
         * Remove zeros
         */

        if(params->verbose)
            std::cout << "Removing zeros" << std::endl;

        if(params->verbose)
            std::cout << MLCommon::arr2Str(final_nnz, n+1, "final_nnz") << std::endl;


        int *orows, *ocols, *cur_rnnz, onnz = 0;
        T *ovals;

        MLCommon::allocate(cur_rnnz, n, true);

        MLCommon::updateHost(&onnz, final_nnz+n, 1); // actual number of nonzero values

        MLCommon::Sparse::coo_sort<T>(n, n, nnz_before_compress, final_rows, final_cols, final_vals);

        if(params->verbose) {
            std::cout << "onnz=" << onnz << std::endl;
            std::cout << "nnz_before_compress=" << nnz_before_compress << std::endl;
            std::cout << MLCommon::arr2Str(final_rows, nnz_before_compress, "final_rows") << std::endl;
            std::cout << MLCommon::arr2Str(final_cols, nnz_before_compress, "final_cols") << std::endl;
            std::cout << MLCommon::arr2Str(final_vals, nnz_before_compress, "final_vals") << std::endl;
        }

        MLCommon::allocate(ocols, onnz, true);
        MLCommon::allocate(ovals, onnz, true);
        MLCommon::allocate(orows, onnz, true);

        MLCommon::Sparse::coo_row_count<TPB_X>(
                final_rows, nnz_before_compress, cur_rnnz, n);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << MLCommon::arr2Str(cur_rnnz, n, "cur_rnnz") << std::endl;
        std::cout << MLCommon::arr2Str(final_nnz, n, "final_rnnz") << std::endl;

        MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(
            final_rows, final_cols, final_vals, nnz_before_compress,
            orows, ocols, ovals,
            final_nnz, cur_rnnz, n);
        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose)
            std::cout << "Done." << std::endl;


        /**
         * Initialize embeddings
         */

        if(params->verbose) {
            std::cout << "init embeddings." << std::endl;
            std::cout << MLCommon::arr2Str(orows, onnz, "orows") << std::endl;
            std::cout << MLCommon::arr2Str(ocols, onnz, "ocols") << std::endl;
            std::cout << MLCommon::arr2Str(ovals, onnz, "ovals") << std::endl;
        }

        InitEmbed::run(X, n, d,
                knn_indices, knn_dists,
                orows, ocols, ovals, onnz,
                params, embeddings, stream, params->init);

        if(params->verbose)
            std::cout << "Done." << std::endl;

        /**
         * Run simplicial set embedding to approximate low-dimensional representation
         */

        if(params->verbose)
            std::cout << "Running simpl set embedding." << std::endl;

        SimplSetEmbed::run<TPB_X, T>(
                X, n, d,
                orows, ocols, ovals, onnz,
                params, embeddings, stream);

        if(params->verbose)
            std::cout << "Done." << std::endl;

        if(params->verbose)
            std::cout << MLCommon::arr2Str(embeddings, n*params->n_components, "embeddings") << std::endl;


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
	                  T *transformed,
                    cudaStream_t stream) {

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
            [] __device__(T input) { return sqrt(input); },
        stream);

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
                rhos, sigmas, params, params->n_neighbors, adjusted_local_connectivity, stream
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

        int *ia, *cur_ia, *ex_scan;
        MLCommon::allocate(ia, n, true);
        MLCommon::allocate(cur_ia, n, true);
        MLCommon::allocate(ex_scan, n, true);

        // COO should be sorted by row at this point- we get the counts and then normalize
        MLCommon::Sparse::coo_row_count<TPB_X>(graph_rows, nnz, ia, n);

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

//        std::cout << "Inside transform " << std::endl;

        MLCommon::Sparse::coo_row_count_nz<TPB_X, T>(graph_rows, graph_vals, nnz, ia, n);
        MLCommon::Sparse::coo_row_count<TPB_X>(graph_rows, nnz, cur_ia, n);

//        std::cout << MLCommon::arr2Str(cur_ia, n, "cur_ia") << std::endl;

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
            },
            stream);

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

        MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(
                graph_rows, graph_cols, graph_vals,nnz,
                crows, ccols, cvals,
                ia, cur_ia, n);

        CUDA_CHECK(cudaPeekAtLastError());

//        std::cout << "Done." << std::endl;
//
        T *epochs_per_sample;
        MLCommon::allocate(epochs_per_sample, nnz);

        SimplSetEmbedImpl::make_epochs_per_sample(cvals, non_zero_vals, params->n_epochs,
                                                    epochs_per_sample, stream);
        CUDA_CHECK(cudaPeekAtLastError());

        SimplSetEmbedImpl::optimize_layout<TPB_X, T>(
            transformed, n,
            embedding, embedding_n,
            crows, ccols, non_zero_vals,
            epochs_per_sample,
            n,
            params->repulsion_strength,
            params,
            n_epochs,
            stream
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
        CUDA_CHECK(cudaFree(cur_ia));
        CUDA_CHECK(cudaFree(ex_scan));

        CUDA_CHECK(cudaFree(epochs_per_sample));
        CUDA_CHECK(cudaFree(crows));
        CUDA_CHECK(cudaFree(ccols));
        CUDA_CHECK(cudaFree(cvals));

        return 0;

	}

}
