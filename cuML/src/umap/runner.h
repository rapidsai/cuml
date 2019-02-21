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
#include "umapparams.h"
#include "optimize.h"

#include "fuzzy_simpl_set/runner.h"
#include "knn_graph/runner.h"
#include "simpl_set_embed/runner.h"
#include "init_embed/runner.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "sparse/csr.h"

#include "cuda_utils.h"

#include <iostream>
#include <cuda_runtime.h>

#pragma once

namespace UMAPAlgo {

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

    template<int TPB_X, typename T>
    __global__ void coo_row_counts(int *rows, T *vals, int nnz,
            int *results, int n) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < n && vals[row] != 0.0)
            atomicAdd(results+row, 1);
    }


    void find_ab(UMAPParams *params) {
        Optimize::find_params_ab(params);
    }


	template<typename T>
	size_t _fit(const T *X,       // input matrix
	            int n,      // rows
	            int d,      // cols
	            UMAPParams *params,
	            T *embeddings) {

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors);
		MLCommon::allocate(knn_dists, n*params->n_neighbors);

        std::cout << "Running knnGraph" << std::endl;

        kNNGraph::run(X, n, d, knn_indices, knn_dists, params);
		CUDA_CHECK(cudaPeekAtLastError());

		std::cout << MLCommon::arr2Str(knn_indices, n*params->n_neighbors, "knn_indices");
        std::cout << MLCommon::arr2Str(knn_dists, n*params->n_neighbors, "knn_dists");

		std::cout << "Finished knnGraph" << std::endl;

		int *graph_rows, *graph_cols;
		T *graph_vals;

		/**
		 * Allocate workspace for fuzzy simplicial set.
		 */
		MLCommon::allocate(graph_rows, n*params->n_neighbors);
		MLCommon::allocate(graph_cols, n*params->n_neighbors);
		MLCommon::allocate(graph_vals, n*params->n_neighbors);

		/**
		 * Run Fuzzy simplicial set
		 */
        std::cout << "Running FuzzySimplSet knnGraph" << std::endl;
		int nnz = 0;

		FuzzySimplSet::run(n, knn_indices, knn_dists,
						   graph_rows,
						   graph_cols,
						   graph_vals,
						   params, &nnz,0);

        std::cout << "ROWS=" << MLCommon::arr2Str<int>(graph_rows, nnz, "rows") << std::endl;
        std::cout << "COLS=" << MLCommon::arr2Str<int>(graph_cols, nnz, "cols") << std::endl;


		CUDA_CHECK(cudaPeekAtLastError());
        std::cout << "Finished FuzzySimplSet" << std::endl;

		T *embedding;
		MLCommon::allocate(embedding, n * params->n_components);

		InitEmbed::run(X, n, d,
		        knn_indices, knn_dists,
		        params, embedding);

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
        std::cout << "Running SimplSetEmbed" << std::endl;
		SimplSetEmbed::run<T, 256>(
		        X, n, d,
		        graph_rows, graph_cols, graph_vals, nnz,
		        params, embedding);

        CUDA_CHECK(cudaPeekAtLastError());
		std::cout << "Finished SimplSetEmbed" << std::endl;

		return 0;
	}



	template<typename T, int TPB_X>
	size_t _transform(const T *X,
	                  int n,
	                  int d,
	                  T *embedding,
	                  int embedding_n,
	                  UMAPParams *params,
	                  T *transformed) {

        dim3 grid(MLCommon::ceildiv(n, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);

	    /**
	     * Perform kNN of X
	     */
        long *knn_indices;
        T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors);
        MLCommon::allocate(knn_dists, n*params->n_neighbors);

        std::cout << "Running knnGraph" << std::endl;

        kNNGraph::run(X, n, d, knn_indices, knn_dists, params);
        CUDA_CHECK(cudaPeekAtLastError());

	    float adjusted_local_connectivity = max(0.0, params->local_connectivity - 1.0);

	    /**
	     * Perform smooth_knn_dist
	     */
        T *sigmas;
        T *rhos;
        MLCommon::allocate(sigmas, n);
        MLCommon::allocate(rhos, n);

        // TODO: Expose this so it can be swapped out.
        FuzzySimplSet::Naive::smooth_knn_dist<TPB_X, T>(n, knn_indices, knn_dists,
                rhos, sigmas, params
        );

        std::cout << MLCommon::arr2Str(rhos, n, "rhos") << std::endl;
        std::cout << MLCommon::arr2Str(sigmas, n, "sigmas") << std::endl;

        /**
         * Compute graph of membership strengths
         */
        int *graph_rows, *graph_cols;
        T *graph_vals;

        /**
         * Allocate workspace for fuzzy simplicial set.
         */
        MLCommon::allocate(graph_rows, n*params->n_neighbors);
        MLCommon::allocate(graph_cols, n*params->n_neighbors);
        MLCommon::allocate(graph_vals, n*params->n_neighbors);


        //TODO: Expose this so it can be swapped out easily
        FuzzySimplSet::Naive::compute_membership_strength_kernel<TPB_X><<<grid, blk>>>(knn_indices,
                knn_dists, sigmas, rhos, graph_vals, graph_rows, graph_cols, n,
                params->n_neighbors);


        int nnz = 0; // TODO: Set this!

        CUDA_CHECK(cudaPeekAtLastError());

        int *ia;
        MLCommon::allocate(ia, n, true);

        int *ex_scan;
        MLCommon::allocate(ex_scan, n, true);


        // COO should be sorted by row- we get the counts and then normalize
        coo_row_counts<TPB_X, T><<<grid, blk>>>(graph_rows, graph_vals, nnz, ia, n);

        thrust::device_ptr<int> dev_ia = thrust::device_pointer_cast(ia);
        thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
        exclusive_scan(dev_ia, dev_ia + n, dev_ex_scan);

         MLCommon::csr_row_normalize_l1<TPB_X, T><<<grid, blk>>>(dev_ex_scan.get(), graph_vals, nnz,
                 n, params->n_neighbors, graph_vals);

        /**
         * Init_transform()
         *
         */
        // cols.shape = (X.shape[0], self._n_neighbors)
        // vas.shape = (X.shape[0], self._n_neighbors)
        T *result;
        MLCommon::allocate(result, n*params->n_components);

        init_transform<TPB_X, T><<<grid,blk>>>(graph_cols, graph_vals, n,
                embedding, embedding_n, params->n_components,
                result, params->n_neighbors);

        /**
         * Find max of data
         */
        thrust::device_ptr<const T> d_ptr = thrust::device_pointer_cast(graph_vals);
        T max = *(thrust::max_element(d_ptr, d_ptr+nnz));

        /**
         * Go through COO values and set everything that's less than
         * vals.max() / params->n_epochs to 0.0
         */
        auto adjust_vals_op = [] __device__(T input, T scalar) {
            if (input < scalar)
                return 0.0f;
            else
                return input;
        };

        MLCommon::LinAlg::unaryOp<T>(graph_vals, graph_vals, (max / params->n_epochs), nnz, adjust_vals_op);

        /**
         * Remove zeros from vals
         *
         * TODO: Create rnnz array (and ex_scan_) for removing the vals
         */


        T *epochs_per_sample = (T*)malloc(nnz*sizeof(T));
        SimplSetEmbed::Algo::make_epochs_per_sample(graph_vals, nnz, params->n_epochs, epochs_per_sample);

        const int *head = graph_rows;
        const int *tail = graph_cols;

        SimplSetEmbed::Algo::optimize_layout(
            result, embedding_n,
            embedding, n,
            head, tail, nnz,
            epochs_per_sample,
            n,
            params
        );

        return 0;

	}

}
