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

#include "cuda_utils.h"

#include <iostream>
#include <cuda_runtime.h>

#pragma once

namespace UMAPAlgo {

	using namespace ML;

    template<int TPB_X, typename T>
	__global__ void init_transform(int *indices, T *weights, int n,
	                    T *embeddings, int embeddings_n, int n_components,
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
	                  const T *embedding,
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
        FuzzySimplSet::Naive::smooth_knn_dist(n, knn_indices, knn_dists,
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


        //TODO: Expose this so it can be swapped out
        FuzzySimplSet::Naive::compute_membership_strength_kernel<TPB_X><<<grid, blk>>>(knn_indices,
                knn_dists, sigmas, rhos, graph_vals, graph_rows, graph_cols, n,
                params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

	    /**
	     * Init_transform()
	     *
	     *         # This was a very specially constructed graph with constant degree.
                    # That lets us do fancy unpacking by reshaping the csr matrix indices
                    # and data. Doing so relies on the constant degree assumption!
                    csr_graph = normalize(graph.tocsr(), norm="l1")
                    inds = csr_graph.indices.reshape(X.shape[0], self._n_neighbors)
                    weights = csr_graph.data.reshape(X.shape[0], self._n_neighbors)
                    embedding = init_transform(inds, weights, self.embedding_)
	     */


//        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
//        graph.eliminate_zeros()
//
//        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
//
//        head = graph.row
//        tail = graph.col
//
//        embedding = optimize_layout(
//            embedding,
//            self.embedding_,
//            head,
//            tail,
//            n_epochs,
//            graph.shape[1],
//            epochs_per_sample,
//            self._a,
//            self._b,
//            rng_state,
//            self.repulsion_strength,
//            self._initial_alpha,
//            self.negative_sample_rate,
//            verbose=self.verbose,
//        )

	}


	void find_ab(UMAPParams *params) {
	    Optimize::find_params_ab(params);
	}
}
