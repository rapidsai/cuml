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

	template<typename T>
	size_t _fit(const T *X,       // input matrix
	            int n,      // rows
	            int d,      // cols
	            UMAPParams *params) {

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

	void find_ab(UMAPParams *params) {
	    Optimize::find_params_ab(params);
	}
}
