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

#include "umap.h"
#include "fuzzy_simpl_set/runner.h"
#include "knn_graph/runner.h"
#include "simpl_set_embed/runner.h"
#include "cuda_utils.h"

namespace UMAPAlgo {

	using namespace ML;

	template<typename T>
	size_t _fit(const T *X, int n, int d, UMAPParams *params, UMAPState<T> *state) {

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

		MLCommon::allocate(knn_indices, n*params->n_neighbors);
		MLCommon::allocate(knn_dists, n*params->n_neighbors);

		kNNGraph::run(X, n, d, knn_indices, knn_dists, params);

		/**
		 * Allocate workspace for fuzzy simplicial set.
		 */
		MLCommon::allocate(state->graph_rows, n*params->n_neighbors);
		MLCommon::allocate(state->graph_cols, n*params->n_neighbors);
		MLCommon::allocate(state->graph_vals, n*params->n_neighbors);

		/**
		 * Run Fuzzy simplicial set
		 */
		FuzzySimplSet::run(knn_indices, knn_dists, n,
						   state->graph_rows,
						   state->graph_cols,
						   state->graph_vals,
						   params, 0);

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
		SimplSetEmbed::run(X, n,
		        state->graph_rows, state->graph_cols, state->graph_vals,
		        params, state);
	}
}
