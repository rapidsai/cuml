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

namespace UMAP {

	using namespace ML;

	template<typename T>
	size_t run(const T *X, int n, int d, UMAPParams *params) {

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

		kNNGraph::run(X, n, d, knn_indices, knn_dists, params);

		/**
		 * Allocate workspace for fuzzy simplicial set.
		 */
		T *sigmas;
		T *rhos;

		/**
		 * Run Fuzzy simplicial set
		 */
		FuzzySimplSet::run(knn_indices, knn_dists, n,
						   sigmas, rhos,
						   params, 0);

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
		SimplSetEmbed::run();
	}
}
