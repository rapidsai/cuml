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

#include "naive.h"
#include "algo.h"

namespace UMAP {

namespace FuzzySimplSet {

	template<typename T>
	void run(const T *X, int n, int n_neighbors,
			 const long *knn_indices, int knn_indices_n,
			 const T *knn_dists, int knn_dists_n,
			 float local_connectivity, int algorithm) {

		switch(algorithm) {
		case 0:
			Naive::launcher();

		case 1:
			Algo::launcher();
		}
	}

}
};
