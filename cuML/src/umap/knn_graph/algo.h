
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

#include "umap/umap.h"
#include "umap/umapparams.h"
#include "knn/knn.h"
#include <iostream>

namespace UMAPAlgo {

namespace kNNGraph {

	namespace Algo {

		using namespace ML;

		/**
		 * Initial implementation calls out to FAISS to do its work.
		 * TODO: cuML kNN implementation should support FAISS' approx NN variants (e.g. IVFPQ GPU).
		 */
		template<typename T>
		void launcher(const float *X, int n, int d,
					  long *knn_indices, T *knn_dists,
					  UMAPParams *params) {

		    std::cout << "Creating knn" << std::endl;

		    kNN knn(d);
			kNNParams *p = new kNNParams[1];
			p[0].ptr = X;
			p[0].N = n;

            std::cout << "Calling fit..." << std::endl;
			knn.fit(p, 1);

			std::cout << "Calling search..." << std::endl;
			knn.search(X, n, knn_indices, knn_dists, params->n_neighbors);

			delete p;
		}
	}
}
};

