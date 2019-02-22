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
#include "umap/umapparams.h"
#include "umap/runner.h"

#include "knn/knn.h"

namespace ML {

    void UMAP::fit(const float *X, int n, int d,
            kNN *knn, float *embeddings) {
		UMAPAlgo::_fit(X, n, d, knn, get_params(), embeddings);
	}

	void UMAP::transform(const float *X, int n, int d,
	        float *embedding, int embedding_n, kNN *knn,
	        float *out) {
	    UMAPAlgo::_transform<float, 256>(X, n, d,
	            embedding, embedding_n, knn,
	            get_params(), out);
	}

	UMAPParams* UMAP::get_params() { return this->params; }
};
