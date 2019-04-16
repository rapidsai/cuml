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
#include "runner.h"
#include "umapparams.h"

#include <iostream>

namespace ML {

    static const int TPB_X = 32;

    UMAP_API::UMAP_API(UMAPParams *params): params(params){
        knn = nullptr;
    };

    UMAP_API::~UMAP_API() {
        delete knn;
    }

    /**
     * Fits a UMAP model
     * @param X
     *        pointer to an array in row-major format (note: this will be col-major soon)
     * @param n
     *        n_samples in X
     * @param d
     *        d_features in X
     * @param embeddings
     *        an array to return the output embeddings of size (n_samples, n_components)
     */
    void UMAP_API::fit(float *X, int n, int d, float *embeddings) {
        this->knn = new kNN(d);
        UMAPAlgo::_fit<float, TPB_X>(X, n, d, knn, get_params(), embeddings);
    }

    /**
     * Project a set of X vectors into the embedding space.
     * @param X
     *        pointer to an array in row-major format (note: this will be col-major soon)
     * @param n
     *        n_samples in X
     * @param d
     *        d_features in X
     * @param embedding
     *        pointer to embedding array of size (embedding_n, n_components) that has been created with fit()
     * @param embedding_n
     *        n_samples in embedding array
     * @param out
     *        pointer to array for storing output embeddings (n, n_components)
     */
    void UMAP_API::transform(float *X, int n, int d,
            float *embedding, int embedding_n,
            float *out) {
        UMAPAlgo::_transform<float, TPB_X>(X, n, d,
                embedding, embedding_n, knn,
                get_params(), out);
    }

    /**
     * Get the UMAPParams instance
     */
    UMAPParams* UMAP_API::get_params()  { return this->params; }
}
