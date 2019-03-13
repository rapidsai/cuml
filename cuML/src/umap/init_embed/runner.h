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

#include "umap/umapparams.h"

#include "random_algo.h"
#include "spectral_algo.h"

#pragma once

namespace UMAPAlgo {

    namespace InitEmbed {

        using namespace ML;

        template<typename T>
        void run(const T *X, int n, int d,
                 const long *knn_indices, const T *knn_dists,
                 int *rows, int *cols, float *vals,
                 int nnz,
                 UMAPParams *params, T* embedding,
                 int algo = 0) {
            switch(algo) {

            /**
             * Initial algo uses FAISS indices
             */
            case 0:
                RandomInit::launcher(X, n, d, knn_indices, knn_dists,  params, embedding);
                break;

            case 1:
                SpectralInit::launcher(X, n, d,
                        knn_indices, knn_dists,
                        rows, cols, vals,
                        nnz,
                        params, embedding);
                break;
            }
        }
    }
};
