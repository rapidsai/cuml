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

#pragma once

#include "umap/umapparams.h"

#include "sparse/coo.h"

#include "spectral/spectral.h"
#include <iostream>

namespace UMAPAlgo {

    namespace InitEmbed {

        namespace SpectralInit {

            using namespace ML;


            /**
             * Performs a spectral layout initialization
             */
            template<typename T>
            void launcher(const cumlHandle &handle,
                          const T *X, int n, int d,
                          const long *knn_indices, const T *knn_dists,
                          MLCommon::Sparse::COO<float> *coo,
                          UMAPParams *params,
                          T *embedding) {

                Spectral::fit_embedding(handle,coo->rows, coo->cols, coo->vals, coo->nnz, n, params->n_components, embedding);
            }
        }
    }
};
