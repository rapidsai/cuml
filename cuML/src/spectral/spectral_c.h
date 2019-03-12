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

#include "spectral.h"

namespace ML {

    void spectral_fit_clusters(float *X, int m, int n, int n_neighbors,
            int n_clusters, float eigen_tol, int *out) {
        fit_clusters(X, m, n, n_neighbors, n_clusters,  eigen_tol, out);
    }

    void spectral_fit_clusters(double *X, int m, int n, int n_neighbors,
            int n_clusters, float eigen_tol, int *out) {
        fit_clusters(X, m, n, n_neighbors, n_clusters, eigen_tol, out);
    }
}
