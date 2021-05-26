/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuml/cluster/kmeans_mg.hpp>
#include "kmeans_mg_impl.cuh"

namespace ML {
namespace kmeans {
namespace opg {

// ----------------------------- fit ---------------------------------//

void fit(const raft::handle_t &handle, const KMeansParams &params,
         const float *X, int n_samples, int n_features, float *centroids,
         float &inertia, int &n_iter) {
  const raft::handle_t &h = handle;

  raft::stream_syncer _(h);
  impl::fit(h, params, X, n_samples, n_features, centroids, inertia, n_iter);
}

void fit(const raft::handle_t &handle, const KMeansParams &params,
         const double *X, int n_samples, int n_features, double *centroids,
         double &inertia, int &n_iter) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);
  impl::fit(h, params, X, n_samples, n_features, centroids, inertia, n_iter);
}

};  // end namespace opg
};  // end namespace kmeans
};  // end namespace ML
