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

#include <cuml/cluster/kmeans.hpp>
#include "sg_impl.cuh"

namespace ML {
namespace kmeans {

// -------------------------- fit_predict --------------------------------//
void fit_predict(const raft::handle_t &handle, const KMeansParams &params,
                 const float *X, int n_samples, int n_features,
                 const float *sample_weight, float *centroids, int *labels,
                 float &inertia, int &n_iter) {
  const raft::handle_t &h = handle;
  std::cout << "Before stream sync" << std::endl;
  raft::stream_syncer _(h);
  std::cout << "After stream sync" << std::endl;

  fit(h, params, X, n_samples, n_features, sample_weight, centroids, inertia,
      n_iter);
  predict(h, params, centroids, X, n_samples, n_features, sample_weight, labels,
          inertia);
}

void fit_predict(const raft::handle_t &handle, const KMeansParams &params,
                 const double *X, int n_samples, int n_features,
                 const double *sample_weight, double *centroids, int *labels,
                 double &inertia, int &n_iter) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);

  fit(h, params, X, n_samples, n_features, sample_weight, centroids, inertia,
      n_iter);
  predict(h, params, centroids, X, n_samples, n_features, sample_weight, labels,
          inertia);
}

// ----------------------------- fit ---------------------------------//

void fit(const raft::handle_t &handle, const KMeansParams &params,
         const float *X, int n_samples, int n_features,
         const float *sample_weight, float *centroids, float &inertia,
         int &n_iter) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);

  fit(h, params, X, n_samples, n_features, sample_weight, centroids, inertia,
      n_iter);
}

void fit(const raft::handle_t &handle, const KMeansParams &params,
         const double *X, int n_samples, int n_features,
         const double *sample_weight, double *centroids, double &inertia,
         int &n_iter) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);

  fit(h, params, X, n_samples, n_features, sample_weight, centroids, inertia,
      n_iter);
}

// ----------------------------- predict ---------------------------------//

void predict(const raft::handle_t &handle, const KMeansParams &params,
             const float *centroids, const float *X, int n_samples,
             int n_features, const float *sample_weight, int *labels,
             float &inertia) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);

  predict(h, params, centroids, X, n_samples, n_features, sample_weight, labels,
          inertia);
}

void predict(const raft::handle_t &handle, const KMeansParams &params,
             const double *centroids, const double *X, int n_samples,
             int n_features, const double *sample_weight, int *labels,
             double &inertia) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);

  predict(h, params, centroids, X, n_samples, n_features, sample_weight, labels,
          inertia);
}

// ----------------------------- transform ---------------------------------//
void transform(const raft::handle_t &handle, const KMeansParams &params,
               const float *centroids, const float *X, int n_samples,
               int n_features, int metric, float *X_new) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);

  transform(h, params, centroids, X, n_samples, n_features, metric, X_new);
}

void transform(const raft::handle_t &handle, const KMeansParams &params,
               const double *centroids, const double *X, int n_samples,
               int n_features, int metric, double *X_new) {
  const raft::handle_t &h = handle;
  raft::stream_syncer _(h);

  transform(h, params, centroids, X, n_samples, n_features, metric, X_new);
}

};  // end namespace kmeans
};  // end namespace ML
