/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/cluster/kmeans_params.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/cluster/kmeans.hpp>

namespace ML {
namespace kmeans {

// ----------------------------- predict ---------------------------------//

template <typename value_t, typename idx_t>
void predict_impl(const raft::handle_t& handle,
                  const KMeansParams& params,
                  const value_t* centroids,
                  const value_t* X,
                  idx_t n_samples,
                  idx_t n_features,
                  const value_t* sample_weight,
                  bool normalize_weights,
                  idx_t* labels,
                  value_t& inertia)
{
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const value_t>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(
      raft::make_device_vector_view<const value_t, idx_t>(sample_weight, n_samples));
  auto centroids_view =
    raft::make_device_matrix_view<const value_t, idx_t>(centroids, params.n_clusters, n_features);
  auto rLabels      = raft::make_device_vector_view<idx_t, idx_t>(labels, n_samples);
  auto inertia_view = raft::make_host_scalar_view<value_t>(&inertia);

  cuvs::cluster::kmeans::predict(
    handle, params.to_cuvs(), X_view, sw, centroids_view, rLabels, normalize_weights, inertia_view);
}

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const float* centroids,
             const float* X,
             int n_samples,
             int n_features,
             const float* sample_weight,
             bool normalize_weights,
             int* labels,
             float& inertia)
{
  predict_impl(handle,
               params,
               centroids,
               X,
               n_samples,
               n_features,
               sample_weight,
               normalize_weights,
               labels,
               inertia);
}

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const double* centroids,
             const double* X,
             int n_samples,
             int n_features,
             const double* sample_weight,
             bool normalize_weights,
             int* labels,
             double& inertia)
{
  predict_impl(handle,
               params,
               centroids,
               X,
               n_samples,
               n_features,
               sample_weight,
               normalize_weights,
               labels,
               inertia);
}

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const float* centroids,
             const float* X,
             int64_t n_samples,
             int64_t n_features,
             const float* sample_weight,
             bool normalize_weights,
             int64_t* labels,
             float& inertia)
{
  predict_impl(handle,
               params,
               centroids,
               X,
               n_samples,
               n_features,
               sample_weight,
               normalize_weights,
               labels,
               inertia);
}

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const double* centroids,
             const double* X,
             int64_t n_samples,
             int64_t n_features,
             const double* sample_weight,
             bool normalize_weights,
             int64_t* labels,
             double& inertia)
{
  predict_impl(handle,
               params,
               centroids,
               X,
               n_samples,
               n_features,
               sample_weight,
               normalize_weights,
               labels,
               inertia);
}

};  // end namespace kmeans
};  // end namespace ML
