/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#include <cuml/cluster/kmeans_params.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cuvs/cluster/kmeans.hpp>

#include <optional>

namespace ML {
namespace kmeans {
namespace opg {

// ----------------------------- fit ---------------------------------//

void fit(const raft::resources& handle,
         const ML::kmeans::KMeansParams& params,
         const float* X,
         int n_samples,
         int n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int& n_iter)
{
  std::optional<raft::device_vector_view<const float, int>> sample_weight_view;
  if (sample_weight != NULL) {
    sample_weight_view = raft::make_device_vector_view<const float, int>(sample_weight, n_samples);
  }

  cuvs::cluster::kmeans::fit(
    handle,
    params.to_cuvs(),
    raft::make_device_matrix_view<const float, int>(X, n_samples, n_features),
    sample_weight_view,
    raft::make_device_matrix_view<float, int>(centroids, params.n_clusters, n_features),
    raft::make_host_scalar_view<float>(&inertia),
    raft::make_host_scalar_view<int>(&n_iter));
}

void fit(const raft::resources& handle,
         const ML::kmeans::KMeansParams& params,
         const double* X,
         int n_samples,
         int n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int& n_iter)
{
  std::optional<raft::device_vector_view<const double, int>> sample_weight_view;
  if (sample_weight != NULL) {
    sample_weight_view = raft::make_device_vector_view<const double, int>(sample_weight, n_samples);
  }

  cuvs::cluster::kmeans::fit(
    handle,
    params.to_cuvs(),
    raft::make_device_matrix_view<const double, int>(X, n_samples, n_features),
    sample_weight_view,
    raft::make_device_matrix_view<double, int>(centroids, params.n_clusters, n_features),
    raft::make_host_scalar_view<double>(&inertia),
    raft::make_host_scalar_view<int>(&n_iter));
}

void fit(const raft::resources& handle,
         const ML::kmeans::KMeansParams& params,
         const float* X,
         int64_t n_samples,
         int64_t n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int64_t& n_iter)
{
  std::optional<raft::device_vector_view<const float, int64_t>> sample_weight_view;
  if (sample_weight != NULL) {
    sample_weight_view =
      raft::make_device_vector_view<const float, int64_t>(sample_weight, n_samples);
  }

  cuvs::cluster::kmeans::fit(
    handle,
    params.to_cuvs(),
    raft::make_device_matrix_view<const float, int64_t>(X, n_samples, n_features),
    sample_weight_view,
    raft::make_device_matrix_view<float, int64_t>(centroids, params.n_clusters, n_features),
    raft::make_host_scalar_view<float>(&inertia),
    raft::make_host_scalar_view<int64_t>(&n_iter));
}

void fit(const raft::resources& handle,
         const ML::kmeans::KMeansParams& params,
         const double* X,
         int64_t n_samples,
         int64_t n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int64_t& n_iter)
{
  std::optional<raft::device_vector_view<const double, int64_t>> sample_weight_view;
  if (sample_weight != NULL) {
    sample_weight_view =
      raft::make_device_vector_view<const double, int64_t>(sample_weight, n_samples);
  }

  cuvs::cluster::kmeans::fit(
    handle,
    params.to_cuvs(),
    raft::make_device_matrix_view<const double, int64_t>(X, n_samples, n_features),
    sample_weight_view,
    raft::make_device_matrix_view<double, int64_t>(centroids, params.n_clusters, n_features),
    raft::make_host_scalar_view<double>(&inertia),
    raft::make_host_scalar_view<int64_t>(&n_iter));
}
};  // end namespace opg
};  // end namespace kmeans
};  // end namespace ML
