/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>

#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>

namespace ML {
namespace kmeans {

// ----------------------------- transform ---------------------------------//
template <typename value_t, typename idx_t>
void transform_impl(const raft::handle_t& handle,
                    const raft::cluster::KMeansParams& params,
                    const value_t* centroids,
                    const value_t* X,
                    idx_t n_samples,
                    idx_t n_features,
                    value_t* X_new)
{
  auto X_view = raft::make_device_matrix_view<const value_t, idx_t>(X, n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<const value_t, idx_t>(centroids, params.n_clusters, n_features);
  auto rX_new = raft::make_device_matrix_view<value_t, idx_t>(X_new, n_samples, n_features);

  raft::cluster::kmeans::transform<value_t, idx_t>(handle, params, X_view, centroids_view, rX_new);
}

void transform(const raft::handle_t& handle,
               const raft::cluster::KMeansParams& params,
               const float* centroids,
               const float* X,
               int n_samples,
               int n_features,
               float* X_new)
{
  transform_impl(handle, params, centroids, X, n_samples, n_features, X_new);
}

void transform(const raft::handle_t& handle,
               const raft::cluster::KMeansParams& params,
               const double* centroids,
               const double* X,
               int n_samples,
               int n_features,
               double* X_new)
{
  transform_impl(handle, params, centroids, X, n_samples, n_features, X_new);
}

void transform(const raft::handle_t& handle,
               const raft::cluster::KMeansParams& params,
               const float* centroids,
               const float* X,
               int64_t n_samples,
               int64_t n_features,
               float* X_new)
{
  transform_impl(handle, params, centroids, X, n_samples, n_features, X_new);
}

void transform(const raft::handle_t& handle,
               const raft::cluster::KMeansParams& params,
               const double* centroids,
               const double* X,
               int64_t n_samples,
               int64_t n_features,
               double* X_new)
{
  transform_impl(handle, params, centroids, X, n_samples, n_features, X_new);
}

};  // end namespace kmeans
};  // end namespace ML
