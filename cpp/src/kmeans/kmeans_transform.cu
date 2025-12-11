/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/kmeans_params.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/cluster/kmeans.hpp>

namespace ML {
namespace kmeans {

// ----------------------------- transform ---------------------------------//
template <typename value_t, typename idx_t>
void transform_impl(const raft::handle_t& handle,
                    const KMeansParams& params,
                    const value_t* centroids,
                    const value_t* X,
                    idx_t n_samples,
                    idx_t n_features,
                    value_t* X_new)
{
  auto X_view = raft::make_device_matrix_view<const value_t, int>(X, n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<const value_t, int>(centroids, params.n_clusters, n_features);
  auto rX_new = raft::make_device_matrix_view<value_t, int>(X_new, n_samples, n_features);

  cuvs::cluster::kmeans::transform(handle, params.to_cuvs(), X_view, centroids_view, rX_new);
}

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const float* centroids,
               const float* X,
               int n_samples,
               int n_features,
               float* X_new)
{
  transform_impl(handle, params, centroids, X, n_samples, n_features, X_new);
}

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const double* centroids,
               const double* X,
               int n_samples,
               int n_features,
               double* X_new)
{
  transform_impl(handle, params, centroids, X, n_samples, n_features, X_new);
}

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const float* centroids,
               const float* X,
               int64_t n_samples,
               int64_t n_features,
               float* X_new)
{
  transform_impl(handle, params, centroids, X, n_samples, n_features, X_new);
}

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
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
