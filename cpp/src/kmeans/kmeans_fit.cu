/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ml_cuda_utils.h"

#include <cuml/cluster/kmeans.hpp>
#include <cuml/cluster/kmeans_params.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuvs/cluster/kmeans.hpp>

#include <optional>

namespace ML {
namespace kmeans {

template <typename value_t, typename idx_t>
void fit_impl(const raft::handle_t& handle,
              const KMeansParams& params,
              const value_t* X,
              idx_t n_samples,
              idx_t n_features,
              const value_t* sample_weight,
              value_t* centroids,
              value_t& inertia,
              idx_t& n_iter)
{
  auto centroids_view =
    raft::make_device_matrix_view<value_t, idx_t>(centroids, params.n_clusters, n_features);
  auto inertia_view = raft::make_host_scalar_view<value_t>(&inertia);

  if (!ML::is_device_or_managed_type(X)) {
    auto n_samples_64  = static_cast<int64_t>(n_samples);
    auto n_features_64 = static_cast<int64_t>(n_features);
    auto X_view =
      raft::make_host_matrix_view<const value_t, int64_t>(X, n_samples_64, n_features_64);
    std::optional<raft::host_vector_view<const value_t, int64_t>> sw = std::nullopt;
    if (sample_weight != nullptr)
      sw = std::make_optional(
        raft::make_host_vector_view<const value_t, int64_t>(sample_weight, n_samples_64));
    // The cuVS host overload still wants a `device_matrix_view<..., int64_t>`
    // for centroids. Rebuild it with the matching index type.
    auto centroids_view_64 =
      raft::make_device_matrix_view<value_t, int64_t>(centroids, params.n_clusters, n_features_64);
    int64_t n_iter_64   = 0;
    auto n_iter_view_64 = raft::make_host_scalar_view<int64_t>(&n_iter_64);

    cuvs::cluster::kmeans::fit(
      handle, params.to_cuvs(), X_view, sw, centroids_view_64, inertia_view, n_iter_view_64);
    n_iter = static_cast<idx_t>(n_iter_64);
    return;
  }

  // Device-resident X: original code path, preserves the caller's `idx_t`.
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const value_t, idx_t>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(
      raft::make_device_vector_view<const value_t, idx_t>(sample_weight, n_samples));
  auto n_iter_view = raft::make_host_scalar_view<idx_t>(&n_iter);

  cuvs::cluster::kmeans::fit(
    handle, params.to_cuvs(), X_view, sw, centroids_view, inertia_view, n_iter_view);
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int n_samples,
         int n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int& n_iter)
{
  fit_impl(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int n_samples,
         int n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int& n_iter)
{
  fit_impl(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int64_t n_samples,
         int64_t n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int64_t& n_iter)
{
  fit_impl(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int64_t n_samples,
         int64_t n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int64_t& n_iter)
{
  fit_impl(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
}

};  // end namespace kmeans
};  // end namespace ML
