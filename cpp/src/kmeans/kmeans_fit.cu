/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/kmeans.hpp>
#include <cuml/cluster/kmeans_params.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/cluster/kmeans.hpp>

#include <optional>
#include <vector>

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
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const value_t, idx_t>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(
      raft::make_device_vector_view<const value_t, idx_t>(sample_weight, n_samples));
  auto centroids_view =
    raft::make_device_matrix_view<value_t, idx_t>(centroids, params.n_clusters, n_features);
  auto inertia_view = raft::make_host_scalar_view<value_t>(&inertia);
  auto n_iter_view  = raft::make_host_scalar_view<idx_t>(&n_iter);

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

template <typename value_t>
void fit_parts_impl(const raft::handle_t& handle,
                    const KMeansParams& params,
                    const value_t** X_parts,
                    const int64_t* n_samples_parts,
                    int64_t n_parts,
                    int64_t n_features,
                    const value_t** sample_weight_parts,
                    value_t* centroids,
                    value_t& inertia,
                    int64_t& n_iter)
{
  std::vector<raft::device_matrix_view<const value_t, int64_t>> X_views;
  X_views.reserve(n_parts);

  std::optional<std::vector<raft::device_vector_view<const value_t, int64_t>>> sw_views =
    std::nullopt;
  if (sample_weight_parts != nullptr) {
    sw_views.emplace();
    sw_views->reserve(n_parts);
  }

  for (int64_t i = 0; i < n_parts; ++i) {
    X_views.emplace_back(raft::make_device_matrix_view<const value_t, int64_t>(
      X_parts[i], n_samples_parts[i], n_features));
    if (sw_views.has_value()) {
      sw_views->emplace_back(raft::make_device_vector_view<const value_t, int64_t>(
        sample_weight_parts[i], n_samples_parts[i]));
    }
  }

  auto centroids_view =
    raft::make_device_matrix_view<value_t, int64_t>(centroids, params.n_clusters, n_features);
  auto inertia_view = raft::make_host_scalar_view<value_t>(&inertia);
  auto n_iter_view  = raft::make_host_scalar_view<int64_t>(&n_iter);

  cuvs::cluster::kmeans::mg::fit(
    handle, params.to_cuvs(), X_views, sw_views, centroids_view, inertia_view, n_iter_view);
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float** X_parts,
         const int64_t* n_samples_parts,
         int64_t n_parts,
         int64_t n_features,
         const float** sample_weight_parts,
         float* centroids,
         float& inertia,
         int64_t& n_iter)
{
  fit_parts_impl(handle,
                 params,
                 X_parts,
                 n_samples_parts,
                 n_parts,
                 n_features,
                 sample_weight_parts,
                 centroids,
                 inertia,
                 n_iter);
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double** X_parts,
         const int64_t* n_samples_parts,
         int64_t n_parts,
         int64_t n_features,
         const double** sample_weight_parts,
         double* centroids,
         double& inertia,
         int64_t& n_iter)
{
  fit_parts_impl(handle,
                 params,
                 X_parts,
                 n_samples_parts,
                 n_parts,
                 n_features,
                 sample_weight_parts,
                 centroids,
                 inertia,
                 n_iter);
}

};  // end namespace kmeans
};  // end namespace ML
