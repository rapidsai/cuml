/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/cluster/specializations.cuh>

#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>

namespace ML {
namespace kmeans {

// -------------------------- fit_predict --------------------------------//
void fit_predict(const raft::handle_t& handle,
                 const raft::cluster::KMeansParams& params,
                 const float* X,
                 int n_samples,
                 int n_features,
                 const float* sample_weight,
                 float* centroids,
                 int* labels,
                 float& inertia,
                 int& n_iter)
{
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const float>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(raft::make_device_vector_view((sample_weight), n_samples));
  auto centroids_opt =
    std::make_optional(raft::make_device_matrix_view(centroids, params.n_clusters, n_features));
  auto rLabels      = raft::make_device_vector_view(labels, n_samples);
  auto inertia_view = raft::make_host_scalar_view(&inertia);
  auto n_iter_view  = raft::make_host_scalar_view(&n_iter);

  raft::cluster::kmeans_fit_predict<float, int>(
    handle, params, X_view, sw, centroids_opt, rLabels, inertia_view, n_iter_view);
}

void fit_predict(const raft::handle_t& handle,
                 const raft::cluster::KMeansParams& params,
                 const double* X,
                 int n_samples,
                 int n_features,
                 const double* sample_weight,
                 double* centroids,
                 int* labels,
                 double& inertia,
                 int& n_iter)
{
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const double>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(raft::make_device_vector_view(sample_weight, n_samples));
  auto centroids_opt =
    std::make_optional(raft::make_device_matrix_view(centroids, params.n_clusters, n_features));
  auto rLabels      = raft::make_device_vector_view(labels, n_samples);
  auto inertia_view = raft::make_host_scalar_view(&inertia);
  auto n_iter_view  = raft::make_host_scalar_view(&n_iter);

  raft::cluster::kmeans_fit_predict<double, int>(
    handle, params, X_view, sw, centroids_opt, rLabels, inertia_view, n_iter_view);
}

// ----------------------------- fit ---------------------------------//

void fit(const raft::handle_t& handle,
         const raft::cluster::KMeansParams& params,
         const float* X,
         int n_samples,
         int n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int& n_iter)
{
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const float>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(raft::make_device_vector_view((sample_weight), n_samples));
  auto centroids_view = raft::make_device_matrix_view(centroids, params.n_clusters, n_features);
  auto inertia_view   = raft::make_host_scalar_view(&inertia);
  auto n_iter_view    = raft::make_host_scalar_view(&n_iter);

  raft::cluster::kmeans_fit<float, int>(
    handle, params, X_view, sw, centroids_view, inertia_view, n_iter_view);
}

void fit(const raft::handle_t& handle,
         const raft::cluster::KMeansParams& params,
         const double* X,
         int n_samples,
         int n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int& n_iter)
{
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const double>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(raft::make_device_vector_view(sample_weight, n_samples));
  auto centroids_view = raft::make_device_matrix_view(centroids, params.n_clusters, n_features);
  auto inertia_view   = raft::make_host_scalar_view(&inertia);
  auto n_iter_view    = raft::make_host_scalar_view(&n_iter);

  raft::cluster::kmeans_fit<double, int>(
    handle, params, X_view, sw, centroids_view, inertia_view, n_iter_view);
}

// ----------------------------- predict ---------------------------------//

void predict(const raft::handle_t& handle,
             const raft::cluster::KMeansParams& params,
             const float* centroids,
             const float* X,
             int n_samples,
             int n_features,
             const float* sample_weight,
             bool normalize_weights,
             int* labels,
             float& inertia)
{
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const float>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(raft::make_device_vector_view(sample_weight, n_samples));
  auto centroids_view = raft::make_device_matrix_view(centroids, params.n_clusters, n_features);
  auto rLabels        = raft::make_device_vector_view(labels, n_samples);
  auto inertia_view   = raft::make_host_scalar_view(&inertia);

  raft::cluster::kmeans_predict<float, int>(
    handle, params, X_view, sw, centroids_view, rLabels, normalize_weights, inertia_view);
}

void predict(const raft::handle_t& handle,
             const raft::cluster::KMeansParams& params,
             const double* centroids,
             const double* X,
             int n_samples,
             int n_features,
             const double* sample_weight,
             bool normalize_weights,
             int* labels,
             double& inertia)
{
  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const double>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(raft::make_device_vector_view(sample_weight, n_samples));
  auto centroids_view = raft::make_device_matrix_view(centroids, params.n_clusters, n_features);
  auto rLabels        = raft::make_device_vector_view(labels, n_samples);
  auto inertia_view   = raft::make_host_scalar_view(&inertia);

  raft::cluster::kmeans_predict<double, int>(
    handle, params, X_view, sw, centroids_view, rLabels, normalize_weights, inertia_view);
}

// ----------------------------- transform ---------------------------------//
void transform(const raft::handle_t& handle,
               const raft::cluster::KMeansParams& params,
               const float* centroids,
               const float* X,
               int n_samples,
               int n_features,
               float* X_new)
{
  auto X_view         = raft::make_device_matrix_view(X, n_samples, n_features);
  auto centroids_view = raft::make_device_matrix_view(centroids, params.n_clusters, n_features);
  auto rX_new         = raft::make_device_matrix_view(X_new, n_samples, n_features);

  raft::cluster::kmeans_transform<float, int>(handle, params, X_view, centroids_view, rX_new);
}

void transform(const raft::handle_t& handle,
               const raft::cluster::KMeansParams& params,
               const double* centroids,
               const double* X,
               int n_samples,
               int n_features,
               double* X_new)
{
  auto X_view         = raft::make_device_matrix_view(X, n_samples, n_features);
  auto centroids_view = raft::make_device_matrix_view(centroids, params.n_clusters, n_features);
  auto rX_new         = raft::make_device_matrix_view(X_new, n_samples, n_features);

  raft::cluster::kmeans_transform<double, int>(handle, params, X_view, centroids_view, rX_new);
}

};  // end namespace kmeans
};  // end namespace ML
