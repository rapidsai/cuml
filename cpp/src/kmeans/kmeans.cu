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

#include "sg_impl.cuh"
#include <cuml/cluster/kmeans.hpp>

namespace ML {
namespace kmeans {

// -------------------------- fit_predict --------------------------------//
void fit_predict(const raft::handle_t& handle,
                 const KMeansParams& params,
                 const float* X,
                 int n_samples,
                 int n_features,
                 const float* sample_weight,
                 float* centroids,
                 int* labels,
                 float& inertia,
                 int& n_iter)
{
  impl::fit(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  impl::predict(
    handle, params, centroids, X, n_samples, n_features, sample_weight, true, labels, inertia);
}

void fit_predict(const raft::handle_t& handle,
                 const KMeansParams& params,
                 const double* X,
                 int n_samples,
                 int n_features,
                 const double* sample_weight,
                 double* centroids,
                 int* labels,
                 double& inertia,
                 int& n_iter)
{
  impl::fit(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  impl::predict(
    handle, params, centroids, X, n_samples, n_features, sample_weight, true, labels, inertia);
}

// ----------------------------- fit ---------------------------------//

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
  impl::fit(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
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
  impl::fit(handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
}

// ----------------------------- predict ---------------------------------//

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
  impl::predict(handle,
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
  impl::predict(handle,
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

// ----------------------------- transform ---------------------------------//
void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const float* centroids,
               const float* X,
               int n_samples,
               int n_features,
               int metric,
               float* X_new)
{
  impl::transform(handle, params, centroids, X, n_samples, n_features, metric, X_new);
}

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const double* centroids,
               const double* X,
               int n_samples,
               int n_features,
               int metric,
               double* X_new)
{
  impl::transform(handle, params, centroids, X, n_samples, n_features, metric, X_new);
}

};  // end namespace kmeans
};  // end namespace ML
