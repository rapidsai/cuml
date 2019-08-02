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

#include "kmeans-inl.cuh"
#include "kmeans-sg-inl.cuh"

namespace ML {
namespace kmeans {

// -------------------------- fit_predict --------------------------------//
void fit_predict(const ML::cumlHandle &handle, const KMeansParams &params,
                 const float *X, int n_samples, int n_features,
                 float *centroids, int *labels, float &inertia, int &n_iter) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  fit(h, params, X, n_samples, n_features, centroids, inertia, n_iter);
  predict(h, params, centroids, X, n_samples, n_features, labels, inertia);
}

void fit_predict(const ML::cumlHandle &handle, const KMeansParams &params,
                 const double *X, int n_samples, int n_features,
                 double *centroids, int *labels, double &inertia, int &n_iter) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  fit(h, params, X, n_samples, n_features, centroids, inertia, n_iter);
  predict(h, params, centroids, X, n_samples, n_features, labels, inertia);
}

// ----------------------------- fit ---------------------------------//

void fit(const ML::cumlHandle &handle, const KMeansParams &params,
         const float *X, int n_samples, int n_features, float *centroids,
         float &inertia, int &n_iter) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  fit(h, params, X, n_samples, n_features, centroids, inertia, n_iter);
}

void fit(const ML::cumlHandle &handle, const KMeansParams &params,
         const double *X, int n_samples, int n_features, double *centroids,
         double &inertia, int &n_iter) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  fit(h, params, X, n_samples, n_features, centroids, inertia, n_iter);
}

// ----------------------------- predict ---------------------------------//

void predict(const ML::cumlHandle &handle, const KMeansParams &params,
             const float *centroids, const float *X, int n_samples,
             int n_features, int *labels, float &inertia) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  predict(h, params, centroids, X, n_samples, n_features, labels, inertia);
}

void predict(const ML::cumlHandle &handle, const KMeansParams &params,
             const double *centroids, const double *X, int n_samples,
             int n_features, int *labels, double &inertia) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  predict(h, params, centroids, X, n_samples, n_features, labels, inertia);
}

// ----------------------------- transform ---------------------------------//
void transform(const ML::cumlHandle &handle, const KMeansParams &params,
               const float *centroids, const float *X, int n_samples,
               int n_features, int metric, float *X_new) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  transform(h, params, centroids, X, n_samples, n_features, metric, X_new);
}

void transform(const ML::cumlHandle &handle, const KMeansParams &params,
               const double *centroids, const double *X, int n_samples,
               int n_features, int metric, double *X_new) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);

  transform(h, params, centroids, X, n_samples, n_features, metric, X_new);
}

};  // end namespace kmeans
};  // end namespace ML
