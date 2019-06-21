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

#include "kmeans.cuh"

namespace ML {
namespace kmeans {

void fit_predict(const ML::cumlHandle &handle, int n_clusters, int metric,
                 kmeans::InitMethod init, int max_iter, double tol, int seed,
                 const float *X, int n_samples, int n_features,
                 float *centroids, int *labels, int verbose) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);
  cudaStream_t stream = h.getStream();

  ML::KMeans<float> kmeans_obj(
    h, n_clusters, static_cast<MLCommon::Distance::DistanceType>(metric), init,
    max_iter, tol, seed, verbose);

  if (kmeans::InitMethod::Array == init) {
    ASSERT(centroids != nullptr,
           "centroids array is null (require a valid array of centroids for "
           "the requested initialization method)");
    kmeans_obj.setCentroids(centroids, n_clusters, n_features);
  }

  kmeans_obj.fit(X, n_samples, n_features);
  if (labels) {
    kmeans_obj.predict(X, n_samples, n_features, labels);
  }

  MLCommon::copy(centroids, kmeans_obj.centroids(), n_clusters * n_features,
                 stream);
}

void fit_predict(const ML::cumlHandle &handle, int n_clusters, int metric,
                 kmeans::InitMethod init, int max_iter, double tol, int seed,
                 const double *X, int n_samples, int n_features,
                 double *centroids, int *labels, int verbose) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);
  cudaStream_t stream = h.getStream();

  ML::KMeans<double> kmeans_obj(
    h, n_clusters, static_cast<MLCommon::Distance::DistanceType>(metric), init,
    max_iter, tol, seed, verbose);

  if (kmeans::InitMethod::Array == init) {
    ASSERT(centroids != nullptr,
           "centroids array is null (require a valid array of centroids for "
           "the requested initialization method)");
    kmeans_obj.setCentroids(centroids, n_clusters, n_features);
  }

  kmeans_obj.fit(X, n_samples, n_features);
  if (labels) {
    kmeans_obj.predict(X, n_samples, n_features, labels);
  }

  MLCommon::copy(centroids, kmeans_obj.centroids(), n_clusters * n_features,
                 stream);
}

void fit(const ML::cumlHandle &handle, int n_clusters, int metric,
         kmeans::InitMethod init, int max_iter, double tol, int seed,
         const float *X, int n_samples, int n_features, float *centroids,
         int verbose) {
  fit_predict(handle, n_clusters, metric, init, max_iter, tol, seed, X,
              n_samples, n_features, centroids, nullptr, verbose);
}

void fit(const ML::cumlHandle &handle, int n_clusters, int metric,
         kmeans::InitMethod init, int max_iter, double tol, int seed,
         const double *X, int n_samples, int n_features, double *centroids,
         int verbose) {
  fit_predict(handle, n_clusters, metric, init, max_iter, tol, seed, X,
              n_samples, n_features, centroids, nullptr, verbose);
}

void predict(const ML::cumlHandle &handle, float *centroids, int n_clusters,
             const float *X, int n_samples, int n_features, int metric,
             int *labels, double *inertia, int verbose) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);
  cudaStream_t stream = h.getStream();

  ML::KMeans<float> kmeans_obj(
    h, n_clusters, static_cast<MLCommon::Distance::DistanceType>(metric));

  kmeans_obj.setCentroids(centroids, n_clusters, n_features);

  kmeans_obj.predict(X, n_samples, n_features, labels);

  const double obj_inertia = -1 * kmeans_obj.getInertia();
  std::memcpy(inertia, &obj_inertia, sizeof(double));
}

void predict(const ML::cumlHandle &handle, double *centroids, int n_clusters,
             const double *X, int n_samples, int n_features, int metric,
             int *labels, double *inertia, int verbose) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);
  cudaStream_t stream = h.getStream();

  ML::KMeans<double> kmeans_obj(
    h, n_clusters, static_cast<MLCommon::Distance::DistanceType>(metric));
  kmeans_obj.setCentroids(centroids, n_clusters, n_features);

  kmeans_obj.predict(X, n_samples, n_features, labels);

  const double obj_inertia = -1 * kmeans_obj.getInertia();
  std::memcpy(inertia, &obj_inertia, sizeof(double));
}

void transform(const ML::cumlHandle &handle, const float *centroids,
               int n_clusters, const float *X, int n_samples, int n_features,
               int metric, float *X_new, double *inertia, int verbose) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);
  cudaStream_t stream = h.getStream();

  ML::KMeans<float> kmeans_obj(
    h, n_clusters, static_cast<MLCommon::Distance::DistanceType>(metric));
  kmeans_obj.setCentroids(centroids, n_clusters, n_features);
  kmeans_obj.transform(X, n_samples, n_features, X_new);

  const double obj_inertia = -1 * kmeans_obj.getInertia();
  std::memcpy(inertia, &obj_inertia, sizeof(double));
}

void transform(const ML::cumlHandle &handle, const double *centroids,
               int n_clusters, const double *X, int n_samples, int n_features,
               int metric, double *X_new, double *inertia, int verbose) {
  const ML::cumlHandle_impl &h = handle.getImpl();
  ML::detail::streamSyncer _(h);
  cudaStream_t stream = h.getStream();

  ML::KMeans<double> kmeans_obj(
    h, n_clusters, static_cast<MLCommon::Distance::DistanceType>(metric));
  kmeans_obj.setCentroids(centroids, n_clusters, n_features);
  kmeans_obj.transform(X, n_samples, n_features, X_new);

  const double obj_inertia = -1 * kmeans_obj.getInertia();
  std::memcpy(inertia, &obj_inertia, sizeof(double));
}

void score(const ML::cumlHandle &handle, float *centroids, int n_clusters,
           const float *X, int n_samples, int n_features, int metric,
           int *labels, double *inertia, int verbose) {
  predict(handle, centroids, n_clusters, X, n_samples, n_features, metric,
          labels, inertia, verbose);
}

void score(const ML::cumlHandle &handle, double *centroids, int n_clusters,
           const double *X, int n_samples, int n_features, int metric,
           int *labels, double *inertia, int verbose) {
  predict(handle, centroids, n_clusters, X, n_samples, n_features, metric,
          labels, inertia, verbose);
}

};  // end namespace kmeans
};  // end namespace ML
