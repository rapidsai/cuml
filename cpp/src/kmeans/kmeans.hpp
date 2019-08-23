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

#pragma once

#include <cuML.hpp>

namespace ML {

namespace kmeans {

struct KMeansParams {
  enum InitMethod { KMeansPlusPlus, Random, Array };

  // The number of clusters to form as well as the number of centroids to
  // generate (default:8).
  int n_clusters = 8;

  /*
   * Method for initialization, defaults to k-means++:
   *  - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
   * to select the initial cluster centers.
   *  - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at
   * random from the input data for the initial centroids.
   *  - InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
   */
  InitMethod init = KMeansPlusPlus;

  // Maximum number of iterations of the k-means algorithm for a single run.
  int max_iter = 300;

  // Relative tolerance with regards to inertia to declare convergence.
  double tol = 1e-4;

  // verbosity mode.
  int verbose = 0;

  // Seed to the random number generator.
  int seed = 0;

  // Metric to use for distance computation. Any metric from
  // MLCommon::Distance::DistanceType can be used
  int metric = 0;

  // Oversampling factor for use in the k-means|| algorithm.
  int oversampling_factor = 2;

  int batch_size = 1 << 15;

  bool inertia_check = false;
};

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 in the input.
 *
 * @param[in]     handle        The handle to the cuML library context that
 manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. It must be noted
 that the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample.
 * @param[in|out] centroids     [in] When init is InitMethod::Array, use
 centroids  as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    labels        Index of the cluster each sample in X belongs
 to.
 * @param[out]    inertia       Sum of squared distances of samples to their
 closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit_predict(const ML::cumlHandle &handle, const KMeansParams &params,
                 const float *X, int n_samples, int n_features,
                 float *centroids, int *labels, float &inertia, int &n_iter);

void fit_predict(const ML::cumlHandle &handle, const KMeansParams &params,
                 const double *X, int n_samples, int n_features,
                 double *centroids, int *labels, double &inertia, int &n_iter);

/**
 * @brief Compute k-means clustering.
 *
 * @param[in]     handle        The handle to the cuML library context that
 manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. It must be noted
 that the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample.
 * @param[in|out] centroids     [in] When init is InitMethod::Array, use
 centroids as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    inertia       Sum of squared distances of samples to their
 closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */

void fit(const ML::cumlHandle &handle, const KMeansParams &params,
         const float *X, int n_samples, int n_features, float *centroids,
         float &inertia, int &n_iter);

void fit(const ML::cumlHandle &handle, const KMeansParams &params,
         const double *X, int n_samples, int n_features, double *centroids,
         double &inertia, int &n_iter);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @param[in]     handle        The handle to the cuML library context that
 * manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     centroids     Cluster centroids. It must be noted that the
 * data must be in row-major format and stored in device accessible location.
 * @param[in]     X             New data to predict.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample in 'X' (value should be same as the dimension for each cluster centers
 * in 'centroids').
 * @param[out]    labels        Index of the cluster each sample in X belongs
 * to.
 * @param[out]    inertia       Sum of squared distances of samples to their
 * closest cluster center.
 */

void predict(const ML::cumlHandle &handle, const KMeansParams &params,
             const float *centroids, const float *X, int n_samples,
             int n_features, int *labels, float &inertia);

void predict(const ML::cumlHandle &handle, const KMeansParams &params,
             const double *centroids, const double *X, int n_samples,
             int n_features, int *labels, double &inertia);

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The handle to the cuML library context that
 * manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     centroids     Cluster centroids. It must be noted that the
 * data must be in row-major format and stored in device accessible location.
 * @param[in]     X             Training instances to cluster. It must be noted
 * that the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample in 'X' (it should be same as the dimension for each cluster centers in
 * 'centroids').
 * @param[in]     metric        Metric to use for distance computation. Any
 * metric from MLCommon::Distance::DistanceType can be used
 * @param[out]    X_new         X transformed in the new space..
 */
void transform(const ML::cumlHandle &handle, const KMeansParams &params,
               const float *centroids, const float *X, int n_samples,
               int n_features, int metric, float *X_new);

void transform(const ML::cumlHandle &handle, const KMeansParams &params,
               const double *centroids, const double *X, int n_samples,
               int n_features, int metric, double *X_new);

};  // end namespace kmeans
};  // end namespace ML
