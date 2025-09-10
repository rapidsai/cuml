/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/cluster/kmeans_params.hpp>

namespace raft {
class handle_t;
}

namespace ML {

namespace kmeans {

/**
 * @brief Compute k-means clustering for each sample in the input.
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
 * @param[in]     sample_weight The weights for each observation in X.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 centroids  as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    inertia       Sum of squared distances of samples to their
 closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int n_samples,
         int n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int& n_iter);

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int n_samples,
         int n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int& n_iter);

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int64_t n_samples,
         int64_t n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int64_t& n_iter);

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int64_t n_samples,
         int64_t n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int64_t& n_iter);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @param[in]     handle            The handle to the cuML library context
 * that manages the CUDA resources.
 * @param[in]     params            Parameters for KMeans model.
 * @param[in]     centroids         Cluster centroids. It must be noted that
 * the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     X                 New data to predict.
 * @param[in]     n_samples         Number of samples in the input X.
 * @param[in]     n_features        Number of features or the dimensions of
 * each sample in 'X' (value should be same as the dimension for each cluster
 * centers in 'centroids').
 * @param[in]     sample_weight     The weights for each observation in X.
 * @param[in]     normalize_weights True if the weights should be normalized
 * @param[out]    labels            Index of the cluster each sample in X
 * belongs to.
 * @param[out]    inertia           Sum of squared distances of samples to
 * their closest cluster center.
 */

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const float* centroids,
             const float* X,
             int n_samples,
             int n_features,
             const float* sample_weight,
             bool normalize_weights,
             int* labels,
             float& inertia);

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const double* centroids,
             const double* X,
             int n_samples,
             int n_features,
             const double* sample_weight,
             bool normalize_weights,
             int* labels,
             double& inertia);
void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const float* centroids,
             const float* X,
             int64_t n_samples,
             int64_t n_features,
             const float* sample_weight,
             bool normalize_weights,
             int64_t* labels,
             float& inertia);

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const double* centroids,
             const double* X,
             int64_t n_samples,
             int64_t n_features,
             const double* sample_weight,
             bool normalize_weights,
             int64_t* labels,
             double& inertia);
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
 * @param[out]    X_new         X transformed in the new space..
 */
void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const float* centroids,
               const float* X,
               int n_samples,
               int n_features,
               float* X_new);

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const double* centroids,
               const double* X,
               int n_samples,
               int n_features,
               double* X_new);
void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const float* centroids,
               const float* X,
               int64_t n_samples,
               int64_t n_features,
               float* X_new);

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const double* centroids,
               const double* X,
               int64_t n_samples,
               int64_t n_features,
               double* X_new);
};  // end namespace kmeans
};  // end namespace ML
