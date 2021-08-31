/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

namespace raft {
class handle_t;
}

namespace ML {
namespace kmeans {
struct KMeansParams;
namespace opg {

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
 * @param[in]     sample_weight The weights for each observation in X.
 * @param[inout]  centroids     When init is InitMethod::Array, use
 centroids as the initial cluster centers
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

};  // end namespace opg
};  // end namespace kmeans
};  // end namespace ML
