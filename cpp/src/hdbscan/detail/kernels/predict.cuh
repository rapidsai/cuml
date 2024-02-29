/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

template <typename value_idx, typename value_t>
CUML_KERNEL void min_mutual_reachability_kernel(value_t* input_core_dists,
                                                value_t* prediction_core_dists,
                                                value_t* pairwise_dists,
                                                value_idx* neighbor_indices,
                                                size_t n_prediction_points,
                                                value_idx neighborhood,
                                                value_t* min_mr_dists,
                                                value_idx* min_mr_indices)
{
  value_idx idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < value_idx(n_prediction_points)) {
    value_t min_mr_dist  = std::numeric_limits<value_t>::max();
    value_idx min_mr_ind = -1;
    for (int i = 0; i < neighborhood; i++) {
      value_t mr_dist = prediction_core_dists[idx];
      if (input_core_dists[neighbor_indices[idx * neighborhood + i]] > mr_dist) {
        mr_dist = input_core_dists[neighbor_indices[idx * neighborhood + i]];
      }
      if (pairwise_dists[idx * neighborhood + i] > mr_dist) {
        mr_dist = pairwise_dists[idx * neighborhood + i];
      }
      if (min_mr_dist > mr_dist) {
        min_mr_dist = mr_dist;
        min_mr_ind  = neighbor_indices[idx * neighborhood + i];
      }
    }
    min_mr_dists[idx]   = min_mr_dist;
    min_mr_indices[idx] = min_mr_ind;
  }
  return;
}

template <typename value_idx, typename value_t>
CUML_KERNEL void cluster_probability_kernel(value_idx* min_mr_indices,
                                            value_t* prediction_lambdas,
                                            value_idx* index_into_children,
                                            value_idx* labels,
                                            value_t* deaths,
                                            value_idx* selected_clusters,
                                            value_idx* parents,
                                            value_t* lambdas,
                                            value_idx n_leaves,
                                            size_t n_prediction_points,
                                            value_idx* predicted_labels,
                                            value_t* cluster_probabilities)
{
  value_idx idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < value_idx(n_prediction_points)) {
    value_idx cluster_label = labels[min_mr_indices[idx]];

    if (cluster_label >= 0 && selected_clusters[cluster_label] > n_leaves &&
        lambdas[index_into_children[selected_clusters[cluster_label]]] < prediction_lambdas[idx]) {
      predicted_labels[idx] = cluster_label;
    } else if (cluster_label >= 0 && selected_clusters[cluster_label] == n_leaves) {
      predicted_labels[idx] = cluster_label;
    } else {
      predicted_labels[idx] = -1;
    }
    if (predicted_labels[idx] >= 0) {
      value_t max_lambda = deaths[selected_clusters[cluster_label] - n_leaves];
      if (max_lambda > 0) {
        cluster_probabilities[idx] =
          (max_lambda < prediction_lambdas[idx] ? max_lambda : prediction_lambdas[idx]) /
          max_lambda;
      } else {
        cluster_probabilities[idx] = 1.0;
      }
    } else {
      cluster_probabilities[idx] = 0.0;
    }
  }
  return;
}

};  // namespace Predict
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
