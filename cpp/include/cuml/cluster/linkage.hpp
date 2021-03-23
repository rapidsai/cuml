/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/linalg/distance_type.h>
#include <raft/sparse/hierarchy/common.h>

#include <cuml/cuml.hpp>

namespace ML {

/**
 * @brief Computes single-linkage hierarchical clustering on a dense input
 * feature matrix and outputs the labels, dendrogram, and minimum spanning tree.
 * Connectivities are constructed using the full n^2 pairwise distance matrix.
 * This can be very fast for smaller datasets when there is enough memory
 * available.
 * @param[in] handle raft handle to encapsulate expensive resources
 * @param[in] X dense feature matrix on device
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[out] out container object for output arrays
 * @param[in] metric distance metric to use. Must be supported by the
 *              dense pairwise distances API.
 * @param[out] n_clusters number of clusters to cut from resulting dendrogram
 */
void single_linkage_pairwise(const raft::handle_t &handle, const float *X,
                             size_t m, size_t n,
                             raft::hierarchy::linkage_output<int, float> *out,
                             raft::distance::DistanceType metric,
                             int n_clusters = 5);

/**
 * @brief Computes single-linkage hierarchical clustering on a dense input
 * feature matrix and outputs the labels, dendrogram, and minimum spanning tree.
 * Connectivities are constructed using a k-nearest neighbors graph. While this
 * strategy enables the algorithm to scale to much higher numbers of rows,
 * it comes with the downside that additional knn steps may need to be
 * executed to connect an otherwise unconnected k-nn graph.
 * @param[in] handle raft handle to encapsulate expensive resources
 * @param[in] X dense feature matrix on device
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[out] out container object for output arrays
 * @param[in] metric distance metric to use. Must be supported by the
 *              dense pairwise distances API.
 * @param[out] c the optimal value of k is guaranteed to be at least log(n) + c
 * where c is some constant. This constant can usually be set to a fairly low
 * value, like 15, and still maintain good performance.
 * @param[out] n_clusters number of clusters to cut from resulting dendrogram
 */
void single_linkage_neighbors(const raft::handle_t &handle, const float *X,
                              size_t m, size_t n,
                              raft::hierarchy::linkage_output<int, float> *out,
                              raft::distance::DistanceType metric =
                                raft::distance::DistanceType::L2Unexpanded,
                              int c = 15, int n_clusters = 5);

void single_linkage_pairwise(
  const raft::handle_t &handle, const float *X, size_t m, size_t n,
  raft::hierarchy::linkage_output<int64_t, float> *out,
  raft::distance::DistanceType metric, int n_clusters = 5);

};  // namespace ML
