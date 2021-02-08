/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>

namespace ML {

/**
 * @defgroup HdbscanCpp C++ implementation of Dbscan algo
 * @brief Fits an HDBSCAN model on an input feature matrix and outputs the labels,
 *        dendrogram, and minimum spanning tree.
 *  TODO: Use a separate type to represent number of edges so we can scale up
 *  number of edges without having to use 64-bit ints for vertices.

 * @param[in] handle
 * @param[in] X
 * @param[in] m
 * @param[in] n
 * @param[in] metric
 * @param[out] out
 */
void single_linkage_pairwise(const raft::handle_t &handle, const float *X,
                             size_t m, size_t n,
                             raft::distance::DistanceType metric,
                             raft::hierarchy::linkage_output<int, float> *out,
                             int c = 15, int n_clusters = 5);

void single_linkage_neighbors(const raft::handle_t &handle, const float *X,
                              size_t m, size_t n,
                              raft::distance::DistanceType metric,
                              raft::hierarchy::linkage_output<int, float> *out,
                              int c = 15, int n_clusters = 5);

void single_linkage_pairwise(
  const raft::handle_t &handle, const float *X, size_t m, size_t n,
  raft::distance::DistanceType metric,
  raft::hierarchy::linkage_output<int64_t, float> *out, int c = 15,
  int n_clusters = 5);

/** @} */

};  // namespace ML
