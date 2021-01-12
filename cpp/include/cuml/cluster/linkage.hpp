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

#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>

namespace ML {

enum LinkageDistance { PAIRWISE = 0, KNN_GRAPH = 1 };

template <typename value_idx, typename value_t>
struct linkage_output {
  value_idx m;
  value_idx n_clusters;

  value_idx n_leaves;
  value_idx n_connected_components;

  value_idx *labels;  // size: m

  value_idx *children;  // size: (m-1, 2)
};

struct linkage_output_float : public linkage_output<int, float> {};

/**
 * @defgroup HdbscanCpp C++ implementation of Dbscan algo
 * @brief Fits an HDBSCAN model on an input feature matrix and outputs the labels,
 *        dendrogram, and minimum spanning tree.

 * @param[in] handle
 * @param[in] X
 * @param[in] m
 * @param[in] n
 * @param[in] metric
 * @param[out] out
 */
void single_linkage(const raft::handle_t &handle, const float *X, size_t m,
                    size_t n, raft::distance::DistanceType metric,
                    LinkageDistance dist_type, linkage_output<int, float> *out,
                    int c = 15, int n_clusters = 5);

/** @} */

}  // namespace ML
