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

template <typename value_idx, typename value_t>
struct hdbscan_output {
  value_idx m;
  value_idx *labels;       // size: m
  value_t *probabilities;  // size: m

  value_idx *mst_src;  // size: m-1
  value_idx *mst_dst;  // size: m-1
  value_t *mst_data;   // size: m-1

  value_idx *linakage_parents;  // size: m
  value_idx *linkage_children;  // size: m
  value_idx *linkage_deltas;    // size: m
  value_idx *linkage_sizes;     // size: m
};

struct hdbscan_output_float : public hdbscan_output<int, float> {};

/**
 * @defgroup HdbscanCpp C++ implementation of Dbscan algo
 * @brief Fits an HDBSCAN model on an input feature matrix and outputs the labels,
 *        dendrogram, and minimum spanning tree.

 * @param[in] handle
 * @param[in] X
 * @param[in] m
 * @param[in] n
 * @param[in] metric
 * @param[in] k
 * @param[in] min_pts
 * @param[in] alpha
 * @param[out] out
 */
void hdbscan(const raft::handle_t &handle, const float *X, int m, int n,
             raft::distance::DistanceType metric, int k, int min_pts,
             float alpha, hdbscan_output<int, float> *out);

/** @} */

}  // namespace ML
