/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>

#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include "../convert/coo.cuh"
#include "../convert/csr.cuh"
#include "../convert/dense.cuh"
#include "../csr.cuh"
#include "../linalg/transpose.h"
#include "../utils.h"

#include "bin_distance.cuh"
#include "ip_distance.cuh"
#include "l2_distance.cuh"
#include "lp_distance.cuh"

#include <cusparse_v2.h>

namespace raft {
namespace sparse {
namespace distance {

/**
 * Compute pairwise distances between A and B, using the provided input
 * configuration and distance function.
 *
 * @param[out] out          dense output array (size A.nrows * B.nrows)
 * @param[in]  input_config input argument configuration
 * @param[in]  metric       distance metric to use
 * @param[in]  metric_arg   The metric argument
 *
 * @tparam value_idx index type
 * @tparam value_t   value type
 */
template <typename value_idx = int, typename value_t = float>
void pairwiseDistance(value_t *out,
                      distances_config_t<value_idx, value_t> input_config,
                      raft::distance::DistanceType metric, float metric_arg) {
  switch (metric) {
    case raft::distance::DistanceType::L2Expanded:
      l2_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::InnerProduct:
      ip_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::L2Unexpanded:
      l2_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::L1:
      l1_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::LpUnexpanded:
      lp_unexpanded_distances_t<value_idx, value_t>(input_config, metric_arg)
        .compute(out);
      break;
    case raft::distance::DistanceType::Linf:
      linf_unexpanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;
    case raft::distance::DistanceType::Canberra:
      canberra_unexpanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;
    case raft::distance::DistanceType::JaccardExpanded:
      jaccard_expanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      cosine_expanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;
    case raft::distance::DistanceType::HellingerExpanded:
      hellinger_expanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;

    default:
      THROW("Unsupported distance: %d", metric);
  }
}

};  // namespace distance
};  // namespace sparse
};  // namespace raft
