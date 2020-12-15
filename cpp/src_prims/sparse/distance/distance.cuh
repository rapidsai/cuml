/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <limits.h>
#include <raft/cudart_utils.h>
#include <sparse/distance/common.h>

#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <common/device_buffer.hpp>

#include <sparse/utils.h>
#include <sparse/csr.cuh>
#include <sparse/distance/ip_distance.cuh>
#include <sparse/distance/l2_distance.cuh>
#include <sparse/distance/semiring.cuh>
#include <sparse/distance/spmv.cuh>

#include <cuml/common/cuml_allocator.hpp>
#include <cuml/neighbors/knn.hpp>

#include <nvfunctional>

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
namespace Distance {

/**
 * Compute pairwise distances between A and B, using the provided
 * input configuration and distance function.
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @param[out] out dense output array (size A.nrows * B.nrows)
 * @param[in] input_config input argument configuration
 * @param[in] metric distance metric to use
 */
template class ip_distances_t<int, float>;
template class l2_distances_t<int, float>;
template class distances_config_t<int, float>;

template <typename value_idx = int, typename value_t = float>
void pairwiseDistance(value_t *out,
                      distances_config_t<value_idx, value_t> input_config,
                      raft::distance::DistanceType metric) {
  CUML_LOG_DEBUG("Running sparse pairwise distances with metric=%d", metric);

  switch (metric) {
    case raft::distance::DistanceType::EucExpandedL2:
      // EucExpandedL2
      l2_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::InnerProduct:
      // InnerProduct
      ip_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::EucUnexpandedL1:
      l1_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::EucUnexpandedL2:
      l2_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::ChebyChev:
      chebychev_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::Canberra:
      canberra_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    default:
      THROW("Unsupported metric: %d", metric);
  }
}

};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
