
/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/distance/distance.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include "pairwise_distance_l1.cuh"

namespace ML {

namespace Metrics {
void pairwise_distance_l1(const raft::handle_t& handle,
                          const double* x,
                          const double* y,
                          double* dist,
                          int m,
                          int n,
                          int k,
                          raft::distance::DistanceType metric,
                          bool isRowMajor,
                          double metric_arg)
{
  // Allocate workspace
  rmm::device_uvector<char> workspace(1, handle.get_stream());
  // Call the distance function
  switch (metric) {
    case raft::distance::DistanceType::L1:
      raft::distance::pairwise_distance_impl<double, int, raft::distance::DistanceType::L1>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    default: THROW("Unknown or unsupported distance metric '%d'!", (int)metric);
  }
}

void pairwise_distance_l1(const raft::handle_t& handle,
                          const float* x,
                          const float* y,
                          float* dist,
                          int m,
                          int n,
                          int k,
                          raft::distance::DistanceType metric,
                          bool isRowMajor,
                          float metric_arg)
{
  // Allocate workspace
  rmm::device_uvector<char> workspace(1, handle.get_stream());
  // Call the distance function
  switch (metric) {
    case raft::distance::DistanceType::L1:
      raft::distance::pairwise_distance_impl<float, int, raft::distance::DistanceType::L1>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    default: THROW("Unknown or unsupported distance metric '%d'!", (int)metric);
  }
}

}  // namespace Metrics
}  // namespace ML
