/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>

namespace ML {
namespace Dbscan {
namespace CorePoints {

/**
 * Compute the core points from the vertex degrees and min_pts criterion
 * @param[in]  handle    cuML handle
 * @param[out] mask      Boolean core point mask
 * @param[in]  N         Number of points
 * @param[in]  start_row Offset for this node
 * @param[in]  stream    CUDA stream
 */
template <typename Index_ = int>
void exchange(
  const raft::handle_t& handle, bool* mask, Index_ N, Index_ start_row, cudaStream_t stream)
{
  const auto& comm = handle.get_comms();
  int my_rank      = comm.get_rank();
  int n_rank       = comm.get_size();

  // Array with the size of the contribution of each worker
  Index_ rows_per_rank           = raft::ceildiv<Index_>(N, n_rank);
  std::vector<size_t> recvcounts = std::vector<size_t>(n_rank, rows_per_rank);
  recvcounts[n_rank - 1]         = N - (n_rank - 1) * rows_per_rank;

  // Array with the displacement of each part
  std::vector<size_t> displs = std::vector<size_t>(n_rank);
  for (int i = 0; i < n_rank; i++)
    displs[i] = i * rows_per_rank;

  // All-gather operation with variable contribution length
  comm.allgatherv<char>(
    (char*)mask + start_row, (char*)mask, recvcounts.data(), displs.data(), stream);
  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");
}

}  // namespace CorePoints
}  // namespace Dbscan
}  // namespace ML
