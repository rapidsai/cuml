/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "pack.h"

#include <raft/core/handle.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace ML {
namespace Dbscan {
namespace AdjGraph {
namespace Algo {

/**
 * @brief Converts a boolean adjacency matrix into CSR format.
 *
 * @tparam[Index_]: indexing arithmetic type
 * @param[in] handle: raft::handle_t
 *
 * @param[in,out] data: A struct containing the adjacency matrix, its number of
 *                      columns, and the vertex degrees.
 *
 * @param[in] batch_size: The number of rows of the adjacency matrix data.adj
 * @param     row_counters: A pre-allocated temporary buffer on the device.
 *            Must be able to contain at least `batch_size` elements.
 * @param[in] stream: CUDA stream
 */
template <typename Index_ = int>
void launcher(const raft::handle_t& handle,
              Pack<Index_> data,
              Index_ batch_size,
              Index_* row_counters,
              cudaStream_t stream)
{
  Index_ num_rows = batch_size;
  Index_ num_cols = data.N;
  bool* adj       = data.adj;  // batch_size x N row-major adjacency matrix

  // Compute the exclusive scan of the vertex degrees
  using namespace thrust;
  device_ptr<Index_> dev_vd      = device_pointer_cast(data.vd);
  device_ptr<Index_> dev_ex_scan = device_pointer_cast(data.ex_scan);
  thrust::exclusive_scan(handle.get_thrust_policy(), dev_vd, dev_vd + batch_size, dev_ex_scan);

  raft::sparse::convert::adj_to_csr(
    handle, adj, data.ex_scan, num_rows, num_cols, row_counters, data.adj_graph);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace Algo
}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML
