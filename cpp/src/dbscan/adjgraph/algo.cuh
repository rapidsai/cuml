/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "../common.cuh"
#include "pack.h"

#include <raft/cuda_utils.cuh>
#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <raft/vectorized.cuh>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace Dbscan {
namespace AdjGraph {
namespace Algo {

/**
 * @brief Convert a boolean adjacency matrix into CSR format.
 *
 * The adj_to_csr kernel converts a boolean adjacency matrix into CSR format.
 * High performance comes at the cost of non-deterministic output: the column
 * indices are not guaranteed to be stored in order.
 *
 * The kernel has been optimized to handle matrices that are non-square, for
 * instance subsets of a full adjacency matrix. In practice, these matrices can
 * be very wide and not very tall. In principle, each row is assigned to one
 * block. If there are more SMs than rows, multiple blocks operate on a single
 * row. To enable cooperation between these blocks, each row is provided a
 * counter where the current output index can be cooperatively (atomically)
 * incremented. As a result, the order of the output indices is not guaranteed
 * to be in order.
 *
 * @param[in] adj: a num_rows x num_cols boolean matrix in contiguous row-major
 *                 format.
 *
 * @param[in] row_ind: an array of length num_rows that indicates at which index
 *                     a row starts in out_col_ind. Equivalently, it is the
 *                     exclusive scan of the number of non-zeros in each row of
 *                     `adj`.
 *
 * @param[in] num_rows: number of rows of adj.
 * @param[in] num_cols: number of columns of adj.
 *
 * @param[in,out] row_counters: a temporary zero-initialized array of length num_rows.
 *
 * @param[out] out_col_ind: an array containing the column indices of the
 *                          non-zero values in `adj`. Size should be at least
 *                          the number of non-zeros in `adj`.
 */
template <typename index_t>
__global__ void adj_to_csr(const bool* adj,         // row-major adjacency matrix
                           const index_t* row_ind,  // precomputed row indices
                           index_t num_rows,        // # rows of adj
                           index_t num_cols,        // # cols of adj
                           index_t* row_counters,   // pre-allocated (zeroed) atomic counters
                           index_t* out_col_ind     // output column indices
)
{
  typedef raft::TxN_t<bool, 16> bool16;

  for (index_t i = blockIdx.y; i < num_rows; i += gridDim.y) {
    // Load row information
    index_t row_base   = row_ind[i];
    index_t* row_count = row_counters + i;
    const bool* row    = adj + i * num_cols;

    // Peeling: process the first j0 elements that are not aligned to a 16-byte
    // boundary.
    index_t j0 = (16 - (((uintptr_t)(const void*)row) % 16)) % 16;
    j0         = min(j0, num_cols);
    if (threadIdx.x < j0 && blockIdx.x == 0) {
      if (row[threadIdx.x]) { out_col_ind[row_base + atomicIncWarp(row_count)] = threadIdx.x; }
    }

    // Process the rest of the row in 16 byte chunks starting at j0.
    // This is a grid-stride loop.
    index_t j = j0 + 16 * (blockIdx.x * blockDim.x + threadIdx.x);
    for (; j + 15 < num_cols; j += 16 * (blockDim.x * gridDim.x)) {
      bool16 chunk;
      chunk.load(row, j);

      for (int k = 0; k < 16; ++k) {
        if (chunk.val.data[k]) { out_col_ind[row_base + atomicIncWarp(row_count)] = j + k; }
      }
    }

    // Remainder: process the last j1 bools in the row individually.
    index_t j1 = (num_cols - j0) % 16;
    if (threadIdx.x < j1 && blockIdx.x == 0) {
      int j = num_cols - j1 + threadIdx.x;
      if (row[j]) { out_col_ind[row_base + atomicIncWarp(row_count)] = j; }
    }
  }
}

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

  // Zero-fill a temporary vector that can be used by the adj_to_csr kernel to
  // keep track of the number of entries added to a row.
  RAFT_CUDA_TRY(cudaMemsetAsync(row_counters, 0, batch_size * sizeof(Index_), stream));

  // Split the grid in the row direction (since each row can be processed
  // independently). If the maximum number of active blocks (num_sms *
  // occupancy) exceeds the number of rows, assign multiple blocks to a single
  // row.
  int threads_per_block = 1024;
  int dev_id, sm_count, blocks_per_sm;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, adj_to_csr<Index_>, threads_per_block, 0);

  Index_ max_active_blocks = sm_count * blocks_per_sm;
  Index_ blocks_per_row    = raft::ceildiv(max_active_blocks, num_rows);
  Index_ grid_rows         = raft::ceildiv(max_active_blocks, blocks_per_row);
  dim3 block(threads_per_block, 1);
  dim3 grid(blocks_per_row, grid_rows);

  adj_to_csr<Index_><<<grid, block, 0, stream>>>(
    adj, data.ex_scan, num_rows, num_cols, row_counters, data.adj_graph);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace Algo
}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML
