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

#include "mgrp_accessor.cuh"

#include <raft/util/device_atomics.cuh>

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace AdjGraph {
namespace Csr {

// Threads per block in adj_to_csr_kernel.
static const constexpr int adj_to_csr_tpb = 512;

/**
 * The implementation is based on
 * https://github.com/rapidsai/raft/blob/branch-23.06/cpp/include/raft/sparse/convert/detail/adj_to_csr.cuh
 */
template <typename index_t>
__global__ void __launch_bounds__(adj_to_csr_tpb) multi_groups_adj_to_csr_kernel(
  const bool* mgrp_adj,  // row-major adjacency matrix
  const std::size_t* adj_offset,
  const index_t* mgrp_row_ind,   // precomputed row indices
  index_t num_groups,            // # groups of adj
  const index_t* mgrp_num_rows,  // # rows of adj
  const index_t* row_start_ids,
  const index_t* adj_col_stride,  // stride of adj
  index_t* mgrp_row_counters,     // pre-allocated (zeroed) atomic counters
  index_t* out_col_ind            // output column indices
)
{
  index_t group_id = blockIdx.z * blockDim.z + threadIdx.z;
  if (group_id >= num_groups) return;

  const int chunk_size = 16;
  typedef raft::TxN_t<bool, chunk_size> chunk_bool;

  index_t num_rows                 = mgrp_num_rows[group_id];
  index_t num_cols                 = adj_col_stride[group_id];
  const bool* adj                  = mgrp_adj + adj_offset[group_id];
  const index_t out_col_ind_offset = row_start_ids[group_id];
  const index_t* row_ind           = mgrp_row_ind + out_col_ind_offset;
  index_t* row_counters            = mgrp_row_counters + out_col_ind_offset;

  for (index_t i = blockIdx.y; i < num_rows; i += gridDim.y) {
    // Load row information
    index_t row_base   = row_ind[i];
    index_t* row_count = row_counters + i;
    const bool* row    = adj + i * num_cols;

    // Peeling: process the first j0 elements that are not aligned to a chunk_size-byte
    // boundary.
    index_t j0 = (chunk_size - (((uintptr_t)(const void*)row) % chunk_size)) % chunk_size;
    j0         = min(j0, num_cols);
    if (threadIdx.x < j0 && blockIdx.x == 0) {
      if (row[threadIdx.x]) {
        out_col_ind[row_base + atomicIncWarp(row_count)] = threadIdx.x + out_col_ind_offset;
      }
    }

    // Process the rest of the row in chunk_size byte chunks starting at j0.
    // This is a grid-stride loop.
    index_t j = j0 + chunk_size * (blockIdx.x * blockDim.x + threadIdx.x);
    for (; j + chunk_size - 1 < num_cols; j += chunk_size * (blockDim.x * gridDim.x)) {
      chunk_bool chunk;
      chunk.load(row, j);
      for (int k = 0; k < chunk_size; ++k) {
        if (chunk.val.data[k]) {
          out_col_ind[row_base + atomicIncWarp(row_count)] = j + k + out_col_ind_offset;
        }
      }
    }

    // Remainder: process the last j1 bools in the row individually.
    index_t j1 = (num_cols - j0) % chunk_size;
    if (threadIdx.x < j1 && blockIdx.x == 0) {
      int j = num_cols - j1 + threadIdx.x;
      if (row[j]) { out_col_ind[row_base + atomicIncWarp(row_count)] = j + out_col_ind_offset; }
    }
  }
}

template <typename index_t = int>
void multi_groups_adj_to_csr(raft::device_resources const& handle,
                             Metadata::AdjGraphAccessor<bool, index_t>& adj_ac,
                             const index_t* row_ind,
                             index_t* row_counter,
                             index_t* out_col_ind,
                             cudaStream_t stream)
{
  index_t n_groups = adj_ac.n_groups;
  index_t num_rows = adj_ac.n_points;
  index_t num_cols = adj_ac.max_nbr;
  // Check inputs and return early if possible.
  if (num_rows == 0 || num_cols == 0) { return; }
  RAFT_EXPECTS(row_counter != nullptr, "adj_to_csr: row_counter workspace may not be null.");

  // Zero-fill a temporary vector that is be used by the kernel to keep track of
  // the number of entries added to a row.
  RAFT_CUDA_TRY(cudaMemsetAsync(row_counter, 0, num_rows * sizeof(index_t), stream));

  // Split the grid in the row direction (since each row can be processed
  // independently). If the maximum number of active blocks (num_sms *
  // occupancy) exceeds the number of rows, assign multiple blocks to a single
  // row.
  int dev_id, sm_count, blocks_per_sm;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, multi_groups_adj_to_csr_kernel<index_t>, adj_to_csr_tpb, 0);

  index_t max_active_blocks = sm_count * blocks_per_sm;
  index_t blocks_per_row    = raft::ceildiv(max_active_blocks, num_rows);
  index_t grid_rows         = raft::ceildiv(max_active_blocks, blocks_per_row);
  index_t grid_groups       = n_groups;
  dim3 block(adj_to_csr_tpb, 1, 1);
  dim3 grid(blocks_per_row, grid_rows, grid_groups);

  const bool* adj                 = adj_ac.adj;
  const std::size_t* adj_offset   = adj_ac.adj_group_offset;
  const index_t* dev_n_rows       = adj_ac.n_rows_ptr;
  const index_t* dev_row_startids = adj_ac.row_start_ids;
  const index_t* adj_col_stride   = adj_ac.adj_col_stride;

  multi_groups_adj_to_csr_kernel<index_t><<<grid, block, 0, stream>>>(adj,
                                                                      adj_offset,
                                                                      row_ind,
                                                                      n_groups,
                                                                      dev_n_rows,
                                                                      dev_row_startids,
                                                                      adj_col_stride,
                                                                      row_counter,
                                                                      out_col_ind);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace Csr
}  // namespace AdjGraph
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML