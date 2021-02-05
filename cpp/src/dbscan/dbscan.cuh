/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <common/nvtx.hpp>
#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include "runner.cuh"

#include <algorithm>

namespace ML {

// Default max mem set to a reasonable value for a 16gb card.
static const size_t DEFAULT_MAX_MEM_MBYTES = 13e3;

template <typename Index_ = int>
Index_ compute_batch_size(size_t &estimated_memory, Index_ n_rows,
                          Index_ n_owned_rows, size_t max_mbytes_per_batch = 0,
                          Index_ neigh_per_row = 0) {
  // In real applications, it's unlikely that the sparse adjacency matrix
  // comes even close to the worst-case memory usage, because if epsilon
  // is so large that all points are connected to 10% or even more of other
  // points, the clusters would probably not be interesting/relevant anymore
  ///@todo: expose `neigh_per_row` to the user

  if (neigh_per_row <= 0) neigh_per_row = n_rows;

  /* Memory needed per batch row:
   *  - Dense adj matrix: n_rows (bool)
   *  - Sparse adj matrix: neigh_per_row (Index_)
   *  - Vertex degrees: 1 (Index_)
   *  - Ex scan: 1 (Index_)
   */
  size_t est_mem_per_row =
    n_rows * sizeof(bool) + (neigh_per_row + 2) * sizeof(Index_);
  /* Memory needed regardless of the batch size:
   *  - Temporary labels: n_rows (Index_)
   *  - Core point mask: n_rows (bool)
   */
  size_t est_mem_fixed = n_rows * (sizeof(Index_) + sizeof(bool));
  // The rest will be so small that it should fit into what we have left over
  // from the over-estimation of the sparse adjacency matrix

  if (max_mbytes_per_batch <= 0) {
    /* using default here as in decision tree, waiting for mem info from device allocator
    size_t total_mem;
	  CUDA_CHECK(cudaMemGetInfo(&max_mbytes_per_batch, &total_mem));
    */
    max_mbytes_per_batch = DEFAULT_MAX_MEM_MBYTES;
  }

  // Batch size determined based on available memory
  Index_ batch_size =
    (max_mbytes_per_batch * 1000000 - est_mem_fixed) / est_mem_per_row;

  // Limit batch size to number of owned rows
  batch_size = std::min(n_owned_rows, batch_size);

  // To avoid overflow, we need: batch_size <= MAX_LABEL / n_rows (floor div)
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  if (batch_size > MAX_LABEL / n_rows) {
    Index_ new_batch_size = MAX_LABEL / n_rows;
    CUML_LOG_WARN(
      "Batch size limited by the chosen integer type (%d bytes). %d -> %d. "
      "Using the larger integer type might result in better performance",
      (int)sizeof(Index_), (int)batch_size, (int)new_batch_size);
    batch_size = new_batch_size;
  }

  // Warn when a smaller index type could be used
  if (sizeof(Index_) > sizeof(int) &&
      batch_size < std::numeric_limits<int>::max() / n_rows) {
    CUML_LOG_WARN(
      "You are using an index type of size (%d bytes) but a smaller index "
      "type (%d bytes) would be sufficient. Using the smaller integer type "
      "might result in better performance.",
      (int)sizeof(Index_), (int)sizeof(int));
  }

  estimated_memory = batch_size * est_mem_per_row + est_mem_fixed;
  return batch_size;
}

template <typename T, typename Index_ = int, bool opg = false>
void dbscanFitImpl(const raft::handle_t &handle, T *input, Index_ n_rows,
                   Index_ n_cols, T eps, Index_ min_pts, Index_ *labels,
                   Index_ *core_sample_indices, size_t max_mbytes_per_batch,
                   cudaStream_t stream, int verbosity) {
  ML::PUSH_RANGE("ML::Dbscan::Fit");
  ML::Logger::get().setLevel(verbosity);
  int algo_vd = 1;
  int algo_adj = 1;
  int algo_ccl = 2;

  int my_rank, n_rank;
  Index_ start_row, n_owned_rows;
  if (opg) {
    const auto &comm = handle.get_comms();
    my_rank = comm.get_rank();
    n_rank = comm.get_size();
    Index_ rows_per_rank = raft::ceildiv<Index_>(n_rows, n_rank);
    start_row = my_rank * rows_per_rank;
    Index_ end_row = min((my_rank + 1) * rows_per_rank, n_rows);
    n_owned_rows = max(Index_(0), end_row - start_row);
    // Note: it is possible for a node to have no work in theory. It won't
    // happen in practice (because n_rows is much greater than n_rank)
  } else {
    my_rank = 0;
    n_rank = 1;
    n_owned_rows = n_rows;
  }

  CUML_LOG_DEBUG("#%d owns %ld rows", (int)my_rank,
                 (unsigned long)n_owned_rows);

  /// TODO: Query device for remaining memory
  size_t estimated_memory;
  Index_ batch_size = compute_batch_size<Index_>(
    estimated_memory, n_rows, n_owned_rows, max_mbytes_per_batch);

  CUML_LOG_DEBUG(
    "Running batched training (batch size: %ld, estimated: %lf MB)",
    (unsigned long)batch_size, (double)estimated_memory * 1e-6);

  size_t workspaceSize = Dbscan::run<T, Index_, opg>(
    handle, input, n_rows, n_cols, start_row, n_owned_rows, eps, min_pts,
    labels, core_sample_indices, algo_vd, algo_adj, algo_ccl, NULL, batch_size,
    stream);

  CUML_LOG_DEBUG("Workspace size: %lf MB", (double)workspaceSize * 1e-6);

  MLCommon::device_buffer<char> workspace(handle.get_device_allocator(), stream,
                                          workspaceSize);
  Dbscan::run<T, Index_, opg>(handle, input, n_rows, n_cols, start_row,
                              n_owned_rows, eps, min_pts, labels,
                              core_sample_indices, algo_vd, algo_adj, algo_ccl,
                              workspace.data(), batch_size, stream);
  ML::POP_RANGE();
}

}  // namespace ML
