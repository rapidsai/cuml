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

#include "runner.cuh"

#include <raft/core/nvtx.hpp>

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/logger.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ML {
namespace Dbscan {

template <typename Index_ = int>
size_t compute_batch_size(size_t& estimated_memory,
                          Index_ n_rows,
                          Index_ n_owned_rows,
                          size_t max_mbytes_per_batch = 0,
                          Index_ neigh_per_row        = 0)
{
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
  size_t est_mem_per_row = n_rows * sizeof(bool) + (neigh_per_row + 2) * sizeof(Index_);
  /* Memory needed regardless of the batch size:
   *  - Temporary labels: n_rows (Index_)
   *  - Core point mask: n_rows (bool)
   */
  size_t est_mem_fixed = n_rows * (sizeof(Index_) + sizeof(bool));
  // The rest will be so small that it should fit into what we have left over
  // from the over-estimation of the sparse adjacency matrix

  // Batch size determined based on available memory
  ASSERT(est_mem_per_row > 0, "Estimated memory per row is 0 for DBSCAN");
  size_t batch_size = (max_mbytes_per_batch * 1000000 - est_mem_fixed) / est_mem_per_row;

  // Limit batch size to number of owned rows
  batch_size = std::min((size_t)n_owned_rows, batch_size);

  // To avoid overflow, we need: batch_size <= MAX_LABEL / n_rows (floor div)
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  if (batch_size > static_cast<std::size_t>(MAX_LABEL / n_rows)) {
    Index_ new_batch_size = MAX_LABEL / n_rows;
    CUML_LOG_WARN(
      "Batch size limited by the chosen integer type (%d bytes). %d -> %d. "
      "Using the larger integer type might result in better performance",
      (int)sizeof(Index_),
      (int)batch_size,
      (int)new_batch_size);
    batch_size = new_batch_size;
  }

  // Warn when a smaller index type could be used
  if ((sizeof(Index_) > sizeof(int)) &&
      (batch_size < std::numeric_limits<int>::max() / static_cast<std::size_t>(n_rows))) {
    CUML_LOG_WARN(
      "You are using an index type of size (%d bytes) but a smaller index "
      "type (%d bytes) would be sufficient. Using the smaller integer type "
      "might result in better performance.",
      (int)sizeof(Index_),
      (int)sizeof(int));
  }

  estimated_memory = batch_size * est_mem_per_row + est_mem_fixed;
  return batch_size;
}

template <typename T, typename Index_ = int, bool opg = false>
void dbscanFitImpl(const raft::handle_t& handle,
                   T* input,
                   Index_ n_rows,
                   Index_ n_cols,
                   T eps,
                   Index_ min_pts,
                   raft::distance::DistanceType metric,
                   Index_* labels,
                   Index_* core_sample_indices,
                   size_t max_mbytes_per_batch,
                   cudaStream_t stream,
                   int verbosity)
{
  raft::common::nvtx::range fun_scope("ML::Dbscan::Fit");
  ML::Logger::get().setLevel(verbosity);
  // XXX: for algo_vd and algo_adj, 0 (naive) is no longer an option and has
  // been removed.
  int algo_vd  = (metric == raft::distance::Precomputed) ? 2 : 1;
  int algo_adj = 1;
  int algo_ccl = 2;

  int my_rank{0};
  int n_rank{1};
  Index_ start_row{0};
  Index_ n_owned_rows{n_rows};

  ASSERT(n_rows > 0, "No rows in the input array. DBSCAN cannot be fitted!");

  if (opg) {
    const auto& comm     = handle.get_comms();
    my_rank              = comm.get_rank();
    n_rank               = comm.get_size();
    Index_ rows_per_rank = raft::ceildiv<Index_>(n_rows, n_rank);
    start_row            = my_rank * rows_per_rank;
    Index_ end_row       = min((my_rank + 1) * rows_per_rank, n_rows);
    n_owned_rows         = max(Index_(0), end_row - start_row);
    // Note: it is possible for a node to have no work in theory. It won't
    // happen in practice (because n_rows is much greater than n_rank)
  }

  CUML_LOG_DEBUG("#%d owns %ld rows", (int)my_rank, (unsigned long)n_owned_rows);

  // Estimate available memory per batch
  // Note: we can't rely on the reported free memory.
  if (max_mbytes_per_batch == 0) {
    // Query memory information to get the total memory on the device
    size_t free_memory, total_memory;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));

    // X can either be a feature matrix or distance matrix
    size_t dataset_memory = (metric == raft::distance::Precomputed)
                              ? ((size_t)n_rows * (size_t)n_rows * sizeof(T))
                              : ((size_t)n_rows * (size_t)n_cols * sizeof(T));

    // The estimate is: 80% * total - dataset
    max_mbytes_per_batch = (80 * total_memory / 100 - dataset_memory) / 1e6;

    CUML_LOG_DEBUG("Dataset memory: %ld MB", (unsigned long long)(dataset_memory / 1e6));

    CUML_LOG_DEBUG("Estimated available memory: %ld / %ld MB",
                   (unsigned long long)max_mbytes_per_batch,
                   (unsigned long long)(total_memory / 1e6));
  }

  size_t estimated_memory;
  size_t batch_size =
    compute_batch_size<Index_>(estimated_memory, n_rows, n_owned_rows, max_mbytes_per_batch);

  CUML_LOG_DEBUG("Running batched training (batch size: %ld, estimated: %lf MB)",
                 (unsigned long)batch_size,
                 (double)estimated_memory * 1e-6);

  size_t workspaceSize = Dbscan::run<T, Index_, opg>(handle,
                                                     input,
                                                     n_rows,
                                                     n_cols,
                                                     start_row,
                                                     n_owned_rows,
                                                     eps,
                                                     min_pts,
                                                     labels,
                                                     core_sample_indices,
                                                     algo_vd,
                                                     algo_adj,
                                                     algo_ccl,
                                                     NULL,
                                                     batch_size,
                                                     stream,
                                                     metric);

  CUML_LOG_DEBUG("Workspace size: %lf MB", (double)workspaceSize * 1e-6);

  rmm::device_uvector<char> workspace(workspaceSize, stream);
  Dbscan::run<T, Index_, opg>(handle,
                              input,
                              n_rows,
                              n_cols,
                              start_row,
                              n_owned_rows,
                              eps,
                              min_pts,
                              labels,
                              core_sample_indices,
                              algo_vd,
                              algo_adj,
                              algo_ccl,
                              workspace.data(),
                              batch_size,
                              stream,
                              metric);
}

template <typename Index_ = int>
void mgrp_dbscan_scheduler(size_t max_mbytes_per_dispatch,
                           Index_ n_groups,
                           Index_* n_rows_ptr,
                           Index_* neigh_rows_ptr,
                           std::vector<std::vector<Index_>>& grouped_row_ids)
{
  std::vector<Index_> group;
  size_t accum_est_mem = 0;
  for (Index_ i = 0; i < n_groups; ++i) {
    Index_ n_rows        = n_rows_ptr[i];
    Index_ neigh_per_row = neigh_rows_ptr[i];

    /* Memory needed per group:
     *  - Dense adj matrix: n_rows * n_rows (bool)
     *  - Sparse adj matrix: n_rows * neigh_per_row (Index_)
     *  - Vertex degrees: n_rows (Index_)
     *  - Ex scan: n_rows (Index_)
     *  - Core point mask: n_rows (bool)
     *  - Temporary buffer: n_rows (Index_)
     */
    size_t est_mem_per_row = (n_rows + 1) * sizeof(bool) + (neigh_per_row + 3) * sizeof(Index_);
    size_t est_mem         = est_mem_per_row * n_rows;
    // The rest will be so small that it should fit into what we have left over
    // from the over-estimation of the sparse adjacency matrix

    accum_est_mem += est_mem;
    if (accum_est_mem >= max_mbytes_per_dispatch * 1000000) {
      if (!group.empty()) {
        grouped_row_ids.emplace_back(std::move(group));
        group.clear();
        accum_est_mem = 0;
      }
    }
    group.emplace_back(i);
  }
  grouped_row_ids.emplace_back(group);
  return;
}

template <typename T, typename Index_ = int, bool opg = false>
void dbscanFitImpl(const raft::handle_t& handle,
                   T* input,
                   Index_ n_groups,
                   Index_* n_rows_ptr,
                   Index_ n_cols,
                   const T* eps_ptr,
                   const Index_* min_pts_ptr,
                   raft::distance::DistanceType metric,
                   Index_* labels,
                   Index_* core_sample_indices,
                   size_t max_mbytes_per_dispatch,
                   cudaStream_t stream,
                   int verbosity,
                   void* custom_workspace        = nullptr,
                   size_t* custom_workspace_size = nullptr)
{
  raft::common::nvtx::range fun_scope("ML::Dbscan::Fit");
  ML::Logger::get().setLevel(verbosity);
  /// NOTE: for algo_vd, 0 (naive) and 2 (precomputed) are not supported.
  ///       for algo_adj, 0 (naive) is not supported.
  int algo_vd  = 1;
  int algo_adj = 1;
  int algo_ccl = 2;

  Index_ accum_rows = thrust::reduce(thrust::host, n_rows_ptr, n_rows_ptr + n_groups);
  ASSERT(accum_rows > 0, "No rows in the input array. DBSCAN cannot be fitted!");

  // Estimate available memory per dispatch
  // Note: we can't rely on the reported free memory.
  if (max_mbytes_per_dispatch == 0) {
    // Query memory information to get the total memory on the device
    size_t free_memory, total_memory;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));

    size_t dataset_memory = ((size_t)accum_rows * (size_t)n_cols * sizeof(T));

    // The estimate is: 80% * total - dataset
    max_mbytes_per_dispatch = (80 * total_memory / 100 - dataset_memory) / 1e6;

    CUML_LOG_DEBUG("Dataset memory: %ld MB", (unsigned long long)(dataset_memory / 1e6));

    CUML_LOG_DEBUG("Estimated available memory: %ld / %ld MB",
                   (unsigned long long)max_mbytes_per_dispatch,
                   (unsigned long long)(total_memory / 1e6));
  }

  CUML_LOG_DEBUG("Scheduling input groups");
  std::vector<std::vector<Index_>> grouped_row_ids;
  mgrp_dbscan_scheduler<Index_>(
    max_mbytes_per_dispatch, n_groups, n_rows_ptr, n_rows_ptr, grouped_row_ids);
  CUML_LOG_DEBUG("Divide input into %lu groups", grouped_row_ids.size());
  if (verbosity >= CUML_LEVEL_DEBUG) {
    std::cout << "row_groups: [ ";
    for (auto group : grouped_row_ids) {
      std::cout << "[";
      for (auto it : group) {
        std::cout << n_rows_ptr[it] << " ";
      }
      std::cout << "];";
    }
    std::cout << std::endl;
  }

  std::vector<Index_> pfx_rows(n_groups, 0);
  thrust::exclusive_scan(thrust::host, n_rows_ptr, n_rows_ptr + n_groups, pfx_rows.data());

  // Get the maximum of workspace size
  size_t max_workspace_size = 0;
  for (size_t i = 0; i < grouped_row_ids.size(); ++i) {
    std::vector<Index_> dispatch_group = grouped_row_ids[i];
    Index_ dispatch_n_groups           = dispatch_group.size();
    Index_ first_row_id                = dispatch_group[0];
    Index_* dispatch_n_rows_ptr        = n_rows_ptr + first_row_id;
    if (dispatch_n_groups != 1) {
      size_t workspaceSize = Dbscan::run<T, Index_, opg>(handle,
                                                         nullptr,
                                                         dispatch_n_groups,
                                                         dispatch_n_rows_ptr,
                                                         n_cols,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         algo_vd,
                                                         algo_adj,
                                                         algo_ccl,
                                                         NULL,
                                                         stream,
                                                         metric);
      max_workspace_size =
        (workspaceSize > max_workspace_size) ? workspaceSize : max_workspace_size;
    }
  }
  CUML_LOG_DEBUG("Workspace size: %lf MB", (double)max_workspace_size * 1e-6);

  if (custom_workspace == nullptr && custom_workspace_size != nullptr) {
    *custom_workspace_size = max_workspace_size;
    return;
  }

  using cuda_async_mr       = rmm::mr::cuda_async_memory_resource;
  void* work_buffer         = nullptr;
  void* mr                  = nullptr;
  bool has_custom_workspace = custom_workspace != nullptr;
  if (has_custom_workspace) {
    work_buffer = custom_workspace;
  } else {
    raft::common::nvtx::push_range("Trace::Dbscan::MemMalloc");
    /*
      rmm::mr::cuda_memory_resource cuda_mr;
      // Construct a resource that uses a coalescing best-fit pool allocator
      rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
      // Updates the current device resource pointer to `pool_mr`
      rmm::mr::set_current_device_resource(&pool_mr);
      // Points to `pool_mr`
      mr = reinterpret_cast<void*>(rmm::mr::get_current_device_resource());
      work_buffer = mr->allocate(max_workspace_size, stream);
     */

    cuda_async_mr* cuda_mr = new cuda_async_mr{max_workspace_size};
    mr                     = reinterpret_cast<void*>(cuda_mr);
    work_buffer            = cuda_mr->allocate(max_workspace_size, stream);
    raft::common::nvtx::pop_range();
  }

  CUML_LOG_DEBUG("work_buffer: %p with size: %lu", work_buffer, max_workspace_size);
  for (size_t i = 0; i < grouped_row_ids.size(); ++i) {
    std::vector<Index_> dispatch_group = grouped_row_ids[i];
    Index_ dispatch_n_groups           = dispatch_group.size();
    Index_ first_row_id                = dispatch_group[0];
    T* dispatch_input                  = input + n_cols * pfx_rows[first_row_id];
    Index_* dispatch_n_rows_ptr        = n_rows_ptr + first_row_id;
    const T* dispatch_eps_ptr          = eps_ptr + first_row_id;
    const Index_* dispatch_min_pts_ptr = min_pts_ptr + first_row_id;
    Index_* dispatch_labels            = labels + pfx_rows[first_row_id];
    Index_* dispatch_core_sample_indices =
      (core_sample_indices == nullptr) ? nullptr : core_sample_indices + pfx_rows[first_row_id];

    if (dispatch_n_groups == 1) {
      CUML_LOG_DEBUG("Running original DBSCAN for single group");
      Dbscan::dbscanFitImpl<T, Index_, opg>(handle,
                                            dispatch_input,
                                            dispatch_n_rows_ptr[0],
                                            n_cols,
                                            dispatch_eps_ptr[0],
                                            dispatch_min_pts_ptr[0],
                                            metric,
                                            dispatch_labels,
                                            dispatch_core_sample_indices,
                                            max_mbytes_per_dispatch,
                                            stream,
                                            verbosity);
    } else {
      Dbscan::run<T, Index_, opg>(handle,
                                  dispatch_input,
                                  dispatch_n_groups,
                                  dispatch_n_rows_ptr,
                                  n_cols,
                                  dispatch_eps_ptr,
                                  dispatch_min_pts_ptr,
                                  dispatch_labels,
                                  dispatch_core_sample_indices,
                                  algo_vd,
                                  algo_adj,
                                  algo_ccl,
                                  work_buffer,
                                  stream,
                                  metric);
    }
  }
  if (!has_custom_workspace) {
    // reinterpret_cast<rmm::mr::device_memory_resource*>(mr)->deallocate(
    //  work_buffer, max_workspace_size, stream);
    reinterpret_cast<cuda_async_mr*>(mr)->deallocate(work_buffer, max_workspace_size, stream);
  }
}

}  // namespace Dbscan
}  // namespace ML
