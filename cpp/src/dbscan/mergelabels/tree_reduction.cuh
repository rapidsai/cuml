/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runner.cuh"

#include <cuml/common/logger.hpp>

#include <raft/core/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

namespace ML {
namespace Dbscan {
namespace MergeLabels {

/**
 * Merges to label arrays according to a given core point mask
 * @param[in]    handle      raft handle
 * @param[inout] labels      Labels computed by this node
 * @param[in]    labels_temp Buffer to receive labels from another node
 * @param[in]    mask        Boolean core point mask
 * @param[in]    work_buffer Working buffer (for R)
 * @param[in]    m           Working flag
 * @param[in]    N           Number of points in the dataset
 * @param[in]    stream      CUDA stream
 */
template <typename Index_ = int>
void tree_reduction(const raft::handle_t& handle,
                    Index_* labels,
                    Index_* labels_temp,
                    const bool* mask,
                    Index_* work_buffer,
                    bool* m,
                    Index_ N,
                    cudaStream_t stream)
{
  const auto& comm = handle.get_comms();
  int my_rank      = comm.get_rank();
  int n_rank       = comm.get_size();

  int s = 1;
  while (s < n_rank) {
    CUML_LOG_DEBUG("Tree reduction, s=", s);

    // Find out whether the node is a receiver / sender / passive
    bool receiver = my_rank % (2 * s) == 0 && my_rank + s < n_rank;
    bool sender   = my_rank % (2 * s) == s;

    // Use device_send/device_recv for GPU memory transfers (via NCCL)
    if (receiver) {
      CUML_LOG_DEBUG("--> Receive labels (from %d)", my_rank + s);
      comm.device_recv(labels_temp, N, my_rank + s, stream);
    } else if (sender) {
      CUML_LOG_DEBUG("--> Send labels (to %d)", my_rank - s);
      comm.device_send(labels, N, my_rank - s, stream);
    }

    // Sync stream to ensure transfer is complete
    if (receiver || sender) { RAFT_CUDA_TRY(cudaStreamSynchronize(stream)); }

    if (receiver) {
      CUML_LOG_DEBUG("--> Merge labels");
      raft::common::nvtx::range fun_scope("Trace::Dbscan::MergeLabels");
      MergeLabels::run<Index_>(handle, labels, labels_temp, mask, work_buffer, m, N, stream);
    }

    s *= 2;
  }
}

}  // namespace MergeLabels
}  // namespace Dbscan
}  // namespace ML
