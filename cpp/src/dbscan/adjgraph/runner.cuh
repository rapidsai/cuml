/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "algo.cuh"
#include "pack.h"

namespace ML {
namespace Dbscan {
namespace AdjGraph {

template <typename Index_ = int>
void run(const raft::handle_t& handle,
         bool* adj,
         Index_* vd,
         Index_* adj_graph,
         Index_ adjnnz,
         Index_* ex_scan,
         Index_ N,
         int algo,
         Index_ batch_size,
         Index_* row_counters,
         cudaStream_t stream)
{
  Pack<Index_> data = {vd, adj, adj_graph, adjnnz, ex_scan, N};
  switch (algo) {
    case 0:
      ASSERT(
        false, "Incorrect algo '%d' passed! Naive version of adjgraph has been removed.", algo);
    case 1: Algo::launcher<Index_>(handle, data, batch_size, row_counters, stream); break;
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML
