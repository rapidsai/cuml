/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <common/allocatorAdapter.hpp>
#include <common/cumlHandle.hpp>
#include <raft/cuda_utils.cuh>
#include "../common.cuh"
#include "pack.h"

#include <sparse/csr.cuh>

using namespace thrust;

namespace Dbscan {
namespace AdjGraph {
namespace Algo {

using namespace MLCommon;

static const int TPB_X = 256;

/**
 * Takes vertex degree array (vd) and CSR row_ind array (ex_scan) to produce the
 * CSR row_ind_ptr array (adj_graph) and filters into a core_pts array based on min_pts.
 */
template <typename Index_ = int>
void launcher(const raft::handle_t &handle, Pack<Index_> data, Index_ batchSize,
              cudaStream_t stream) {
  device_ptr<Index_> dev_vd = device_pointer_cast(data.vd);
  device_ptr<Index_> dev_ex_scan = device_pointer_cast(data.ex_scan);

  ML::thrustAllocatorAdapter alloc(handle.get_device_allocator(), stream);
  exclusive_scan(thrust::cuda::par(alloc).on(stream), dev_vd,
                 dev_vd + batchSize, dev_ex_scan);

  bool *core_pts = data.core_pts;
  int minPts = data.minPts;
  Index_ *vd = data.vd;

  MLCommon::Sparse::csr_adj_graph_batched<Index_, TPB_X>(
    data.ex_scan, data.N, data.adjnnz, batchSize, data.adj, data.adj_graph,
    stream,
    [core_pts, minPts, vd] __device__(Index_ row, Index_ start_idx,
                                      Index_ stop_idx) {
      // fuse the operation of core points construction
      core_pts[row] = (vd[row] >= minPts);
    });

  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Algo
}  // namespace AdjGraph
}  // namespace Dbscan
