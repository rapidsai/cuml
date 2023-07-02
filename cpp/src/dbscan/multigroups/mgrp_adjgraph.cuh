/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
#include "mgrp_csr.cuh"

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace AdjGraph {

template <typename Index_ = int>
void run(const raft::handle_t& handle,
         Metadata::AdjGraphAccessor<bool, Index_>& adj_ac,
         const Metadata::VertexDegAccessor<Index_, Index_>& vd_ac,
         Index_* adj_graph,
         Index_ adjnnz,
         Index_* ex_scan,
         Index_* row_counters,
         cudaStream_t stream)
{
  Index_* vd      = vd_ac.vd;
  Index_ n_points = vd_ac.n_points;

  // Compute the exclusive scan of the vertex degrees
  using namespace thrust;
  device_ptr<Index_> dev_vd      = device_pointer_cast(vd);
  device_ptr<Index_> dev_ex_scan = device_pointer_cast(ex_scan);
  thrust::exclusive_scan(handle.get_thrust_policy(), dev_vd, dev_vd + n_points, dev_ex_scan);

  Csr::multi_groups_adj_to_csr(handle, adj_ac, ex_scan, row_counters, adj_graph, stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace AdjGraph
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML