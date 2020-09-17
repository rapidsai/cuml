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

#include <common/cumlHandle.hpp>
#include "algo.cuh"
#include "naive.cuh"
#include "pack.h"

namespace Dbscan {
namespace AdjGraph {

template <typename Index_ = int>
void run(const raft::handle_t& handle, bool* adj, Index_* vd, Index_* adj_graph,
         Index_ adjnnz, Index_* ex_scan, Index_ N, Index_ minpts,
         bool* core_pts, int algo, Index_ batchSize, cudaStream_t stream) {
  Pack<Index_> data = {vd,      adj,      adj_graph, adjnnz,
                       ex_scan, core_pts, N,         minpts};
  switch (algo) {
    case 0:
      Naive::launcher<Index_>(handle, data, batchSize, stream);
      break;
    case 1:
      Algo::launcher<Index_>(handle, data, batchSize, stream);
      break;
    default:
      ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace AdjGraph
}  // namespace Dbscan
