/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include "algo.h"
#include "naive.h"
#include "pack.h"

namespace Dbscan {
namespace AdjGraph {

template <typename Type, typename Index_ = int>
void run(const ML::cumlHandle_impl& handle, bool* adj, int* vd, Type* adj_graph,
         Type adjnnz, Type* ex_scan, Index_ N, Type minpts, bool* core_pts,
         int algo, Index_ batchSize, cudaStream_t stream) {
  Pack<Type, Index_> data = {vd,      adj,      adj_graph, adjnnz,
                             ex_scan, core_pts, N,         minpts};
  switch (algo) {
    case 0:
      Naive::launcher<Type, Index_>(handle, data, batchSize, stream);
      break;
    case 1:
      Algo::launcher<Type, Index_>(handle, data, batchSize, stream);
      break;
    default:
      ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace AdjGraph
}  // namespace Dbscan
