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

#include "algo.cuh"
#include "pack.h"
#include "precomputed.cuh"

namespace ML {
namespace Dbscan {
namespace VertexDeg {

template <typename Type_f, typename Index_ = int>
void run(const raft::handle_t& handle,
         bool* adj,
         Index_* vd,
         const Type_f* x,
         Type_f eps,
         Index_ N,
         Index_ D,
         int algo,
         Index_ start_vertex_id,
         Index_ batch_size,
         cudaStream_t stream,
         raft::distance::DistanceType metric)
{
  Pack<Type_f, Index_> data = {vd, adj, x, eps, N, D};
  switch (algo) {
    case 0:
      ASSERT(
        false, "Incorrect algo '%d' passed! Naive version of vertexdeg has been removed.", algo);
    case 1:
      Algo::launcher<Type_f, Index_>(handle, data, start_vertex_id, batch_size, stream, metric);
      break;
    case 2:
      Precomputed::launcher<Type_f, Index_>(handle, data, start_vertex_id, batch_size, stream);
      break;
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace VertexDeg
}  // namespace Dbscan
}  // namespace ML
