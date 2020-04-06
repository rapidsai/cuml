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

#include <math.h>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <distance/epsilon_neighborhood.cuh>
#include "cuda_runtime.h"
#include "cuda_utils.h"

#include "pack.h"

namespace Dbscan {
namespace VertexDeg {
namespace Algo {

/**
 * Calculates the vertex degree array and the epsilon neighborhood adjacency matrix for the batch.
 */
template <typename value_t, typename index_t = int>
void launcher(const ML::cumlHandle_impl &handle, Pack<value_t, index_t> data,
              index_t startVertexId, index_t batchSize, cudaStream_t stream) {
  data.resetArray(stream, batchSize + 1);

  ASSERT(sizeof(index_t) == 4 || sizeof(index_t) == 8,
         "index_t should be 4 or 8 bytes");

  index_t m = data.N;
  index_t n = min(data.N - startVertexId, batchSize);
  index_t k = data.D;
  value_t eps2 = data.eps * data.eps;

  MLCommon::Distance::epsUnexpL2SqNeighborhood<value_t, index_t>(
    data.adj, data.vd, data.x, data.x + startVertexId * k, m, n, k, eps2,
    stream);
}

}  // namespace Algo
}  // end namespace VertexDeg
};  // end namespace Dbscan
