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
#include <raft/cuda_utils.cuh>
#include "pack.h"

namespace Dbscan {
namespace VertexDeg {
namespace Naive {

using namespace MLCommon;

/** number of threads in a CTA along X dim */
static const int TPB_X = 32;
/** number of threads in a CTA along Y dim */
static const int TPB_Y = 8;

/**
 * @brief Naive distance matrix evaluation and epsilon neighborhood construction
 * @param data input struct containing vertex degree computation params
 * @param startVertexId which vertex to begin the computations from
 * @param batchSize number of vertices in this batch
 */
template <typename Type, typename Index_ = int>
__global__ void vertex_degree_kernel(Pack<Type, Index_> data,
                                     Index_ startVertexId, Index_ batchSize) {
  const Type Zero = (Type)0;
  Index_ row = (blockIdx.y * TPB_Y) + threadIdx.y;
  Index_ col = (blockIdx.x * TPB_X) + threadIdx.x;
  Index_ N = data.N;
  if ((row >= batchSize) || (col >= N)) return;
  Type eps = data.eps;
  Type eps2 = eps * eps;
  Type sum = Zero;
  Index_ D = data.D;
  Type *x = data.x;
  bool *adj = data.adj;
  Index_ *vd = data.vd;
  for (Index_ d = 0; d < D; ++d) {
    Type a = __ldg(x + (row + startVertexId) * D + d);
    Type b = __ldg(x + col * D + d);
    Type diff = a - b;
    sum += (diff * diff);
  }
  Index_ res = (sum <= eps2);
  adj[row * N + col] = res;

  if (sizeof(Index_) == 4) {
    raft::myAtomicAdd((int *)(vd + row), (int)res);
    raft::myAtomicAdd((int *)(vd + batchSize), (int)res);
  } else if (sizeof(Index_) == 8) {
    raft::myAtomicAdd<unsigned long long>((unsigned long long *)(vd + row),
                                          res);
    raft::myAtomicAdd<unsigned long long>(
      (unsigned long long *)(vd + batchSize), res);
  }
}

template <typename Type, typename Index_ = int>
void launcher(Pack<Type, Index_> data, Index_ startVertexId, Index_ batchSize,
              cudaStream_t stream) {
  ASSERT(sizeof(Index_) == 4 || sizeof(Index_) == 8,
         "index_t should be 4 or 8 bytes");

  dim3 grid(raft::ceildiv(data.N, (Index_)TPB_X),
            raft::ceildiv(batchSize, (Index_)TPB_Y), 1);
  dim3 blk(TPB_X, TPB_Y, 1);
  data.resetArray(stream, batchSize + 1);
  vertex_degree_kernel<<<grid, blk, 0, stream>>>(data, startVertexId,
                                                 batchSize);
}

}  // namespace Naive
}  // namespace VertexDeg
}  // namespace Dbscan
