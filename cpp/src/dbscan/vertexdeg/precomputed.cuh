/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math.h>
#include <raft/core/cudart_utils.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/device_atomics.cuh>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/reduce.cuh>

#include "pack.h"

namespace ML {
namespace Dbscan {
namespace VertexDeg {
namespace Precomputed {

/**
 * Calculates the vertex degree array and the epsilon neighborhood adjacency matrix for the batch.
 */
template <typename value_t, typename index_t = int>
void launcher(const raft::handle_t& handle,
              Pack<value_t, index_t> data,
              index_t start_vertex_id,
              index_t batch_size,
              cudaStream_t stream)
{
  // Regarding index types, a special index type is used here for indices in
  // the distance matrix due to its dimensions (that are independent of the
  // batch size)
  using long_index_t = long long int;

  // The last position of data.vd is the sum of all elements in this array
  // (excluding it). Hence, its length is one more than the number of points
  // Initialize it to zero.
  index_t* d_nnz = data.vd + batch_size;
  RAFT_CUDA_TRY(cudaMemsetAsync(d_nnz, 0, sizeof(index_t), stream));

  long_index_t N              = data.N;
  long_index_t cur_batch_size = min(data.N - start_vertex_id, batch_size);

  const value_t& eps = data.eps;
  raft::linalg::unaryOp<value_t>(
    data.adj,
    data.x + (long_index_t)start_vertex_id * N,
    cur_batch_size * N,
    [eps] __device__(value_t dist) { return (dist <= eps); },
    stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Reduction of adj to compute the vertex degrees
  raft::linalg::coalescedReduction<bool, index_t, long_index_t>(
    data.vd,
    data.adj,
    data.N,
    batch_size,
    (index_t)0,
    stream,
    false,
    [] __device__(bool adj_ij, long_index_t idx) { return static_cast<index_t>(adj_ij); },
    raft::Sum<index_t>(),
    [d_nnz] __device__(index_t degree) {
      atomicAdd(d_nnz, degree);
      return degree;
    });
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace Precomputed
}  // end namespace VertexDeg
}  // end namespace Dbscan
}  // namespace ML
