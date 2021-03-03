/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <math.h>
#include <raft/cudart_utils.h>
#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/unary_op.cuh>

#include "pack.h"

namespace ML {
namespace Dbscan {
namespace VertexDeg {
namespace Precomputed {

template <typename value_t, typename index_t>
__global__ void dist_to_adj_kernel(const value_t* X, bool* adj, index_t N,
                                   index_t start_vertex_id, index_t batch_size,
                                   value_t eps) {
  for (index_t i = threadIdx.x; i < batch_size; i += blockDim.x) {
    adj[batch_size * blockIdx.x + i] =
      X[N * blockIdx.x + start_vertex_id + i] <= eps;
  }
}

/**
 * Calculates the vertex degree array and the epsilon neighborhood adjacency matrix for the batch.
 */
template <typename value_t, typename index_t = int>
void launcher(const raft::handle_t& handle, Pack<value_t, index_t> data,
              index_t start_vertex_id, index_t batch_size,
              cudaStream_t stream) {
  const value_t& eps = data.eps;

  /// TODO: case where less remaining elements than batch_size?

  // Note: the matrix is symmetric. We take advantage of this to have two
  //       coalesced kernels:
  //  - The reduction works on a column-major N*B matrix to compute a B vector
  //    with a cub-BlockReduce-based primitive.
  //    The final_op is used to compute the total number of non-zero elements
  //  - The conversion to a boolean matrix works on a column-major B*N matrix
  //    (coalesced 2d copy + transform).

  // Reduction to compute the vertex degrees
  index_t* d_nnz = data.vd + batch_size;
  CUDA_CHECK(cudaMemsetAsync(d_nnz, 0, sizeof(index_t), stream));
  raft::linalg::coalescedReduction(
    data.vd, data.x + start_vertex_id * data.N, data.N, batch_size, (index_t)0,
    stream, false,
    [eps] __device__(value_t dist, index_t idx) {
      return static_cast<index_t>(dist <= eps);
    },
    raft::Sum<index_t>(),
    [d_nnz] __device__(index_t degree) {
      atomicAdd(d_nnz, degree);
      return degree;
    });

  // Transform the distance matrix into a neighborhood matrix
  dist_to_adj_kernel<<<data.N, std::min(batch_size, (index_t)256), 0, stream>>>(
    data.x, data.adj, data.N, start_vertex_id, batch_size, data.eps);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Precomputed
}  // end namespace VertexDeg
}  // end namespace Dbscan
}  // namespace ML
