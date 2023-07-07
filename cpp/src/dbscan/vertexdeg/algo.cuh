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

#include "pack.h"
#include <cuda_runtime.h>
#include <math.h>
#include <raft/neighbors/epsilon_neighborhood.cuh>

#include "pack.h"
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/util/device_atomics.cuh>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace Dbscan {
namespace VertexDeg {
namespace Algo {

/**
 * Calculates the vertex degree array and the epsilon neighborhood adjacency matrix for the batch.
 */
template <typename value_t, typename index_t = int>
void launcher(const raft::handle_t& handle,
              Pack<value_t, index_t> data,
              index_t start_vertex_id,
              index_t batch_size,
              cudaStream_t stream,
              raft::distance::DistanceType metric)
{
  // The last position of data.vd is the sum of all elements in this array
  // (excluding it). Hence, its length is one more than the number of points
  // Initialize it to zero.
  index_t* d_nnz = data.vd + batch_size;
  RAFT_CUDA_TRY(cudaMemsetAsync(d_nnz, 0, sizeof(index_t), stream));

  ASSERT(sizeof(index_t) == 4 || sizeof(index_t) == 8, "index_t should be 4 or 8 bytes");

  index_t m = data.N;
  index_t n = min(data.N - start_vertex_id, batch_size);
  index_t k = data.D;
  value_t eps2;

  // Compute adjacency matrix `adj` using Cosine or L2 metric.
  if (metric == raft::distance::DistanceType::CosineExpanded) {
    rmm::device_uvector<value_t> rowNorms(m, stream);

    raft::linalg::rowNorm(rowNorms.data(),
                          data.x,
                          k,
                          m,
                          raft::linalg::NormType::L2Norm,
                          true,
                          stream,
                          [] __device__(value_t in) { return sqrtf(in); });

    /* Cast away constness because the output matrix for normalization cannot be of const type.
     * Input matrix will be modified due to normalization.
     */
    raft::linalg::matrixVectorOp(
      const_cast<value_t*>(data.x),
      data.x,
      rowNorms.data(),
      k,
      m,
      true,
      true,
      [] __device__(value_t mat_in, value_t vec_in) { return mat_in / vec_in; },
      stream);

    eps2 = 2 * data.eps;

    raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t, index_t>(
      data.adj, nullptr, data.x + start_vertex_id * k, data.x, n, m, k, eps2, stream);

    /**
     * Restoring the input matrix after normalization.
     */
    raft::linalg::matrixVectorOp(
      const_cast<value_t*>(data.x),
      data.x,
      rowNorms.data(),
      k,
      m,
      true,
      true,
      [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
      stream);
  } else {
    eps2 = data.eps * data.eps;

    // 1. The output matrix adj is now an n x m matrix (row-major order)
    // 2. Do not compute the vertex degree in epsUnexpL2SqNeighborhood (pass a
    // nullptr)
    raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t, index_t>(
      data.adj, nullptr, data.x + start_vertex_id * k, data.x, n, m, k, eps2, stream);
  }

  // Reduction of adj to compute the vertex degrees
  raft::linalg::coalescedReduction<bool, index_t, index_t>(
    data.vd,
    data.adj,
    data.N,
    batch_size,
    (index_t)0,
    stream,
    false,
    [] __device__(bool adj_ij, index_t idx) { return static_cast<index_t>(adj_ij); },
    raft::Sum<index_t>(),
    [d_nnz] __device__(index_t degree) {
      atomicAdd(d_nnz, degree);
      return degree;
    });
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace Algo
}  // end namespace VertexDeg
}  // end namespace Dbscan
}  // namespace ML
