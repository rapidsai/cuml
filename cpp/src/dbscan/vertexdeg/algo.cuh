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
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace ML {
namespace Dbscan {
namespace VertexDeg {
namespace Algo {

template <typename index_t = int>
struct column_counter : public thrust::unary_function<index_t, index_t> {
  index_t* ia_;
  index_t n_;

  column_counter(index_t* ia, index_t n) : ia_(ia), n_(n) {}

  __host__ __device__ index_t operator()(const index_t& input) const
  {
    return (input < n_) ? ia_[input + 1] - ia_[input] : ia_[n_];
  }
};

template <typename math_t, typename index_t = int, int tpb = 128, int warpsize = 32>
static __global__ void accumulateWeights(const index_t* ia,
                                         const index_t num_rows,
                                         const index_t* ja,
                                         const math_t* col_weights,
                                         math_t* weight_sums)
{
  constexpr int warps_per_block = tpb / warpsize;

  // Setup WarpReduce shared memory for all warps
  typedef cub::WarpReduce<math_t> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];

  // all threads in a warp are responsible for one line of csr
  const int thread_in_warp = threadIdx.x % warpsize;
  int warp_id              = threadIdx.x / 32;

  int idx = blockIdx.x * warps_per_block + warp_id;

  math_t weight_sum_tmp = 0;
  int rowStartIdx       = (idx < num_rows) ? ia[idx] : 0;
  int rowEndIdx         = (idx < num_rows) ? ia[idx + 1] : 0;
  for (int pos = rowStartIdx + thread_in_warp; pos < rowEndIdx; pos += warpsize) {
    weight_sum_tmp += col_weights[ja[pos]];
  }

  math_t weight_sum = WarpReduce(temp_storage[warp_id]).Sum(weight_sum_tmp);

  if (thread_in_warp == 0 && idx < num_rows && weight_sum > 0) { weight_sums[idx] = weight_sum; }
}

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

    if (data.rbc_index != nullptr) {
      raft::neighbors::ball_cover::epsUnexpL2NeighborhoodRbc<index_t, value_t, index_t>(
        handle, *data.rbc_index, data.ia, data.ja, data.x + start_vertex_id * k, n, k, sqrtf(eps2));
    } else {
      raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t, index_t>(
        data.adj, nullptr, data.x + start_vertex_id * k, data.x, n, m, k, eps2, stream);
    }

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

    if (data.rbc_index != nullptr) {
      raft::neighbors::ball_cover::epsUnexpL2NeighborhoodRbc<index_t, value_t, index_t>(
        handle, *data.rbc_index, data.ia, data.ja, data.x + start_vertex_id * k, n, k, data.eps);
    } else {
      raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t, index_t>(
        data.adj, nullptr, data.x + start_vertex_id * k, data.x, n, m, k, eps2, stream);
    }
  }

  if (data.rbc_index != nullptr) {
    auto thrust_exec_policy = handle.get_thrust_policy();
    thrust::transform(thrust_exec_policy,
                      thrust::make_counting_iterator<index_t>(0),
                      thrust::make_counting_iterator<index_t>(n + 1),
                      data.vd,
                      column_counter(data.ia, n));
  } else {
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

  if (data.weight_sum != nullptr && data.sample_weight != nullptr) {
    if (data.rbc_index != nullptr) {
      accumulateWeights<value_t, index_t, 128, 32>
        <<<raft::ceildiv(n, (index_t)4), 128, 0, stream>>>(
          data.ia, n, data.ja, data.sample_weight, data.weight_sum);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {
      const value_t* sample_weight = data.sample_weight;
      // Reduction of adj to compute the weighted vertex degrees
      raft::linalg::coalescedReduction<bool, value_t, index_t>(
        data.weight_sum,
        data.adj,
        data.N,
        batch_size,
        (value_t)0,
        stream,
        false,
        [sample_weight] __device__(bool adj_ij, index_t j) {
          return adj_ij ? sample_weight[j] : (value_t)0;
        },
        raft::Sum<value_t>());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }
}

}  // namespace Algo
}  // end namespace VertexDeg
}  // end namespace Dbscan
}  // namespace ML
