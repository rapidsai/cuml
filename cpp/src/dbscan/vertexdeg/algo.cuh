/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cuml/common/utils.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/neighbors/epsilon_neighborhood.cuh>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <math.h>

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
CUML_KERNEL void accumulateWeights(const index_t* ia,
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

template <typename value_t, typename index_t = int>
void eps_nn(const raft::handle_t& handle,
            Pack<value_t, index_t> data,
            index_t start_vertex_id,
            index_t batch_size,
            cudaStream_t stream,
            value_t eps)
{
  index_t n = min(data.N - start_vertex_id, batch_size);
  index_t k = data.D;

  index_t spare_elemets_per_row =
    data.max_k > 0 ? (batch_size * data.N - data.ja->capacity()) / n : 0;

  if (data.max_k > 0 && data.max_k < spare_elemets_per_row) {
    ASSERT(data.ja != nullptr, "column pointer should be valid");

    index_t max_k = data.max_k;
    raft::neighbors::ball_cover::eps_nn<index_t, value_t, index_t, index_t>(
      handle,
      *data.rbc_index,
      raft::make_device_vector_view<index_t, index_t>(data.ia, n + 1),
      raft::make_device_vector_view<index_t, index_t>(data.ja->data(), n * data.N),
      raft::make_device_vector_view<index_t, index_t>(nullptr, n + 1),
      raft::make_device_matrix_view<const value_t, index_t>(data.x + start_vertex_id * k, n, k),
      eps,
      raft::make_host_scalar_view<index_t, index_t>(&max_k));
    ASSERT(max_k == data.max_k, "given maximum rowsize was not sufficient");
  } else {
    raft::neighbors::ball_cover::eps_nn<index_t, value_t, index_t, index_t>(
      handle,
      *data.rbc_index,
      raft::make_device_vector_view<index_t, index_t>(data.ia, n + 1),
      raft::make_device_vector_view<index_t, index_t>(nullptr, 0),
      raft::make_device_vector_view<index_t, index_t>(data.vd, n + 1),
      raft::make_device_matrix_view<const value_t, index_t>(data.x + start_vertex_id * k, n, k),
      eps);

    if (data.ja != nullptr) {
      // no need to re-compute in second batch loop - ja has already been resized
      if (data.vd != nullptr) {
        index_t curradjlen = 0;
        raft::update_host(&curradjlen, data.vd + n, 1, stream);
        handle.sync_stream(stream);
        data.ja->resize(curradjlen, stream);
      }

      raft::neighbors::ball_cover::eps_nn<index_t, value_t, index_t, index_t>(
        handle,
        *data.rbc_index,
        raft::make_device_vector_view<index_t, index_t>(data.ia, n + 1),
        raft::make_device_vector_view<index_t, index_t>(data.ja->data(), n * data.N),
        raft::make_device_vector_view<index_t, index_t>(nullptr, n + 1),
        raft::make_device_matrix_view<const value_t, index_t>(data.x + start_vertex_id * k, n, k),
        eps);
    }
  }
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
      eps_nn(handle, data, start_vertex_id, batch_size, stream, (value_t)sqrtf(eps2));
    } else {
      raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t, index_t>(
        data.adj, data.vd, data.x + start_vertex_id * k, data.x, n, m, k, eps2, stream);
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
    if (data.rbc_index != nullptr) {
      eps_nn(handle, data, start_vertex_id, batch_size, stream, data.eps);
    } else {
      raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t, index_t>(
        data.adj, data.vd, data.x + start_vertex_id * k, data.x, n, m, k, eps2, stream);
    }
  }

  if (data.weight_sum != nullptr && data.sample_weight != nullptr) {
    if (data.rbc_index != nullptr) {
      accumulateWeights<value_t, index_t, 128, 32>
        <<<raft::ceildiv(n, (index_t)4), 128, 0, stream>>>(
          data.ia, n, data.ja->data(), data.sample_weight, data.weight_sum);
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
        raft::add_op());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }
}

}  // namespace Algo
}  // end namespace VertexDeg
}  // end namespace Dbscan
}  // namespace ML
