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

#include "mgrp_accessor.cuh"
#include "mgrp_epsilon_neighborhood.cuh"
#include "mgrp_reduce.cuh"

#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/util/device_atomics.cuh>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace VertexDeg {

template <typename value_t, typename index_t = int>
void launcher(const raft::handle_t& handle,
              Metadata::AdjGraphAccessor<bool, index_t>& adj_ac,
              Metadata::VertexDegAccessor<index_t, index_t>& vd_ac,
              const Metadata::PointAccessor<value_t, index_t>& x_ac,
              value_t* eps,
              cudaStream_t stream,
              raft::distance::DistanceType metric)
{
  ASSERT(sizeof(index_t) == 4 || sizeof(index_t) == 8, "index_t should be 4 or 8 bytes");

  const index_t n_groups = x_ac.n_groups;
  index_t m              = x_ac.n_points;
  index_t k              = x_ac.feat_size;
  const value_t* x       = x_ac.pts;

  // Compute adjacency matrix `adj` using Cosine or L2 metric.
  if (metric == raft::distance::DistanceType::CosineExpanded) {
    auto counting = thrust::make_counting_iterator<index_t>(0);
    thrust::for_each(
      handle.get_thrust_policy(), counting, counting + n_groups, [=] __device__(index_t idx) {
        eps[idx] *= 2;
      });
    rmm::device_uvector<value_t> rowNorms(m, stream);

    raft::linalg::rowNorm(
      rowNorms.data(),
      x,
      k,
      m,
      raft::linalg::NormType::L2Norm,
      true,
      stream,
      [] __device__(value_t in) { return (in > FLT_EPSILON) ? sqrtf(in) : FLT_EPSILON; });

    /* Cast away constness because the output matrix for normalization cannot be of const type.
     * Input matrix will be modified due to normalization.
     */
    raft::linalg::matrixVectorOp(
      const_cast<value_t*>(x),
      x,
      rowNorms.data(),
      k,
      m,
      true,
      false,
      [] __device__(value_t mat_in, value_t vec_in) { return mat_in / vec_in; },
      stream);

    EpsNeighborhood::MultiGroupEpsUnexpL2SqNeighborhood<value_t, index_t>(
      adj_ac, vd_ac, x_ac, x_ac, eps, false, stream);

    /**
     * Restoring the input matrix after normalization.
     */
    raft::linalg::matrixVectorOp(
      const_cast<value_t*>(x),
      x,
      rowNorms.data(),
      k,
      m,
      true,
      false,
      [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
      stream);
  } else {
    auto counting = thrust::make_counting_iterator<index_t>(0);
    thrust::for_each(
      handle.get_thrust_policy(), counting, counting + n_groups, [=] __device__(index_t idx) {
        eps[idx] = eps[idx] * eps[idx];
      });

    // 1. The output matrix adj is now an n x m matrix (row-major order)
    // 2. Do not compute the vertex degree in epsUnexpL2SqNeighborhood (pass a
    // nullptr)
    EpsNeighborhood::MultiGroupEpsUnexpL2SqNeighborhood<value_t, index_t>(
      adj_ac, vd_ac, x_ac, x_ac, eps, false, stream);
  }
  // Reduction
  index_t* vd       = vd_ac.vd;
  index_t* vd_group = vd_ac.vd_group;
  index_t* vd_all   = vd_ac.vd_all;

  Reduce::MultiGroupCoalescedReduction<bool, index_t, index_t>(
    vd_ac.vd,
    adj_ac.adj,
    n_groups,
    adj_ac.max_nbr,
    adj_ac.max_nbr,
    adj_ac.adj_col_stride,
    adj_ac.n_rows_ptr,
    vd_ac.row_start_ids,
    adj_ac.adj_group_offset,
    (index_t)0,
    stream,
    false,
    [] __device__(bool adj_ij, index_t idx) { return static_cast<index_t>(adj_ij); },
    raft::Sum<index_t>(),
    [vd_group, vd_all] __device__(index_t degree) {
      index_t group_id       = blockIdx.z * blockDim.z + threadIdx.z;
      index_t* vd_group_base = vd_group + group_id;
      atomicAdd(vd_group_base, degree);
      atomicAdd(vd_all, degree);
      return degree;
    });
}

template <typename Type_f, typename Index_ = int>
void run(const raft::handle_t& handle,
         Metadata::AdjGraphAccessor<bool, Index_>& adj_ac,
         Metadata::VertexDegAccessor<Index_, Index_>& vd_ac,
         const Metadata::PointAccessor<Type_f, Index_>& x_ac,
         Type_f* eps,
         int algo,
         cudaStream_t stream,
         raft::distance::DistanceType metric)
{
  switch (algo) {
    case 0:
      ASSERT(
        false, "Incorrect algo '%d' passed! Naive version of vertexdeg has been removed.", algo);
    case 1: launcher<Type_f, Index_>(handle, adj_ac, vd_ac, x_ac, eps, stream, metric); break;
    case 2:
      ASSERT(false,
             "Incorrect algo '%d' passed! Precomputed version of vertexdeg is not supported for "
             "multi-groups vertexdeg.",
             algo);
    default: ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace VertexDeg
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML