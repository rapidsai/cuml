/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <common/allocatorAdapter.hpp>

#include <distance/distance.cuh>

#include <rmm/device_uvector.hpp>

#include <raft/linalg/distance_type.h>
#include <raft/mr/device/buffer.hpp>

// TODO: Not a good strategy for pluggability but will be
// removed once our dense pairwise distance API is in RAFT
#include <raft/sparse/hierarchy/common.h>
#include <raft/sparse/hierarchy/detail/connectivities.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <limits>

namespace raft {
namespace hierarchy {
namespace detail {

template <typename value_idx>
__global__ void fill_indices2(value_idx *indices, size_t m, size_t nnz) {
  value_idx tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nnz) return;
  value_idx v = tid % m;
  indices[tid] = v;
}

/**
 * Compute connected CSR of pairwise distances
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param X
 * @param m
 * @param n
 * @param metric
 * @param[out] indptr
 * @param[out] indices
 * @param[out] data
 */
template <typename value_idx, typename value_t>
void pairwise_distances(const raft::handle_t &handle, const value_t *X,
                        size_t m, size_t n, raft::distance::DistanceType metric,
                        value_idx *indptr, value_idx *indices, value_t *data) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  value_idx nnz = m * m;

  value_idx blocks = raft::ceildiv(nnz, (value_idx)256);
  fill_indices2<value_idx><<<blocks, 256, 0, stream>>>(indices, m, nnz);

  thrust::sequence(thrust::cuda::par.on(stream), indptr, indptr + m, 0, (int)m);

  raft::update_device(indptr + m, &nnz, 1, stream);

  // TODO: Keeping raft device buffer here for now until our
  // dense pairwise distances API is finished being refactored
  raft::mr::device::buffer<char> workspace(d_alloc, stream, (size_t)0);

  // TODO: It would ultimately be nice if the MST could accept
  // dense inputs directly so we don't need to double the memory
  // usage to hand it a sparse array here.
  MLCommon::Distance::pairwise_distance<value_t, value_idx>(
    X, X, data, m, m, n, workspace, metric, stream);
}

/**
 * Connectivities specialization for pairwise distances
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct distance_graph_impl<raft::hierarchy::LinkageDistance::PAIRWISE,
                           value_idx, value_t> {
  void run(const raft::handle_t &handle, const value_t *X, size_t m, size_t n,
           raft::distance::DistanceType metric,
           rmm::device_uvector<value_idx> &indptr,
           rmm::device_uvector<value_idx> &indices,
           rmm::device_uvector<value_t> &data, int c) {
    auto d_alloc = handle.get_device_allocator();
    auto stream = handle.get_stream();

    size_t nnz = m * m;

    indices.resize(nnz, stream);
    data.resize(nnz, stream);

    pairwise_distances(handle, X, m, n, metric, indptr.data(), indices.data(),
                       data.data());
  }
};

};  // namespace detail
};  // end namespace hierarchy
};  // end namespace raft