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

#include <raft/core/cudart_utils.hpp>
#include <raft/cuda_utils.cuh>

#include <cuml/metrics/metrics.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/distance/distance_types.hpp>

// TODO: Not a good strategy for pluggability but will be
// removed once our dense pairwise distance API is in RAFT
#include <raft/cluster/detail/connectivities.cuh>
#include <raft/cluster/single_linkage_types.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>

namespace raft {
namespace cluster {
namespace detail {

template <typename value_idx>
__global__ void fill_indices2(value_idx* indices, size_t m, size_t nnz)
{
  value_idx tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nnz) return;
  value_idx v  = tid % m;
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
void pairwise_distances(const raft::handle_t& handle,
                        const value_t* X,
                        size_t m,
                        size_t n,
                        raft::distance::DistanceType metric,
                        value_idx* indptr,
                        value_idx* indices,
                        value_t* data)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  value_idx nnz = m * m;

  value_idx blocks = raft::ceildiv(nnz, (value_idx)256);
  fill_indices2<value_idx><<<blocks, 256, 0, stream>>>(indices, m, nnz);

  thrust::sequence(thrust::cuda::par.on(stream), indptr, indptr + m, 0, (int)m);

  raft::update_device(indptr + m, &nnz, 1, stream);

  // TODO: It would ultimately be nice if the MST could accept
  // dense inputs directly so we don't need to double the memory
  // usage to hand it a sparse array here.
  ML::Metrics::pairwise_distance(handle, X, X, data, m, m, n, metric);
  // self-loops get max distance
  auto transform_in =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), data));

  thrust::transform(exec_policy,
                    transform_in,
                    transform_in + nnz,
                    data,
                    [=] __device__(const thrust::tuple<value_idx, value_t>& tup) {
                      value_idx idx  = thrust::get<0>(tup);
                      bool self_loop = idx % m == idx / m;
                      return (self_loop * std::numeric_limits<value_t>::max()) +
                             (!self_loop * thrust::get<1>(tup));
                    });
}

/**
 * Connectivities specialization for pairwise distances
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct distance_graph_impl<raft::cluster::LinkageDistance::PAIRWISE, value_idx, value_t> {
  void run(const raft::handle_t& handle,
           const value_t* X,
           size_t m,
           size_t n,
           raft::distance::DistanceType metric,
           rmm::device_uvector<value_idx>& indptr,
           rmm::device_uvector<value_idx>& indices,
           rmm::device_uvector<value_t>& data,
           int c)
  {
    auto stream = handle.get_stream();

    size_t nnz = m * m;

    indices.resize(nnz, stream);
    data.resize(nnz, stream);

    pairwise_distances(handle, X, m, n, metric, indptr.data(), indices.data(), data.data());
  }
};

};  // namespace detail
};  // namespace cluster
};  // end namespace raft
