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

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>

#include <distance/distance.cuh>
#include <cuml/neighbors/knn.hpp>

namespace ML {
namespace HDBSCAN {
namespace Reachability {

template <typename value_t>
__global__ void core_distances_kernel(value_t *knn_dists,
                                      int min_pts,
                                      value_t *out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  out[row] = knn_dists[row + min_pts];
}

/**
 * Extract core distances from KNN graph. This is essentially
 * performing a knn_dists[:,min_pts]
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @tparam tpb block size for kernel
 * @param knn_dists knn distance array
 * @param min_pts
 * @param n
 * @param out
 * @param stream
 */
template <typename value_t, int tpb = 1024>
void core_distances(value_t *knn_dists, int min_pts, size_t n, value_t *out,
                    cudaStream_t stream) {
  int blocks = raft::ceildiv(n * min_pts, (size_t)tpb);
  core_distances_kernel<value_t>
    <<<blocks, tpb, 0, stream>>>(knn_dists, min_pts, out);
}

template <typename value_idx, typename value_t>
__global__ void mutual_reachability_kernel(value_t *pw_dists,
                                           value_t *core_dists,
                                           size_t m,
                                           size_t n,
                                           value_t *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / n;
  int col = idx % n;

  value_idx a = row;
  value_idx b = col;

  value_t core_a = core_dists[a];
  value_t core_b = core_dists[b];

  value_t dist = pw_dists[row + col];

  // TODO: Really only need to output one of the triangles
  out[row + col] = max(core_a, core_b, dist);
}

template <typename value_idx, typename value_t, int tpb = 1024>
void mutual_reachability(value_t *pw_dists,
                         value_t *core_dists,
                         size_t m,
                         cudaStream_t stream) {
  int blocks = raft::ceildiv(m * m, (size_t)tpb);

  mutual_reachability_kernel<value_idx, value_t>
    <<<blocks, tpb, 0, stream>>>(pw_dists, core_dists, m, m, stream);
}

/**
 * Constructs a mutual reachability of the pairwise distances. This is
 * a naive implementation, which has a O(n^2) memory & complexity scaling.
 *
 * @TODO: Investigate constructing the MST by forming the mutual reachaiblity
 * graph directly from the KNN graph.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle
 * @param[in] X
 * @param[in] m
 * @param[in] n
 * @param[in] metric
 * @param[out] pw_dists
 * @param[in] k
 */
template <typename value_idx, typename value_t>
void pairwise_mutual_reachability_graph(const raft::handle_t &handle,
                                        const value_t *X,
                                        size_t m,
                                        size_t n,
                                        raft::distance::DistanceType metric,
                                        int min_pts,
                                        int k,
                                        value_t *pw_dists) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  raft::mr::device::buffer<value_idx> inds(d_alloc, stream, k);
  raft::mr::device::buffer<value_t> dists(d_alloc, stream, k);

  // perform knn
  brute_force_knn(handle, {X}, {m}, n, X, m, inds.data(), dists.data(), k, true,
                  true, metric);

  raft::mr::device::buffer<value_t> core_dists(d_alloc, stream, m);

  core_distances(dists.data(), min_pts, m, core_dists.data(), stream);

  inds.release();
  dists.release();

  // @TODO: This is super expensive. Future versions need to eliminate
  //   the pairwise distance matrix, use KNN, or an MST based on the KNN graph
  pairwise_distance(X, X, pw_dists, m, m, n, nullptr, metric, stream);

  mutual_reachability<value_idx, value_t>(inds.data(), dists.data(),
                                          core_dists.data(), m, n, stream);
}

};  // end namespace Reachability
};  // end namespace HDBSCAN
};  // end namespace ML