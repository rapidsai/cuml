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
#include <common/cumlHandle.hpp>
#include <raft/cuda_utils.cuh>

#include <raft/mr/device/buffer.hpp>

#include <cuml/neighbors/knn.hpp>
#include <distance/distance.cuh>

namespace ML {
namespace HDBSCAN {
namespace Reachability {

template <typename value_t>
__global__ void core_distances_kernel(value_t *knn_dists, int min_pts,
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
                                           value_t *core_dists, size_t m,
                                           size_t n, value_t *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / n;
  int col = idx % n;

  value_idx a = row;
  value_idx b = col;

  value_t core_a = core_dists[a];
  value_t core_b = core_dists[b];

  value_t dist = pw_dists[row + col];

  out[row + col] = max(core_a, core_b, dist);
}

template <typename value_idx, typename value_t, int tpb = 1024>
void mutual_reachability(value_t *pw_dists, value_t *core_dists, size_t m,
                         cudaStream_t stream) {
  int blocks = raft::ceildiv(m * m, (size_t)tpb);

  mutual_reachability_kernel<value_idx, value_t>
    <<<blocks, tpb, 0, stream>>>(pw_dists, core_dists, m, m, stream);
}

/**
 * Constructs a mutual reachability graph, which is a k-nearest neighbors
 * graph projected into mutual reachability space using the following
 * function for each data point, where core_distance is the distance
 * to the kth neighbor: max(core_distance(a), core_distance(b), d(a, b))
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
void mutual_reachability_dists(const raft::handle_t &handle,
                               const value_t *X, size_t m, size_t n,
                               raft::distance::DistanceType metric,
                               int k, value_idx *inds, value_t *dists,
                               value_t *core_dists) {
  auto stream = handle.get_stream();

  // perform knn
  brute_force_knn(handle, {X}, {m}, n, X, m, inds, dists, k, true,
                  true, metric);

  // Slice core distances (distances to kth nearest neighbor)
  core_distances(dists, k, m, core_dists, stream);

  // Project into mutual reachability space.
  // Note that it's not guaranteed the knn graph will be connected
  // at this point so the core distances will need to be returned
  // so additional points can be added to the graph and projected
  // ito mutual reachability space later.
  mutual_reachability<value_idx, value_t>(inds, dists,
                                          core_dists, m, n, stream);
}

};  // end namespace Reachability
};  // end namespace HDBSCAN
};  // end namespace ML