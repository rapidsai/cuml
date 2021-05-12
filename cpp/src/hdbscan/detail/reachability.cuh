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

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <raft/mr/device/buffer.hpp>

#include <raft/linalg/unary_op.cuh>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/hierarchy/detail/connectivities.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/selection/knn_graph.cuh>

#include <rmm/device_uvector.hpp>

#include <cuml/neighbors/knn.hpp>
#include <raft/distance/distance.cuh>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {

template <typename value_t>
__global__ void core_distances_kernel(value_t *knn_dists, int k, int min_samples, size_t n,
                                      value_t *out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) out[row] = knn_dists[row * k + (min_samples - 1)];
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
template <typename value_t, int tpb = 256>
void core_distances(value_t *knn_dists, int k, int num_points, size_t n, value_t *out,
                    cudaStream_t stream) {
  int blocks = raft::ceildiv(n, (size_t)tpb);
  core_distances_kernel<value_t>
    <<<blocks, tpb, 0, stream>>>(knn_dists, k, num_points, n, out);
}

template <typename value_idx, typename value_t>
__global__ void mutual_reachability_kernel(value_t *dists, value_idx *inds,
                                           value_t *core_dists, int k, size_t n,
                                           float alpha, value_t *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / k;

  if (idx < k * n) {
    value_idx a = row;
    value_idx b = inds[idx];

    value_t core_a = core_dists[a];
    value_t core_b = core_dists[b];

    value_t dist = alpha * dists[idx];

    out[idx] = max(core_a, max(core_b, dist));
  }
}

template <typename value_idx, typename value_t, int tpb = 256>
void mutual_reachability(value_t *pw_dists, value_idx *inds,
                         value_t *core_dists, int k, size_t n, float alpha,
                         cudaStream_t stream) {
  int blocks = raft::ceildiv(k * n, (size_t)tpb);

  mutual_reachability_kernel<value_idx, value_t><<<blocks, tpb, 0, stream>>>(
    pw_dists, inds, core_dists, k, n, alpha, pw_dists);
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
void mutual_reachability_graph(const raft::handle_t &handle, const value_t *X,
                               size_t m, size_t n,
                               raft::distance::DistanceType metric, int k, int min_samples,
                               float alpha, value_idx *indptr,
                               value_t *core_dists,
                               raft::sparse::COO<value_t, value_idx> &out) {
  auto stream = handle.get_stream();

  std::vector<value_t *> inputs;
  inputs.push_back(const_cast<value_t *>(X));

  std::vector<int> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  rmm::device_uvector<value_idx> coo_rows(k * m, stream);
  rmm::device_uvector<int64_t> int64_indices(k * m, stream);
  rmm::device_uvector<value_idx> inds(k * m, stream);
  rmm::device_uvector<value_t> dists(k * m, stream);

  // perform knn
  brute_force_knn(handle, inputs, sizes, n, const_cast<value_t *>(X), m,
                  int64_indices.data(), dists.data(), k, true, true, metric);

  // convert from current knn's 64-bit to 32-bit.
  raft::sparse::selection::conv_indices(int64_indices.data(), inds.data(),
                                        k * m, stream);

  raft::linalg::unaryOp<value_t>(
    dists.data(), dists.data(), k * m,
    [] __device__(value_t input) {
      if (input == 0.0)
        return std::numeric_limits<value_t>::max();
      else
        return input;
    },
    stream);


  // Slice core distances (distances to kth nearest neighbor)
  core_distances(dists.data(), k, min_samples, m, core_dists, stream);

  // Project into mutual reachability space.
  // Note that it's not guaranteed the knn graph will be connected
  // at this point so the core distances will need to be returned
  // so additional points can be added to the graph and projected
  // into mutual reachability space later.
  mutual_reachability<value_idx, value_t>(dists.data(), inds.data(), core_dists,
                                          k, m, alpha, stream);

  raft::sparse::selection::fill_indices<value_idx>
    <<<raft::ceildiv(k * m, (size_t)256), 256, 0, stream>>>(coo_rows.data(), k,
                                                            m * k);

  raft::sparse::linalg::symmetrize(handle, coo_rows.data(), inds.data(),
                                   dists.data(), m, m, k * m, out);

  raft::sparse::convert::sorted_coo_to_csr(
    out.rows(), out.nnz, indptr, m + 1, handle.get_device_allocator(), stream);
}


};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML