/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "reachability_faiss.cuh"

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <raft/linalg/unary_op.cuh>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuml/neighbors/knn.hpp>
#include <raft/distance/distance.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {

/**
 * Extract core distances from KNN graph. This is essentially
 * performing a knn_dists[:,min_pts]
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @tparam tpb block size for kernel
 * @param[in] knn_dists knn distance array (size n * k)
 * @param[in] min_samples this neighbor will be selected for core distances
 * @param[in] n_neighbors the number of neighbors of each point in the knn graph
 * @param[in] n number of samples
 * @param[out] out output array (size n)
 * @param[in] stream stream for which to order cuda operations
 */
template <typename value_idx, typename value_t, int tpb = 256>
void core_distances(
  value_t* knn_dists, int min_samples, int n_neighbors, size_t n, value_t* out, cudaStream_t stream)
{
  ASSERT(n_neighbors >= min_samples,
         "the size of the neighborhood should be greater than or equal to min_samples");

  int blocks = raft::ceildiv(n, (size_t)tpb);

  auto exec_policy = rmm::exec_policy(stream);

  auto indices = thrust::make_counting_iterator<value_idx>(0);

  thrust::transform(exec_policy, indices, indices + n, out, [=] __device__(value_idx row) {
    return knn_dists[row * n_neighbors + (min_samples - 1)];
  });
}

/**
 * Wraps the brute force knn API, to be used for both training and prediction
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[out] inds nearest neighbor indices (size n_search_items * k)
 * @param[out] dists nearest neighbor distances (size n_search_items * k)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] search_items array of items to search of dimensionality D (size n_search_items * n)
 * @param[in] n_search_items number of rows in search_items
 * @param[in] k number of nearest neighbors
 * @param[in] metric distance metric to use
 */
template <typename value_idx, typename value_t>
void compute_knn(const raft::handle_t& handle,
                 const value_t* X,
                 value_idx* inds,
                 value_t* dists,
                 size_t m,
                 size_t n,
                 const value_t* search_items,
                 size_t n_search_items,
                 int k,
                 raft::distance::DistanceType metric)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();
  std::vector<value_t*> inputs;
  inputs.push_back(const_cast<value_t*>(X));

  std::vector<int> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  rmm::device_uvector<int64_t> int64_indices(k * n_search_items, stream);

  // perform knn
  brute_force_knn(handle,
                  inputs,
                  sizes,
                  n,
                  const_cast<value_t*>(search_items),
                  n_search_items,
                  int64_indices.data(),
                  dists,
                  k,
                  true,
                  true,
                  metric);

  // convert from current knn's 64-bit to 32-bit.
  thrust::transform(exec_policy,
                    int64_indices.data(),
                    int64_indices.data() + int64_indices.size(),
                    inds,
                    [] __device__(int64_t in) -> value_idx { return in; });
}

/**
 * Constructs a mutual reachability graph, which is a k-nearest neighbors
 * graph projected into mutual reachability space using the following
 * function for each data point, where core_distance is the distance
 * to the kth neighbor: max(core_distance(a), core_distance(b), d(a, b))
 *
 * Unfortunately, points in the tails of the pdf (e.g. in sparse regions
 * of the space) can have very large neighborhoods, which will impact
 * nearby neighborhoods. Because of this, it's possible that the
 * radius for points in the main mass, which might have a very small
 * radius initially, to expand very large. As a result, the initial
 * knn which was used to compute the core distances may no longer
 * capture the actual neighborhoods after projection into mutual
 * reachability space.
 *
 * For the experimental version, we execute the knn twice- once
 * to compute the radii (core distances) and again to capture
 * the final neighborhoods. Future iterations of this algorithm
 * will work improve upon this "exact" version, by using
 * more specialized data structures, such as space-partitioning
 * structures. It has also been shown that approximate nearest
 * neighbors can yield reasonable neighborhoods as the
 * data sizes increase.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metric to use
 * @param[in] k neighborhood size
 * @param[in] min_samples this neighborhood will be selected for core distances
 * @param[in] alpha weight applied when internal distance is chosen for
 *            mutual reachability (value of 1.0 disables the weighting)
 * @param[out] indptr CSR indptr of output knn graph (size m + 1)
 * @param[out] core_dists output core distances array (size m)
 * @param[out] out COO object, uninitialized on entry, on exit it stores the
 *             (symmetrized) maximum reachability distance for the k nearest
 *             neighbors.
 */
template <typename value_idx, typename value_t>
void mutual_reachability_graph(const raft::handle_t& handle,
                               const value_t* X,
                               size_t m,
                               size_t n,
                               raft::distance::DistanceType metric,
                               int min_samples,
                               value_t alpha,
                               value_idx* indptr,
                               value_t* core_dists,
                               raft::sparse::COO<value_t, value_idx>& out)
{
  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  rmm::device_uvector<value_idx> coo_rows(min_samples * m, stream);
  rmm::device_uvector<value_idx> inds(min_samples * m, stream);
  rmm::device_uvector<value_t> dists(min_samples * m, stream);

  // perform knn
  compute_knn(handle, X, inds.data(), dists.data(), m, n, X, m, min_samples, metric);

  // Slice core distances (distances to kth nearest neighbor)
  core_distances<value_idx>(dists.data(), min_samples, min_samples, m, core_dists, stream);

  /**
   * Compute L2 norm
   */
  mutual_reachability_knn_l2(
    handle, inds.data(), dists.data(), X, m, n, min_samples, core_dists, (value_t)1.0 / alpha);

  // self-loops get max distance
  auto coo_rows_counting_itr = thrust::make_counting_iterator<value_idx>(0);
  thrust::transform(exec_policy,
                    coo_rows_counting_itr,
                    coo_rows_counting_itr + (m * min_samples),
                    coo_rows.data(),
                    [min_samples] __device__(value_idx c) -> value_idx { return c / min_samples; });

  raft::sparse::linalg::symmetrize(
    handle, coo_rows.data(), inds.data(), dists.data(), m, m, min_samples * m, out);

  raft::sparse::convert::sorted_coo_to_csr(out.rows(), out.nnz, indptr, m + 1, stream);

  // self-loops get max distance
  auto transform_in =
    thrust::make_zip_iterator(thrust::make_tuple(out.rows(), out.cols(), out.vals()));

  thrust::transform(exec_policy,
                    transform_in,
                    transform_in + out.nnz,
                    out.vals(),
                    [=] __device__(const thrust::tuple<value_idx, value_idx, value_t>& tup) {
                      return thrust::get<0>(tup) == thrust::get<1>(tup)
                               ? std::numeric_limits<value_t>::max()
                               : thrust::get<2>(tup);
                    });
}

};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML