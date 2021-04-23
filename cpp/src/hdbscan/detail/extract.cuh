/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <label/classlabels.cuh>

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>

#include <rmm/device_uvector.hpp>

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include "common.h"


#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Extract {

template <typename value_idx>
__global__ void propagate_cluster_negation(const value_idx *indptr,
                                           const value_idx *children,
                                           bool *frontier, bool *is_cluster,
                                           int n_clusters) {
  int cluster = blockDim.x * blockIdx.x + threadIdx.x;

  if (cluster < n_clusters && frontier[cluster]) {
    frontier[cluster] = false;

    value_idx children_start = indptr[cluster];
    value_idx children_stop = indptr[cluster];
    for (int i = 0; i < children_stop - children_start; i++) {
      value_idx child = children[i];
      frontier[child] = true;
      is_cluster[child] = false;
    }
  }
}

template <typename value_t>
struct transform_functor {

public:
  transform_functor(value_t *stabilities_, value_t *births_) :
    stabilities(stabilities_),
    births(births_) {
  }

  __device__ value_t operator()(const int &idx) {
    return stabilities[idx] - births[idx];
  }

 private:
  value_t *stabilities, *births;
};

template <typename value_idx, typename value_t, typename CUBReduceFunc>
void segmented_reduce(const value_t *in, value_t *out, int n_segments,
                      const value_idx *offsets, cudaStream_t stream, CUBReduceFunc cub_reduce_func) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub_reduce_func(d_temp_storage, temp_storage_bytes, in, out, n_segments,
                  offsets, offsets + 1, stream, false);
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub_reduce_func(d_temp_storage, temp_storage_bytes, in, out, n_segments,
                  offsets, offsets + 1, stream, false);
  CUDA_CHECK(cudaFree(d_temp_storage));
}

template <typename value_idx, typename value_t>
void compute_stabilities(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  value_t  *stabilities) {
  auto parents = condensed_tree.get_parents();
  auto children = condensed_tree.get_children();
  auto lambdas = condensed_tree.get_lambdas();
  auto n_edges = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();

  auto stream = handle.get_stream();
  auto thrust_policy = rmm::exec_policy(stream);

  // TODO: Reverse topological sort (e.g. sort hierarchy, lambdas, and sizes by lambda)
  rmm::device_uvector<value_idx> sorted_child(n_edges, stream);
  raft::copy_async(sorted_child.data(), children, n_edges, stream);
  rmm::device_uvector<value_t> sorted_lambdas(n_edges, stream);
  raft::copy_async(sorted_lambdas.data(), lambdas, n_edges, stream);

  rmm::device_uvector<value_idx> sorted_parent(n_edges, stream);
  raft::copy_async(sorted_parent.data(), parents, n_edges, stream);
  thrust::sort_by_key(thrust_policy->on(stream), sorted_parent.begin(), sorted_parent.end(),
                      sorted_child.begin());

  raft::copy_async(sorted_parent.data(), parents, n_edges, stream);
  thrust::sort_by_key(thrust_policy->on(stream), sorted_parent.begin(), sorted_parent.end(),
                      sorted_lambdas.begin());
  // TODO: Segmented reduction on min_lambda within each cluster
  // TODO: Converting child array to CSR offset and using CUB Segmented Reduce
  // Investigate use of a kernel like coo_spmv
  rmm::device_uvector<value_t> births(n_clusters, stream);
  thrust::fill(thrust_policy->on(stream), births.begin(), births.end(), 0.0f);

  rmm::device_uvector<value_idx> sorted_child_offsets(n_edges + 1, stream);

  raft::sparse::convert::sorted_coo_to_csr(
    sorted_child.data(), n_edges, sorted_child_offsets.data(), n_clusters,
    handle.get_device_allocator(), handle.get_stream());

  segmented_reduce(
    lambdas, births.data(), n_clusters, sorted_child_offsets.data(), stream, cub::DeviceSegmentedReduce::Min<const value_t*, value_t*, const value_idx*>);

  // TODO: Embarassingly parallel construction of output
  // TODO: It can be done with same coo_spmv kernel
  // Or naive kernel, atomically write to cluster stability
  thrust::fill(thrust_policy->on(stream), stabilities, stabilities+n_clusters, 0.0f);

  segmented_reduce(
    lambdas, stabilities, n_clusters, sorted_child_offsets.data(),
    stream, cub::DeviceSegmentedReduce::Sum<const value_t*, value_t*, const value_idx*>);

  // now transform, and calculate summation lambda(point) - lambda(birth)
  auto transform_op =
    transform_functor<value_t>(stabilities, births.data());
  thrust::transform(thrust_policy->on(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(n_clusters),
                    stabilities, transform_op);
}

/**
 * Computes the excess of mass. This is a cluster extraction
 * strategy that iterates upwards from the leaves of the cluster
 * tree toward the root, selecting a cluster
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param condensed_tree
 * @param stability
 * @param is_cluster
 * @param n_clusters
 * @param max_cluster_size
 */
template <typename value_idx, typename value_t, int tpb = 256>
void excess_of_mass(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  value_t *stability, bool *is_cluster, value_idx n_clusters,
  value_idx max_cluster_size) {

  cudaStream_t stream = handle.get_stream();

  /**
   * 1. Build CSR of cluster tree from condensed tree by filtering condensed tree for
   *    only those entries w/ lambda > 1 and constructing a CSR from the result
   */

  std::vector<value_idx> cluster_sizes;

  value_idx cluster_tree_edges = thrust::transform_reduce(
    thrust::cuda::par.on(stream), condensed_tree.get_lambdas(),
    condensed_tree.get_lambdas() + condensed_tree.get_n_edges(),
    [=] __device__ (value_t a) {return a > 1.0;}, 0, thrust::plus<value_idx>());

  rmm::device_uvector<value_idx> parents(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> children(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> sizes(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> indptr(n_clusters + 1, stream);

  auto in = thrust::make_zip_iterator(
    thrust::make_tuple(condensed_tree.get_parents(), condensed_tree.get_children(),
                       condensed_tree.get_sizes()));

  auto out = thrust::make_zip_iterator(
    thrust::make_tuple(parents.data(), children.data(), sizes.data()));

  thrust::copy_if(thrust::cuda::par.on(stream), in,
                  in + (condensed_tree.get_n_edges()),
                  condensed_tree.get_lambdas(), out,
                  [=] __device__ (value_t a) {return a > 1.0;});

  raft::sparse::op::coo_sort(
    0, 0, cluster_tree_edges, parents.data(), children.data(), sizes.data(),
    handle.get_device_allocator(), handle.get_stream());

  raft::sparse::convert::sorted_coo_to_csr(
    parents.data(), cluster_tree_edges, indptr.data(), n_clusters + 1,
    handle.get_device_allocator(), handle.get_stream());

  /**
   * 2. Iterate through each level from leaves back to root. Use the cluster
   *    tree CSR and warp-level reduction to sum stabilities and test whether
   *    or not current cluster should continue to be its own
   */

  bool is_cluster_h[n_clusters];
  bool frontier_h[n_clusters];

  std::vector<value_idx> indptr_h(indptr.size());
  raft::update_host(indptr_h.data(), indptr.data(), indptr.size(), stream);
  // don't need to sync here- thrust should take care of it.

  for (value_idx node = 0; node < n_clusters; node++) {
    value_t node_stability;
    raft::update_host(&node_stability, stability + node, 1, stream);

    value_t subtree_stability = thrust::transform_reduce(
      thrust::cuda::par.on(stream), children.data() + indptr_h[node],
      children.data() + indptr_h[node] + 1,
      [=] __device__(value_idx a) { return stability[a]; }, 0,
      thrust::plus<value_t>());

    if (subtree_stability > stability[node] ||
        cluster_sizes[node] > max_cluster_size) {
      // Deselect / merge cluster with children
      raft::update_device(stability + node, &subtree_stability, 1, stream);
      is_cluster[node] = false;
    } else {
      // Mark children to be deselected
      is_cluster_h[node] = false;
    }
  }

  /**
   * 3. Perform BFS through is_cluster, propagating cluster
   * "deselection" through subtrees
   */
  rmm::device_uvector<bool> frontier(n_clusters, stream);
  raft::update_device(is_cluster, is_cluster_h, n_clusters, stream);
  raft::update_device(frontier.data(), frontier_h, n_clusters, stream);

  thrust::transform(thrust::cuda::par.on(stream), is_cluster,
                    is_cluster + n_clusters, frontier.data(),
                    [=] __device__(value_t a) { return !a; });

  value_idx n_elements_to_traverse =
    thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                   frontier.data() + frontier.size(), 0);

  // TODO: Investigate whether it's worth gathering the sparse frontier into
  // a dense form for purposes of uniform workload/thread scheduling

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(frontier.size(), (size_t)tpb);

  while (n_elements_to_traverse > 0) {
    propagate_cluster_negation<<<grid, tpb, 0, stream>>>(
      indptr.data(), children.data(), frontier.data(), is_cluster, n_clusters);

    n_elements_to_traverse =
      thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                     frontier.data() + frontier.size(), 0);
  }
}

template <typename value_idx, typename value_t>
void get_stability_scores(const raft::handle_t &handle, const value_idx *labels,
                          const value_t *stability, const value_idx *clusters,
                          size_t n_clusters, value_t max_lambda,
                          size_t n_leaves, value_t *result) {
  /**
   * 1. Populate cluster sizes
   */
  rmm::device_uvector<value_idx> cluster_sizes(n_clusters, handle.get_stream());
  value_idx *sizes = cluster_sizes.data();
  thrust::for_each(thrust::cuda::par.on(handle.get_stream()), labels,
                   labels + n_leaves,
                   [=] __device__(value_idx v) { atomicAdd(sizes + v, 1); });

  /**
   * Compute stability scores
   */
  auto enumeration = thrust::make_zip_iterator(
    thrust::make_tuple(clusters, cluster_sizes.data()));
  thrust::transform(
    thrust::cuda::par.on(handle.get_stream()), enumeration,
    enumeration + n_clusters, result,
    [=] __device__(thrust::tuple<value_idx, value_idx> tup) {
      value_idx size = thrust::get<1>(tup);
      value_idx c = thrust::get<0>(tup);

      bool expr = max_lambda == std::numeric_limits<value_t>::max() ||
                  max_lambda == 0.0 || size == 0;
      return (!expr * (stability[c] / size * max_lambda)) + (expr * 1.0);
    });
}

template <typename value_idx, typename value_t>
void do_labelling() {
  // TODO: This can be done efficiently on host.
}

template <typename value_idx, typename value_t>
void get_probabilities(const raft::handle_t &handle, value_t *probabilities) {
  // TODO: Compute deaths array similarly to compute_stabilities

  // TODO: Embarassingly parallel
}

template<typename value_idx, typename value_t>
void extract_clusters(const raft::handle_t &handle,
                      Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
                      size_t n_leaves,
                      value_idx *labels,
                      value_t *stabilities,
                      value_t *probabilities) {

  rmm::device_uvector<value_t> tree_stabilities(condensed_tree.get_n_clusters(), handle.get_stream());

  compute_stabilities(handle, condensed_tree, tree_stabilities.data());

  rmm::device_uvector<bool> is_cluster(condensed_tree.get_n_clusters(), handle.get_stream());

  value_idx max_cluster_size = -1; // TODO
  excess_of_mass(handle, condensed_tree, tree_stabilities.data(), is_cluster.data(),
                 condensed_tree.get_n_clusters(), max_cluster_size);

  // TODO: create final clusters array based on excess of mass
  rmm::device_uvector<value_idx> clusters(0, handle.get_stream());

  // TODO: Fill this in
  do_labelling<value_idx, value_t>();

  // TODO: Fill this in
  get_probabilities<value_idx, value_t>(handle, probabilities);


  value_t max_lambda = -1; //TODO Fill this in
  rmm::device_uvector<value_idx> stability_scores(0, handle.get_stream());
  get_stability_scores(handle, labels, tree_stabilities.data(), clusters.data(),
                       clusters.size(), max_lambda, n_leaves, stabilities);
}


};  // end namespace Extract
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
