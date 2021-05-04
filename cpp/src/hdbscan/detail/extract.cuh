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

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.cuh>

#include <algorithm>

#include <hdbscan/condensed_hierarchy.cu>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Extract {

template <typename value_idx>
class TreeUnionFind {
 public:
  TreeUnionFind(value_idx size) : data(size * 2, 0) {
    for (int i = 0; i < size; i++) {
      data[i * 2] = i;
    }
  }

  void perform_union(value_idx x, value_idx y) {
    value_idx x_root = find(x);
    value_idx y_root = find(y);

    if (data[x_root * 2 + 1] < data[y_root * 2 + 1])
      data[x_root * 2] = y_root;
    else if (data[x_root * 2 + 1] > data[y_root * 2 + 1])
      data[y_root * 2] = x_root;
    else {
      data[y_root * 2] = x_root;
      data[x_root * 2 + 1] += 1;
    }
  }

  value_idx find(value_idx x) {
    if (data[x * 2] != x) data[x * 2] = find(data[x * 2]);

    return data[x * 2];
  }

  value_idx *get_data() { return data.data(); }

 private:
  std::vector<value_idx> data;
};

template <typename value_idx>
__global__ void propagate_cluster_negation(const value_idx *indptr,
                                           const value_idx *children,
                                           int *frontier, int *is_cluster,
                                           int n_clusters) {
  int cluster = blockDim.x * blockIdx.x + threadIdx.x;

  if (cluster < n_clusters && frontier[cluster]) {
    frontier[cluster] = false;

    value_idx children_start = indptr[cluster];
    value_idx children_stop = indptr[cluster + 1];
    for (int i = children_start; i < children_stop; i++) {
      value_idx child = children[i];
      frontier[child] = true;
      is_cluster[child] = false;
    }
  }
}

template <typename value_idx, typename value_t, typename CUBReduceFunc>
void segmented_reduce(const value_t *in, value_t *out, int n_segments,
                      const value_idx *offsets, cudaStream_t stream,
                      CUBReduceFunc cub_reduce_func) {
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
struct stabilities_functor {
 public:
  stabilities_functor(value_t *stabilities_, const value_t *births_,
                      const value_idx *parents_, const value_idx *children_,
                      const value_t *lambdas_, const value_idx *sizes_,
                      const value_idx n_leaves_)
    : stabilities(stabilities_),
      births(births_),
      parents(parents_),
      children(children_),
      lambdas(lambdas_),
      sizes(sizes_),
      n_leaves(n_leaves_) {}

  __device__ void operator()(const int &idx) {
    auto parent = parents[idx] - n_leaves;

    atomicAdd(&stabilities[parent],
              (lambdas[idx] - births[parent]) * sizes[idx]);
  }

 private:
  value_t *stabilities;
  const value_t *births, *lambdas;
  const value_idx *parents, *children, *sizes, n_leaves;
};

template <typename value_idx, typename value_t>
void compute_stabilities(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  value_t *stabilities) {
  auto parents = condensed_tree.get_parents();
  auto children = condensed_tree.get_children();
  auto lambdas = condensed_tree.get_lambdas();
  auto sizes = condensed_tree.get_sizes();
  auto n_edges = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves = condensed_tree.get_n_leaves();

  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  // TODO: Reverse topological sort (e.g. sort hierarchy, lambdas, and sizes by lambda)
  rmm::device_uvector<value_t> sorted_lambdas(n_edges, stream);
  raft::copy_async(sorted_lambdas.data(), lambdas, n_edges, stream);

  rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  thrust::sort_by_key(exec_policy, sorted_parents.begin(), sorted_parents.end(),
                      sorted_lambdas.begin());

  // 0-index sorted parents by subtracting n_leaves for offsets and birth/stability indexing
  auto index_op = [n_leaves] __device__(const auto &x) { return x - n_leaves; };
  thrust::transform(exec_policy, sorted_parents.begin(), sorted_parents.end(),
                    sorted_parents.begin(), index_op);

  rmm::device_uvector<value_idx> sorted_parents_offsets(n_edges + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    sorted_parents.data(), n_edges, sorted_parents_offsets.data(),
    n_clusters + 1, handle.get_device_allocator(), handle.get_stream());

  // Segmented reduction on min_lambda within each cluster
  // TODO: Converting child array to CSR offset and using CUB Segmented Reduce
  // Investigate use of a kernel like coo_spmv

  // This is to consider the case where a child may also be a parent
  // in which case, births for that parent are initialized to
  // lambda for that child
  rmm::device_uvector<value_t> births(n_clusters, stream);
  thrust::fill(exec_policy, births.begin(), births.end(), 0.0f);
  auto births_init_op = [n_leaves, children, lambdas,
                         births = births.data()] __device__(const auto &idx) {
    auto child = children[idx];
    if (child >= n_leaves) {
      births[child - n_leaves] = lambdas[idx];
    }
  };

  // this is to find minimum lambdas of all children under a prent
  rmm::device_uvector<value_t> births_parent_min(n_clusters, stream);
  thrust::fill(exec_policy, births.begin(), births.end(), 0.0f);
  thrust::for_each(exec_policy, thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges), births_init_op);
  segmented_reduce(sorted_lambdas.data(), births_parent_min.data() + 1,
                   n_clusters - 1, sorted_parents_offsets.data() + 1, stream,
                   cub::DeviceSegmentedReduce::Min<const value_t *, value_t *,
                                                   const value_idx *>);

  // finally, we find minimum between initialized births where parent=child
  // and births of parents for their childrens
  auto births_zip = thrust::make_zip_iterator(
    thrust::make_tuple(births.data(), births_parent_min.data()));
  auto min_op =
    [] __device__(const thrust::tuple<value_t, value_t> &birth_pair) {
      auto birth = thrust::get<0>(birth_pair);
      auto births_parent_min = thrust::get<1>(birth_pair);

      return birth < births_parent_min ? birth : births_parent_min;
    };
  thrust::transform(exec_policy, births_zip, births_zip + n_clusters,
                    births.begin(), min_op);
  raft::print_device_vector("Births", births.data(), n_clusters, std::cout);

  thrust::fill(exec_policy, stabilities, stabilities + n_clusters, 0.0f);

  // for each child, calculate summation (lambda[child] - lambda[birth[parent]]) * sizes[child]
  stabilities_functor<value_idx, value_t> stabilities_op(
    stabilities, births.data(), parents, children, lambdas, sizes, n_leaves);
  thrust::for_each(exec_policy, thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges), stabilities_op);

  raft::print_device_vector("Stabilities", stabilities, n_clusters, std::cout);
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
 * @param stability an array of nodes from the cluster tree and their
 *                  corresponding stabilities
 * @param is_cluster
 * @param n_clusters
 * @param max_cluster_size
 */
template <typename value_idx, typename value_t, int tpb = 256>
void excess_of_mass(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  value_t *stability, int *is_cluster, value_idx n_clusters,
  value_idx max_cluster_size) {
  cudaStream_t stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  /**
   * 1. Build CSR of cluster tree from condensed tree by filtering condensed tree for
   *    only those entries w/ lambda > 1 and constructing a CSR from the result
   */
  value_idx cluster_tree_edges = thrust::transform_reduce(
    exec_policy, condensed_tree.get_sizes(),
    condensed_tree.get_sizes() + condensed_tree.get_n_edges(),
    [=] __device__(value_t a) { return a > 1.0; }, 0,
    thrust::plus<value_idx>());

  rmm::device_uvector<value_idx> parents(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> children(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> sizes(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> indptr(n_clusters + 1, stream);
  rmm::device_uvector<value_idx> cluster_sizes(n_clusters, stream);

  thrust::fill(exec_policy, cluster_sizes.data(),
               cluster_sizes.data() + cluster_sizes.size(), 0);

  auto in = thrust::make_zip_iterator(thrust::make_tuple(
    condensed_tree.get_parents(), condensed_tree.get_children(),
    condensed_tree.get_sizes()));

  value_idx n_leaves = condensed_tree.get_n_leaves();

  auto out = thrust::make_zip_iterator(
    thrust::make_tuple(parents.data(), children.data(), sizes.data()));

  thrust::copy_if(exec_policy, in, in + (condensed_tree.get_n_edges()),
                  condensed_tree.get_sizes(), out,
                  [=] __device__(value_t a) { return a > 1.0; });

  value_idx *cluster_sizes_ptr = cluster_sizes.data();

  thrust::transform(exec_policy, parents.data(),
                    parents.data() + parents.size(), parents.data(),
                    [=] __device__(value_idx a) { return a - n_leaves; });
  thrust::transform(exec_policy, children.data(),
                    children.data() + children.size(), children.data(),
                    [=] __device__(value_idx a) { return a - n_leaves; });

  thrust::for_each(
    exec_policy, out, out + children.size(),
    [=] __device__(const thrust::tuple<value_idx, value_idx, value_idx> &tup) {
      cluster_sizes_ptr[thrust::get<1>(tup)] = thrust::get<2>(tup);
    });

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
  std::vector<int> is_cluster_h(n_clusters, true);
  std::vector<int> frontier_h(n_clusters, false);
  std::vector<value_idx> cluter_sizes_h(n_clusters);

  std::vector<value_idx> indptr_h(indptr.size());
  raft::update_host(indptr_h.data(), indptr.data(), indptr.size(), stream);
  raft::update_host(cluter_sizes_h.data(), cluster_sizes.data(),
                    cluster_sizes.size(), stream);
  // don't need to sync here- thrust should take care of it.

  // Loop through stabilities in "reverse topological order" (e.g. reverse sorted order)
  for (value_idx node = n_clusters - 1; node >= 0; node--) {
    value_t node_stability = 0;
    raft::update_host(&node_stability, stability + node, 1, stream);

    value_t subtree_stability = 0;

    if (indptr_h[node + 1] - indptr_h[node] > 0) {
      subtree_stability = thrust::transform_reduce(
        exec_policy, children.data() + indptr_h[node],
        children.data() + indptr_h[node + 1],
        [=] __device__(value_idx a) { return stability[a]; }, 0,
        thrust::plus<value_t>());
    }

    if (subtree_stability > node_stability ||
        cluter_sizes_h[node] > max_cluster_size) {
      // Deselect / merge cluster with children
      raft::update_device(stability + node, &subtree_stability, 1, stream);
      is_cluster_h[node] = false;
    } else {
      // Mark children to be deselected
      frontier_h[node] = true;
    }
  }

  /**
   * 3. Perform BFS through is_cluster, propagating cluster
   * "deselection" through subtrees
   */
  rmm::device_uvector<int> cluster_propagate(n_clusters, stream);
  rmm::device_uvector<int> frontier(n_clusters, stream);

  raft::update_device(is_cluster, is_cluster_h.data(), n_clusters, stream);
  raft::update_device(frontier.data(), frontier_h.data(), n_clusters, stream);

  value_idx n_elements_to_traverse = thrust::reduce(
    exec_policy, frontier.data(), frontier.data() + frontier.size(), 0);

  // TODO: Investigate whether it's worth gathering the sparse frontier into
  // a dense form for purposes of uniform workload/thread scheduling

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(frontier.size(), (size_t)tpb);

  while (n_elements_to_traverse > 0) {
    propagate_cluster_negation<<<grid, tpb, 0, stream>>>(
      indptr.data(), children.data(), frontier.data(), is_cluster, n_clusters);

    n_elements_to_traverse = thrust::reduce(
      exec_policy, frontier.data(), frontier.data() + frontier.size(), 0);
  }
}

template <typename value_idx, typename value_t>
void get_stability_scores(const raft::handle_t &handle, const value_idx *labels,
                          const value_t *stability, size_t n_clusters,
                          value_t max_lambda, size_t n_leaves,
                          value_t *result) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  /**
   * 1. Populate cluster sizes
   */
  rmm::device_uvector<value_idx> cluster_sizes(n_clusters, handle.get_stream());
  value_idx *sizes = cluster_sizes.data();
  thrust::for_each(exec_policy, labels, labels + n_leaves,
                   [=] __device__(value_idx v) { atomicAdd(sizes + v, 1); });

  /**
   * Compute stability scores
   */
  auto enumeration = thrust::make_zip_iterator(thrust::make_tuple(
    thrust::make_counting_iterator(0), cluster_sizes.data()));
  thrust::transform(
    exec_policy, enumeration, enumeration + n_clusters, result,
    [=] __device__(thrust::tuple<value_idx, value_idx> tup) {
      value_idx size = thrust::get<1>(tup);
      value_idx c = thrust::get<0>(tup);

      bool expr = max_lambda == std::numeric_limits<value_t>::max() ||
                  max_lambda == 0.0 || size == 0;
      return (!expr * (stability[c] / size * max_lambda)) + (expr * 1.0);
    });
}

template <typename value_idx, typename value_t>
void do_labelling_on_host(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  std::set<value_idx> &clusters, value_idx n_leaves, bool allow_single_cluster,
  value_idx *labels) {
  auto stream = handle.get_stream();

  std::vector<value_idx> children_h(condensed_tree.get_n_edges());
  std::vector<value_t> lambda_h(condensed_tree.get_n_edges());
  std::vector<value_idx> parent_h(condensed_tree.get_n_edges());

  raft::update_host(children_h.data(), condensed_tree.get_children(),
                    condensed_tree.get_n_edges(), stream);
  raft::update_host(parent_h.data(), condensed_tree.get_parents(),
                    condensed_tree.get_n_edges(), stream);
  raft::update_host(lambda_h.data(), condensed_tree.get_lambdas(),
                    condensed_tree.get_n_edges(), stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx size = *std::max_element(parent_h.begin(), parent_h.end());

  std::vector<value_idx> result(n_leaves);
  std::vector<value_t> parent_lambdas(size + 1, 0);

  auto union_find = TreeUnionFind<value_idx>(size + 1);

  for (int i = 0; i < condensed_tree.get_n_edges(); i++) {
    value_idx child = children_h[i];
    value_idx parent = parent_h[i];

    if (clusters.find(child) == clusters.end()) {
      union_find.perform_union(parent, child);
    }

    parent_lambdas[parent_h[i]] = max(parent_lambdas[parent_h[i]], lambda_h[i]);
  }

  for (int i = 0; i < n_leaves; i++) {
    value_idx cluster = union_find.find(i);

    if (cluster < n_leaves)
      result[i] = -1;
    else if (cluster == n_leaves) {
      //TODO: Implement the cluster_selection_epsilon / epsilon_search
      if (clusters.size() == 1 && allow_single_cluster) {
        auto it = std::find(children_h.begin(), children_h.end(), i);
        auto child_idx = std::distance(children_h.begin(), it);
        value_idx child_lambda = lambda_h[child_idx];
        if (child_lambda >= parent_lambdas[cluster])
          result[i] = cluster - n_leaves;
        else
          result[i] = -1;
      } else {
        result[i] = -1;
      }
    } else {
      result[i] = cluster - n_leaves;
    }
  }

  raft::update_device(labels, result.data(), n_leaves, stream);
}

/**
 * Extracts flattened labels using a cut point (epsilon) and minimum cluster
 * size. This is useful for Robust Single Linkage and DBSCAN labeling.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param children
 * @param deltas
 * @param sizes
 * @param n_leaves
 * @param cut
 * @param min_cluster_size
 * @param labels
 */
template <typename value_idx, typename value_t>
void do_labelling_at_cut(const raft::handle_t &handle,
                         const value_idx *children, const value_t *deltas,
                         value_idx n_leaves, double cut,
                         int min_cluster_size, value_idx *labels) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  // Root is the last edge in the dendrogram
  value_idx root = 2 * (n_leaves - 1);
  value_idx num_points = root / 2 + 1;

  std::vector<value_idx> labels_h(n_leaves);

  auto union_find = TreeUnionFind<value_idx>(root + 1);

  std::vector<value_idx> children_h(n_leaves * 2);
  std::vector<value_t> delta_h(n_leaves);

  raft::update_host(children_h.data(), children, n_leaves * 2, stream);
  raft::update_host(delta_h.data(), deltas, n_leaves, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx cluster = num_points;

  // Perform union on host to label parents / clusters
  for (int row = 0; row < n_leaves; row++) {
    if (delta_h[row] < cut) {
      union_find.perform_union(children_h[row * 2], cluster);
      union_find.perform_union(children_h[row * 2 + 1], cluster);
    }
    cluster += 1;
  }

  // Label points in parallel
  rmm::device_uvector<value_idx> union_find_data((root + 1) * 2, stream);
  raft::update_device(union_find_data.data(), union_find.get_data(),
                      union_find_data.size(), stream);

  rmm::device_uvector<value_idx> cluster_sizes(cluster, stream);
  thrust::fill(exec_policy, cluster_sizes.data(), cluster_sizes.data(), 0);

  auto seq = thrust::make_counting_iterator<value_idx>(0);

  value_idx *union_find_data_ptr = union_find_data.data();
  value_idx *cluster_sizes_ptr = cluster_sizes.data();

  thrust::for_each(exec_policy, seq, seq + n_leaves,
                   [=] __device__(value_idx leaf) {
                     // perform find using tree-union find
                     value_idx cur_find = union_find_data_ptr[leaf * 2];
                     while (cur_find != leaf)
                       cur_find = union_find_data_ptr[cur_find * 2];

                     labels[leaf] = cur_find;
                     atomicAdd(cluster_sizes_ptr + cur_find, 1);
                   });

  // Label noise points
  thrust::transform(exec_policy, labels, labels + n_leaves, labels,
                    [=] __device__(value_idx cluster) {
                      bool too_small =
                        cluster_sizes_ptr[cluster] < min_cluster_size;
                      return (too_small * -1) + (!too_small * cluster);
                    });

  // Draw non-noise points from a monotonically increasing set
  raft::label::make_monotonic(
    labels, labels, n_leaves, stream,
    [] __device__(value_idx label) { return label == -1; },
    handle.get_device_allocator(), true);
}

template <typename value_idx, typename value_t>
struct probabilities_functor {
 public:
  probabilities_functor(value_t *probabilities_, const value_t *deaths_,
                        const value_idx *parents_, const value_idx *children_,
                        const value_t *lambdas_, const value_idx *labels_,
                        const value_idx root_cluster_)
    : probabilities(probabilities_),
      deaths(deaths_),
      parents(parents_),
      children(children_),
      lambdas(lambdas_),
      labels(labels_),
      root_cluster(root_cluster_) {}

  __device__ void operator()(const value_idx &idx) {
    auto child = children[idx];

    // intermediate nodes
    if (child >= root_cluster) {
      return;
    }

    auto cluster = labels[child];

    // noise
    if (cluster == -1) {
      return;
    }

    auto cluster_death = deaths[cluster];
    auto child_lambda = lambdas[idx];
    if (cluster_death == 0.0 || isnan(child_lambda)) {
      probabilities[child] = 1.0;
    } else {
      auto min_lambda = min(child_lambda, cluster_death);
      probabilities[child] = min_lambda / cluster_death;
    }
  }

 private:
  value_t *probabilities;
  const value_t *deaths, *lambdas;
  const value_idx *parents, *children, *labels, root_cluster;
};

template <typename value_idx, typename value_t>
void get_probabilities(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  const value_idx *labels, value_t *probabilities) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  auto parents = condensed_tree.get_parents();
  auto children = condensed_tree.get_children();
  auto lambdas = condensed_tree.get_lambdas();
  auto n_edges = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves = condensed_tree.get_n_leaves();

  rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  rmm::device_uvector<value_t> sorted_lambdas(n_edges, stream);
  raft::copy_async(sorted_lambdas.data(), lambdas, n_edges, stream);

  thrust::sort_by_key(exec_policy, sorted_parents.begin(), sorted_parents.end(),
                      sorted_lambdas.begin());

  // 0-index sorted parents by subtracting n_leaves for offsets and birth/stability indexing
  auto index_op = [n_leaves] __device__(const auto &x) { return x - n_leaves; };
  thrust::transform(exec_policy, sorted_parents.begin(), sorted_parents.end(),
                    sorted_parents.begin(), index_op);

  rmm::device_uvector<value_idx> sorted_parents_offsets(n_edges + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    sorted_parents.data(), n_edges, sorted_parents_offsets.data(),
    n_clusters + 1, handle.get_device_allocator(), handle.get_stream());

  // this is to find maximum lambdas of all children under a prent
  rmm::device_uvector<value_t> deaths(n_clusters, stream);
  thrust::fill(exec_policy, deaths.begin(), deaths.end(), 0.0f);

  segmented_reduce(sorted_lambdas.data(), deaths.data(), n_clusters,
                   sorted_parents_offsets.data(), stream,
                   cub::DeviceSegmentedReduce::Max<const value_t *, value_t *,
                                                   const value_idx *>);
  raft::print_device_vector("Deaths", deaths.data(), n_clusters, std::cout);

  // Calculate probability per point
  thrust::fill(exec_policy, probabilities, probabilities + n_leaves, 0.0f);

  std::cout << "root cluster: " << n_leaves << std::endl;
  probabilities_functor<value_idx, value_t> probabilities_op(
    probabilities, deaths.data(), parents, children, lambdas, labels, n_leaves);
  thrust::for_each(exec_policy, thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges), probabilities_op);

  raft::print_device_vector("Probabilities", probabilities, n_leaves,
                            std::cout);
}

template <typename value_idx, typename value_t>
void extract_clusters(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  size_t n_leaves, value_idx *labels, value_t *stabilities,
  value_t *probabilities, bool allow_single_cluster = true,
  value_idx max_cluster_size = 0) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  rmm::device_uvector<value_t> tree_stabilities(condensed_tree.get_n_clusters(),
                                                handle.get_stream());

  compute_stabilities(handle, condensed_tree, tree_stabilities.data());

  rmm::device_uvector<int> is_cluster(condensed_tree.get_n_clusters(),
                                      handle.get_stream());

  if (max_cluster_size <= 0)
    max_cluster_size = n_leaves;  // negates the max cluster size

  excess_of_mass(handle, condensed_tree, tree_stabilities.data(),
                 is_cluster.data(), condensed_tree.get_n_clusters(),
                 max_cluster_size);

  std::vector<int> is_cluster_h(is_cluster.size());
  raft::update_host(is_cluster_h.data(), is_cluster.data(), is_cluster_h.size(),
                    stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::set<value_idx> clusters;
  for (int i = 0; i < is_cluster_h.size(); i++)
    if (is_cluster_h[i] != 0) clusters.insert(i + n_leaves);

  do_labelling_on_host<value_idx, value_t>(
    handle, condensed_tree, clusters, n_leaves, allow_single_cluster, labels);

  get_probabilities<value_idx, value_t>(handle, condensed_tree, labels,
                                        probabilities);

  raft::label::make_monotonic(labels, labels, n_leaves, stream,
                              handle.get_device_allocator(), true);

  value_t max_lambda = *(thrust::max_element(
    exec_policy, condensed_tree.get_lambdas(),
    condensed_tree.get_lambdas() + condensed_tree.get_n_edges()));

  get_stability_scores(handle, labels, tree_stabilities.data(), clusters.size(),
                       max_lambda, n_leaves, stabilities);
}

};  // end namespace Extract
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
