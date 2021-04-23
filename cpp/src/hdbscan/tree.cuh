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

#include "detail/tree_kernels.cuh"

#include <src_prims/label/classlabels.cuh>

#include <raft/cudart_utils.h>

#include <rmm/device_uvector.hpp>

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace ML {
namespace HDBSCAN {
namespace Tree {

struct Not_Empty {

template <typename value_t>
__host__ __device__ __forceinline__ value_t operator()(value_t a) {
  return a != -1;
}
};


template<typename value_idx, typename value_t>
struct CondensedHierarchy {

  CondensedHierarchy(const raft::handle_t &handle_, value_idx n_leaves_):
               handle(handle_), n_leaves(n_leaves_), parents(0, handle.get_stream()), children(0, handle.get_stream()),
               lambdas(0, handle.get_stream()), sizes(0, handle.get_stream()) {}

  void condense(value_idx *full_parents, value_idx *full_children,
                value_t *full_lambdas, value_idx *full_sizes) {

    auto stream = handle.get_stream();

    n_edges = thrust::transform_reduce(thrust::cuda::par.on(stream),
                                       full_parents, full_parents + (n_leaves * 2),
                                       Not_Empty(), 0, thrust::plus<value_idx>());

    parents.resize(n_edges, stream);
    children.resize(n_edges, stream);
    lambdas.resize(n_edges, stream);
    sizes.resize(n_edges, stream);

    thrust::copy_if(thrust::cuda::par.on(stream), full_parents,
                    full_parents + (n_leaves * 2), parents.data(), Not_Empty());
    thrust::copy_if(thrust::cuda::par.on(stream),
                    full_children, full_children + (n_leaves * 2), children.data(), Not_Empty());
    thrust::copy_if(thrust::cuda::par.on(stream),
                    full_lambdas, full_lambdas + (n_leaves * 2), lambdas.data(), Not_Empty());
    thrust::copy_if(thrust::cuda::par.on(stream),
                    full_sizes, full_sizes + (n_leaves * 2), sizes.data(), Not_Empty());

    n_clusters = MLCommon::Label::make_monotonic(handle, parents.data(), parents.begin(), parents.end());
  }

  value_idx *get_parents() {
    return parents.data();
  }

  value_idx *get_children() {
    return children.data()
  }

  value_t *get_lambdas() {
    return lambdas.data();
  }

  value_idx *get_sizes() {
    return sizes.data();
  }

  value_idx get_n_edges() {
    return n_edges;
  }

  int get_n_clusters() {
    return n_clusters;
  }

 private:
  const raft::handle_t &handle;

  rmm::device_uvector<value_idx> parents;
  rmm::device_uvector<value_idx> children;
  rmm::device_uvector<value_t> lambdas;
  rmm::device_uvector<value_idx> sizes;

  value_idx n_edges;
  value_idx n_leaves;
  int n_clusters;

};

/**
 * Condenses a binary tree dendrogram in the Scipy format
 * by merging labels that fall below a minimum cluster size.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param[in] children
 * @param[in] delta
 * @param[in] sizes
 * @param[in] min_cluster_size
 * @param[in] n_leaves
 * @param[out] out_parent
 * @param[out] out_child
 * @param[out] out_lambda
 * @param[out] out_size
 */
template <typename value_idx, typename value_t, int tpb = 256>
void condense_hierarchy(const raft::handle_t &handle, const value_idx *src,
                        const value_idx *dst,
                        const value_t *delta, const value_idx *sizes,
                        int min_cluster_size, int n_leaves,
                        CondensedHierarchy<value_idx, value_t> &condensed_tree) {

  cudaStream_t stream = handle.get_stream();

  rmm::device_uvector<bool> frontier(n_leaves * 2, stream);
  rmm::device_uvector<bool> ignore(n_leaves * 2, stream);

  rmm::device_uvector<value_idx> out_parent(n_leaves * 2, stream);
  rmm::device_uvector<value_idx> out_child(n_leaves * 2, stream);
  rmm::device_uvector<value_t> out_lambda(n_leaves * 2, stream);
  rmm::device_uvector<value_idx> out_size(n_leaves * 2, stream);

  int root = 2 * n_leaves;
  int num_points = floor(root / 2.0) + 1;

  thrust::fill(thrust::cuda::par.on(stream), out_parent.data(),
               out_parent.data()+(n_leaves*2), -1);
  thrust::fill(thrust::cuda::par.on(stream), out_child.data(),
               out_child.data()+(n_leaves*2), -1);
  thrust::fill(thrust::cuda::par.on(stream), out_lambda.data(),
               out_lambda.data()+(n_leaves*2), -1);
  thrust::fill(thrust::cuda::par.on(stream), out_size.data(),
               out_size.data()+(n_leaves*2), -1);

  rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());
  raft::update_device(relabel.data()+root, root, 1, handle.get_stream());

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(n_leaves * 2, (size_t)tpb);

  value_idx n_elements_to_traverse =
    thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                   frontier.data() + (n_leaves * 2), 0);

  while (n_elements_to_traverse > 0) {
    detail::condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(
      frontier.data(), ignore.data(), next_label.data(), relabel.data(),
      src, dst, delta, sizes, n_leaves, num_points, min_cluster_size);

    n_elements_to_traverse =
      thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                     frontier.data() + (n_leaves * 2), 0);
  }

  // TODO: Normalize labels so they are drawn from a monotonically increasing set.

  condensed_tree.condense(out_parent.data(), out_child.data(),
                          out_lambda.data(), out_size.data());

}

template<typename value_t>
struct transform_functor {

public:
  transform_op(value_t *stabilities_, value_t *births_) :
    stabilities(stabilities_),
    births(births_) {

  }

  __device__ value_t operator()(const &idx) {
    return stabilities[idx] - births[idx];
  }

private:
  value_t *stabilities, *births;
};

template <typename value_idx, typename value_t, typename cub_reduce_func>
void segmented_reduce(const value_t *in, value_t *out, const value_idx *offsets, cudaStream_t stream) {
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub_reduce_func(d_temp_storage, temp_storage_bytes, in, out,
    n_clusters, offsets, offsets + 1, stream);
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub_reduce_func(d_temp_storage, temp_storage_bytes, in, out,
    n_clusters, offsets, offsets + 1, stream);
  CUDA_CHECK(cudaFree(d_temp_storage));
}

template <typename value_idx, typename value_t>
void compute_stabilities(const raft::handle_t &handle,
  const CondensedHierarchy<value_idx, value_t> &condensed_tree,
  rmm::device_uvector<value_t> &stabilities) {

  auto parents = condensed_tree.get_parents();
  auto children = condensed_tree.get_children();
  auto lambdas = condensed_tree.get_lambdas();
  auto n_edges = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();

  auto stream = handle.get_stream();
  auto thrust_policy = rmm::exec_policy(stream);

  // TODO: Reverse topological sort (e.g. sort hierarchy, lambdas, and sizes by lambda)
  rmm::device_uvector<value_idx> sorted_child(condensed_child, n_edges, stream);
  rmm::device_uvector<value_t> sorted_lambdas(lambdas, n_edges, stream);

  auto children_lambda_zip = thrust::make_zip_iterator(thrust::make_tuple(sorted_child.begin(), sorted_lambdas.begin()));
  thrust::sort_by_key(policy, parents, parents + n_edges, children_lambda_zip);

  // TODO: sort hierarchy, lambdas, and sizes by lambda

  // TODO: Segmented reduction on min_lambda within each cluster
  // TODO: Converting child array to CSR offset and using CUB Segmented Reduce
  // Investigate use of a kernel like coo_spmv
  rmm::device_uvector<value_idx> births(n_clusters, stream);
  thrust::fill(thrust_policy, births.begin(), births.end(), 0);

  rmm::device_uvector<value_idx> sorted_child_offsets(n_edges + 1, stream);

  raft::sparse::convert::sorted_coo_to_csr(sorted_child.data(), n_edges, sorted_child_offsets.data(), n_clusters, handle.get_stream(), handle.get_device_allocator());

  segmented_reduce<value_idx, value_t, cub::DeviceSegmentedReduce::Min>(lambdas, births.data(), sorted_child_offsets.data(), stream);

  // TODO: Embarassingly parallel construction of output
  // TODO: It can be done with same coo_spmv kernel
  // Or naive kernel, atomically write to cluster stability
  thrust::fill(thrust_policy, stabilities.begin(), stabilities.end(), 0);

  segmented_reduce<value_idx, value_t, cub::DeviceSegmentedReduce::Sum>(lambdas, stabilities.data(), sorted_child_offsets.data(), stream);

  // now transform, and calculate summation lambda(point) - lambda(birth)
  auto transform_op = transform_functor<value_t>(stabilities.data(), birth.data());
  thrust::transform(policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_clusters), stabilities.begin(), transform_op);

  return stabilities;
}

struct Greater_Than_One {

  template <typename value_t>
  __host__ __device__ __forceinline__ value_t operator()(value_t a) {
    return a > 1;
  }
};

template<typename value_idx, typename value_t>
void excess_of_mass(const raft::handle_t &handle,
                    const CondensedHierarchy<value_idx, value_t> &condensed_tree,
                    value_t *stability, bool *is_cluster, value_idx n_clusters) {

  /**
   * - If the sum of the stabilities of the child clusters is greater than the
   * stability of the cluster, then we set the cluster stability to be the
   * sum of the child stabilities.
   * - If, on the other hand, the cluster’s stability is greater than the sum
   * of its children then we declare the  cluster to be a selected cluster
   * and unselect all its descendants.
   * - Once we reach the root node we call the current set of selected clusters
   * our flat clustering and return that.
   */

  cudaStream_t stream = handle.get_stream();

  /**
   * 1. Build CSR of cluster tree from condensed tree by filtering condensed tree for
   *    only those entries w/ lambda > 1 and constructing a CSR from the result
   */


  value_idx cluster_tree_edges = thrust::transform_reduce(thrust::cuda::par.on(stream),
                                     condensed_tree.get_lambdas(),
                                     condensed_tree.get_lambdas() + condensed_tree.get_n_edges(),
                                     Greater_Than_One(), 0, thrust::plus<value_idx>());

  rmm::device_uvector<value_idx> parents(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> children(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> sizes(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> indptr(n_clusters, stream);

  thrust::copy_if(thrust::cuda::par.on(stream), condensed_tree.get_parents(),
                  condensed_tree.get_parents() + (condensed_tree.get_n_edges()), condensed_tree.get_lambdas(),
                  parents.data(), Greater_Than_One());

  thrust::copy_if(thrust::cuda::par.on(stream), condensed_tree.get_children(),
                  condensed_tree.get_children() + (condensed_tree.get_n_edges()), condensed_tree.get_lambdas(),
                  children.data(), Greater_Than_One());

  thrust::copy_if(thrust::cuda::par.on(stream), condensed_tree.get_sizes(),
                  condensed_tree.get_sizes() + (condensed_tree.get_n_edges()), condensed_tree.get_lambdas(),
                  sizes.data(), Greater_Than_One());

  raft::sparse::op::coo_sort(0, 0, cluster_tree_edges, parents.data(), children.data(), sizes.data(),
                             handle.get_device_allocator(), handle.get_stream());

  raft::sparse::convert::sorted_coo_to_csr(parents.data(), cluster_tree_edges, indptr.data(), n_clusters,
                                           handle.get_device_allocator(), handle.get_stream());

  /**
   * 2. Iterate through each level from leaves back to root. Use the cluster
   *    tree CSR and warp-level reduction to sum stabilities and test whether
   *    or not current cluster should continue to be its own
   */
   /**
    * Copy indptr to host
    * For each node in sorted stability keys,
    *    - transformed reducet
    */

  /**
   * 3. Perform BFS through is_cluster, propagating cluster "deselection" to leaves
   */

}

template<typename value_idx, typename value_t>
void get_stability_scores() {

  // TODO: Perform segmented reduction to compute cluster_size

  // TODO: Embarassingly parallel
}

template<typename value_idx, typename value_t>
void do_labelling() {

  // TODO: Similar to SLHC dendrogram construction, this one is probably best done
  // on host, at least for the first iteration
}


template<typename value_idx, typename value_t>
void get_probabilities() {

  // TODO: Compute deaths array similarly to compute_stabilities

  // TODO: Embarassingly parallel
}

};  // end namespace Tree
};  // end namespace HDBSCAN
};  // end namespace ML