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

#include <raft/cudart_utils.h>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace ML {
namespace HDBSCAN {
namespace Tree {

template <typename value_idx, typename value_t>
__device__ value_t get_lambda(value_idx node, value_idx num_points,
                              value_t *deltas) {
  value_t delta = deltas[node - num_points];
  if (delta > 0.0) return 1.0 / delta;
  return std::numeric_limits<value_t>::max();
}

/**
 *
 * @tparam value_idx
 * @tparam value_t
 * @param frontier
 * @param ignore Should be initialized to -1
 * @param next_label
 * @param relabel
 * @param hierarchy
 * @param deltas
 * @param sizes
 * @param n_leaves
 * @param num_points
 * @param min_cluster_size
 */
template <typename value_idx, typename value_t>
__global__ void condense_hierarchy_kernel(
  bool *frontier, value_idx *ignore, value_idx *relabel,
  const value_idx *hierarchy, const value_t *deltas,
  const value_idx *sizes, int n_leaves,
  int num_points, int min_cluster_size,
  value_idx *out_parent, value_idx *out_child,
  value_t *out_lambda, value_idx *out_count) {

  int node = blockDim.x * blockIdx.x + threadIdx.x;

  // If node is in frontier, flip frontier for children
  if(node > n_leaves * 2 || !frontier[node])
    return;

  frontier[node] = false;

  // TODO: Check bounds
  value_idx left_child = hierarchy[(node - num_points) * 2];
  value_idx right_child = hierarchy[((node - num_points) * 2) + 1];

  frontier[left_child] = true;
  frontier[right_child] = true;

  bool ignore_val = ignore[node];
  bool should_ignore = ignore_val > -1;

  // If the current node is being ignored (e.g. > -1) then propagate the ignore
  // to children, if any
  ignore[left_child] = (should_ignore * ignore_val) + (!should_ignore * -1);
  ignore[right_child] = (should_ignore * ignore_val) + (!should_ignore * -1);

  if (node < num_points) {
    out_parent[node] = relabel[should_ignore];
    out_child[node] = node;
    out_lambda[node] = get_lambda(should_ignore, num_points, deltas);
    out_count[node] = 1;
  }

  // If node is not ignored and is not a leaf, condense its children
  // if necessary
  else if (!should_ignore and node >= num_points) {
    value_idx left_child = hierarchy[(node - num_points) * 2];
    value_idx right_child = hierarchy[((node - num_points) * 2) + 1];

    value_t lambda_value = get_lambda(node, num_points, deltas);

    int left_count =
      left_child >= num_points ? sizes[left_child - num_points] : 1;
    int right_count =
      right_child >= num_points ? sizes[right_child - num_points] : 1;

    // If both children are large enough, they should be relabeled and
    // included directly in the output hierarchy.
    if (left_count >= min_cluster_size && right_count >= min_cluster_size) {
      relabel[left_child] = node;
      out_parent[node] = relabel[node];
      out_child[node] = relabel[left_child];
      out_lambda[node] = lambda_value;
      out_count[node] = left_count;

      relabel[right_child] = node;
      out_parent[node] = relabel[node];
      out_child[node] = relabel[right_child];
      out_lambda[node] = lambda_value;
      out_count[node] = left_count;
    }

    // Consume left or right child as necessary
    bool left_child_too_small = left_count < min_cluster_size;
    bool right_child_too_small = right_count < min_cluster_size;
    ignore[left_child] =
      (left_child_too_small * node) + (!left_child_too_small * -1);
    ignore[right_child] =
      (right_child_too_small * node) + (!right_child_too_small * -1);

    // If only left or right child is too small, consume it and relabel the other
    // (to it can be its own cluster)
    bool only_left_child_too_small =
      left_child_too_small && !right_child_too_small;
    bool only_right_child_too_small =
      !left_child_too_small && right_child_too_small;

    relabel[right_child] = (only_left_child_too_small * relabel[node]) +
                           (!only_left_child_too_small * -1);
    relabel[left_child] = (only_right_child_too_small * relabel[node]) +
                          (!only_right_child_too_small * -1);
  }
}

struct Not_Empty {

template <typename value_t>
__host__ __device__ __forceinline__ value_t operator()(value_t a) {
  return a != -1;
}
};


template<typename value_idx, typename value_t>
struct CondensedHierarchy {

  CondensedHierarchy(value_idx n_leaves_, cudaStream_t stream_):
               n_leaves(n_leaves_), parents(0, stream_), children(0, stream_),
               lambdas(0, stream_), sizes(0, stream_) {}

  void condense(value_idx *full_parents, value_idx *full_children,
                value_t *full_lambdas, value_idx *full_sizes) {

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

 private:
  rmm::device_uvector<value_idx> parents;
  rmm::device_uvector<value_idx> children;
  rmm::device_uvector<value_t> lambdas;
  rmm::device_uvector<value_idx> sizes;

  cudaStream_t stream;
  value_idx n_edges;
  value_idx n_leaves;

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
void condense_hierarchy(raft::handle_t &handle, const value_idx *children,
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

  thrust::fill(thrust::cuda::par.on(stream), out_parent.data(), out_parent.data()+(n_leaves*2), -1);
  thrust::fill(thrust::cuda::par.on(stream), out_child.data(), out_parent.data()+(n_leaves*2), -1);
  thrust::fill(thrust::cuda::par.on(stream), out_lambda.data(), out_parent.data()+(n_leaves*2), -1);
  thrust::fill(thrust::cuda::par.on(stream), out_size.data(), out_parent.data()+(n_leaves*2), -1);

  rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());
  raft::update_device(relabel.data()+root, root, 1, handle.get_stream());

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(n_leaves * 2, (size_t)tpb);

  value_idx n_elements_to_traverse =
    thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                   frontier.data() + (n_leaves * 2), 0);

  while (n_elements_to_traverse > 0) {
    condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(
      frontier.data(), ignore.data(), next_label.data(), relabel.data(),
      children, delta, sizes, n_leaves, num_points, min_cluster_size);

    n_elements_to_traverse =
      thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                     frontier.data() + (n_leaves * 2), 0);
  }

  // TODO: Normalize labels so they are drawn from a monotonically increasing set.

  condensed_tree.condense(out_parent.data(), out_child.data(), out_lambda.data(), out_size.data());
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

template<typename value_idx, typename value_t>
rmm::device_uvector<value_t> compute_stabilities(value_idx *condensed_parent,
                         value_idx *condensed_child,
                         value_t *lambdas,
                         value_idx *sizes,
                         int n_points,
                         int n_leaves) {

  auto thrust_policy = rmm::exec_policy(stream);

  // TODO: Reverse topological sort (e.g. sort hierarchy, lambdas, and sizes by lambda)
  rmm::device_uvector<value_idx> sorted_child(condensed_child, n_points, stream);
  rmm::device_uvector<value_t> sorted_lambdas(lambdas, n_points, stream);

  auto children_lambda_zip = thrust::make_zip_iterator(thrust::make_tuple(sorted_child.begin(), sorted_lambdas.begin()));
  thrust::sort_by_key(policy, condensed_parent, condensed_parent + n_points, children_lambda_zip);

  // TODO: Segmented reduction on min_lambda within each cluster
  // TODO: Converting child array to CSR offset and using CUB Segmented Reduce
  // Investigate use of a kernel like coo_spmv
  auto n_clusters = // max label in parent - n_leaves, which make_monotonic will provide with
  rmm::device_uvector<value_idx> birth(n_clusters, stream);
  thrust::fill(thrust_policy, birth.begin(), birth.end(), 0);

  rmm::device_uvector<value_idx> sorted_child_offsets(n_points + 1, stream);
  auto start_offset = 0;
  sorted_child_offsets.set_element_async(0, start_offset, stream);
  thrust::inclusive_scan(thrust_policy, sorted_child.begin(), sorted_child.end(), sorted_child_offsets.begin() + 1);

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, lambdas.begin(), birth.begin(),
    n_clusters, sorted_child_offsets.begin(), sorted_child_offsets.begin() + 1);
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, lambdas.begin(), birth.begin(),
    n_clusters, sorted_child_offsets.begin(), sorted_child_offsets.begin() + 1);
  CUDA_CHECK(cudaFree(d_temp_storage));

  // TODO: Embarassingly parallel construction of output
  // TODO: It can be done with same coo_spmv kernel
  // Or naive kernel, atomically write to cluster stability
  rmm::device_uvector<value_t> stabilities(n_clusters, stream);
  thrust::fill(thrust_policy, stabilities.begin(), stabilities.end(), 0);

  *d_temp_storage = NULL;
  temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, lambdas.begin(), stabilities.begin(),
    n_clusters, sorted_child_offsets.begin(), sorted_child_offsets.begin() + 1);
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, lambdas.begin(), stabilities.begin(),
    n_clusters, sorted_child_offsets.begin(), sorted_child_offsets.begin() + 1);
  CUDA_CHECK(cudaFree(d_temp_storage));

  // now transform
  auto transform_op = transform_functor<value_t>(stabilities.data(), birth.data());
  thrust::transform(policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_clusters), stabilities.begin(), transform_op);

  return stabilities;
}

template<typename value_idx, typename value_t>
void excess_of_mass() {

  // TODO: Build CSR of cluster tree with stabilities of each child as the weights

  // TODO: Segmented reduction over CSR of cluster tree

  // TODO: Perform bfs, starting at root-
  // TODO:    Maintain frontier and is_cluster array.
  // TODO:    In each iteration, children are added to tree
  // TODO:    If node has is_cluster[node] = false, set children to false
  // TODO:    else subtree stability > stability[node] or cluster_sizes[node] > max_cluster_size
  // TODO:          set is_cluster[node] = false and stability[node] = subtree_stability
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