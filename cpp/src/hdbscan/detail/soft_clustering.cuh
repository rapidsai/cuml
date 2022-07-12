// /*
//  * Copyright (c) 2021-2022, NVIDIA CORPORATION.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#pragma once

#include "kernels/soft_clustering.cuh"
#include "utils.h"

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>

#include <raft/sparse/convert/csr.hpp>
#include <raft/sparse/op/sort.hpp>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.hpp>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/matrix_vector_op.hpp>
#include <raft/linalg/norm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Membership {

//////////////
template <typename value_idx, typename value_t>
value_idx preprocess(const raft::handle_t& handle,
                     Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                     const value_idx* labels,
                     value_idx* clusters,
                     value_idx* rearranged_lambdas,
                     value_idx* leaf_idx,
                     value_idx* sorted_label_offsets)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto lambdas    = condensed_tree.get_lambdas();
  auto n_leaves   = condensed_tree.get_n_leaves();

  rmm::device_uvector<value_idx> sorted_labels(n_leaves, stream);
  raft::copy_async(sorted_labels.data(), labels, n_leaves, stream);

  thrust::sequence(exec_policy, leaf_idx, leaf_idx + n_leaves, 1);
  thrust::sort_by_key(exec_policy, sorted_labels.begin(), sorted_labels.end(), leaf_idx.data());

  auto counting = thrust::make_counting_iterator<int>(0);

  auto offsets_end_ptr = thrust::unique_by_key_copy(exec_policy,
                                                    sorted_labels.data(),
                                                    sorted_labels.data() + n_leaves,
                                                    counting,
                                                    clusters,
                                                    sorted_label_offsets);

  auto n_groups = offsets_end_ptr.first - sorted_labels_unique.data();
  
  value_idx outlier_offset = n_groups - n_selected_clusters;
  sorted_label_offsets.set_element(n_groups, n_leaves, stream);

  auto rearrange_op =
    [rearranged_lambdas,
     lambdas,
     leaf_idx] __device__(auto idx) {
       rearranged_lambdas[idx] = lambdas[leaf_idx[idx]];
       return;
     };

  thrust::for_each(
    exec_policy,
    counting,
    counting + n_leaves,
    rearrange_op);
  
  return outlier_offset;
}

template <typename value_idx, typename value_t>
value_idx get_exemplar_lambdas(const raft::handle_t& handle,
                               value_idx n_selected_clusters,
                               value_idx* rearranged_lambdas,
                               value_idx outlier_offset,
                               value_idx* sorted_label_offsets)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<int>(0);

  Utils::cub_segmented_reduce(
    rearranged_lambdas,
    deaths.data(),
    n_selected_clusters,
    sorted_label_offsets + outlier_offset,
    stream,
    cub::DeviceSegmentedReduce::Max<const value_t*, value_t*, const value_idx*, const value_idx*>);
}

template <typename value_idx, typename value_t>
value_idx get_exemplars(const raft::handle_t& handle,
                        value_idx n_leaves,
                        const value_idx* labels,
                        const value_idx* label_map,
                        value_idx n_selected_clusters,
                        value_idx* rearranged_lambdas,
                        value_idx* leaf_idx,
                        value_idx* sorted_labels,
                        value_t* deaths,
                        value_idx* exemplar_idx,
                        value_idx* exemplar_label_offsets)
{ 
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();
  rmm::device_uvector<value_idx> is_exemplar(n_leaves, stream);
  auto exemplar_op =
    [is_exemplar = is_exemplar.data(),
     rearranged_lambdas,
     label_map,
     deaths,
     sorted_labels,
     leaf_idx] __device__(auto idx) {
      is_exemplar[idx] = (sorted_labels[idx] >= 0 && rearranged_lambdas[idx] ==
      deaths[label_map[sorted_labels[idx]]] ? leaf_idx[idx] : -1);
      return;
     };

  thrust::for_each(exec_policy,
                   counting,
                   counting + n_leaves,
                   exemplar_op);
 
  auto exemplar_idx_end_ptr = thrust::copy_if(exec_policy,
                                              is_exemplar.data(),
                                              is_exemplar.data() + n_leaves,
                                              exemplar_idx,
                                              [] __device__(auto idx) { return idx >= 0; });
 
  auto n_exemplars = exemplar_idx_end_ptr - exemplar_idx;

  rmm::device_uvector<value_idx>exemplar_labels(n_exemplars, stream);

  thrust::transform(
    exec_policy,
    exemplar_idx,
    exemplar_idx + n_exemplars,
    exemplar_labels.data(),
    [labels] __device__(auto idx) { return labels[idx]; });

  raft::sparse::convert::sorted_coo_to_csr(exemplar_labels.data(), n_exemplars, exemplar_label_offsets, n_selected_clusters + 1, stream);

  return n_exemplars;
}

template <typename value_idx, typename value_t>
value_idx all_points_dist_membership_vector(const raft::handle_t& handle,
                                            const value_t* X,
                                            size_t m,
                                            size_t n,
                                            size_t n_exemplars,
                                            size_t n_selected_clusters,
                                            value_idx* exemplar_idx,
                                            value_idx* exemplar_label_offsets,
                                            value_t* dist_membership_vec,
                                            raft::distance::DistanceType metric,
                                            bool softmax = false)
{
  auto stream = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<value_idx>(0);
 
  rmm::device_uvector<value_t> exemplars_dense(n_exemplars * n, stream);

  raft::matrix::copyRows<value_t, value_idx, size_t>(X,
                                                     n_exemplars,
                                                     n,
                                                     exemplars_dense.data(),
                                                     exemplar_idx,
                                                     n_exemplars,
                                                     stream,
                                                     true);
  
  rmm::device_uvector<value_t> dist(m * n_exemplars, stream);
  raft::distance::distance<metric, value_t, value_t, value_t, int>(
    X, exemplars_dense.data(), dist.data(), m, n_exemplars, n, stream, true);

  rmm::device_uvector<value_t> min_dist(m * n_selected_clusters, stream);
  thrust::fill(exec_policy, min_dist.begin(), min_dist.end(), std::numeric_limits<value_t>::max());
  
  auto reduction_op =
    [dist = dist.data(),
     n_selected_clusters,
     n_exemplars,
     exemplar_label_offsets,
     min_dist = min_dist.data()]
     __device__(auto idx) {
      auto col = idx % n_selected_clusters;
      auto row = idx / n_selected_clusters;
      auto start = exemplar_label_offsets[col];
      auto end = exemplar_label_offsets[col + 1];
    
      for(value_idx i = start; i < end; i++){
        if (dist[row * n_exemplars + i] < min_dist[row * n_selected_clusters + col]){
          min_dist[row * n_selected_clusters + col] = dist[row * n_exemplars + i];
        }
      }
       return;
     };
  
  thrust::for_each(
    exec_policy,
    counting,
    counting + m * n_selected_clusters,
    reduction_op
    );
  
  if (softmax){
    thrust::transform(
      exec_policy,
      min_dist.data(),
      min_dist.data() + m * n_selected_clusters,
      dist_membership_vec,
      [=] __device__(value_t val){
        if(val != 0){
          return exp(value_t(1.0/val - std::numeric_limits<value_t>::max()));
        }
        return 1.0;
      }
    );
  }

  else{
    thrust::transform(
      exec_policy,
      min_dist.data(),
      min_dist.data() + m * n_selected_clusters,
      dist_membership_vec,
      [=] __device__(value_t val){
        if(val != 0){
          return value_t(1.0/val);
        }
        return value_t(std::numeric_limits<value_t>::max()/n_selected_clusters);
      }
    );
  }

  rmm::device_uvector<value_t> dist_membership_vec_sums(m, stream);
  thrust::fill(exec_policy, dist_membership_vec_sums.begin(), dist_membership_vec_sums.end(), 1.0);

  raft::linalg::rowNorm(
    dist_membership_vec_sums.data(),
    dist_membership_vec,
    n_selected_clusters,
    m,
    raft::linalg::L1Norm,
    true,
    stream
  );

  raft::linalg::matrixVectorOp(
    dist_membership_vec,
    dist_membership_vec,
    dist_membership_vec_sums.data(),
    n_selected_clusters,
    m,
    true,
    true,
    [] __device__(value_t mat_in, value_t vec_in) { return mat_in / vec_in; },
    stream
  );
};

template <typename value_idx, typename value_t, int tpb = 256>
value_idx get_merge_heights(const raft::handle_t& handle,
                            Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                            value_idx* selected_clusters,
                            size_t m,
                            size_t n_selected_clusters,
                            value_t* merge_heights)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  rmm::device_uvector<value_idx> index_into_children(n_edges, stream);
  auto index_op = [index_into_children = index_into_children.data()] __device__(auto t) {
    index_into_children[thrust::get<0>(t)] = thrust::get<1>(t);
    return;
  };
  thrust::for_each(
    exec_policy,
    thrust::make_zip_iterator(thrust::make_tuple(children, counting)),
    thrust::make_zip_iterator(thrust::make_tuple(children + n_edges, counting + n_edges)),
    index_op
  );

  int n_blocks = (m * n_selected_clusters) / tpb;
  merge_height_kernel<<<n_blocks, tpb>>>(merge_heights,
                                         index_into_children.data(),
                                         parents,
                                         m,
                                         n_selected_clusters,
                                         selected_clusters)
}

template <typename value_idx, typename value_t, int tpb = 256>
value_idx all_points_outlier_membership_vector(
  const raft::handle_t& handle,
  Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
  value_idx* selected_clusters,
  size_t m,
  size_t n_selected_clusters,
  value_t* merge_heights
)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  rmm::device_uvector<value_idx> sorted_parents_offsets(n_selected_clusters + 1, stream);
  Utils::parent_csr(handle, condensed_tree, sorted_parents.data(), sorted_parents_offsets.data());

  // this is to find maximum lambdas of all children under a parent
  rmm::device_uvector<value_t> deaths(n_selected_clusters, stream);
  thrust::fill(exec_policy, deaths.begin(), deaths.end(), 0.0f);

  Utils::cub_segmented_reduce(
    lambdas,
    deaths.data(),
    n_selected_clusters,
    sorted_parents_offsets.data(),
    stream,
    cub::DeviceSegmentedReduce::Max<const value_t*, value_t*, const value_idx*, const value_idx*>);

  raft::linalg::matrixVectorOp(
    outlier_membership_vec,
    merge_heights,
    deaths.data(),
    m,
    n_selected_clusters,
    true,
    true,
    [] __device__(value_t mat_in, value_t vec_in) { return exp(-mat_in / (vec_in + 1e-8)); }, //+ 1e-8 to avoid zero lambda
    stream
  );

  if (softmax){
    thrust::transform(
      exec_policy,
      outlier_membership_vec,
      outlier_membership_vec + m * n_selected_clusters,
      outlier_membership_vec,
      [=] __device__(value_t val){
          return exp(value_t(val - std::numeric_limits<value_t>::max()));
      }
    );
  }

  rmm::device_uvector<value_idx> height_argmax(m, stream);
  rmm::device_uvector<value_idx> point_offsets(m + 1, stream);
  thrust::sequence(exec_policy, point_offsets.data(), point_offsets.data() + m + 1, n_selected_clusters);
  Utils::cub_segmented_reduce(
    merge_heights,
    height_argmax.data(),
    n_selected_clusters,
    point_offsets.data(),
    stream,
    cub::DeviceSegmentedReduce::ArgMax<const value_t*, value_t*, const value_idx*, const value_idx*>
  );

  auto prob_in_some_cluster_op =
    [height_argmax = height_argmax.data(),
     lambdas] __device__(auto idx) {
       prob_in_some_cluster[idx] = merge_heights[height_argmax[idx]] / max_lambdas[selected_clusters[height_argmax[idx]] + n_leaves];
       return;
     };

  raft::linalg::matrixVectorOp(
    membership_vec,
    membership_vec,
    prob_in_some_cluster,
    m,
    n_selected_clusters,
    true,
    true,
    [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
    stream
  );
}

void all_points_membership_vector(
)
{
  rmm::device_uvector<value_idx> leaf_idx(n_leaves, stream);
  rmm::device_uvector<value_idx> clusters(n_selected_clusters + 1, stream);
  rmm::device_uvector<value_idx> sorted_label_offsets(n_selected_clusters + 2, stream);
  rmm::device_uvector<value_t> rearranged_lambdas(n_leaves, stream);
  value_idx outlier_offset = preprocess(handle,
                                        condensed_tree,
                                        labels,
                                        clusters.data(),
                                        rearranged_lambdas.data(),
                                        leaf_idx.data(),
                                        sorted_label_offsets.data());
  
  rmm::device_uvector<value_t> deaths(n_selected_clusters, stream);
  thrust::fill(exec_policy, deaths.begin(), deaths.end(), 0.0f);
  get_exemplar_lambdas(handle,
                       n_selected_clusters,
                       rearranged_lambdas,
                       outlier_offset,
                       sorted_label_offsets,
                       deaths.data());

  value_idx n_exemplars = get_exemplars(handle,
                                        n_leaves,
                                        labels,
                                        label_map,
                                        n_selected_clusters,
                                        rearranged_lambdas.data(),
                                        leaf_idx.data(),
                                        sorted_labels.data(),
                                        sorted_label_offsets.data(),
                                        deaths.data(),
                                        exemplar_idx.data(),
                                        exemplar_label_offsets.data());
  
  rmm::device_uvector<value_t> dist_membership_vec(m * n_selected_clusters, stream);
  all_points_dist_membership_vector(handle,
                                    X,
                                    m,
                                    n,
                                    n_exemplars,
                                    n_selected_clusters,
                                    exemplar_idx.data(),
                                    exemplar_label_offsets.data(),
                                    dist_membership_vec.data(),
                                    metric,
                                    bool softmax = false);

  all_points_outlier_membership_vector(handle,
                                       condensed_tree,
                                       label_map,
                                       n_selected_clusters,
                                       merge_heights.data(),
                                       outlier_membership_vec.data(),
                                       softmax = false);
}

};  // namespace Membership
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
