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

template <typename value_idx, typename value_t>
value_idx get_exemplars(const raft::handle_t& handle,
                        Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                        const value_idx* labels,
                        value_idx n_clusters,
                        value_idx* exemplar_idx,
                        value_idx* exemplar_label_offsets)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto lambdas    = condensed_tree.get_lambdas();
  auto n_leaves   = condensed_tree.get_n_leaves();

  rmm::device_uvector<value_idx> sorted_labels(n_leaves, stream);
  raft::copy_async(sorted_labels.data(), labels, n_leaves, stream);

  rmm::device_uvector<value_idx> leaf_idx(n_leaves, stream);
  thrust::sequence(exec_policy, leaf_idx.begin(), leaf_idx.end(), 1);
  thrust::sort_by_key(exec_policy, sorted_labels.begin(), sorted_labels.end(), leaf_idx.data());

  rmm::device_uvector<value_idx> sorted_labels_unique(n_clusters + 1, stream);
  rmm::device_uvector<value_idx> sorted_label_offsets(n_clusters + 2, stream);
  auto offsets_end_ptr = thrust::unique_by_key_copy(exec_policy,
                                                    sorted_labels.data(),
                                                    sorted_labels.data() + n_leaves,
                                                    thrust::make_counting_iterator(0),
                                                    sorted_labels_unique.data(),
                                                    sorted_label_offsets.begin());

  auto n_groups = offsets_end_ptr.first - sorted_labels_unique.data();
  
  value_idx outlier_offset = 0;
  if(sorted_labels_unique.element(0, stream) < 0){
    outlier_offset = sorted_labels_unique.element(1, stream);
    sorted_label_offsets.set_element(n_groups--, n_leaves, stream);
  }
  else{
    sorted_label_offsets.set_element(n_groups, n_leaves, stream);
  }
  
  CUML_LOG_DEBUG("n_clusters: %d\n", n_groups);

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  rmm::device_uvector<value_t> rearranged_lambdas(n_leaves, stream);
  auto rearrange_op =
    [rearranged_lambdas = rearranged_lambdas.data(),
     lambdas,
     leaf_idx = leaf_idx.data()] __device__(auto idx) {
       rearranged_lambdas[idx] = lambdas[leaf_idx[idx]];
       return;
     };

  thrust::for_each(
    exec_policy,
    counting,
    counting + n_leaves,
    rearrange_op);

  rmm::device_uvector<value_t> deaths(n_groups, stream);
  thrust::fill(exec_policy, deaths.begin(), deaths.end(), 0.0f);

  Utils::cub_segmented_reduce(
    rearranged_lambdas.data(),
    deaths.data(),
    n_groups,
    sorted_label_offsets.data() + (outlier_offset > 0),
    stream,
    cub::DeviceSegmentedReduce::Max<const value_t*, value_t*, const value_idx*, const value_idx*>);
  
  rmm::device_uvector<value_idx> label_map(n_leaves, stream);
  thrust::fill(exec_policy, label_map.begin(), label_map.end(), -1);
  auto label_map_op =
    [label_map = label_map.data()] __device__(auto t) {
       label_map[thrust::get<0>(t)] = thrust::get<1>(t);
       return;
       };
  thrust::for_each(
    exec_policy,
    thrust::make_zip_iterator(thrust::make_tuple(sorted_labels_unique.begin() + (outlier_offset > 0), counting)),
    thrust::make_zip_iterator(thrust::make_tuple(sorted_labels_unique.end(), counting + n_groups)),
    label_map_op
  );

  // for(int i = 0; i < n_groups; i++){
  //     CUML_LOG_DEBUG("%d", sorted_label_offsets.element(i + (outlier_offset > 0), stream));
  // }
  
  // for(int i = 0; i < n_leaves; i++){
  //     CUML_LOG_DEBUG("%d", label_map.element(i, stream));
  //   }
  rmm::device_uvector<value_idx> is_exemplar(n_leaves, stream);
  auto exemplar_op =
    [is_exemplar = is_exemplar.data(),
     rearranged_lambdas = rearranged_lambdas.data(),
     label_map = label_map.data(),
     deaths = deaths.data(),
     sorted_labels = sorted_labels.data(),
     leaf_idx = leaf_idx.data()]
     __device__(auto idx) {
       is_exemplar[idx] = (sorted_labels[idx] >= 0 && rearranged_lambdas[idx] ==
        deaths[label_map[sorted_labels[idx]]] ? leaf_idx[idx] : -1);
       return;
     };
  
  // CUML_LOG_DEBUG("%d %d", outlier_offset, n_groups);
  // // for(int i = 0; i < n_groups; i++){
  // //   CUML_LOG_DEBUG("%d %f", sorted_labels_unique.element(i + (outlier_offset > 0), stream), deaths.element(i, stream));
  // // }
  for(int i = 0; i < n_leaves; i++){
    if(sorted_labels.element(i, stream) >= 0 && deaths.element(label_map.element(sorted_labels.element(i, stream), stream), stream) == rearranged_lambdas.element(i, stream)){
      CUML_LOG_DEBUG("%d %d %d\n", i, leaf_idx.element(i, stream), sorted_labels.element(i, stream));
    }
  }
  thrust::for_each(exec_policy,
                   counting,
                   counting + n_leaves,
                   exemplar_op);
  
  // for(int i = 0; i < n_leaves; i++){
  //   if(is_exemplar.element(i, stream) >= 0)CUML_LOG_DEBUG("%f %f", deaths.element(label_map.element(sorted_labels.element(i, stream), stream), stream), rearranged_lambdas.element(i, stream));
  // }
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

  thrust::unique_by_key_copy(exec_policy,
                             exemplar_labels.data(),
                             exemplar_labels.data() + n_exemplars,
                             thrust::make_counting_iterator(0),
                             thrust::make_discard_iterator(),
                             exemplar_label_offsets);
  thrust::fill(exec_policy,
               exemplar_label_offsets + n_exemplars,
               exemplar_label_offsets + n_exemplars + 1,
               n_exemplars);

  // for(int i = 0; i < n_exemplars + 1; i++){
  //   CUML_LOG_DEBUG("%d", exemplar_label_offsets.element(i, stream));
  // }
  return n_exemplars;
}

template <typename value_idx, typename value_t>
value_idx dist_membership_vector(const raft::handle_t& handle,
                                 const value_t* X,
                                 size_t m,
                                 size_t n,
                                 size_t n_exemplars,
                                 size_t n_selected_clusters,
                                 bool softmax,
                                 value_idx* exemplar_idx,
                                 value_idx* exemplar_label_offsets,
                                 value_t* dist_membership_vec,
                                 raft::distance::DistanceType metric
                                 )
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
  thrust::fill(exec_policy, min_dist.begin(), min_dist.end(), DBL_MAX);
  
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
          return exp(value_t(1.0/val));
        }
        return std::numeric_limits<value_t>::max();
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

  // TODO: Compute distance membership vector sums correctly

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

};  // namespace Membership
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
