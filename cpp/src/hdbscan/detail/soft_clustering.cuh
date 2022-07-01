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

#include <algorithm>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
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

template <typename value_idx, typename value_t, int tpb = 256>
value_idx get_exemplars(const raft::handle_t& handle,
                        Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                        const value_idx* labels,
                        value_idx n_selected_clusters,
                        value_idx* exemplar_idx)
                        // value_idx* exemplar_offsets)
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

  rmm::device_uvector<value_idx> sorted_labels_unique(n_selected_clusters + 1, stream);
  rmm::device_uvector<value_idx> sorted_label_offsets(n_selected_clusters + 2, stream);
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
  
  CUML_LOG_DEBUG("n_selected_clusters: %d\n", n_groups);

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

  // size_t nblocks = raft::ceildiv(n_leaves, tpb);
  // rearrange_kernel<<<nblocks, tpb, 0, stream>>>(
  //   leaf_idx.data(), lambdas, rearranged_lambdas.data(), n_leaves);

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
 
  auto n_exemplar_indices = exemplar_idx_end_ptr - exemplar_idx;

  // rmm::device_uvector<value_idx>exemplar_labels(n_exemplar_indices, stream);

  // thrust::transform(
  //   exec_policy,
  //   exemplar_idx,
  //   exemplar_idx + n_exemplar_indices,
  //   exemplar_labels.data(),
  //   [labels] __device__(auto idx) { return labels[idx]; });

  // rmm::device_uvector<value_idx> exemplar_label_offsets(n_exemplar_indices + 1, stream);
  // thrust::unique_by_key_copy(exec_policy,
  //                            exemplar_labels.data(),
  //                            exemplar_labels.data() + n_exemplar_indices,
  //                            thrust::make_counting_iterator(0),
  //                            thrust::make_discard_iterator(),
  //                            exemplar_label_offsets.begin());
  // exemplar_label_offsets.set_element(n_exemplar_indices, n_exemplar_indices, stream);
  return n_exemplar_indices;
}

template <typename value_idx, typename value_t, int tpb = 256>
value_idx dist_membership_vector(const raft::handle_t& handle,
                                 Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                 const value_t* X,
                                 value_idx* exemplar_idx)
{
  raft::matrix::copyRows<value_t, value_idx, size_t>(
    X,
    index.m,
    index.n,
  index.get_R(),
  R_1nn_cols2.data(),
  index.n_landmarks,
  handle.get_stream(),
  true);
  
  raft::distance::distance<metric, value_idx, value_idx, value_idx, int>(
    x, y, dist, m, n, k, handle.get_stream(), isRowMajor);
};  // namespace Membership
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
