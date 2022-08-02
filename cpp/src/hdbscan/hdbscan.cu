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

#include "detail/condense.cuh"
#include <cuml/cluster/hdbscan.hpp>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include "runner.h"


namespace ML {

void hdbscan(const raft::handle_t& handle,
             const float* X,
             size_t m,
             size_t n,
             raft::distance::DistanceType metric,
             HDBSCAN::Common::HDBSCANParams& params,
             HDBSCAN::Common::hdbscan_output<int, float>& out,
             bool prediction_data,
             HDBSCAN::Common::PredictionData<int, float>& pred_data)
{
  HDBSCAN::_fit_hdbscan(handle, X, m, n, metric, params, out, prediction_data, pred_data);
}

void build_condensed_hierarchy(const raft::handle_t& handle,
                               const int* children,
                               const float* delta,
                               const int* sizes,
                               int min_cluster_size,
                               int n_leaves,
                               HDBSCAN::Common::CondensedHierarchy<int, float>& condensed_tree)
{
  HDBSCAN::detail::Condense::build_condensed_hierarchy(
    handle, children, delta, sizes, min_cluster_size, n_leaves, condensed_tree);
}

void _extract_clusters(const raft::handle_t& handle,
                       size_t n_leaves,
                       int n_edges,
                       int* parents,
                       int* children,
                       float* lambdas,
                       int* sizes,
                       int* labels,
                       float* probabilities,
                       HDBSCAN::Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                       bool allow_single_cluster,
                       int max_cluster_size,
                       float cluster_selection_epsilon)
{
  HDBSCAN::Common::CondensedHierarchy condensed_tree(
    handle, n_leaves, n_edges, parents, children, lambdas, sizes);

  rmm::device_uvector<float> stabilities(condensed_tree.get_n_clusters(), handle.get_stream());
  rmm::device_uvector<int> label_map(n_leaves, handle.get_stream());

  HDBSCAN::detail::Extract::extract_clusters(handle,
                                             condensed_tree,
                                             n_leaves,
                                             labels,
                                             stabilities.data(),
                                             probabilities,
                                             label_map.data(),
                                             cluster_selection_method,
                                             allow_single_cluster,
                                             max_cluster_size,
                                             cluster_selection_epsilon);
}

void _all_points_membership_vectors(const raft::handle_t& handle,
                                    HDBSCAN::Common::CondensedHierarchy<int, float>& condensed_tree,
                                    HDBSCAN::Common::PredictionData<int, float>& prediction_data,
                                    float* membership_vec,
                                    const float* X,
                                    raft::distance::DistanceType metric)
{
  HDBSCAN::detail::Membership::all_points_membership_vectors(handle,
                                                             condensed_tree,
                                                             prediction_data,
                                                             membership_vec,
                                                             X,
                                                             metric);
}

template <typename value_idx, typename value_t>
void HDBSCAN::Common::PredictionData<value_idx, value_t>::cache(const raft::handle_t& handle,
           value_idx n_exemplars_,
           value_idx n_clusters,
           value_idx n_selected_clusters_,
           value_t* deaths_,
           value_idx* exemplar_idx_,
           value_idx* exemplar_label_offsets_,
           value_idx* selected_clusters_)
{
  this-> n_exemplars = n_exemplars_;
  this-> n_selected_clusters = n_selected_clusters_;
  exemplar_idx.resize(n_exemplars, handle.get_stream());
  exemplar_label_offsets.resize(n_selected_clusters_ + 1, handle.get_stream());
  deaths.resize(n_clusters, handle.get_stream());
  selected_clusters.resize(n_selected_clusters, handle.get_stream());
  raft::copy(exemplar_idx.begin(), exemplar_idx_, n_exemplars_, handle.get_stream());
  raft::copy(exemplar_label_offsets.begin(), exemplar_label_offsets_, n_selected_clusters_ + 1, handle.get_stream());
  raft::copy(deaths.begin(), deaths_, n_clusters, handle.get_stream());
  raft::copy(selected_clusters.begin(), selected_clusters_, n_selected_clusters_, handle.get_stream());
}

};  // end namespace ML
