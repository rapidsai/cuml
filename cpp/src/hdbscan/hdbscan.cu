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

#include "runner.h"

namespace ML {

void hdbscan(const raft::handle_t& handle,
             const float* X,
             size_t m,
             size_t n,
             raft::distance::DistanceType metric,
             HDBSCAN::Common::HDBSCANParams& params,
             HDBSCAN::Common::hdbscan_output<int, float>& out)
{
  HDBSCAN::_fit_hdbscan(handle, X, m, n, metric, params, out);
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

};  // end namespace ML
