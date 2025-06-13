/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include <cuml/cluster/linkage.hpp>
#include <cuml/cluster/single_linkage_output.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/cluster/agglomerative.hpp>

namespace ML {

void single_linkage_pairwise(const raft::handle_t& handle,
                             const float* X,
                             size_t m,
                             size_t n,
                             ML::single_linkage_output<int>* out,
                             ML::distance::DistanceType metric,
                             int n_clusters)
{
  auto X_view = raft::make_device_matrix_view<const float, int, raft::row_major>(
    X, static_cast<int>(m), static_cast<int>(n));
  cuvs::cluster::agglomerative::single_linkage(handle,
                                               X_view,
                                               out->get_children(),
                                               out->get_labels(),
                                               static_cast<cuvs::distance::DistanceType>(metric),
                                               static_cast<size_t>(n_clusters),
                                               cuvs::cluster::agglomerative::Linkage::PAIRWISE,
                                               0);
}

void single_linkage_neighbors(const raft::handle_t& handle,
                              const float* X,
                              size_t m,
                              size_t n,
                              ML::single_linkage_output<int>* out,
                              ML::distance::DistanceType metric,
                              int c,
                              int n_clusters)
{
  auto X_view = raft::make_device_matrix_view<const float, int, raft::row_major>(
    X, static_cast<int>(m), static_cast<int>(n));
  cuvs::cluster::agglomerative::single_linkage(handle,
                                               X_view,
                                               out->get_children(),
                                               out->get_labels(),
                                               static_cast<cuvs::distance::DistanceType>(metric),
                                               static_cast<size_t>(n_clusters),
                                               cuvs::cluster::agglomerative::Linkage::KNN_GRAPH,
                                               c);
}

};  // end namespace ML
