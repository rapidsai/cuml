/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "pw_dist_graph.cuh"
#include <cuml/cluster/linkage.hpp>
#include <raft/sparse/hierarchy/single_linkage.hpp>

namespace raft {
class handle_t;
}

namespace ML {

void single_linkage_pairwise(const raft::handle_t& handle,
                             const float* X,
                             size_t m,
                             size_t n,
                             raft::hierarchy::linkage_output<int, float>* out,
                             raft::distance::DistanceType metric,
                             int n_clusters)
{
  raft::hierarchy::single_linkage<int, float, raft::hierarchy::LinkageDistance::PAIRWISE>(
    handle, X, m, n, metric, out, 0, n_clusters);
}

void single_linkage_neighbors(const raft::handle_t& handle,
                              const float* X,
                              size_t m,
                              size_t n,
                              raft::hierarchy::linkage_output<int, float>* out,
                              raft::distance::DistanceType metric,
                              int c,
                              int n_clusters)
{
  raft::hierarchy::single_linkage<int, float, raft::hierarchy::LinkageDistance::KNN_GRAPH>(
    handle, X, m, n, metric, out, c, n_clusters);
}

struct distance_graph_impl_int_float
  : public raft::hierarchy::detail::
      distance_graph_impl<raft::hierarchy::LinkageDistance::PAIRWISE, int, float> {
};
struct distance_graph_impl_int_double
  : public raft::hierarchy::detail::
      distance_graph_impl<raft::hierarchy::LinkageDistance::PAIRWISE, int, double> {
};

};  // end namespace ML
