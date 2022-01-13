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

#pragma once

#include <raft/cudart_utils.h>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cuml/common/logger.hpp>

#include <raft/sparse/coo.hpp>
#include <raft/sparse/hierarchy/detail/agglomerative.cuh>
#include <raft/sparse/hierarchy/detail/mst.cuh>

#include "detail/condense.cuh"
#include "detail/extract.cuh"
#include "detail/reachability.cuh"
#include <cuml/cluster/hdbscan.hpp>

namespace ML {
namespace HDBSCAN {

/**
 * Functor with reduction ops for performing fused 1-nn
 * computation and guaranteeing only cross-component
 * neighbors are considered.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct FixConnectivitiesRedOp {
  value_idx* colors;
  value_t* core_dists;
  value_idx m;

  FixConnectivitiesRedOp(value_idx* colors_, value_t* core_dists_, value_idx m_)
    : colors(colors_), core_dists(core_dists_), m(m_){};

  typedef typename cub::KeyValuePair<value_idx, value_t> KVP;
  DI void operator()(value_idx rit, KVP* out, const KVP& other)
  {
    if (rit < m && other.value < std::numeric_limits<value_t>::max() &&
        colors[rit] != colors[other.key]) {
      value_t core_dist_rit   = core_dists[rit];
      value_t core_dist_other = max(core_dist_rit, max(core_dists[other.key], other.value));

      value_t core_dist_out;
      if (out->key > -1) {
        core_dist_out = max(core_dist_rit, max(core_dists[out->key], out->value));
      } else {
        core_dist_out = out->value;
      }

      bool smaller = core_dist_other < core_dist_out;
      out->key     = smaller ? other.key : out->key;
      out->value   = smaller ? core_dist_other : core_dist_out;
    }
  }

  DI KVP operator()(value_idx rit, const KVP& a, const KVP& b)
  {
    if (rit < m && a.key > -1 && colors[rit] != colors[a.key]) {
      value_t core_dist_rit = core_dists[rit];
      value_t core_dist_a   = max(core_dist_rit, max(core_dists[a.key], a.value));

      value_t core_dist_b;
      if (b.key > -1) {
        core_dist_b = max(core_dist_rit, max(core_dists[b.key], b.value));
      } else {
        core_dist_b = b.value;
      }

      return core_dist_a < core_dist_b ? KVP(a.key, core_dist_a) : KVP(b.key, core_dist_b);
    }

    return b;
  }

  DI void init(value_t* out, value_t maxVal) { *out = maxVal; }
  DI void init(KVP* out, value_t maxVal)
  {
    out->key   = -1;
    out->value = maxVal;
  }
};

/**
 * Constructs a linkage by computing mutual reachability, mst, and
 * dendrogram. This is shared by HDBSCAN and Robust Single Linkage
 * since the two algorithms differ only in the cluster
 * selection and extraction.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points (size m * n)
 * @param[in] m number of rows
 * @param[in] n number of columns
 * @param[in] metric distance metric to use
 * @param[in] params hyper parameters
 * @param[out] out output container object
 */
template <typename value_idx = int64_t, typename value_t = float>
void build_linkage(const raft::handle_t& handle,
                   const value_t* X,
                   size_t m,
                   size_t n,
                   raft::distance::DistanceType metric,
                   Common::HDBSCANParams& params,
                   Common::robust_single_linkage_output<value_idx, value_t>& out)
{
  auto stream = handle.get_stream();

  /**
   * Mutual reachability graph
   */
  rmm::device_uvector<value_idx> mutual_reachability_indptr(m + 1, stream);
  raft::sparse::COO<value_t, value_idx> mutual_reachability_coo(stream, params.min_samples * m * 2);
  rmm::device_uvector<value_t> core_dists(m, stream);

  detail::Reachability::mutual_reachability_graph(handle,
                                                  X,
                                                  (size_t)m,
                                                  (size_t)n,
                                                  metric,
                                                  params.min_samples,
                                                  params.alpha,
                                                  mutual_reachability_indptr.data(),
                                                  core_dists.data(),
                                                  mutual_reachability_coo);

  /**
   * Construct MST sorted by weights
   */

  rmm::device_uvector<value_idx> color(m, stream);
  FixConnectivitiesRedOp<value_idx, value_t> red_op(color.data(), core_dists.data(), m);
  // during knn graph connection
  raft::hierarchy::detail::build_sorted_mst(handle,
                                            X,
                                            mutual_reachability_indptr.data(),
                                            mutual_reachability_coo.cols(),
                                            mutual_reachability_coo.vals(),
                                            m,
                                            n,
                                            out.get_mst_src(),
                                            out.get_mst_dst(),
                                            out.get_mst_weights(),
                                            color.data(),
                                            mutual_reachability_coo.nnz,
                                            red_op,
                                            metric,
                                            (size_t)10);

  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = m - 1;

  raft::hierarchy::detail::build_dendrogram_host(handle,
                                                 out.get_mst_src(),
                                                 out.get_mst_dst(),
                                                 out.get_mst_weights(),
                                                 n_edges,
                                                 out.get_children(),
                                                 out.get_deltas(),
                                                 out.get_sizes());
}

template <typename value_idx = int64_t, typename value_t = float>
void _fit_hdbscan(const raft::handle_t& handle,
                  const value_t* X,
                  size_t m,
                  size_t n,
                  raft::distance::DistanceType metric,
                  Common::HDBSCANParams& params,
                  Common::hdbscan_output<value_idx, value_t>& out)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  int min_cluster_size = params.min_cluster_size;

  build_linkage(handle, X, m, n, metric, params, out);

  /**
   * Condense branches of tree according to min cluster size
   */
  detail::Condense::build_condensed_hierarchy(handle,
                                              out.get_children(),
                                              out.get_deltas(),
                                              out.get_sizes(),
                                              min_cluster_size,
                                              m,
                                              out.get_condensed_tree());

  /**
   * Extract labels from stability
   */

  rmm::device_uvector<value_t> tree_stabilities(out.get_condensed_tree().get_n_clusters(),
                                                handle.get_stream());

  rmm::device_uvector<value_idx> label_map(m, stream);

  std::vector<value_idx> label_set;
  value_idx n_selected_clusters =
    detail::Extract::extract_clusters(handle,
                                      out.get_condensed_tree(),
                                      m,
                                      out.get_labels(),
                                      tree_stabilities.data(),
                                      out.get_probabilities(),
                                      label_map.data(),
                                      params.cluster_selection_method,
                                      params.allow_single_cluster,
                                      params.max_cluster_size,
                                      params.cluster_selection_epsilon);

  out.set_n_clusters(n_selected_clusters);

  auto lambdas_ptr   = thrust::device_pointer_cast(out.get_condensed_tree().get_lambdas());
  value_t max_lambda = *(thrust::max_element(
    exec_policy, lambdas_ptr, lambdas_ptr + out.get_condensed_tree().get_n_edges()));

  detail::Stability::get_stability_scores(handle,
                                          out.get_labels(),
                                          tree_stabilities.data(),
                                          out.get_condensed_tree().get_n_clusters(),
                                          max_lambda,
                                          m,
                                          out.get_stabilities(),
                                          label_map.data());

  /**
   * Normalize labels so they are drawn from a monotonically increasing set
   * starting at 0 even in the presence of noise (-1)
   */

  value_idx* label_map_ptr = label_map.data();
  thrust::transform(exec_policy,
                    out.get_labels(),
                    out.get_labels() + m,
                    out.get_labels(),
                    [=] __device__(value_idx label) {
                      if (label != -1) return label_map_ptr[label];
                      return -1;
                    });
}

};  // end namespace HDBSCAN
};  // end namespace ML
