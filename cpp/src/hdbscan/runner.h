/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "detail/condense.cuh"
#include "detail/extract.cuh"
#include "detail/reachability.cuh"
#include "detail/soft_clustering.cuh"

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/logger.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cuvs/neighbors/reachability.hpp>

namespace ML {
namespace HDBSCAN {

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
 * @param[in] core_dists buffer for storing core distances (size m)
 * @param[out] out output container object
 */
template <typename value_idx = int64_t, typename value_t = float>
void build_linkage(const raft::handle_t& handle,
                   const value_t* X,
                   size_t m,
                   size_t n,
                   cuvs::distance::DistanceType metric,
                   Common::HDBSCANParams& params,
                   value_t* core_dists,
                   Common::robust_single_linkage_output<value_idx, value_t>& out)
{
  auto stream = handle.get_stream();

  /**
   * Mutual reachability graph
   */
  rmm::device_uvector<value_idx> mutual_reachability_indptr(m + 1, stream);
  // Note that (min_samples+1) is parsed while allocating space for the COO matrix and to the
  // mutual_reachability_graph function. This was done to account for self-loops in the knn graph
  // and be consistent with Scikit learn Contrib.
  size_t mutual_reachability_nnz = ((params.min_samples + 1) * m * 2);
  raft::sparse::COO<value_t, value_idx> mutual_reachability_coo(stream, mutual_reachability_nnz);

  cuvs::neighbors::reachability::mutual_reachability_graph(
    handle,
    raft::make_device_matrix_view<const value_t, int64_t>(X, m, n),
    params.min_samples + 1,
    raft::make_device_vector_view<value_idx>(mutual_reachability_indptr.data(), m + 1),
    raft::make_device_vector_view<value_t>(core_dists, m),
    mutual_reachability_coo,
    metric,
    params.alpha);

  value_idx n_edges = m - 1;

  cuvs::neighbors::reachability::helpers::build_single_linkage_dendrogram(
    handle,
    raft::make_device_matrix_view<const value_t, value_idx>(X, m, n),
    metric,
    raft::make_device_vector_view<value_idx, value_idx>(mutual_reachability_indptr.data(), m + 1),
    raft::make_device_coo_matrix_view<value_t, value_idx, value_idx, size_t>(
      mutual_reachability_coo.vals(),
      raft::make_device_coordinate_structure_view(mutual_reachability_coo.rows(),
                                                  mutual_reachability_coo.cols(),
                                                  value_idx(m),
                                                  value_idx(m),
                                                  mutual_reachability_nnz)),
    raft::make_device_vector_view<value_t, value_idx>(core_dists, m),
    raft::make_device_coo_matrix_view<value_t, value_idx, value_idx, value_idx>(
      out.get_mst_weights(),
      raft::make_device_coordinate_structure_view(
        out.get_mst_src(), out.get_mst_dst(), value_idx(m), value_idx(m), n_edges)),
    raft::make_device_matrix_view<value_idx, value_idx>(out.get_children(), n_edges, 2),
    raft::make_device_vector_view<value_t, value_idx>(out.get_deltas(), n_edges),
    raft::make_device_vector_view<value_idx, value_idx>(out.get_sizes(), n_edges));
}

template <typename value_idx = int64_t, typename value_t = float>
void _fit_hdbscan(const raft::handle_t& handle,
                  const value_t* X,
                  size_t m,
                  size_t n,
                  cuvs::distance::DistanceType metric,
                  Common::HDBSCANParams& params,
                  value_idx* labels,
                  value_t* core_dists,
                  Common::hdbscan_output<value_idx, value_t>& out)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  int min_cluster_size = params.min_cluster_size;

  build_linkage(handle, X, m, n, metric, params, core_dists, out);

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

  rmm::device_uvector<value_idx> label_map(out.get_condensed_tree().get_n_clusters(),
                                           handle.get_stream());
  value_idx n_selected_clusters =
    detail::Extract::extract_clusters(handle,
                                      out.get_condensed_tree(),
                                      m,
                                      labels,
                                      tree_stabilities.data(),
                                      out.get_probabilities(),
                                      label_map.data(),
                                      params.cluster_selection_method,
                                      out._get_inverse_label_map(),
                                      params.allow_single_cluster,
                                      params.max_cluster_size,
                                      params.cluster_selection_epsilon);

  out.set_n_clusters(n_selected_clusters);

  auto lambdas_ptr   = thrust::device_pointer_cast(out.get_condensed_tree().get_lambdas());
  value_t max_lambda = *(thrust::max_element(
    exec_policy, lambdas_ptr, lambdas_ptr + out.get_condensed_tree().get_n_edges()));

  detail::Stability::get_stability_scores(handle,
                                          labels,
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

  thrust::transform(exec_policy,
                    labels,
                    labels + m,
                    out.get_labels(),
                    [label_map = label_map.data()] __device__(value_idx label) {
                      if (label != -1) return label_map[label];
                      return -1;
                    });
}

};  // end namespace HDBSCAN
};  // end namespace ML
