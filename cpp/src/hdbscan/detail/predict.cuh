/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "kernels/predict.cuh"
#include "reachability.cuh"

#include <raft/core/cudart_utils.hpp>
#include <raft/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

/**
Find the nearest mutual reachability neighbor of a point, and  compute
the associated lambda value for the point, given the mutual reachability
distance to a nearest neighbor.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[in] input_core_dists an array of core distances for all points (size m)
 * @param[in] prediction_core_dists an array of core distances for all prediction points (size
n_prediction_points)
 * @param[in] knn_dists knn distance array (size n_prediction_points * neighborhood)
 * @param[in] knn_inds knn indices array (size n_prediction_points * neighborhood)
 * @param[in] n_prediction_points number of prediction points
 * @param[in] neighborhood the neighborhood of prediction points
 * @param[out] min_mr_inds indices of points with the minimum mutual reachability distance (size
n_prediction_points)
 * @param[out] prediction_lambdas lambda values for prediction points (size n_prediction_points)
 */
template <typename value_idx, typename value_t, int tpb = 256>
void _find_neighbor_and_lambda(const raft::handle_t& handle,
                               value_t* input_core_dists,
                               value_t* prediction_core_dists,
                               value_t* knn_dists,
                               value_idx* knn_inds,
                               size_t n_prediction_points,
                               int neighborhood,
                               value_idx* min_mr_inds,
                               value_t* prediction_lambdas)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  // Buffer for storing the minimum mutual reachability distances
  rmm::device_uvector<value_t> min_mr_dists(n_prediction_points, stream);

  int n_blocks = raft::ceildiv((int)n_prediction_points, tpb);

  // get nearest neighbors for each prediction point in mutual reachability space
  min_mutual_reachability_kernel<<<n_blocks, tpb, 0, stream>>>(input_core_dists,
                                                               prediction_core_dists,
                                                               knn_dists,
                                                               knn_inds,
                                                               n_prediction_points,
                                                               neighborhood,
                                                               min_mr_dists.data(),
                                                               min_mr_inds);

  // obtain lambda values from minimum mutual reachability distances
  thrust::transform(exec_policy,
                    min_mr_dists.data(),
                    min_mr_dists.data() + n_prediction_points,
                    prediction_lambdas,
                    [] __device__(value_t dist) {
                      if (dist > 0) return (1 / dist);
                      return std::numeric_limits<value_t>::max();
                    });
}

/**
 Return the cluster label (of the original clustering) and membership
 probability of a new data point.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] n_prediction_points number of prediction points
 * @param[in] min_mr_inds indices of points with the minimum mutual reachability distance (size
 n_prediction_points)
 * @param[in] prediction_lambdas lambda values for prediction points (size n_prediction_points)
 * @param[in] labels monotonic labels of all points
 * @param[out] out_labels output cluster labels
 * @param[out] out_probabilities output probabilities
 */
template <typename value_idx, typename value_t, int tpb = 256>
void _find_cluster_and_probability(const raft::handle_t& handle,
                                   Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                   Common::PredictionData<value_idx, value_t>& prediction_data,
                                   size_t n_prediction_points,
                                   value_idx* min_mr_inds,
                                   value_t* prediction_lambdas,
                                   value_idx* labels,
                                   value_idx* out_labels,
                                   value_t* out_probabilities)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents     = condensed_tree.get_parents();
  auto children    = condensed_tree.get_children();
  value_t* lambdas = condensed_tree.get_lambdas();
  auto n_edges     = condensed_tree.get_n_edges();
  auto n_leaves    = condensed_tree.get_n_leaves();

  value_t* deaths                = prediction_data.get_deaths();
  value_idx* selected_clusters   = prediction_data.get_selected_clusters();
  value_idx* index_into_children = prediction_data.get_index_into_children();

  int n_blocks = raft::ceildiv((int)n_prediction_points, tpb);

  cluster_probability_kernel<<<n_blocks, tpb, 0, stream>>>(min_mr_inds,
                                                           prediction_lambdas,
                                                           index_into_children,
                                                           labels,
                                                           deaths,
                                                           selected_clusters,
                                                           parents,
                                                           lambdas,
                                                           n_leaves,
                                                           n_prediction_points,
                                                           out_labels,
                                                           out_probabilities);
}
/**
 * Predict the cluster label and the probability of the label for new points.
 * The returned labels are those of the original clustering,
 * and therefore are not (necessarily) the cluster labels that would
 * be found by clustering the original data combined with
 * the prediction points, hence the 'approximate' label.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] X input data points (size m * n)
 * @param[in] labels converted monotonic labels of the input data points
 * @param[in] points_to_predict input prediction points (size n_prediction_points * n)
 * @param[in] n_prediction_points number of prediction points
 * @param[in] metric distance metric to use
 * @param[in] min_samples neighborhood size during training (includes self-loop)
 * @param[out] out_labels output cluster labels
 * @param[out] out_probabilities output probabilities
 */
template <typename value_idx, typename value_t, int tpb = 256>
void approximate_predict(const raft::handle_t& handle,
                         Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                         Common::PredictionData<value_idx, value_t>& prediction_data,
                         const value_t* X,
                         value_idx* labels,
                         const value_t* points_to_predict,
                         size_t n_prediction_points,
                         raft::distance::DistanceType metric,
                         int min_samples,
                         value_idx* out_labels,
                         value_t* out_probabilities)
{
  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  size_t m                  = prediction_data.n_rows;
  size_t n                  = prediction_data.n_cols;
  value_t* input_core_dists = prediction_data.get_core_dists();

  // this is the neighborhood of prediction points for which MR distances are computed
  int neighborhood = (min_samples - 1) * 2;

  rmm::device_uvector<value_idx> inds(neighborhood * n_prediction_points, stream);
  rmm::device_uvector<value_t> dists(neighborhood * n_prediction_points, stream);
  rmm::device_uvector<value_t> prediction_core_dists(n_prediction_points, stream);

  // perform knn
  Reachability::compute_knn(handle,
                            X,
                            inds.data(),
                            dists.data(),
                            m,
                            n,
                            points_to_predict,
                            n_prediction_points,
                            neighborhood,
                            metric);

  // Slice core distances (distances to kth nearest neighbor). The index of the neighbor is
  // consistent with Scikit-learn Contrib
  Reachability::core_distances<value_idx>(dists.data(),
                                          min_samples,
                                          neighborhood,
                                          n_prediction_points,
                                          prediction_core_dists.data(),
                                          stream);

  // Obtain lambdas for each prediction point using the closest point in mutual reachability space
  rmm::device_uvector<value_t> prediction_lambdas(n_prediction_points, stream);
  rmm::device_uvector<value_idx> min_mr_inds(n_prediction_points, stream);
  _find_neighbor_and_lambda(handle,
                            input_core_dists,
                            prediction_core_dists.data(),
                            dists.data(),
                            inds.data(),
                            n_prediction_points,
                            neighborhood,
                            min_mr_inds.data(),
                            prediction_lambdas.data());

  // Using the nearest neighbor indices, find the assigned cluster label and probability
  _find_cluster_and_probability(handle,
                                condensed_tree,
                                prediction_data,
                                n_prediction_points,
                                min_mr_inds.data(),
                                prediction_lambdas.data(),
                                labels,
                                out_labels,
                                out_probabilities);
}

};  // end namespace Predict
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML