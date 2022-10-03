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
#include "detail/predict.cuh"
#include <cuml/cluster/hdbscan.hpp>
#include <raft/spatial/knn/specializations.hpp>

#include <raft/core/cudart_utils.hpp>
#include <raft/cuda_utils.cuh>

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
  rmm::device_uvector<int> labels(m, handle.get_stream());
  rmm::device_uvector<int> label_map(m, handle.get_stream());
  rmm::device_uvector<float> core_dists(m, handle.get_stream());
  HDBSCAN::_fit_hdbscan(
    handle, X, m, n, metric, params, labels.data(), label_map.data(), core_dists.data(), out);
}

void hdbscan(const raft::handle_t& handle,
             const float* X,
             size_t m,
             size_t n,
             raft::distance::DistanceType metric,
             HDBSCAN::Common::HDBSCANParams& params,
             HDBSCAN::Common::hdbscan_output<int, float>& out,
             HDBSCAN::Common::PredictionData<int, float>& prediction_data_)
{
  rmm::device_uvector<int> labels(m, handle.get_stream());
  rmm::device_uvector<int> label_map(m, handle.get_stream());
  HDBSCAN::_fit_hdbscan(handle,
                        X,
                        m,
                        n,
                        metric,
                        params,
                        labels.data(),
                        label_map.data(),
                        prediction_data_.get_core_dists(),
                        out);

  HDBSCAN::Common::build_prediction_data(handle,
                                         out.get_condensed_tree(),
                                         labels.data(),
                                         label_map.data(),
                                         out.get_n_clusters(),
                                         prediction_data_);
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

void compute_all_points_membership_vectors(
  const raft::handle_t& handle,
  HDBSCAN::Common::CondensedHierarchy<int, float>& condensed_tree,
  HDBSCAN::Common::PredictionData<int, float>& prediction_data,
  const float* X,
  raft::distance::DistanceType metric,
  float* membership_vec)
{
  HDBSCAN::detail::Predict::all_points_membership_vectors(
    handle, condensed_tree, prediction_data, X, metric, membership_vec);
}

void out_of_sample_predict(const raft::handle_t& handle,
                           HDBSCAN::Common::CondensedHierarchy<int, float>& condensed_tree,
                           HDBSCAN::Common::PredictionData<int, float>& prediction_data,
                           const float* X,
                           int* labels,
                           const float* points_to_predict,
                           size_t n_prediction_points,
                           raft::distance::DistanceType metric,
                           int min_samples,
                           int* out_labels,
                           float* out_probabilities)
{
  // Note that (min_samples+1) is parsed to the approximate_predict function. This was done for the
  // core distance computation to consistent with Scikit learn Contrib.
  HDBSCAN::detail::Predict::approximate_predict(handle,
                                                condensed_tree,
                                                prediction_data,
                                                X,
                                                labels,
                                                points_to_predict,
                                                n_prediction_points,
                                                metric,
                                                min_samples + 1,
                                                out_labels,
                                                out_probabilities);
}
};  // end namespace ML
