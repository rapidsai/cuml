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

#include "cuml/common/logger.hpp"
#include "kernels/soft_clustering.cuh"
#include "select.cuh"
#include "utils.h"

#include <cub/cub.cuh>
#include <common/fast_int_div.cuh>

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/argmax.cuh>

#include <algorithm>
#include <cmath>
#include <limits>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

// This function is used for both -- all_points_membership_vectors and membership_vector
template <typename value_idx, typename value_t>
void dist_membership_vector(const raft::handle_t& handle,
                            const value_t* X,
                            const value_t* query,
                            size_t n_queries,
                            size_t n,
                            size_t n_exemplars,
                            value_idx n_selected_clusters,
                            value_idx* exemplar_idx,
                            value_idx* exemplar_label_offsets,
                            value_t* dist_membership_vec,
                            raft::distance::DistanceType metric,
                            int batch_size,
                            bool softmax = false)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  size_t free_memory, total_memory;

  rmm::device_uvector<value_t> exemplars_dense(n_exemplars * n, stream);

  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("After dense exemplars matrix (n_exemplars * n_features) was declared . Free memory: %zu; Total memory: %zu", free_memory, total_memory);
  CUML_LOG_INFO("n_exemplars %d", n_exemplars);

  // use the exemplar point indices to obtain the exemplar points as a dense array
  raft::matrix::copyRows<value_t, value_idx, size_t>(
    X, n_exemplars, n, exemplars_dense.data(), exemplar_idx, n_exemplars, stream, true);

  // compute the distances using raft API
  value_idx n_batches;

  if (batch_size == 0) {
    n_batches = 1;
    batch_size = n_queries;
  }
  else {
    n_batches = raft::ceildiv((int)n_queries, (int)batch_size);
  }
  CUML_LOG_INFO("n_batches %d", n_batches);
  for(value_idx bid = 0; bid < n_batches; bid++) {
    value_idx samples_per_batch = min(batch_size, (int)n_queries - bid*batch_size);
    value_idx batch_offset = bid * batch_size;
    CUML_LOG_INFO("bid %d, size %d", bid, samples_per_batch);
    rmm::device_uvector<value_t> dist(samples_per_batch * n_exemplars, stream);

    handle.sync_stream(stream);
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
    CUML_LOG_INFO("After distance matrix (n_samples * n_exemplars) was generated . Free memory: %zu; Total memory: %zu", free_memory, total_memory);

    switch (metric) {
      case raft::distance::DistanceType::L2SqrtExpanded:
        raft::distance::
          distance<raft::distance::DistanceType::L2SqrtExpanded, value_t, value_t, value_t, int>(
          handle, query + batch_offset * n, exemplars_dense.data(), dist.data(), samples_per_batch, n_exemplars, n, true);
      break;
    case raft::distance::DistanceType::L1:
      raft::distance::distance<raft::distance::DistanceType::L1, value_t, value_t, value_t, int>(
        handle, query + batch_offset * n, exemplars_dense.data(), dist.data(), samples_per_batch, n_exemplars, n, true);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      raft::distance::
        distance<raft::distance::DistanceType::CosineExpanded, value_t, value_t, value_t, int>(
          handle, query + batch_offset * n, exemplars_dense.data(), dist.data(), samples_per_batch, n_exemplars, n, true);
      break;
    default: ASSERT(false, "Incorrect metric passed!");
  }

  // compute the minimum distances to exemplars of each cluster
  rmm::device_uvector<value_t> min_dist(samples_per_batch * n_selected_clusters, stream);

  thrust::fill(exec_policy, min_dist.begin(), min_dist.end(), std::numeric_limits<value_t>::max());

  auto reduction_op = [dist = dist.data(),
                       divisor = MLCommon::FastIntDiv(n_selected_clusters),
                       n_selected_clusters,
                       n_exemplars,
                       exemplar_label_offsets,
                       min_dist = min_dist.data()] __device__(auto idx) {
    auto col   = idx % divisor;
    auto row   = idx / divisor;
    auto start = exemplar_label_offsets[col];
    auto end   = exemplar_label_offsets[col + 1];

    for (value_idx i = start; i < end; i++) {
      if (dist[row * n_exemplars + i] < min_dist[row * n_selected_clusters + col]) {
        min_dist[row * n_selected_clusters + col] = dist[row * n_exemplars + i];
      }
    }
    return;
  };

  thrust::for_each(exec_policy, counting, counting + samples_per_batch * n_selected_clusters, reduction_op);

  // Softmax computation is ignored in distance membership
  if (softmax) {
    thrust::transform(exec_policy,
                      min_dist.begin(),
                      min_dist.end(),
                      dist_membership_vec + batch_offset * n_selected_clusters,
                      [=] __device__(value_t val) {
                        if (val != 0) { return value_t(exp(1.0 / val)); }
                        return std::numeric_limits<value_t>::max();
                      });
  }

  // Transform the distances to obtain membership based on proximity to exemplars
  else {
    thrust::transform(exec_policy,
                      min_dist.begin(),
                      min_dist.end(),
                      dist_membership_vec + batch_offset * n_selected_clusters,
                      [=] __device__(value_t val) {
                        if (val > 0) { return value_t(1.0 / val); }
                        return std::numeric_limits<value_t>::max() / n_selected_clusters;
                      });
  }
  }

  handle.sync_stream(stream);
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
    CUML_LOG_INFO("Total memory usage during dist_membership_vec computation . Free memory: %zu; Total memory: %zu", free_memory, total_memory);

  // Normalize the obtained result to sum to 1.0
  Utils::normalize(dist_membership_vec, n_selected_clusters, n_queries, stream);
};

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_outlier_membership_vector(
  const raft::handle_t& handle,
  Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
  value_t* deaths,
  value_idx* selected_clusters,
  value_idx* index_into_children,
  size_t m,
  int n_selected_clusters,
  value_t* merge_heights,
  value_t* outlier_membership_vec,
  bool softmax)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents      = condensed_tree.get_parents();
  auto children     = condensed_tree.get_children();
  value_t* lambdas  = condensed_tree.get_lambdas();
  value_idx n_edges = condensed_tree.get_n_edges();
  auto n_clusters   = condensed_tree.get_n_clusters();
  auto n_leaves     = condensed_tree.get_n_leaves();

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  int n_blocks = raft::ceildiv(int(m * n_selected_clusters), tpb);
  merge_height_kernel<<<n_blocks, tpb, 0, stream>>>(merge_heights,
                                                    lambdas,
                                                    index_into_children,
                                                    parents,
                                                    m,
                                                    n_selected_clusters,
                                                    MLCommon::FastIntDiv(n_selected_clusters),
                                                    selected_clusters);

  size_t free_memory, total_memory;
  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("Before leaf_max_lambdas was declared (size n_leaves). Free memory: %zu; Total memory: %zu", free_memory, total_memory);

  rmm::device_uvector<value_t> leaf_max_lambdas(n_leaves, stream);

  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("After leaf_max_lambdas was declared (size n_leaves). Free memory: %zu; Total memory: %zu", free_memory, total_memory);

  thrust::for_each(exec_policy,
                   counting,
                   counting + n_leaves,
                   [deaths,
                    parents,
                    index_into_children,
                    leaf_max_lambdas = leaf_max_lambdas.data(),
                    n_leaves] __device__(auto idx) {
                     leaf_max_lambdas[idx] = deaths[parents[index_into_children[idx]] - n_leaves];
                   });

  raft::linalg::matrixVectorOp(
    outlier_membership_vec,
    merge_heights,
    leaf_max_lambdas.data(),
    n_selected_clusters,
    (value_idx)m,
    true,
    false,
    [] __device__(value_t mat_in, value_t vec_in) {
      return exp(-(vec_in + 1e-8) / mat_in);
    },  //+ 1e-8 to avoid zero lambda
    stream);

  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("Total memory usage during outlier membership computation (before normalizing and softmax). Free memory: %zu; Total memory: %zu", free_memory, total_memory);

  if (softmax) { Utils::softmax(handle, outlier_membership_vec, n_selected_clusters, m); }

  Utils::normalize(outlier_membership_vec, n_selected_clusters, m, stream);
}

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_prob_in_some_cluster(const raft::handle_t& handle,
                                     Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                     value_t* deaths,
                                     value_idx* selected_clusters,
                                     value_idx* index_into_children,
                                     size_t m,
                                     value_idx n_selected_clusters,
                                     value_t* merge_heights,
                                     value_t* prob_in_some_cluster)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  value_t* lambdas = condensed_tree.get_lambdas();
  auto n_leaves    = condensed_tree.get_n_leaves();
  auto n_edges     = condensed_tree.get_n_edges();
  auto children    = condensed_tree.get_children();

  rmm::device_uvector<value_idx> height_argmax(m, stream);

  auto merge_heights_view = raft::make_device_matrix_view<const value_t, value_idx, raft::row_major>(merge_heights, (int)m, n_selected_clusters);
  auto height_argmax_view = raft::make_device_vector_view<value_idx, value_idx>(height_argmax.data(), (int)m);

  raft::matrix::argmax(
    handle, merge_heights_view, height_argmax_view);

  int n_blocks = raft::ceildiv((int)m, tpb);
  prob_in_some_cluster_kernel<<<n_blocks, tpb, 0, stream>>>(merge_heights,
                                                            height_argmax.data(),
                                                            deaths,
                                                            index_into_children,
                                                            selected_clusters,
                                                            lambdas,
                                                            prob_in_some_cluster,
                                                            n_selected_clusters,
                                                            n_leaves,
                                                            m);
  size_t free_memory, total_memory;
  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("Total memory during prob_in_some_cluster computation (just before returning from the function). Free memory: %zu; Total memory: %zu", free_memory, total_memory);
}

template <typename value_idx, typename value_t, int tpb = 256>
void outlier_membership_vector(const raft::handle_t& handle,
                               Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                               value_t* deaths,
                               value_idx* min_mr_inds,
                               value_t* prediction_lambdas,
                               value_idx* selected_clusters,
                               value_idx* index_into_children,
                               size_t n_prediction_points,
                               int n_selected_clusters,
                               value_t* merge_heights,
                               value_t* outlier_membership_vec,
                               bool softmax)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents      = condensed_tree.get_parents();
  auto children     = condensed_tree.get_children();
  value_t* lambdas  = condensed_tree.get_lambdas();
  value_idx n_edges = condensed_tree.get_n_edges();
  auto n_clusters   = condensed_tree.get_n_clusters();
  auto n_leaves     = condensed_tree.get_n_leaves();

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  // Using the nearest neighbor indices, compute outlier membership
  int n_blocks = raft::ceildiv(int(n_prediction_points * n_selected_clusters), tpb);
  merge_height_kernel<<<n_blocks, tpb, 0, stream>>>(merge_heights,
                                                    lambdas,
                                                    prediction_lambdas,
                                                    min_mr_inds,
                                                    index_into_children,
                                                    parents,
                                                    n_prediction_points,
                                                    n_selected_clusters,
                                                    MLCommon::FastIntDiv(n_selected_clusters),
                                                    selected_clusters);

  // fetch the max lambda of the cluster to which the nearest MR neighbor belongs in the condensed
  // hierarchy
  rmm::device_uvector<value_t> nearest_cluster_max_lambda(n_prediction_points, stream);

  thrust::for_each(exec_policy,
                   counting,
                   counting + n_prediction_points,
                   [deaths,
                    parents,
                    index_into_children,
                    min_mr_inds,
                    nearest_cluster_max_lambda = nearest_cluster_max_lambda.data(),
                    n_leaves] __device__(auto idx) {
                     nearest_cluster_max_lambda[idx] =
                       deaths[parents[index_into_children[min_mr_inds[idx]]] - n_leaves];
                   });

  raft::linalg::matrixVectorOp(
    outlier_membership_vec,
    merge_heights,
    nearest_cluster_max_lambda.data(),
    n_selected_clusters,
    (value_idx)n_prediction_points,
    true,
    false,
    [] __device__(value_t mat_in, value_t vec_in) {
      value_t denominator = vec_in - mat_in;
      if (denominator <= 0) { denominator = 1e-8; }
      return vec_in / denominator;
    },
    stream);

  if (softmax) {
    Utils::softmax(handle, outlier_membership_vec, n_selected_clusters, n_prediction_points);
  }
  Utils::normalize(outlier_membership_vec, n_selected_clusters, n_prediction_points, stream);
}

template <typename value_idx, typename value_t, int tpb = 256>
void prob_in_some_cluster(const raft::handle_t& handle,
                          Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                          value_t* deaths,
                          value_idx* selected_clusters,
                          value_idx* index_into_children,
                          size_t n_prediction_points,
                          value_idx n_selected_clusters,
                          value_idx* min_mr_indices,
                          value_t* merge_heights,
                          value_t* prediction_lambdas,
                          value_t* prob_in_some_cluster)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  value_t* lambdas = condensed_tree.get_lambdas();
  auto n_leaves    = condensed_tree.get_n_leaves();
  auto n_edges     = condensed_tree.get_n_edges();
  auto children    = condensed_tree.get_children();

  rmm::device_uvector<value_idx> height_argmax(n_prediction_points, stream);

  auto merge_heights_view = raft::make_device_matrix_view<const value_t, value_idx, raft::row_major>(merge_heights, (int)n_prediction_points, n_selected_clusters);
  auto height_argmax_view = raft::make_device_vector_view<value_idx, value_idx>(height_argmax.data(), (int)n_prediction_points);
  raft::matrix::argmax(
    handle, merge_heights_view, height_argmax_view);

  int n_blocks = raft::ceildiv((int)n_prediction_points, tpb);

  prob_in_some_cluster_kernel<<<n_blocks, tpb, 0, stream>>>(merge_heights,
                                                            height_argmax.data(),
                                                            prediction_lambdas,
                                                            deaths,
                                                            index_into_children,
                                                            min_mr_indices,
                                                            selected_clusters,
                                                            lambdas,
                                                            prob_in_some_cluster,
                                                            n_selected_clusters,
                                                            n_leaves,
                                                            n_prediction_points);
}

/**
 * Predict soft cluster membership vectors for all points in the original dataset the clusterer was
 * trained on
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] X all points (size m * n)
 * @param[in] metric distance metric
 * @param[out] membership_vec output membership vectors (size m * n_selected_clusters)
 * @param[in] batch_size batch size to be used while computing distance membership vector
 */
template <typename value_idx, typename value_t>
void all_points_membership_vectors(const raft::handle_t& handle,
                                   Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                   Common::PredictionData<value_idx, value_t>& prediction_data,
                                   const value_t* X,
                                   raft::distance::DistanceType metric,
                                   value_t* membership_vec,
                                   value_idx batch_size = 0)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  size_t m = prediction_data.n_rows;
  size_t n = prediction_data.n_cols;
  ASSERT(0 <= batch_size && batch_size <= m, "Invalid batch_size. batch_size should be >= 0 and <= the number of samples in the training data");

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  value_idx n_selected_clusters  = prediction_data.get_n_selected_clusters();
  value_t* deaths                = prediction_data.get_deaths();
  value_idx* selected_clusters   = prediction_data.get_selected_clusters();
  value_idx* index_into_children = prediction_data.get_index_into_children();
  value_idx n_exemplars          = prediction_data.get_n_exemplars();

  // Compute membership vectors only if the number of selected clusters is non-zero. This is done to
  // avoid CUDA run-time errors in raft primitives for pairwise distances and other kernel
  // invocations.
  if (n_selected_clusters > 0) {
    size_t free_memory, total_memory;

    handle.sync_stream(stream);
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
    CUML_LOG_INFO("Before dist_membership_vec was declared . Free memory: %zu; Total memory: %zu", free_memory, total_memory);

    rmm::device_uvector<value_t> dist_membership_vec(m * n_selected_clusters, stream);

    handle.sync_stream(stream);
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
    CUML_LOG_INFO("after dist_membership_vec was declared . Free memory: %zu; Total memory: %zu", free_memory, total_memory);
                            
    dist_membership_vector(handle,
                           X,
                           X,
                           m,
                           n,
                           n_exemplars,
                           n_selected_clusters,
                           prediction_data.get_exemplar_idx(),
                           prediction_data.get_exemplar_label_offsets(),
                           dist_membership_vec.data(),
                           metric,
                           batch_size);

    handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("before merge heights matrix was generated . Free memory: %zu; Total memory: %zu", free_memory, total_memory);

    rmm::device_uvector<value_t> merge_heights(m * n_selected_clusters, stream);


  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("after merge heights matrix was generated . Free memory: %zu; Total memory: %zu", free_memory, total_memory);
    all_points_outlier_membership_vector(handle,
                                         condensed_tree,
                                         deaths,
                                         selected_clusters,
                                         index_into_children,
                                         m,
                                         n_selected_clusters,
                                         merge_heights.data(),
                                         membership_vec,
                                         true);

    handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("before probability in some cluster vector (size m) was generated . Free memory: %zu; Total memory: %zu", free_memory, total_memory);
    rmm::device_uvector<value_t> prob_in_some_cluster(m, stream);
    all_points_prob_in_some_cluster(handle,
                                    condensed_tree,
                                    deaths,
                                    selected_clusters,
                                    index_into_children,
                                    m,
                                    n_selected_clusters,
                                    merge_heights.data(),
                                    prob_in_some_cluster.data());

     handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("after probability in some cluster vector (size m) was generated . Free memory: %zu; Total memory: %zu", free_memory, total_memory);
    thrust::transform(exec_policy,
                      dist_membership_vec.begin(),
                      dist_membership_vec.end(),
                      membership_vec,
                      membership_vec,
                      thrust::multiplies<value_t>());

     handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  CUML_LOG_INFO("before normalizing final membership vector . Free memory: %zu; Total memory: %zu", free_memory, total_memory);
    // Normalize to obtain probabilities conditioned on points belonging to some cluster
    Utils::normalize(membership_vec, n_selected_clusters, m, stream);

    // Multiply with probabilities of points belonging to some cluster to obtain joint distribution
    raft::linalg::matrixVectorOp(
      membership_vec,
      membership_vec,
      prob_in_some_cluster.data(),
      n_selected_clusters,
      (value_idx)m,
      true,
      false,
      [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
      stream);
  }
}

/**
 * Predict soft cluster membership vectors for new points (not in the training data). 
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] X all points (size m * n)
 * @param[in] points_to_predict input prediction points (size n_prediction_points * n)
 * @param[in] n_prediction_points number of prediction points
 * @param[in] metric distance metric
 * @param[in] min_samples neighborhood size during training (includes self-loop)
 * @param[out] membership_vec output membership vectors (size n_prediction_points * n_selected_clusters)
 */
template <typename value_idx, typename value_t, int tpb = 256>
void membership_vector(const raft::handle_t& handle,
                       Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                       Common::PredictionData<value_idx, value_t>& prediction_data,
                       const value_t* X,
                       const value_t* points_to_predict,
                       size_t n_prediction_points,
                       raft::distance::DistanceType metric,
                       int min_samples,
                       value_t* membership_vec)
{
  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  size_t m                       = prediction_data.n_rows;
  size_t n                       = prediction_data.n_cols;
  value_idx n_selected_clusters  = prediction_data.get_n_selected_clusters();
  value_t* deaths                = prediction_data.get_deaths();
  value_idx* selected_clusters   = prediction_data.get_selected_clusters();
  value_idx* index_into_children = prediction_data.get_index_into_children();
  value_idx n_exemplars          = prediction_data.get_n_exemplars();
  value_t* lambdas               = condensed_tree.get_lambdas();

  auto counting = thrust::make_counting_iterator<value_idx>(0);
  rmm::device_uvector<value_t> dist_membership_vec(n_prediction_points * n_selected_clusters,
                                                   stream);

  dist_membership_vector(handle,
                         X,
                         points_to_predict,
                         n_prediction_points,
                         n,
                         n_exemplars,
                         n_selected_clusters,
                         prediction_data.get_exemplar_idx(),
                         prediction_data.get_exemplar_label_offsets(),
                         dist_membership_vec.data(),
                         raft::distance::DistanceType::L2SqrtExpanded,
                         0);

  rmm::device_uvector<value_t> prediction_lambdas(n_prediction_points, stream);
  rmm::device_uvector<value_idx> min_mr_inds(n_prediction_points, stream);

  _compute_knn_and_nearest_neighbor(handle,
                                    prediction_data,
                                    X,
                                    points_to_predict,
                                    min_samples,
                                    n_prediction_points,
                                    min_mr_inds.data(),
                                    prediction_lambdas.data(),
                                    metric);

  thrust::for_each(exec_policy,
                   counting,
                   counting + n_prediction_points,
                   [lambdas,
                    index_into_children,
                    min_mr_inds        = min_mr_inds.data(),
                    prediction_lambdas = prediction_lambdas.data()] __device__(auto idx) {
                     value_t neighbor_lambda = lambdas[index_into_children[min_mr_inds[idx]]];
                     if (neighbor_lambda < prediction_lambdas[idx])
                       prediction_lambdas[idx] = neighbor_lambda;
                   });

  rmm::device_uvector<value_t> merge_heights(n_prediction_points * n_selected_clusters, stream);

  outlier_membership_vector(handle,
                            condensed_tree,
                            deaths,
                            min_mr_inds.data(),
                            prediction_lambdas.data(),
                            selected_clusters,
                            index_into_children,
                            n_prediction_points,
                            n_selected_clusters,
                            merge_heights.data(),
                            membership_vec,
                            true);

  auto combine_op = [membership_vec,
                     dist_membership_vec = dist_membership_vec.data()] __device__(auto idx) {
    membership_vec[idx] = pow(membership_vec[idx], 2) * pow(dist_membership_vec[idx], 0.5);
    return;
  };

  thrust::for_each(
    exec_policy, counting, counting + n_prediction_points * n_selected_clusters, combine_op);

  // Normalize to obtain probabilities conditioned on points belonging to some cluster
  Utils::normalize(membership_vec, n_selected_clusters, n_prediction_points, stream);

  rmm::device_uvector<value_t> prob_in_some_cluster_(n_prediction_points, stream);

  prob_in_some_cluster(handle,
                       condensed_tree,
                       deaths,
                       selected_clusters,
                       index_into_children,
                       n_prediction_points,
                       n_selected_clusters,
                       min_mr_inds.data(),
                       merge_heights.data(),
                       prediction_lambdas.data(),
                       prob_in_some_cluster_.data());

  // Multiply conditional probabilities with probability of point belonging to some cluster. This
  // gives the joint distribution.
  raft::linalg::matrixVectorOp(
    membership_vec,
    membership_vec,
    prob_in_some_cluster_.data(),
    n_selected_clusters,
    (value_idx)n_prediction_points,
    true,
    false,
    [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
    stream);
}

};  // namespace Predict
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
