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

#include "kernels/soft_clustering.cuh"
#include "select.cuh"
#include "utils.h"

#include <cub/cub.cuh>

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <raft/sparse/convert/csr.hpp>
#include <raft/sparse/op/sort.hpp>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/label/classlabels.hpp>
#include <raft/linalg/matrix_vector_op.hpp>
#include <raft/linalg/norm.hpp>
#include <raft/matrix/math.hpp>

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

template <typename value_idx, typename value_t>
void all_points_dist_membership_vector(const raft::handle_t& handle,
                                       const value_t* X,
                                       size_t m,
                                       size_t n,
                                       size_t n_exemplars,
                                       value_idx n_selected_clusters,
                                       value_idx* exemplar_idx,
                                       value_idx* exemplar_label_offsets,
                                       value_t* dist_membership_vec,
                                       raft::distance::DistanceType metric,
                                       bool softmax = false)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  rmm::device_uvector<value_t> exemplars_dense(n_exemplars * n, stream);

  // use the exemplar point indices to obtain the exemplar points as a dense array
  raft::matrix::copyRows<value_t, value_idx, size_t>(
    X, n_exemplars, n, exemplars_dense.data(), exemplar_idx, n_exemplars, stream, true);

  // compute the distances using raft API
  rmm::device_uvector<value_t> dist(m * n_exemplars, stream);

  switch (metric) {
    case raft::distance::DistanceType::L2SqrtExpanded:
      raft::distance::
        distance<raft::distance::DistanceType::L2SqrtExpanded, value_t, value_t, value_t, int>(
          X, exemplars_dense.data(), dist.data(), m, n_exemplars, n, stream, true);
      break;
    case raft::distance::DistanceType::L1:
      raft::distance::distance<raft::distance::DistanceType::L1, value_t, value_t, value_t, int>(
        X, exemplars_dense.data(), dist.data(), m, n_exemplars, n, stream, true);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      raft::distance::
        distance<raft::distance::DistanceType::CosineExpanded, value_t, value_t, value_t, int>(
          X, exemplars_dense.data(), dist.data(), m, n_exemplars, n, stream, true);
      break;
    default: ASSERT(false, "Incorrect metric passed!");
  }

  // compute the minimum distances to exemplars of each cluster
  rmm::device_uvector<value_t> min_dist(m * n_selected_clusters, stream);
  thrust::fill(exec_policy, min_dist.begin(), min_dist.end(), std::numeric_limits<value_t>::max());

  auto reduction_op = [dist = dist.data(),
                       n_selected_clusters,
                       n_exemplars,
                       exemplar_label_offsets,
                       min_dist = min_dist.data()] __device__(auto idx) {
    auto col   = idx % n_selected_clusters;
    auto row   = idx / n_selected_clusters;
    auto start = exemplar_label_offsets[col];
    auto end   = exemplar_label_offsets[col + 1];

    for (value_idx i = start; i < end; i++) {
      if (dist[row * n_exemplars + i] < min_dist[row * n_selected_clusters + col]) {
        min_dist[row * n_selected_clusters + col] = dist[row * n_exemplars + i];
      }
    }
    return;
  };

  thrust::for_each(exec_policy, counting, counting + m * n_selected_clusters, reduction_op);

  // Softmax computation is ignored in distance membership
  if (softmax) {
    thrust::transform(exec_policy,
                      min_dist.data(),
                      min_dist.data() + m * n_selected_clusters,
                      dist_membership_vec,
                      [=] __device__(value_t val) {
                        if (val != 0) { return value_t(exp(1.0 / val)); }
                        return std::numeric_limits<value_t>::max();
                      });
  }

  // Transform the distances to obtain membership based on proximity to exemplars
  else {
    thrust::transform(exec_policy,
                      min_dist.data(),
                      min_dist.data() + m * n_selected_clusters,
                      dist_membership_vec,
                      [=] __device__(value_t val) {
                        if (val > 0) { return value_t(1.0 / val); }
                        return std::numeric_limits<value_t>::max() / n_selected_clusters;
                      });
  }

  // Normalize the obtained result to sum to 1.0
  Utils::normalize(dist_membership_vec, n_selected_clusters, m, stream);
};

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_outlier_membership_vector(
  const raft::handle_t& handle,
  Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
  value_t* deaths,
  value_idx* selected_clusters,
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

  rmm::device_uvector<value_idx> index_into_children(n_edges + 1, stream);
  auto index_op = [index_into_children = index_into_children.data()] __device__(auto t) {
    index_into_children[thrust::get<0>(t)] = thrust::get<1>(t);
    return;
  };

  auto it = thrust::make_zip_iterator(thrust::make_tuple(children, counting));
  thrust::for_each(exec_policy, it, it + n_edges, index_op);

  int n_blocks = raft::ceildiv(int(m * n_selected_clusters), tpb);
  merge_height_kernel<<<n_blocks, tpb, 0, stream>>>(merge_heights,
                                                    lambdas,
                                                    index_into_children.data(),
                                                    parents,
                                                    m,
                                                    n_selected_clusters,
                                                    selected_clusters);

  rmm::device_uvector<value_t> leaf_max_lambdas(n_leaves, stream);

  thrust::for_each(exec_policy,
                   counting,
                   counting + n_leaves,
                   [deaths,
                    parents,
                    index_into_children = index_into_children.data(),
                    leaf_max_lambdas    = leaf_max_lambdas.data(),
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

  if (softmax) {
    thrust::transform(exec_policy,
                      outlier_membership_vec,
                      outlier_membership_vec + m * n_selected_clusters,
                      outlier_membership_vec,
                      [=] __device__(value_t val) { return exp(val); });
  }

  Utils::normalize(outlier_membership_vec, n_selected_clusters, m, stream);
}

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_prob_in_some_cluster(const raft::handle_t& handle,
                                     Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                     value_t* deaths,
                                     value_idx* selected_clusters,
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

  rmm::device_uvector<value_t> height_argmax(m, stream);

  raft::matrix::argmax(merge_heights, n_selected_clusters, m, height_argmax.data(), stream);

  rmm::device_uvector<value_idx> index_into_children(n_edges + 1, stream);
  auto counting = thrust::make_counting_iterator<value_idx>(0);

  auto index_op = [index_into_children = index_into_children.data()] __device__(auto t) {
    index_into_children[thrust::get<0>(t)] = thrust::get<1>(t);
    return;
  };
  thrust::for_each(
    exec_policy,
    thrust::make_zip_iterator(thrust::make_tuple(children, counting)),
    thrust::make_zip_iterator(thrust::make_tuple(children + n_edges, counting + n_edges)),
    index_op);

  int n_blocks = raft::ceildiv((int)m, tpb);
  prob_in_some_cluster_kernel<<<n_blocks, tpb, 0, stream>>>(merge_heights,
                                                            height_argmax.data(),
                                                            deaths,
                                                            index_into_children.data(),
                                                            selected_clusters,
                                                            lambdas,
                                                            prob_in_some_cluster,
                                                            n_selected_clusters,
                                                            n_leaves,
                                                            m);
}

template <typename value_idx, typename value_t>
void all_points_membership_vectors(const raft::handle_t& handle,
                                   Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                   Common::PredictionData<value_idx, value_t>& prediction_data,
                                   value_t* membership_vec,
                                   const value_t* X,
                                   raft::distance::DistanceType metric)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  size_t m                      = prediction_data.n_rows;
  size_t n                      = prediction_data.n_cols;
  value_idx n_selected_clusters = prediction_data.get_n_selected_clusters();
  value_t* deaths               = prediction_data.get_deaths();
  value_idx* selected_clusters  = prediction_data.get_selected_clusters();
  value_idx n_exemplars         = prediction_data.get_n_exemplars();

  rmm::device_uvector<value_t> dist_membership_vec(m * n_selected_clusters, stream);

  all_points_dist_membership_vector(handle,
                                    X,
                                    m,
                                    n,
                                    n_exemplars,
                                    n_selected_clusters,
                                    prediction_data.get_exemplar_idx(),
                                    prediction_data.get_exemplar_label_offsets(),
                                    dist_membership_vec.data(),
                                    metric);

  rmm::device_uvector<value_t> merge_heights(m * n_selected_clusters, stream);

  all_points_outlier_membership_vector(handle,
                                       condensed_tree,
                                       deaths,
                                       selected_clusters,
                                       m,
                                       n_selected_clusters,
                                       merge_heights.data(),
                                       membership_vec,
                                       true);

  rmm::device_uvector<value_t> prob_in_some_cluster(m, stream);
  all_points_prob_in_some_cluster(handle,
                                  condensed_tree,
                                  deaths,
                                  selected_clusters,
                                  m,
                                  n_selected_clusters,
                                  merge_heights.data(),
                                  prob_in_some_cluster.data());

  thrust::transform(exec_policy,
                    dist_membership_vec.begin(),
                    dist_membership_vec.end(),
                    membership_vec,
                    membership_vec,
                    thrust::multiplies<value_t>());

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

};  // namespace Predict
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
