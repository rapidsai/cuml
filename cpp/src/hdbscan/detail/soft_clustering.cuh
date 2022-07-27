// /*
//  * Copyright (c) 2021-2022, NVIDIA CORPORATION.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#pragma once

#include "kernels/soft_clustering.cuh"
#include "utils.h"
#include "select.cuh"

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <raft/sparse/convert/csr.hpp>
#include <raft/sparse/op/sort.hpp>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.hpp>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/matrix_vector_op.hpp>
#include <raft/linalg/norm.hpp>
#include <raft/linalg/transpose.hpp>
#include <raft/matrix/math.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
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
namespace Membership {

template <typename value_idx, typename value_t>
void build_prediction_data(const raft::handle_t& handle,
                                Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                value_idx* labels,
                                value_idx* label_map,
                                value_idx n_selected_clusters,
                                Common::PredictionData<value_idx, value_t>& prediction_data)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  CUML_LOG_INFO("Building prediction data");
  
  auto counting = thrust::make_counting_iterator<int>(0);
  
    auto parents    = condensed_tree.get_parents();
    auto children   = condensed_tree.get_children();
    auto lambdas    = condensed_tree.get_lambdas();
    auto n_edges    = condensed_tree.get_n_edges();
    auto n_clusters = condensed_tree.get_n_clusters();
    auto n_leaves   = condensed_tree.get_n_leaves();
    auto sizes      = condensed_tree.get_sizes();
    
    CUML_LOG_INFO("n_clusters: %d", n_clusters);
    CUML_LOG_INFO("n_edges: %d", n_edges);
    CUML_LOG_INFO("n_leaves: %d", n_leaves);

    rmm::device_uvector<value_t> deaths(n_clusters, stream);

    rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
    raft::copy_async(sorted_parents.data(), parents, n_edges, stream);
  
    rmm::device_uvector<value_idx> sorted_parents_offsets(n_clusters + 1, stream);
    Utils::parent_csr(handle, condensed_tree, sorted_parents.data(), sorted_parents_offsets.data());
  
    // this is to find maximum lambdas of all children under a parent
    Utils::cub_segmented_reduce(
      lambdas,
      deaths.data(),
      n_clusters,
      sorted_parents_offsets.data(),
      stream,
      cub::DeviceSegmentedReduce::Max<const value_t*, value_t*, const value_idx*, const value_idx*>);
    
    handle.sync_stream(stream);
    cudaDeviceSynchronize();
    CUML_LOG_INFO("Deaths computed successfully");
  
    rmm::device_uvector<int> is_leaf_cluster(n_clusters, stream);
    thrust::fill(exec_policy, is_leaf_cluster.begin(), is_leaf_cluster.end(), 1);
  
    auto leaf_cluster_op =
      [is_leaf_cluster = is_leaf_cluster.data(),
       parents,
       sizes,
       n_leaves] __device__(auto idx) {
        if (sizes[idx] > 1) {
          is_leaf_cluster[parents[idx] - n_leaves] = 0;
        }
        return;
       };
  
    thrust::for_each(exec_policy,
                     counting,
                     counting + n_edges,
                     leaf_cluster_op);
  
    rmm::device_uvector<value_idx> is_exemplar(n_leaves, stream);
    rmm::device_uvector<value_idx> exemplar_idx(n_leaves, stream);
    rmm::device_uvector<value_idx> exemplar_label_offsets(n_selected_clusters, stream);
    rmm::device_uvector<value_idx> selected_clusters(n_selected_clusters, stream);
    auto exemplar_op =
      [is_exemplar = is_exemplar.data(),
       lambdas,
       is_leaf_cluster = is_leaf_cluster.data(),
       parents,
       children,
       n_leaves,
       deaths = deaths.data()] __device__(auto idx) {
        if (children[idx] < n_leaves) {
          is_exemplar[children[idx]] = (is_leaf_cluster[parents[idx] - n_leaves] && lambdas[idx] == deaths[parents[idx] - n_leaves]);
        return;
        }
       };
  
    thrust::for_each(exec_policy,
                     counting,
                     counting + n_edges,
                     exemplar_op);
  
    auto exemplar_idx_end_ptr = thrust::copy_if(exec_policy,
                                                counting,
                                                counting + n_leaves,
                                                exemplar_idx.data(),
                                                [is_exemplar = is_exemplar.data()] __device__(auto idx) { return is_exemplar[idx]; });
   
    value_idx n_exemplars = exemplar_idx_end_ptr - exemplar_idx.data();
    // return n_exemplars;

    rmm::device_uvector<value_idx> exemplar_labels(n_exemplars, stream);

  thrust::transform(
    exec_policy,
    exemplar_idx.data(),
    exemplar_idx.data() + n_exemplars,
    exemplar_labels.data(),
    [labels] __device__(auto idx) { return labels[idx]; });

  thrust::sort_by_key(exec_policy,
                      exemplar_labels.data(),
                      exemplar_labels.data() + n_exemplars,
                      exemplar_idx.data());
  raft::print_device_vector("exemplars", exemplar_idx.data(), n_exemplars, std::cout);
  rmm::device_uvector<value_idx> converted_exemplar_labels(n_exemplars, stream);
  thrust::transform(
    exec_policy,
    exemplar_labels.begin(),
    exemplar_labels.end(),
    converted_exemplar_labels.data(),
    [label_map] __device__(auto idx) { return label_map[idx]; });
  
  raft::sparse::convert::sorted_coo_to_csr(converted_exemplar_labels.data(), n_exemplars, exemplar_label_offsets.data(), n_selected_clusters + 1, stream);

  thrust::transform(
    exec_policy,
    exemplar_label_offsets.data(),
    exemplar_label_offsets.data() + n_selected_clusters,
    selected_clusters.data(),
    [exemplar_labels = exemplar_labels.data(), n_leaves] __device__(auto idx) { return exemplar_labels[idx] + n_leaves; });
  
  handle.sync_stream(stream);
  cudaDeviceSynchronize();
    // raft::print_device_vector("prob_in_some_cluster", prob_in_some_cluster.data(), 3, std::cout);
  CUML_LOG_INFO("Prediction Data successfully constructed");

  prediction_data.cache(handle, n_exemplars, n_clusters, n_selected_clusters, deaths.data(), exemplar_idx.data(), exemplar_label_offsets.data(), selected_clusters.data());
}

template <typename value_idx, typename value_t>
void all_points_dist_membership_vector(const raft::handle_t& handle,
                                       const value_t* X,
                                       size_t m,
                                       size_t n,
                                       size_t n_exemplars,
                                       size_t n_selected_clusters,
                                       value_idx* exemplar_idx,
                                       value_idx* exemplar_label_offsets,
                                       value_t* dist_membership_vec,
                                       raft::distance::DistanceType metric,
                                       bool softmax = false)
{
  auto stream = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<value_idx>(0);
 
  rmm::device_uvector<value_t> exemplars_dense(n_exemplars * n, stream);

  raft::matrix::copyRows<value_t, value_idx, size_t>(X,
                                                     n_exemplars,
                                                     n,
                                                     exemplars_dense.data(),
                                                     exemplar_idx,
                                                     n_exemplars,
                                                     stream,
                                                     true);
  
  rmm::device_uvector<value_t> dist(m * n_exemplars, stream);
  raft::distance::distance<raft::distance::DistanceType::L2SqrtExpanded, value_t, value_t, value_t, int>(
    X, exemplars_dense.data(), dist.data(), m, n_exemplars, n, stream, true);
  
  // for(int i = 0; i < 3 * n_exemplars; i++){
  //     CUML_LOG_DEBUG("%f\n", dist.element(i, stream));
  // }

  rmm::device_uvector<value_t> min_dist(m * n_selected_clusters, stream);
  thrust::fill(exec_policy, min_dist.begin(), min_dist.end(), std::numeric_limits<value_t>::max());
  
  auto reduction_op =
    [dist = dist.data(),
     n_selected_clusters,
     n_exemplars,
     exemplar_label_offsets,
     min_dist = min_dist.data()]
     __device__(auto idx) {
      auto col = idx % n_selected_clusters;
      auto row = idx / n_selected_clusters;
      auto start = exemplar_label_offsets[col];
      auto end = exemplar_label_offsets[col + 1];
    
      for(value_idx i = start; i < end; i++){
        if (dist[row * n_exemplars + i] < min_dist[row * n_selected_clusters + col]){
          min_dist[row * n_selected_clusters + col] = dist[row * n_exemplars + i];
        }
      }
       return;
     };

  thrust::for_each(
    exec_policy,
    counting,
    counting + m * n_selected_clusters,
    reduction_op);
    // for(int i = 0; i < 10 * n_selected_clusters; i++){
    //   CUML_LOG_DEBUG("%f\n", min_dist.element(i, stream));
    // }  
  if (softmax){
    thrust::transform(
      exec_policy,
      min_dist.data(),
      min_dist.data() + m * n_selected_clusters,
      dist_membership_vec,
      [=] __device__(value_t val){
        if(val != 0){
          return value_t(exp(1.0/val - std::numeric_limits<value_t>::max()));
        }
        return value_t(1.0);
      }
    );
  }

  else{
    thrust::transform(
      exec_policy,
      min_dist.data(),
      min_dist.data() + m * n_selected_clusters,
      dist_membership_vec,
      [=] __device__(value_t val){
        if(val > 0){
          return value_t(1.0 / val);
        }
        return std::numeric_limits<value_t>::max() / n_selected_clusters;
        // return value_t(DBL_MAX/n_selected_clusters);
      }
    );
  }

  Utils::normalize(dist_membership_vec, n_selected_clusters, m, stream);
};

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_outlier_membership_vector(
  const raft::handle_t& handle,
  Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
  value_t* deaths,
  value_idx* selected_clusters,
  int m,
  int n_selected_clusters,
  value_t* merge_heights,
  value_t* outlier_membership_vec,
  bool softmax
)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  value_t* lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  rmm::device_uvector<value_idx> index_into_children(n_edges, stream);
  auto index_op = [index_into_children = index_into_children.data()] __device__(auto t) {
    index_into_children[thrust::get<0>(t)] = thrust::get<1>(t);
    return;
  };
  thrust::for_each(
    exec_policy,
    thrust::make_zip_iterator(thrust::make_tuple(children, counting)),
    thrust::make_zip_iterator(thrust::make_tuple(children + n_edges, counting + n_edges)),
    index_op
  );

  // raft::print_device_vector("index_into_children", index_into_children.data(), 30, std::cout);
  // raft::print_device_vector("children", children, 30, std::cout);

  int n_blocks = raft::ceildiv(m * n_selected_clusters, tpb);
  CUML_LOG_DEBUG("n_blocks %d", n_blocks);
  CUML_LOG_DEBUG("tpb %d", tpb);
  merge_height_kernel<<<n_blocks, tpb, 0, stream>>>(merge_heights,
                                                    lambdas,
                                                    index_into_children.data(),
                                                    parents,
                                                    m,
                                                    n_selected_clusters,
                                                    selected_clusters);

  rmm::device_uvector<value_t> leaf_max_lambdas(n_leaves, stream);

  thrust::for_each(
    exec_policy,
    counting,
    counting + n_leaves,
    [deaths,
    parents,
    index_into_children = index_into_children.data(),
    leaf_max_lambdas = leaf_max_lambdas.data(),
    n_leaves] __device__(auto idx) { 
      leaf_max_lambdas[idx] = deaths[parents[index_into_children[idx]] - n_leaves];});

  raft::linalg::matrixVectorOp(
    outlier_membership_vec,
    merge_heights,
    leaf_max_lambdas.data(),
    n_selected_clusters,
    m,
    true,
    false,
    [] __device__(value_t mat_in, value_t vec_in) { return exp(-(vec_in + 1e-8) / mat_in); }, //+ 1e-8 to avoid zero lambda
    stream);
    
    handle.sync_stream(stream);
    cudaDeviceSynchronize();
    
    // raft::print_device_vector("leaf_max_lambdas", leaf_max_lambdas.data() + 14, 10, std::cout);
    // for(int i = 0; i < 10; i++){
    //   raft::print_device_vector("merge_heights", merge_heights + i * n_selected_clusters, n_selected_clusters, std::cout);
    //   raft::print_device_vector("outlier_membership_vec", outlier_membership_vec + i * n_selected_clusters, n_selected_clusters, std::cout);
    // }
    
  if (softmax){
    thrust::transform(
      exec_policy,
      outlier_membership_vec,
      outlier_membership_vec + m * n_selected_clusters,
      outlier_membership_vec,
      [=] __device__(value_t val){
          return exp(val);
      }
    );
  }

  Utils::normalize(outlier_membership_vec, n_selected_clusters, m, stream);
  // handle.sync_stream(stream);
  //   cudaDeviceSynchronize();
  // for(int i = 0; i < 10; i++){
  //     raft::print_device_vector("outlier_membership_vec", outlier_membership_vec + i * n_selected_clusters, n_selected_clusters, std::cout);
  //   }
}

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_prob_in_some_cluster(const raft::handle_t& handle,
                                     Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                     value_t* deaths,
                                     value_idx* selected_clusters,
                                     int m,
                                     int n_selected_clusters,
                                     value_t* merge_heights,
                                     value_t* prob_in_some_cluster)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  value_t* lambdas = condensed_tree.get_lambdas();
  auto n_leaves = condensed_tree.get_n_leaves();
  auto n_edges = condensed_tree.get_n_edges();
  auto children = condensed_tree.get_children();
  rmm::device_uvector<value_t> height_argmax(m, stream);
  raft::matrix::argmax(merge_heights, n_selected_clusters, m, height_argmax.data(), stream);

  // handle.sync_stream(stream);
  // cudaDeviceSynchronize();
  // raft::print_device_vector("heights_vertical", tmp_merge_heights.data(), n_selected_clusters, std::cout);  
  // for(int i = 0; i < n_selected_clusters; i++){
  //   raft::print_device_vector("heights_vertical", tmp_merge_heights.data() + m * i, 1, std::cout);  
  // }
  // for(int i = 0; i < 15; i++){
  //   raft::print_device_vector("heights", merge_heights + n_selected_clusters * i, n_selected_clusters, std::cout);  
  // }
  // raft::print_device_vector("height_argmax", height_argmax.data(), 15, std::cout);

  rmm::device_uvector<value_idx> index_into_children(n_edges, stream);
  auto counting = thrust::make_counting_iterator<value_idx>(0);

  auto index_op = [index_into_children = index_into_children.data()] __device__(auto t) {
    index_into_children[thrust::get<0>(t)] = thrust::get<1>(t);
    return;
  };
  thrust::for_each(
    exec_policy,
    thrust::make_zip_iterator(thrust::make_tuple(children, counting)),
    thrust::make_zip_iterator(thrust::make_tuple(children + n_edges, counting + n_edges)),
    index_op
  );
  
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

  handle.sync_stream(stream);
  cudaDeviceSynchronize();
  // for(int i = 0; i < 15; i++){
  //     raft::print_device_vector("heights", merge_heights + n_selected_clusters * i, n_selected_clusters, std::cout);
  //     raft::print_device_vector("heights", merge_heights + i * n_selected_clusters + (int)height_argmax.element(i, stream), 1, std::cout);
  //   }
  // raft::print_device_vector("prob_in_some_cluster", prob_in_some_cluster, 10, std::cout);
    // raft::print_device_vector("height_argmax", height_argmax.data(), 15, std::cout);
    

}

template <typename value_idx, typename value_t>
void all_points_membership_vectors(const raft::handle_t& handle,
                                   Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                   Common::PredictionData<value_idx, value_t>& prediction_data,
                                   value_t* membership_vec,
                                   const value_t* X,
                                   raft::distance::DistanceType metric)
{
  // auto stream      = handle.get_stream();
  // auto exec_policy = handle.get_thrust_policy();

  // auto parents    = condensed_tree.get_parents();
  // auto children   = condensed_tree.get_children();
  // value_t* lambdas    = condensed_tree.get_lambdas();
  // auto n_edges    = condensed_tree.get_n_edges();
  // auto n_clusters = condensed_tree.get_n_clusters();
  // auto n_leaves   = condensed_tree.get_n_leaves();
  
  // auto m = prediction_data.n_rows;
  // auto n = prediction_data.n_cols;
  // auto n_selected_clusters = prediction_data.get_n_selected_clusters();
  // auto deaths = prediction_data.get_deaths();
  // auto selected_clusters = prediction_data.get_selected_clusters();
  // auto n_exemplars = prediction_data.get_n_exemplars();
  // // CUML_LOG_DEBUG("n_selected_clusters: %d", n_selected_clusters);
  // // CUML_LOG_DEBUG("n_exemplars: %d", n_exemplars);
  // // raft::print_device_vector("selected_clusters", prediction_data.get_selected_clusters(), n_selected_clusters, std::cout);

  // // rmm::device_uvector<value_t> deaths(n_clusters, stream);
  // // rmm::device_uvector<value_idx> selected_clusters(n_selected_clusters, stream);

  // // compute_deaths(handle, condensed_tree, deaths.data());

  // rmm::device_uvector<value_t> dist_membership_vec(m * n_selected_clusters, stream);
  // all_points_dist_membership_vector(handle,
  //                                   X,
  //                                   m,
  //                                   n,
  //                                   n_exemplars,
  //                                   n_selected_clusters,
  //                                   prediction_data.get_exemplar_idx(),
  //                                   prediction_data.get_exemplar_label_offsets(),
  //                                   dist_membership_vec.data(),
  //                                   metric);

  // // raft::print_device_vector("parents", cluster_tree_parents, cluster_tree_edges, std::cout);
  // // raft::print_device_vector("children", cluster_tree_children, cluster_tree_edges, std::cout);
  // rmm::device_uvector<value_t> merge_heights(m * n_selected_clusters, stream);

  // all_points_outlier_membership_vector(handle,
  //                                      condensed_tree,
  //                                      deaths,
  //                                      selected_clusters,
  //                                      m,
  //                                      n_selected_clusters,
  //                                      merge_heights.data(),
  //                                      membership_vec,
  //                                      true);
  // // raft::print_device_vector("merge_heights", merge_heights.data(), 2*n_selected_clusters, std::cout);
  // rmm::device_uvector<value_t> prob_in_some_cluster(m, stream);
  // all_points_prob_in_some_cluster(handle,
  //                                 condensed_tree,
  //                                 deaths,
  //                                 selected_clusters,
  //                                 m,
  //                                 n_selected_clusters,
  //                                 merge_heights.data(),
  //                                 prob_in_some_cluster.data());
  // thrust::transform(exec_policy, dist_membership_vec.begin(),
  //   dist_membership_vec.end(),
  //   membership_vec,
  //   membership_vec,
  //   thrust::multiplies<value_t>());
  
  //   raft::print_device_vector("membership_vec_1", membership_vec + 15, 30, std::cout);
  //   Utils::normalize(membership_vec, n_selected_clusters, m, stream);
  
  // raft::print_device_vector("membership_vec_2", membership_vec + 15, 30, std::cout);
  // raft::linalg::matrixVectorOp(
  //   membership_vec,
  //   membership_vec,
  //   prob_in_some_cluster.data(),
  //   n_selected_clusters,
  //   m,
  //   true,
  //   false,
  //   [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
  //   stream);

  //   handle.sync_stream(stream);
  //   cudaDeviceSynchronize();
  //   // raft::print_device_vector("prob_in_some_cluster", prob_in_some_cluster.data(), 3, std::cout);
  //   raft::print_device_vector("membership_vec_from_device(for data[6:11])", membership_vec + 30, 30, std::cout);
}

};  // namespace Membership
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
