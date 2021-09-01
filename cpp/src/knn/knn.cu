/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <raft/cuda_utils.cuh>
#include <raft/label/classlabels.cuh>
#include <raft/spatial/knn/ann.hpp>
#include <raft/spatial/knn/knn.hpp>
#include <raft/spatial/knn/ball_cover.hpp>
#include <rmm/device_uvector.hpp>

#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn.hpp>
#include <ml_mg_utils.cuh>
#include <selection/knn.cuh>

#include <cstddef>
#include <sstream>
#include <vector>

namespace ML {

void brute_force_knn(const raft::handle_t &handle, std::vector<float *> &input,
                     std::vector<int> &sizes, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k, bool rowMajorIndex,
                     bool rowMajorQuery, raft::distance::DistanceType metric,
                     float metric_arg) {
  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors must be the same size");

  if (D == 2 && (metric == raft::distance::DistanceType::Haversine ||
                 metric == raft::distance::DistanceType::L2Expanded ||
                 metric == raft::distance::DistanceType::L2Unexpanded ||
                 metric == raft::distance::DistanceType::L2SqrtExpanded ||
                 metric == raft::distance::DistanceType::L2SqrtUnexpanded) {

    raft::spatial::knn::BallCoverIndex<value_idx, value_t> index(
      handle, input[0], sizs[0], D, metric);

    raft::spatial::knn::rbc_build_index(handle, index, k);
    raft::spatial::knn::rbc_knn_query(handle, index, k, search_items,
                                      n, res_I, res_D);
  } else {
    raft::spatial::knn::brute_force_knn(
      handle, input, sizes, D, search_items, n, res_I, res_D, k, rowMajorIndex,
      rowMajorQuery, nullptr, metric, metric_arg);
  }
}

void approx_knn_build_index(raft::handle_t& handle,
                            raft::spatial::knn::knnIndex* index,
                            raft::spatial::knn::knnIndexParam* params,
                            raft::distance::DistanceType metric,
                            float metricArg,
                            float* index_array,
                            int n,
                            int D)
{
  raft::spatial::knn::approx_knn_build_index(
    handle, index, params, metric, metricArg, index_array, n, D);
}

void approx_knn_search(raft::handle_t& handle,
                       float* distances,
                       int64_t* indices,
                       raft::spatial::knn::knnIndex* index,
                       int k,
                       float* query_array,
                       int n)
{
  raft::spatial::knn::approx_knn_search(handle, distances, indices, index, k, query_array, n);
}

void knn_classify(raft::handle_t& handle,
                  int* out,
                  int64_t* knn_indices,
                  std::vector<int*>& y,
                  size_t n_index_rows,
                  size_t n_query_rows,
                  int k)
{
  cudaStream_t stream = handle.get_stream();

  std::vector<rmm::device_uvector<int>> uniq_labels_v;
  std::vector<int*> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (std::size_t i = 0; i < y.size(); i++) {
    uniq_labels_v.emplace_back(0, stream);
    n_unique[i]    = raft::label::getUniquelabels(uniq_labels_v.back(), y[i], n_index_rows, stream);
    uniq_labels[i] = uniq_labels_v[i].data();
  }

  MLCommon::Selection::knn_classify(
    out, knn_indices, y, n_index_rows, n_query_rows, k, uniq_labels, n_unique, stream);
}

void knn_regress(raft::handle_t& handle,
                 float* out,
                 int64_t* knn_indices,
                 std::vector<float*>& y,
                 size_t n_index_rows,
                 size_t n_query_rows,
                 int k)
{
  MLCommon::Selection::knn_regress(
    out, knn_indices, y, n_index_rows, n_query_rows, k, handle.get_stream());
}

void knn_class_proba(raft::handle_t& handle,
                     std::vector<float*>& out,
                     int64_t* knn_indices,
                     std::vector<int*>& y,
                     size_t n_index_rows,
                     size_t n_query_rows,
                     int k)
{
  cudaStream_t stream = handle.get_stream();

  std::vector<rmm::device_uvector<int>> uniq_labels_v;
  std::vector<int*> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (std::size_t i = 0; i < y.size(); i++) {
    uniq_labels_v.emplace_back(0, stream);
    n_unique[i]    = raft::label::getUniquelabels(uniq_labels_v.back(), y[i], n_index_rows, stream);
    uniq_labels[i] = uniq_labels_v[i].data();
  }

  MLCommon::Selection::class_probs(
    out, knn_indices, y, n_index_rows, n_query_rows, k, uniq_labels, n_unique, stream);
}

};  // END NAMESPACE ML
