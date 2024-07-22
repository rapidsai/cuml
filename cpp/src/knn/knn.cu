/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/handle.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/spatial/knn/ann.cuh>
#include <raft/spatial/knn/ball_cover.cuh>
#include <raft/spatial/knn/knn.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cuvs/neighbors/brute_force.hpp>
#include <ml_mg_utils.cuh>
#include <selection/knn.cuh>

#include <cstddef>
#include <sstream>
#include <vector>

namespace ML {
void brute_force_knn(const raft::handle_t& handle,
                     std::vector<float*>& input,
                     std::vector<int>& sizes,
                     int D,
                     float* search_items,
                     int n,
                     int64_t* res_I,
                     float* res_D,
                     int k,
                     bool rowMajorIndex,
                     bool rowMajorQuery,
                     cuvs::distance::DistanceType metric,
                     float metric_arg,
                     std::vector<int64_t>* translations)
{
  ASSERT(input.size() == sizes.size(), "input and sizes vectors must be the same size");

  // The cuvs api doesn't support having multiple input values to search against.
  auto userStream = raft::resource::get_cuda_stream(handle);

  ASSERT(input.size() == sizes.size(), "input and sizes vectors should be the same size");

  std::vector<int64_t>* id_ranges;
  if (translations == nullptr) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    id_ranges       = new std::vector<int64_t>();
    int64_t total_n = 0;
    for (size_t i = 0; i < input.size(); i++) {
      id_ranges->push_back(total_n);
      total_n += sizes[i];
    }
  } else {
    // otherwise, use the given translations
    id_ranges = translations;
  }

  rmm::device_uvector<int64_t> trans(id_ranges->size(), userStream);
  raft::update_device(trans.data(), id_ranges->data(), id_ranges->size(), userStream);

  rmm::device_uvector<float> all_D(0, userStream);
  rmm::device_uvector<int64_t> all_I(0, userStream);

  float* out_D   = res_D;
  int64_t* out_I = res_I;

  if (input.size() > 1) {
    all_D.resize(input.size() * k * n, userStream);
    all_I.resize(input.size() * k * n, userStream);

    out_D = all_D.data();
    out_I = all_I.data();
  }

  // Make other streams from pool wait on main stream
  raft::resource::wait_stream_pool_on_stream(handle);

  for (size_t i = 0; i < input.size(); i++) {
    float* out_d_ptr   = out_D + (i * k * n);
    int64_t* out_i_ptr = out_I + (i * k * n);

    auto stream         = raft::resource::get_next_usable_stream(handle, i);
    auto current_handle = raft::device_resources(stream);

    // build the brute_force index (precalculates norms etc)
    std::optional<cuvs::neighbors::brute_force::index<float>> idx;
    if (rowMajorIndex) {
      idx = cuvs::neighbors::brute_force::build(
        current_handle,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(input[i], sizes[i], D),
        metric,
        metric_arg);

    } else {
      idx = cuvs::neighbors::brute_force::build(
        current_handle,
        raft::make_device_matrix_view<const float, int64_t, raft::col_major>(input[i], sizes[i], D),
        metric,
        metric_arg);
    }

    // query the index
    if (rowMajorQuery) {
      cuvs::neighbors::brute_force::search(
        current_handle,
        *idx,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(search_items, n, D),
        raft::make_device_matrix_view<int64_t, int64_t>(out_i_ptr, n, k),
        raft::make_device_matrix_view<float, int64_t>(out_d_ptr, n, k),
        std::nullopt);
    } else {
      cuvs::neighbors::brute_force::search(
        current_handle,
        *idx,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(search_items, n, D),
        raft::make_device_matrix_view<int64_t, int64_t>(out_i_ptr, n, k),
        raft::make_device_matrix_view<float, int64_t>(out_d_ptr, n, k),
        std::nullopt);
    }
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  raft::resource::sync_stream_pool(handle);

  if (input.size() > 1 || translations != nullptr) {
    // This is necessary for proper index translations. If there are
    // no translations or partitions to combine, it can be skipped.
    // TODO: sort out where this knn_merge_parts should live
    raft::spatial::knn::knn_merge_parts(
      out_D, out_I, res_D, res_I, n, input.size(), k, userStream, trans.data());
  }

  if (translations == nullptr) delete id_ranges;
}

void rbc_build_index(const raft::handle_t& handle,
                     raft::spatial::knn::BallCoverIndex<int64_t, float, uint32_t>& index)
{
  raft::spatial::knn::rbc_build_index(handle, index);
}

void rbc_knn_query(const raft::handle_t& handle,
                   raft::spatial::knn::BallCoverIndex<int64_t, float, uint32_t>& index,
                   uint32_t k,
                   const float* search_items,
                   uint32_t n_search_items,
                   int64_t* out_inds,
                   float* out_dists)
{
  raft::spatial::knn::rbc_knn_query(
    handle, index, k, search_items, n_search_items, out_inds, out_dists);
}

void approx_knn_build_index(raft::handle_t& handle,
                            raft::spatial::knn::knnIndex* index,
                            raft::spatial::knn::knnIndexParam* params,
                            cuvs::distance::DistanceType metric,
                            float metricArg,
                            float* index_array,
                            int n,
                            int D)
{
  raft::spatial::knn::approx_knn_build_index(handle,
                                             index,
                                             params,
                                             static_cast<raft::distance::DistanceType>(metric),
                                             metricArg,
                                             index_array,
                                             n,
                                             D);
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
    handle, out, knn_indices, y, n_index_rows, n_query_rows, k, uniq_labels, n_unique);
}

void knn_regress(raft::handle_t& handle,
                 float* out,
                 int64_t* knn_indices,
                 std::vector<float*>& y,
                 size_t n_index_rows,
                 size_t n_query_rows,
                 int k)
{
  MLCommon::Selection::knn_regress(handle, out, knn_indices, y, n_index_rows, n_query_rows, k);
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
    handle, out, knn_indices, y, n_index_rows, n_query_rows, k, uniq_labels, n_unique);
}

};  // END NAMESPACE ML
