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

#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn.hpp>

#include <ml_mg_utils.cuh>

#include <label/classlabels.cuh>
#include <raft/spatial/knn/knn.hpp>
#include <selection/knn.cuh>

#include <cuda_runtime.h>
#include <raft/cuda_utils.cuh>

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

  raft::spatial::knn::brute_force_knn(
    handle, input, sizes, D, search_items, n, res_I, res_D, k, rowMajorIndex,
    rowMajorQuery, nullptr, metric, metric_arg);
}

void approx_knn_build_index(raft::handle_t &handle, ML::knnIndex *index,
                            ML::knnIndexParam *params,
                            raft::distance::DistanceType metric,
                            float metricArg, float *index_array, int n, int D) {
  MLCommon::Selection::approx_knn_build_index(handle, index, params, metric,
                                              metricArg, index_array, n, D);
}

void approx_knn_search(raft::handle_t &handle, float *distances,
                       int64_t *indices, ML::knnIndex *index, int k,
                       float *query_array, int n) {
  MLCommon::Selection::approx_knn_search(handle, distances, indices, index, k,
                                         query_array, n);
}

void knn_classify(raft::handle_t &handle, int *out, int64_t *knn_indices,
                  std::vector<int *> &y, size_t n_index_rows,
                  size_t n_query_rows, int k) {
  auto d_alloc = handle.get_device_allocator();
  cudaStream_t stream = handle.get_stream();

  std::vector<int *> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (int i = 0; i < y.size(); i++) {
    MLCommon::Label::getUniqueLabels(y[i], n_index_rows, &(uniq_labels[i]),
                                     &(n_unique[i]), stream, d_alloc);
  }

  MLCommon::Selection::knn_classify(out, knn_indices, y, n_index_rows,
                                    n_query_rows, k, uniq_labels, n_unique,
                                    d_alloc, stream);
}

void knn_regress(raft::handle_t &handle, float *out, int64_t *knn_indices,
                 std::vector<float *> &y, size_t n_index_rows,
                 size_t n_query_rows, int k) {
  MLCommon::Selection::knn_regress(out, knn_indices, y, n_index_rows,
                                   n_query_rows, k, handle.get_stream());
}

void knn_class_proba(raft::handle_t &handle, std::vector<float *> &out,
                     int64_t *knn_indices, std::vector<int *> &y,
                     size_t n_index_rows, size_t n_query_rows, int k) {
  auto d_alloc = handle.get_device_allocator();
  cudaStream_t stream = handle.get_stream();

  std::vector<int *> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (int i = 0; i < y.size(); i++) {
    MLCommon::Label::getUniqueLabels(y[i], n_index_rows, &(uniq_labels[i]),
                                     &(n_unique[i]), stream, d_alloc);
  }

  MLCommon::Selection::class_probs(out, knn_indices, y, n_index_rows,
                                   n_query_rows, k, uniq_labels, n_unique,
                                   d_alloc, stream);
}

};  // END NAMESPACE ML
