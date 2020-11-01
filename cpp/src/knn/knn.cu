/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <common/cumlHandle.hpp>

#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn.hpp>

#include <ml_mg_utils.cuh>

#include <label/classlabels.cuh>
#include <selection/knn.cuh>

#include <cuda_runtime.h>
#include <raft/cuda_utils.cuh>

#include <sstream>
#include <vector>

namespace ML {

void brute_force_knn(raft::handle_t &handle, std::vector<float *> &input,
                     std::vector<int> &sizes, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k, bool rowMajorIndex,
                     bool rowMajorQuery, MetricType metric, float metric_arg,
                     bool expanded) {
  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors must be the same size");

  std::vector<cudaStream_t> int_streams = handle.get_internal_streams();

  MLCommon::Selection::brute_force_knn(
    input, sizes, D, search_items, n, res_I, res_D, k,
    handle.get_device_allocator(), handle.get_stream(), int_streams.data(),
    handle.get_num_internal_streams(), rowMajorIndex, rowMajorQuery, nullptr,
    metric, metric_arg, expanded);
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

/**
 * @brief Flat C API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param[in] handle the cuml handle to use
 * @param[in] input an array of pointers to the input arrays
 * @param[in] sizes an array of sizes of input arrays
 * @param[in] n_params array size of input and sizes
 * @param[in] D the dimensionality of the arrays
 * @param[in] search_items array of items to search of dimensionality D
 * @param[in] n number of rows in search_items
 * @param[out] res_I the resulting index array of size n * k
 * @param[out] res_D the resulting distance array of size n * k
 * @param[in] k the number of nearest neighbors to return
 * @param[in] rowMajorIndex is the index array in row major layout?
 * @param[in] rowMajorQuery is the query array in row major layout?
 */
extern "C" cumlError_t knn_search(const cumlHandle_t handle, float **input,
                                  int *sizes, int n_params, int D,
                                  float *search_items, int n, int64_t *res_I,
                                  float *res_D, int k, bool rowMajorIndex,
                                  bool rowMajorQuery, int metric_type,
                                  float metric_arg, bool expanded) {
  cumlError_t status;

  raft::handle_t *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);

  std::vector<cudaStream_t> int_streams = handle_ptr->get_internal_streams();

  std::vector<float *> input_vec(n_params);
  std::vector<int> sizes_vec(n_params);
  for (int i = 0; i < n_params; i++) {
    input_vec.push_back(input[i]);
    sizes_vec.push_back(sizes[i]);
  }

  if (status == CUML_SUCCESS) {
    try {
      MLCommon::Selection::brute_force_knn(
        input_vec, sizes_vec, D, search_items, n, res_I, res_D, k,
        handle_ptr->get_device_allocator(), handle_ptr->get_stream(),
        int_streams.data(), handle_ptr->get_num_internal_streams(),
        rowMajorIndex, rowMajorQuery, nullptr, (ML::MetricType)metric_type,
        metric_arg, expanded);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
};  // END NAMESPACE ML
