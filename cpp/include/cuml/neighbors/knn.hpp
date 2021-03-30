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

#pragma once

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <raft/linalg/distance_type.h>
#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>

namespace ML {
struct knnIndex {
  faiss::gpu::GpuIndex *index;
  raft::distance::DistanceType metric;
  float metricArg;

  faiss::gpu::StandardGpuResources *gpu_res;
  int device;
  ~knnIndex() {
    delete index;
    delete gpu_res;
  }
};

typedef enum {
  QT_8bit,
  QT_4bit,
  QT_8bit_uniform,
  QT_4bit_uniform,
  QT_fp16,
  QT_8bit_direct,
  QT_6bit
} QuantizerType;

struct knnIndexParam {
  virtual ~knnIndexParam() {}
};

struct IVFParam : knnIndexParam {
  int nlist;
  int nprobe;
};

struct IVFFlatParam : IVFParam {};

struct IVFPQParam : IVFParam {
  int M;
  int n_bits;
  bool usePrecomputedTables;
};

struct IVFSQParam : IVFParam {
  QuantizerType qtype;
  bool encodeResidual;
};

/**
 * @brief Flat C++ API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param[in] handle RAFT handle
 * @param[in] input vector of pointers to the input arrays
 * @param[in] sizes vector of sizes of input arrays
 * @param[in] D the dimensionality of the arrays
 * @param[in] search_items array of items to search of dimensionality D
 * @param[in] n number of rows in search_items
 * @param[out] res_I the resulting index array of size n * k
 * @param[out] res_D the resulting distance array of size n * k
 * @param[in] k the number of nearest neighbors to return
 * @param[in] rowMajorIndex are the index arrays in row-major order?
 * @param[in] rowMajorQuery are the query arrays in row-major order?
 * @param[in] metric distance metric to use. Euclidean (L2) is used by
 * 			   default
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances. This
 * 					 is ignored if the metric_type is not Minkowski.
 */
void brute_force_knn(const raft::handle_t &handle, std::vector<float *> &input,
                     std::vector<int> &sizes, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k,
                     bool rowMajorIndex = false, bool rowMajorQuery = false,
                     raft::distance::DistanceType metric =
                       raft::distance::DistanceType::L2Expanded,
                     float metric_arg = 2.0f);

/**
 * @brief Flat C++ API function to build an approximate nearest neighbors index
 * from an index array and a set of parameters.
 *
 * @param[in] handle RAFT handle
 * @param[out] index index to be built
 * @param[in] params parametrization of the index to be built
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 * @param[in] metricArg metric argument
 * @param[in] index_array the index array to build the index with
 * @param[in] n number of rows in the index array
 * @param[in] D the dimensionality of the index array
 */
void approx_knn_build_index(raft::handle_t &handle, ML::knnIndex *index,
                            ML::knnIndexParam *params,
                            raft::distance::DistanceType metric,
                            float metricArg, float *index_array, int n, int D);

/**
 * @brief Flat C++ API function to perform an approximate nearest neighbors
 * search from previously built index and a query array
 *
 * @param[in] handle RAFT handle
 * @param[out] distances distances of the nearest neighbors toward
 *                       their query point
 * @param[out] indices indices of the nearest neighbors
 * @param[in] index index to perform a search with
 * @param[in] k the number of nearest neighbors to search for
 * @param[in] query_array the query to perform a search with
 * @param[in] n number of rows in the query array
 */
void approx_knn_search(raft::handle_t &handle, float *distances,
                       int64_t *indices, ML::knnIndex *index, int k,
                       float *query_array, int n);

/**
 * @brief Flat C++ API function to perform a knn classification using a
 * given a vector of label arrays. This supports multilabel classification
 * by classifying on multiple label arrays. Note that each label is
 * classified independently, as is done in scikit-learn.
 *
 * @param[in] handle RAFT handle
 * @param[out] out output array on device (size n_samples * size of y vector)
 * @param[in] knn_indices index array on device resulting from knn query (size n_samples * k)
 * @param[in] y vector of label arrays on device vector size is number of (size n_samples)
 * @param[in] n_index_rows number of vertices in index (eg. size of each y array)
 * @param[in] n_query_rows number of samples in knn_indices
 * @param[in] k number of nearest neighbors in knn_indices
 */
void knn_classify(raft::handle_t &handle, int *out, int64_t *knn_indices,
                  std::vector<int *> &y, size_t n_index_rows,
                  size_t n_query_rows, int k);

/**
 * @brief Flat C++ API function to perform a knn regression using
 * a given a vector of label arrays. This supports multilabel
 * regression by clasifying on multiple label arrays. Note that
 * each label is classified independently, as is done in scikit-learn.
 *
 * @param[in] handle RAFT handle
 * @param[out] out output array on device (size n_samples)
 * @param[in] knn_indices array on device of knn indices (size n_samples * k)
 * @param[in] y array of labels on device (size n_samples)
 * @param[in] n_index_rows number of vertices in index (eg. size of each y array)
 * @param[in] n_query_rows number of samples in knn_indices and out
 * @param[in] k number of nearest neighbors in knn_indices
 */
void knn_regress(raft::handle_t &handle, float *out, int64_t *knn_indices,
                 std::vector<float *> &y, size_t n_index_rows,
                 size_t n_query_rows, int k);

/**
 * @brief Flat C++ API function to compute knn class probabilities
 * using a vector of device arrays containing discrete class labels.
 * Note that the output is a vector, which is
 *
 * @param[in] handle RAFT handle
 * @param[out] out vector of output arrays on device. vector size = n_outputs.
 * Each array should have size(n_samples, n_classes)
 * @param[in] knn_indices array on device of knn indices (size n_samples * k)
 * @param[in] y array of labels on device (size n_samples)
 * @param[in] n_index_rows number of labels in y
 * @param[in] n_query_rows number of rows in knn_indices and out
 * @param[in] k number of nearest neighbors in knn_indices
 */
void knn_class_proba(raft::handle_t &handle, std::vector<float *> &out,
                     int64_t *knn_indices, std::vector<int *> &y,
                     size_t n_index_rows, size_t n_query_rows, int k);
};  // namespace ML
