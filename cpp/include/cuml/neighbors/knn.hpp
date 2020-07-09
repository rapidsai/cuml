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

#pragma once

#include <common/cumlHandle.hpp>
#include <cuml/common/logger.hpp>

namespace ML {

enum MetricType {
  METRIC_INNER_PRODUCT = 0,
  METRIC_L2,
  METRIC_L1,
  METRIC_Linf,
  METRIC_Lp,

  METRIC_Canberra = 20,
  METRIC_BrayCurtis,
  METRIC_JensenShannon,

  METRIC_Cosine = 100,
  METRIC_Correlation
};

/**
   * @brief Flat C++ API function to perform a brute force knn on
   * a series of input arrays and combine the results into a single
   * output array for indexes and distances.
   *
   * @param handle the cuml handle to use
   * @param input vector of pointers to the input arrays
   * @param sizes vector of sizes of input arrays
   * @param D the dimensionality of the arrays
   * @param search_items array of items to search of dimensionality D
   * @param n number of rows in search_items
   * @param res_I the resulting index array of size n * k
   * @param res_D the resulting distance array of size n * k
   * @param k the number of nearest neighbors to return
   * @param rowMajorIndex are the index arrays in row-major order?
   * @param rowMajorQuery are the query arrays in row-major order?
   * @param metric distance metric to use. Euclidean (L2) is used by
   * 			   default
 * @param metric_arg the value of `p` for Minkowski (l-p) distances. This
 * 					 is ignored if the metric_type is not Minkowski.
 * @param expanded should lp-based distances be returned in their expanded
 * 					 form (e.g., without raising to the 1/p power).
   */
void brute_force_knn(cumlHandle &handle, std::vector<float *> &input,
                     std::vector<int> &sizes, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k,
                     bool rowMajorIndex = false, bool rowMajorQuery = false,
                     MetricType metric = MetricType::METRIC_L2,
                     float metric_arg = 2.0f, bool expanded = false);

/**
 * @brief Flat C++ API function to perform a knn classification using a
 * given a vector of label arrays. This supports multilabel classification
 * by classifying on multiple label arrays. Note that each label is
 * classified independently, as is done in scikit-learn.
 *
 * @param handle the cuml handle to use
 * @param out output array on device (size n_samples * size of y vector)
 * @param knn_indices index array on device resulting from knn query (size n_samples * k)
 * @param y vector of label arrays on device vector size is number of (size n_samples)
 * @param n_index_rows number of vertices in index (eg. size of each y array)
 * @param n_samples number of samples in knn_indices
 * @param k number of nearest neighbors in knn_indices
 */
void knn_classify(cumlHandle &handle, int *out, int64_t *knn_indices,
                  std::vector<int *> &y, size_t n_index_rows, size_t n_samples,
                  int k);

/**
 * @brief Flat C++ API function to perform a knn regression using
 * a given a vector of label arrays. This supports multilabel
 * regression by clasifying on multiple label arrays. Note that
 * each label is classified independently, as is done in scikit-learn.
 *
 * @param handle the cuml handle to use
 * @param out output array on device (size n_samples)
 * @param knn_indices array on device of knn indices (size n_samples * k)
 * @param y array of labels on device (size n_samples)
 * @param n_index_rows number of vertices in index (eg. size of each y array)
 * @param n_samples number of samples in knn_indices and out
 * @param k number of nearest neighbors in knn_indices
 */
void knn_regress(cumlHandle &handle, float *out, int64_t *knn_indices,
                 std::vector<float *> &y, size_t n_query_rows, size_t n_samples,
                 int k);

/**
 * @brief Flat C++ API function to compute knn class probabilities
 * using a vector of device arrays containing discrete class labels.
 * Note that the output is a vector, which is
 *
 * @param handle the cuml handle to use
 * @param out vector of output arrays on device. vector size = n_outputs.
 * Each array should have size(n_samples, n_classes)
 * @param knn_indices array on device of knn indices (size n_samples * k)
 * @param y array of labels on device (size n_samples)
 * @param n_index_rows number of labels
 * @param n_samples number of samples in knn_indices and out
 * @param k number of nearest neighbors in knn_indices
 */
void knn_class_proba(cumlHandle &handle, std::vector<float *> &out,
                     int64_t *knn_indices, std::vector<int *> &y,
                     size_t n_index_rows, size_t n_samples, int k);
};  // namespace ML
