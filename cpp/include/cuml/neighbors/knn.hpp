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
   */
void brute_force_knn(cumlHandle &handle, std::vector<float *> &input,
                     std::vector<int> &sizes, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k,
                     bool rowMajorIndex = false, bool rowMajorQuery = false);

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
 * @param n_labels number of vertices in index (eg. size of each y array)
 * @param n_samples number of samples in knn_indices
 * @param k number of nearest neighbors in knn_indices
 */
void knn_classify(cumlHandle &handle, int *out, int64_t *knn_indices,
                  std::vector<int *> &y, size_t n_labels, size_t n_samples,
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
 * @param n_labels number of vertices in index (eg. size of each y array)
 * @param n_samples number of samples in knn_indices and out
 * @param k number of nearest neighbors in knn_indices
 */
void knn_regress(cumlHandle &handle, float *out, int64_t *knn_indices,
                 std::vector<float *> &y, size_t n_labels, size_t n_samples,
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
 * @param n_labels number of labels
 * @param n_samples number of samples in knn_indices and out
 * @param k number of nearest neighbors in knn_indices
 */
void knn_class_proba(cumlHandle &handle, std::vector<float *> &out,
                     int64_t *knn_indices, std::vector<int *> &y,
                     size_t n_labels, size_t n_samples, int k);

class kNN {
  float **ptrs;
  int *sizes;

  int total_n;
  int indices;
  int D;

  bool rowMajorIndex;

  cumlHandle *handle;

 public:
  /**
   * Build a kNN object for training and querying a k-nearest neighbors model.
   * @param[in] handle    cuml handle
   * @param[in] D         number of features in each vector
   * @param[in] verbosity verbosity level for logging messages during execution
   */
  kNN(const cumlHandle &handle, int D, int verbosity = CUML_LEVEL_INFO);
  ~kNN();

  void reset();

  /**
     * Search the kNN for the k-nearest neighbors of a set of query vectors
     * @param search_items      set of vectors to query for neighbors
     * @param search_items_size number of items in search_items
     * @param res_I             pointer to device memory for returning k nearest indices
     * @param res_D             pointer to device memory for returning k nearest distances
     * @param k                 number of neighbors to query
     * @param rowMajor          is the query array in row major layout?
     */
  void search(float *search_items, int search_items_size, int64_t *res_I,
              float *res_D, int k, bool rowMajor = false);

  /**
     * Fit a kNN model by creating separate indices for multiple given
     * instances of kNNParams.
     * @param input    an array of pointers to data on (possibly different) devices
     * @param sizes    number of items in input array.
     * @param rowMajor is the index array in rowMajor layout?
     */
  void fit(std::vector<float *> &input, std::vector<int> &sizes,
           bool rowMajor = false);
};
};  // namespace ML
