/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <faiss/Heap.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <iostream>

namespace ML {

/**
   * @brief Flat C++ API function to perform a brute force knn on
   * a series of input arrays and combine the results into a single
   * output array for indexes and distances.
   *
   * @param handle the cuml handle to use
   * @param input an array of pointers to the input arrays
   * @param sizes an array of sizes of input arrays
   * @param n_params array size of input and sizes
   * @param D the dimensionality of the arrays
   * @param search_items array of items to search of dimensionality D
   * @param n number of rows in search_items
   * @param res_I the resulting index array of size n * k
   * @param res_D the resulting distance array of size n * k
   * @param k the number of nearest neighbors to return
   */
void brute_force_knn(cumlHandle &handle, float **input, int *sizes,
                     int n_params, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k,
                     bool rowMajorIndex = false, bool rowMajorQuery = false);

/**
 * @brief Flat C++ API function to perform a knn classification using a
 * given array of labels.
 *
 * @param handle the cuml handle to use
 * @param out output array on device (size n_samples)
 * @param knn_indices array on device of knn indices (size n_samples * k)
 * @param y array of labels on device (size n_samples)
 * @param n_samples number of samples in knn_indices and out
 * @param k number of nearest neighbors in knn_indices
 * @param output_offset to support multiple outputs- specifies the
 *                      offset to use.
 */
void knn_classify(cumlHandle &handle, int *out, int64_t *knn_indices, int **y,
                  size_t n_samples, int k, int n_parts);

/**
 * @brief Flat C++ API function to perform a knn regression using
 * a given array of labels
 *
 * @param handle the cuml handle to use
 * @param out output array on device (size n_samples)
 * @param knn_indices array on device of knn indices (size n_samples * k)
 * @param y array of labels on device (size n_samples)
 * @param n_samples number of samples in knn_indices and out
 * @param k number of nearest neighbors in knn_indices
 */
void knn_regress(cumlHandle &handle, float *out, int64_t *knn_indices, float *y,
                 size_t n_samples, int k);

/**
 * @brief Flat C++ API function to compute knn class probabilities
 * using a given array of discrete class labels
 */
void knn_class_proba(cumlHandle &handle, float *out, int64_t *knn_indices,
                     int **y, size_t n_samples, int k, int n_parts);

class kNN {
  float **ptrs;
  int *sizes;

  int total_n;
  int indices;
  int D;
  bool verbose;

  bool rowMajorIndex;

  cumlHandle *handle;

 public:
  /**
	     * Build a kNN object for training and querying a k-nearest neighbors model.
	     * @param D     number of features in each vector
	     */
  kNN(const cumlHandle &handle, int D, bool verbose = false);
  ~kNN();

  void reset();

  /**
     * Search the kNN for the k-nearest neighbors of a set of query vectors
     * @param search_items set of vectors to query for neighbors
     * @param n            number of items in search_items
     * @param res_I        pointer to device memory for returning k nearest indices
     * @param res_D        pointer to device memory for returning k nearest distances
     * @param k            number of neighbors to query
     * @param rowMajor     is the query array in row major layout?
     */
  void search(float *search_items, int search_items_size, int64_t *res_I,
              float *res_D, int k, bool rowMajor = false);

  /**
     * Fit a kNN model by creating separate indices for multiple given
     * instances of kNNParams.
     * @param input  an array of pointers to data on (possibly different) devices
     * @param N      number of items in input array.
     * @param rowMajor is the index array in rowMajor layout?
     */
  void fit(float **input, int *sizes, int N, bool rowMajor = false);
};
};  // namespace ML
