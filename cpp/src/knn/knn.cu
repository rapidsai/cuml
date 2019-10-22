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

#include "common/cumlHandle.hpp"

#include <cuml/neighbors/knn.hpp>

#include "ml_mg_utils.h"

#include "selection/knn.h"

#include <cuda_runtime.h>
#include "cuda_utils.h"

#include <sstream>
#include <vector>

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
                     int64_t *res_I, float *res_D, int k, bool rowMajorIndex,
                     bool rowMajorQuery) {
  MLCommon::Selection::brute_force_knn(
    input, sizes, n_params, D, search_items, n, res_I, res_D, k,
    handle.getImpl().getStream(), rowMajorIndex, rowMajorQuery);
}

/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
kNN::kNN(const cumlHandle &handle, int D, bool verbose)
  : D(D), total_n(0), indices(0), verbose(verbose) {
  this->handle = const_cast<cumlHandle *>(&handle);
  sizes = nullptr;
  ptrs = nullptr;
}

kNN::~kNN() {
  if (this->indices > 0) {
    reset();
  }
}

void kNN::reset() {
  if (this->indices > 0) {
    this->indices = 0;
    this->total_n = 0;

    delete[] this->ptrs;
    delete[] this->sizes;
  }
}

/**
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 * @param rowMajor is the input in rowMajor?
	 */
void kNN::fit(float **input, int *sizes, int N, bool rowMajor) {
  this->rowMajorIndex = rowMajor;

  if (this->verbose) std::cout << "N=" << N << std::endl;

  reset();

  this->indices = N;
  this->ptrs = (float **)malloc(N * sizeof(float *));
  this->sizes = (int *)malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    this->ptrs[i] = input[i];
    this->sizes[i] = sizes[i];
  }
}

/**
	 * Search the kNN for the k-nearest neighbors of a set of query vectors
	 * @param search_items set of vectors to query for neighbors
	 * @param n 		   number of items in search_items
	 * @param res_I 	   pointer to device memory for returning k nearest indices
	 * @param res_D		   pointer to device memory for returning k nearest distances
	 * @param k			   number of neighbors to query
	 */
void kNN::search(float *search_items, int n, int64_t *res_I, float *res_D,
                 int k, bool rowMajor) {
  ASSERT(this->indices > 0, "Cannot search before model has been trained.");

  MLCommon::Selection::brute_force_knn(
    ptrs, sizes, indices, D, search_items, n, res_I, res_D, k,
    handle->getImpl().getStream(), this->rowMajorIndex, rowMajor);
}
};  // namespace ML

/**
 * @brief Flat C API function to perform a brute force knn on
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
extern "C" cumlError_t knn_search(const cumlHandle_t handle, float **input,
                                  int *sizes, int n_params, int D,
                                  float *search_items, int n, int64_t *res_I,
                                  float *res_D, int k, bool rowMajorIndex,
                                  bool rowMajorQuery) {
  cumlError_t status;

  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      MLCommon::Selection::brute_force_knn(
        input, sizes, n_params, D, search_items, n, res_I, res_D, k,
        handle_ptr->getImpl().getStream(), rowMajorIndex, rowMajorQuery);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
