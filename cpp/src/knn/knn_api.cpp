/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/common/distance_type.hpp>
#include <cuml/neighbors/knn.hpp>
#include <cuml/neighbors/knn_api.h>

#include <vector>

extern "C" {

namespace ML {

/**
 * @brief Flat C API function to perform a brute force knn on a series of input
 * arrays and combine the results into a single output array for indexes and
 * distances.
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
 * @param[in] metric_type distance metric to use. Specify the metric using the
 *    integer value of the enum `ML::MetricType`.
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances. This
 *    is ignored if the metric_type is not Minkowski.
 * @param[in] expanded should lp-based distances be returned in their expanded
 *    form (e.g., without raising to the 1/p power).
 */
cumlError_t knn_search(const cumlHandle_t handle,
                       float** input,
                       int* sizes,
                       int n_params,
                       int D,
                       float* search_items,
                       int n,
                       int64_t* res_I,
                       float* res_D,
                       int k,
                       bool rowMajorIndex,
                       bool rowMajorQuery,
                       int metric_type,
                       float metric_arg,
                       bool expanded)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  ML::distance::DistanceType metric_distance_type =
    static_cast<ML::distance::DistanceType>(metric_type);

  std::vector<float*> input_vec(n_params);
  std::vector<int> sizes_vec(n_params);
  for (int i = 0; i < n_params; i++) {
    input_vec.push_back(input[i]);
    sizes_vec.push_back(sizes[i]);
  }

  if (status == CUML_SUCCESS) {
    try {
      ML::brute_force_knn(*handle_ptr,
                          input_vec,
                          sizes_vec,
                          D,
                          search_items,
                          n,
                          res_I,
                          res_D,
                          k,
                          rowMajorIndex,
                          rowMajorQuery,
                          metric_distance_type,
                          metric_arg);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
};  // END NAMESPACE ML
}
