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

#pragma once

#include "smo_sets.cuh"

#include <cuml/common/utils.hpp>

namespace ML {
namespace SVM {

/**
 * Mark elements as unavailable if they are in the the idx list.
 * \param [out] available flag whether an idx is available, size [n_rows]
 * \param [in] n_rows number of training vectors
 * \param [in] idx list of indices already selected, size [n_selected]
 * \param [in] n_selected number of elements in the idx list
 */
CUML_KERNEL void set_unavailable(bool* available, int n_rows, const int* idx, int n_selected)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_selected) { available[idx[tid]] = false; }
}

/** Set availability to true for elements in the upper set, otherwise false.
 * @param [out] available size [n]
 * @param [in] n of elements in the working set
 * @param [in] alpha dual coefficients, size [n]
 * @param [in] y class label, must be +/-1, size [n]
 * @param [in] C penalty factor
 */
template <typename math_t>
CUML_KERNEL void set_upper(
  bool* available, int n, const math_t* alpha, const math_t* y, const math_t* C)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) available[tid] = in_upper(alpha[tid], y[tid], C[tid]);
}

/** Set availability to true for elements in the lower set, otherwise false.
 * @param [out] available size [n]
 * @param [in] n of elements in the working set
 * @param [in] alpha dual coefficients, size [n]
 * @param [in] y class label, must be +/-1, size [n]
 * @param [in] C penalty factor
 */
template <typename math_t>
CUML_KERNEL void set_lower(
  bool* available, int n, const math_t* alpha, const math_t* y, const math_t* C)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) available[tid] = in_lower(alpha[tid], y[tid], C[tid]);
}
/**
 * Get the priority of the elements that are selected by new_idx.
 *
 * We look up these indices from the old working set (idx), and return their
 * priority increased by one.
 *
 * @param [out] new_priority size [n_selected]
 * @param [in] n_selected (less equal n_ws)
 * @param [in] new_idx size [n_selected]
 * @param [in] n_ws working set size
 * @param [in] idx indices in the old working set, size [n_ws]
 * @param [in] priority of elements in the old working set, size [n_ws]
 */
CUML_KERNEL void update_priority(int* new_priority,
                                 int n_selected,
                                 const int* new_idx,
                                 int n_ws,
                                 const int* idx,
                                 const int* priority)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_selected) {
    int my_new_idx = new_idx[tid];
    // The working set size is limited (~1024 elements) so we just loop through it
    for (int i = 0; i < n_ws; i++) {
      if (idx[i] == my_new_idx) new_priority[tid] = priority[i] + 1;
    }
  }
}
}  // namespace SVM
}  // namespace ML
