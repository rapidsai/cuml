/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "cuml/tsa/divide_batch.h"

#include "common/cumlHandle.hpp"
#include "timeSeries/divide_batch.h"

namespace ML {

int divide_batch_build_index(const cumlHandle& handle, const bool* d_mask,
                             int* d_index, int batch_size, int n_obs) {
  cudaStream_t stream = handle.getStream();
  auto allocator = handle.getDeviceAllocator();
  return MLCommon::TimeSeries::divide_batch_build_index(
    d_mask, d_index, batch_size, n_obs, allocator, stream);
}

template <typename DataT>
void divide_batch_execute_helper(const cumlHandle& handle, const DataT* d_in,
                                 const bool* d_mask, const int* d_index,
                                 DataT* d_out0, DataT* d_out1, int batch_size,
                                 int n_obs) {
  cudaStream_t stream = handle.getStream();
  MLCommon::TimeSeries::divide_batch_execute(d_in, d_mask, d_index, d_out0,
                                             d_out1, batch_size, n_obs, stream);
}

void divide_batch_execute(const cumlHandle& handle, const float* d_in,
                          const bool* d_mask, const int* d_index, float* d_out0,
                          float* d_out1, int batch_size, int n_obs) {
  divide_batch_execute_helper(handle, d_in, d_mask, d_index, d_out0, d_out1,
                              batch_size, n_obs);
}

void divide_batch_execute(const cumlHandle& handle, const double* d_in,
                          const bool* d_mask, const int* d_index,
                          double* d_out0, double* d_out1, int batch_size,
                          int n_obs) {
  divide_batch_execute_helper(handle, d_in, d_mask, d_index, d_out0, d_out1,
                              batch_size, n_obs);
}

void divide_batch_execute(const cumlHandle& handle, const int* d_in,
                          const bool* d_mask, const int* d_index, int* d_out0,
                          int* d_out1, int batch_size, int n_obs) {
  divide_batch_execute_helper(handle, d_in, d_mask, d_index, d_out0, d_out1,
                              batch_size, n_obs);
}

}  // namespace ML
