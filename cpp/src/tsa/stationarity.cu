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

#include "stationarity.hpp"

#include "../../src_prims/timeSeries/stationarity.h"
#include "common/cumlHandle.hpp"

namespace ML {

void stationarity(const ML::cumlHandle& handle, const double* y_d, int* d,
                  int n_batches, int n_samples, double pval_threshold) {
  const ML::cumlHandle_impl& handle_impl = handle.getImpl();
  cudaStream_t stream = handle_impl.getStream();
  cublasHandle_t cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  MLCommon::TimeSeries::stationarity(y_d, d, n_batches, n_samples, allocator,
                                     stream, cublas_handle, pval_threshold);
}

}  // namespace ML