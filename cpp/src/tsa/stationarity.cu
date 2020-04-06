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

#include "cuml/tsa/stationarity.h"

#include "../../src_prims/timeSeries/stationarity.h"
#include "common/cumlHandle.hpp"

namespace ML {

namespace Stationarity {

template <typename DataT>
int stationarity_helper(const cumlHandle& handle, const DataT* y_d, int* d,
                        int n_batches, int n_samples, DataT pval_threshold) {
  const auto& handle_impl = handle.getImpl();
  cudaStream_t stream = handle_impl.getStream();
  auto allocator = handle_impl.getDeviceAllocator();

  return MLCommon::TimeSeries::stationarity(y_d, d, n_batches, n_samples,
                                            allocator, stream, pval_threshold);
}

int stationarity(const cumlHandle& handle, const float* y_d, int* d,
                 int n_batches, int n_samples, float pval_threshold) {
  return stationarity_helper<float>(handle, y_d, d, n_batches, n_samples,
                                    pval_threshold);
}

int stationarity(const cumlHandle& handle, const double* y_d, int* d,
                 int n_batches, int n_samples, double pval_threshold) {
  return stationarity_helper<double>(handle, y_d, d, n_batches, n_samples,
                                     pval_threshold);
}

}  // namespace Stationarity
}  // namespace ML
