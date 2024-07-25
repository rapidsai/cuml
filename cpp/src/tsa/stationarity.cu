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

#include <cuml/tsa/stationarity.h>

#include <raft/core/handle.hpp>

#include <timeSeries/stationarity.cuh>

namespace ML {

namespace Stationarity {

template <typename DataT>
inline void kpss_test_helper(const raft::handle_t& handle,
                             const DataT* d_y,
                             bool* results,
                             int batch_size,
                             int n_obs,
                             int d,
                             int D,
                             int s,
                             DataT pval_threshold)
{
  const auto& handle_impl = handle;
  cudaStream_t stream     = handle_impl.get_stream();

  MLCommon::TimeSeries::kpss_test(d_y, results, batch_size, n_obs, d, D, s, stream, pval_threshold);
}

void kpss_test(const raft::handle_t& handle,
               const float* d_y,
               bool* results,
               int batch_size,
               int n_obs,
               int d,
               int D,
               int s,
               float pval_threshold)
{
  kpss_test_helper<float>(handle, d_y, results, batch_size, n_obs, d, D, s, pval_threshold);
}

void kpss_test(const raft::handle_t& handle,
               const double* d_y,
               bool* results,
               int batch_size,
               int n_obs,
               int d,
               int D,
               int s,
               double pval_threshold)
{
  kpss_test_helper<double>(handle, d_y, results, batch_size, n_obs, d, D, s, pval_threshold);
}

}  // namespace Stationarity
}  // namespace ML
