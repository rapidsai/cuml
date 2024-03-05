/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cuml/datasets/make_arima.hpp>

#include <raft/core/handle.hpp>

#include <random/make_arima.cuh>

namespace ML {
namespace Datasets {

template <typename DataT, typename IdxT>
inline void make_arima_helper(const raft::handle_t& handle,
                              DataT* out,
                              IdxT batch_size,
                              IdxT n_obs,
                              ARIMAOrder order,
                              DataT scale,
                              DataT noise_scale,
                              DataT intercept_scale,
                              uint64_t seed)
{
  auto stream = handle.get_stream();

  MLCommon::Random::make_arima(
    out, batch_size, n_obs, order, stream, scale, noise_scale, intercept_scale, seed);
}

void make_arima(const raft::handle_t& handle,
                float* out,
                int batch_size,
                int n_obs,
                ARIMAOrder order,
                float scale,
                float noise_scale,
                float intercept_scale,
                uint64_t seed)
{
  make_arima_helper(
    handle, out, batch_size, n_obs, order, scale, noise_scale, intercept_scale, seed);
}

void make_arima(const raft::handle_t& handle,
                double* out,
                int batch_size,
                int n_obs,
                ARIMAOrder order,
                double scale,
                double noise_scale,
                double intercept_scale,
                uint64_t seed)
{
  make_arima_helper(
    handle, out, batch_size, n_obs, order, scale, noise_scale, intercept_scale, seed);
}

}  // namespace Datasets
}  // namespace ML
