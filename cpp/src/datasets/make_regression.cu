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

#include <common/cumlHandle.hpp>
#include <cuml/datasets/make_regression.hpp>
#include "random/make_regression.h"

namespace ML {
namespace Datasets {

template <typename DataT, typename IdxT>
void make_regression_helper(const cumlHandle& handle, DataT* out, DataT* values,
                            IdxT n_rows, IdxT n_cols, IdxT n_informative,
                            DataT* coef, IdxT n_targets, DataT bias,
                            IdxT effective_rank, DataT tail_strength,
                            DataT noise, bool shuffle, uint64_t seed) {
  const auto& handle_impl = handle.getImpl();
  cudaStream_t stream = handle_impl.getStream();
  cublasHandle_t cublas_handle = handle_impl.getCublasHandle();
  cusolverDnHandle_t cusolver_handle = handle_impl.getcusolverDnHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  MLCommon::Random::make_regression(
    out, values, n_rows, n_cols, n_informative, cublas_handle, cusolver_handle,
    allocator, stream, coef, n_targets, bias, effective_rank, tail_strength,
    noise, shuffle, seed);
}

void make_regression(const cumlHandle& handle, float* out, float* values,
                     int64_t n_rows, int64_t n_cols, int64_t n_informative,
                     float* coef, int64_t n_targets, float bias,
                     int64_t effective_rank, float tail_strength, float noise,
                     bool shuffle, uint64_t seed) {
  make_regression_helper(handle, out, values, n_rows, n_cols, n_informative,
                         coef, n_targets, bias, effective_rank, tail_strength,
                         noise, shuffle, seed);
}

void make_regression(const cumlHandle& handle, double* out, double* values,
                     int64_t n_rows, int64_t n_cols, int64_t n_informative,
                     double* coef, int64_t n_targets, double bias,
                     int64_t effective_rank, double tail_strength, double noise,
                     bool shuffle, uint64_t seed) {
  make_regression_helper(handle, out, values, n_rows, n_cols, n_informative,
                         coef, n_targets, bias, effective_rank, tail_strength,
                         noise, shuffle, seed);
}

void make_regression(const cumlHandle& handle, float* out, float* values,
                     int n_rows, int n_cols, int n_informative, float* coef,
                     int n_targets, float bias, int effective_rank,
                     float tail_strength, float noise, bool shuffle,
                     uint64_t seed) {
  make_regression_helper(handle, out, values, n_rows, n_cols, n_informative,
                         coef, n_targets, bias, effective_rank, tail_strength,
                         noise, shuffle, seed);
}

void make_regression(const cumlHandle& handle, double* out, double* values,
                     int n_rows, int n_cols, int n_informative, double* coef,
                     int n_targets, double bias, int effective_rank,
                     double tail_strength, double noise, bool shuffle,
                     uint64_t seed) {
  make_regression_helper(handle, out, values, n_rows, n_cols, n_informative,
                         coef, n_targets, bias, effective_rank, tail_strength,
                         noise, shuffle, seed);
}

}  // namespace Datasets
}  // namespace ML
