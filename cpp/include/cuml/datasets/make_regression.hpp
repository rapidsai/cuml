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

#include <cuml/cuml.hpp>

namespace ML {
namespace Datasets {

/**
 * @todo docs 
 */
void make_regression(const cumlHandle& handle, float* out, float* values,
                     int64_t n_rows, int64_t n_cols, int64_t n_informative,
                     int64_t n_targets = 1LL, float bias = 0.0f,
                     int64_t effective_rank = -1LL, float tail_strength = 0.5f,
                     float noise = 0.0f, bool shuffle = true,
                     uint64_t seed = 0ULL);

void make_regression(const cumlHandle& handle, double* out, double* values,
                     int64_t n_rows, int64_t n_cols, int64_t n_informative,
                     int64_t n_targets = 1LL, double bias = 0.0,
                     int64_t effective_rank = -1LL, double tail_strength = 0.5,
                     double noise = 0.0, bool shuffle = true,
                     uint64_t seed = 0ULL);

void make_regression(const cumlHandle& handle, float* out, float* values,
                     int n_rows, int n_cols, int n_informative,
                     int n_targets = 1LL, float bias = 0.0f,
                     int effective_rank = -1LL, float tail_strength = 0.5f,
                     float noise = 0.0f, bool shuffle = true,
                     uint64_t seed = 0ULL);

void make_regression(const cumlHandle& handle, double* out, double* values,
                     int n_rows, int n_cols, int n_informative,
                     int n_targets = 1LL, double bias = 0.0,
                     int effective_rank = -1LL, double tail_strength = 0.5,
                     double noise = 0.0, bool shuffle = true,
                     uint64_t seed = 0ULL);

/// @todo other variants

}  // namespace Datasets
}  // namespace ML
