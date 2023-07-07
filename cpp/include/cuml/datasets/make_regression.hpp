/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cstdint>

namespace raft {
class handle_t;
}

namespace ML {
namespace Datasets {

/**
 * @brief GPU-equivalent of sklearn.datasets.make_regression as documented at:
 * https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
 *
 * @param[in]   handle          cuML handle
 * @param[out]  out             Row-major (samples, features) matrix to store
 *                              the problem data
 * @param[out]  values          Row-major (samples, targets) matrix to store
 *                              the values for the regression problem
 * @param[in]   n_rows          Number of samples
 * @param[in]   n_cols          Number of features
 * @param[in]   n_informative   Number of informative features (non-zero
 *                              coefficients)
 * @param[out]  coef            Row-major (features, targets) matrix to store
 *                              the coefficients used to generate the values
 *                              for the regression problem. If nullptr is
 *                              given, nothing will be written
 * @param[in]   n_targets       Number of targets (generated values per sample)
 * @param[in]   bias            A scalar that will be added to the values
 * @param[in]   effective_rank  The approximate rank of the data matrix (used
 *                              to create correlations in the data). -1 is the
 *                              code to use well-conditioned data
 * @param[in]   tail_strength   The relative importance of the fat noisy tail
 *                              of the singular values profile if
 *                              effective_rank is not -1
 * @param[in]   noise           Standard deviation of the gaussian noise
 *                              applied to the output
 * @param[in]   shuffle         Shuffle the samples and the features
 * @param[in]   seed            Seed for the random number generator
 */
void make_regression(const raft::handle_t& handle,
                     float* out,
                     float* values,
                     int64_t n_rows,
                     int64_t n_cols,
                     int64_t n_informative,
                     float* coef            = nullptr,
                     int64_t n_targets      = 1LL,
                     float bias             = 0.0f,
                     int64_t effective_rank = -1LL,
                     float tail_strength    = 0.5f,
                     float noise            = 0.0f,
                     bool shuffle           = true,
                     uint64_t seed          = 0ULL);

void make_regression(const raft::handle_t& handle,
                     double* out,
                     double* values,
                     int64_t n_rows,
                     int64_t n_cols,
                     int64_t n_informative,
                     double* coef           = nullptr,
                     int64_t n_targets      = 1LL,
                     double bias            = 0.0,
                     int64_t effective_rank = -1LL,
                     double tail_strength   = 0.5,
                     double noise           = 0.0,
                     bool shuffle           = true,
                     uint64_t seed          = 0ULL);

void make_regression(const raft::handle_t& handle,
                     float* out,
                     float* values,
                     int n_rows,
                     int n_cols,
                     int n_informative,
                     float* coef         = nullptr,
                     int n_targets       = 1LL,
                     float bias          = 0.0f,
                     int effective_rank  = -1LL,
                     float tail_strength = 0.5f,
                     float noise         = 0.0f,
                     bool shuffle        = true,
                     uint64_t seed       = 0ULL);

void make_regression(const raft::handle_t& handle,
                     double* out,
                     double* values,
                     int n_rows,
                     int n_cols,
                     int n_informative,
                     double* coef         = nullptr,
                     int n_targets        = 1LL,
                     double bias          = 0.0,
                     int effective_rank   = -1LL,
                     double tail_strength = 0.5,
                     double noise         = 0.0,
                     bool shuffle         = true,
                     uint64_t seed        = 0ULL);

}  // namespace Datasets
}  // namespace ML
