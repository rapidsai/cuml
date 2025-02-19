/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "lars_impl.cuh"

#include <cuml/solvers/lars.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace Solver {
namespace Lars {

// Explicit instantiation
template void larsFit<float, int>(const raft::handle_t& handle,
                                  float* X,
                                  int n_rows,
                                  int n_cols,
                                  const float* y,
                                  float* beta,
                                  int* active_idx,
                                  float* alphas,
                                  int* n_active,
                                  float* Gram,
                                  int max_iter,
                                  float* coef_path,
                                  rapids_logger::level_enum verbosity,
                                  int ld_X,
                                  int ld_G,
                                  float eps);

template void larsFit<double, int>(const raft::handle_t& handle,
                                   double* X,
                                   int n_rows,
                                   int n_cols,
                                   const double* y,
                                   double* beta,
                                   int* active_idx,
                                   double* alphas,
                                   int* n_active,
                                   double* Gram,
                                   int max_iter,
                                   double* coef_path,
                                   rapids_logger::level_enum verbosity,
                                   int ld_X,
                                   int ld_G,
                                   double eps);

template void larsPredict(const raft::handle_t& handle,
                          const float* X,
                          int n_rows,
                          int n_cols,
                          int ld_X,
                          const float* beta,
                          int n_active,
                          int* active_idx,
                          float intercept,
                          float* preds);

template void larsPredict(const raft::handle_t& handle,
                          const double* X,
                          int n_rows,
                          int n_cols,
                          int ld_X,
                          const double* beta,
                          int n_active,
                          int* active_idx,
                          double intercept,
                          double* preds);
};  // namespace Lars
};  // namespace Solver
};  // end namespace ML
