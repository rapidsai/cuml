/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
