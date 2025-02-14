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

#pragma once

#include <cuml/common/logger.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace Solver {
namespace Lars {

/**
 * @brief Train a regressor using LARS method.
 *
 * @param handle RAFT handle
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols]. Note that the columns of X will be permuted if
 *     the Gram matrix is not specified. It is expected that X is normalized so
 *     that each column has zero mean and unit variance.
 * @param n_rows number of training samples
 * @param n_cols number of feature columns
 * @param y device array of the regression targets, size [n_rows]. y should
 *     be normalized to have zero mean.
 * @param beta device array of regression coefficients, has to be allocated on
 *     entry, size [max_iter]
 * @param active_idx device array containing the indices of active variables.
 *     Must be allocated on entry. Size [max_iter]
 * @param alphas the maximum correlation along the regularization path are
 *    returned here. Must be a device array allocated on entry Size [max_iter].
 * @param n_active host pointer to return the number of active elements, scalar.
 * @param Gram device array containing Gram matrix (X.T * X). Can be nullptr.
 *    Size [n_cols * ld_G]
 * @param max_iter maximum number of iterations, this equals with the maximum
 *    number of coefficients returned. max_iter <= n_cols.
 * @param coef_path coefficients along the regularization path are returned
 *    here. Must be nullptr, or a device array already allocated on entry.
 *    Size [max_iter * max_iter].
 * @param verbosity verbosity level
 * @param ld_X leading dimension of X (stride of columns, ld_X >= n_rows).
 * @param ld_G leading dimesion of G (ld_G >= n_cols)
 * @param eps numeric parameter for Cholesky rank one update
 */
template <typename math_t, typename idx_t>
void larsFit(const raft::handle_t& handle,
             math_t* X,
             idx_t n_rows,
             idx_t n_cols,
             const math_t* y,
             math_t* beta,
             idx_t* active_idx,
             math_t* alphas,
             idx_t* n_active,
             math_t* Gram,
             int max_iter,
             math_t* coef_path,
             rapids_logger::level_enum verbosity,
             idx_t ld_X,
             idx_t ld_G,
             math_t eps);

/**
 * @brief Predict with LARS regressor.
 *
 * @param handle RAFT handle
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols].
 * @param n_rows number of training samples
 * @param n_cols number of feature columns
 * @param ld_X leading dimension of X (stride of columns)
 * @param beta device array of regression coefficients, size [n_active]
 * @param n_active the number of regression coefficients
 * @param active_idx device array containing the indices of active variables.
 *     Only these columns of X will be used for prediction, size [n_active].
 * @param intercept
 * @param preds device array to store the predictions, size [n_rows]. Must be
 *     allocated on entry.
 */
template <typename math_t, typename idx_t>
void larsPredict(const raft::handle_t& handle,
                 const math_t* X,
                 idx_t n_rows,
                 idx_t n_cols,
                 idx_t ld_X,
                 const math_t* beta,
                 idx_t n_active,
                 idx_t* active_idx,
                 math_t intercept,
                 math_t* preds);
};  // namespace Lars
};  // namespace Solver
};  // end namespace ML
