/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>

namespace ML {
namespace Solver {
namespace Lars {

/**
 * @brief      Train a regressor using Least Angre Regression.
 *
 *             Least Angle Regression (LAR or LARS) is a model selection
 *             algorithm. It builds up the model using the following algorithm:
 *
 * 1. We start with all the coefficients equal to zero.
 * 2. At each step we select the predictor that has the largest absolute
 *    correlation with the residual.
 * 3. We take the largest step possible in the direction which is equiangular
 *    with all the predictors selected so far. The largest step is determined
 *    such that using this step a new predictor will have as much correlation
 *    with the residual as any of the currently active predictors.
 * 4. Stop if max_iter reached or all the predictors are used, or if the
 *    correlation between any unused predictor and the residual is lower than a
 *    tolerance.
 *
 *             The solver is based on [1]. The equations referred in the
 *             comments correspond to the equations in the paper.
 *
 *             Note: this algorithm assumes that the offset is removed from X
 *             and y, and each feature is normalized:
 * - sum_i y_i = 0,
 * - sum_i x_{i,j} = 0, sum_i x_{i,j}^2=1 for j=0..n_col-1
 *
 *             References: [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani,
 *             Least Angle Regression The Annals of Statistics (2004) Vol 32, No
 *             2, 407-499 http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param      handle      RAFT handle
 * @param      X           device array of training vectors in column major
 *                         format, size [n_rows * n_cols]. Note that the columns
 *                         of X will be permuted if the Gram matrix is not
 *                         specified. It is expected that X is normalized so
 *                         that each column has zero mean and unit variance.
 * @param      n_rows      number of training samples
 * @param      n_cols      number of feature columns
 * @param      y           device array of the regression targets, size
 *                         [n_rows]. y should be normalized to have zero mean.
 * @param      beta        device array of regression coefficients, has to be
 *                         allocated on entry, size [max_iter]
 * @param      active_idx  device array containing the indices of active
 *                         variables. Must be allocated on entry. Size
 *                         [max_iter]
 * @param      alphas      device array to return the maximum correlation along
 *                         the regularization path. Must be allocated on entry,
 *                         size [max_iter+1].
 * @param      n_active    host pointer to return the number of active elements
 *                         (scalar)
 * @param      Gram        device array containing Gram matrix containing X.T *
 *                         X. Can be nullptr.
 * @param      max_iter    maximum number of iterations, this equals with the
 *                         maximum number of coefficients returned. max_iter <=
 *                         n_cols.
 * @param      coef_path   coefficients along the regularization path are
 *                         returned here. Must be nullptr, or a device array
 *                         already allocated on entry. Size [max_iter *
 *                         (max_iter+1)].
 * @param      verbosity   verbosity level
 * @param      ld_X        leading dimension of X (stride of columns)
 * @param      ld_G        leading dimesion of G
 * @param      eps         numeric parameter for Cholesky rank one update
 *
 * @tparam     math_t      { description }
 * @tparam     idx_t       { description }
 */
template <typename math_t, typename idx_t>
void larsFit(const raft::handle_t& handle, math_t* X, idx_t n_rows,
             idx_t n_cols, const math_t* y, math_t* beta, idx_t* active_idx,
             math_t* alphas, idx_t* n_active, math_t* Gram = nullptr,
             int max_iter = 500, math_t* coef_path = nullptr, int verbosity = 0,
             idx_t ld_X = 0, idx_t ld_G = 0, math_t eps = -1);

/**
 * @brief      Predict with least angle regressor.
 *
 * @param      handle      RAFT handle
 * @param      X           device array of training vectors in column major
 *                         format, size [n_rows * n_cols].
 * @param      n_rows      number of training samples
 * @param      n_cols      number of feature columns
 * @param      ld_X        leading dimension of X (stride of columns)
 * @param      beta        device array of regression coefficients, size
 *                         [n_active]
 * @param      n_active    the number of regression coefficients
 * @param      active_idx  device array containing the indices of active
 *                         variables. Only these columns of X will be used for
 *                         prediction, size [n_active].
 * @param      intercept   The intercept
 * @param      preds       device array to store the predictions, size [n_rows].
 *                         Must be allocated on entry.
 *
 * @tparam     math_t      { description }
 * @tparam     idx_t       { description }
 */
template <typename math_t, typename idx_t>
void larsPredict(const raft::handle_t& handle, const math_t* X, idx_t n_rows,
                 idx_t n_cols, idx_t ld_X, const math_t* beta, idx_t n_active,
                 idx_t* active_idx, math_t intercept, math_t* preds);
};  // namespace Lars
};  // namespace Solver
};  // end namespace ML
