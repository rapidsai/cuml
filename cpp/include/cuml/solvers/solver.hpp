/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

namespace raft {
class handle_t;
}

namespace ML {
namespace Solver {

void sgdFit(raft::handle_t& handle,
            float* input,
            int n_rows,
            int n_cols,
            float* labels,
            float* coef,
            float* intercept,
            bool fit_intercept,
            int batch_size,
            int epochs,
            int lr_type,
            float eta0,
            float power_t,
            int loss,
            int penalty,
            float alpha,
            float l1_ratio,
            bool shuffle,
            float tol,
            int n_iter_no_change);

void sgdFit(raft::handle_t& handle,
            double* input,
            int n_rows,
            int n_cols,
            double* labels,
            double* coef,
            double* intercept,
            bool fit_intercept,
            int batch_size,
            int epochs,
            int lr_type,
            double eta0,
            double power_t,
            int loss,
            int penalty,
            double alpha,
            double l1_ratio,
            bool shuffle,
            double tol,
            int n_iter_no_change);

void sgdPredict(raft::handle_t& handle,
                const float* input,
                int n_rows,
                int n_cols,
                const float* coef,
                float intercept,
                float* preds,
                int loss);

void sgdPredict(raft::handle_t& handle,
                const double* input,
                int n_rows,
                int n_cols,
                const double* coef,
                double intercept,
                double* preds,
                int loss);

void sgdPredictBinaryClass(raft::handle_t& handle,
                           const float* input,
                           int n_rows,
                           int n_cols,
                           const float* coef,
                           float intercept,
                           float* preds,
                           int loss);

void sgdPredictBinaryClass(raft::handle_t& handle,
                           const double* input,
                           int n_rows,
                           int n_cols,
                           const double* coef,
                           double intercept,
                           double* preds,
                           int loss);

/**
 * Fits a linear, lasso, and elastic-net regression model using Coordinate Descent solver.
 *
 * i.e. finds coefficients that minimize the following loss function:
 *
 * f(coef) = 1/2 * || labels - input * coef ||^2
 *         + 1/2 * alpha * (1 - l1_ratio) * ||coef||^2
 *         +       alpha *    l1_ratio    * ||coef||_1
 *
 *
 * @param handle
 *        Reference of raft::handle_t
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param labels
 *        pointer to an array for labels (size of n_rows)
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). This will be filled with
 * coefficients once the function is executed.
 * @param intercept
 *        pointer to a scalar for intercept. This will be filled
 *        once the function is executed
 * @param fit_intercept
 *        boolean parameter to control if the intercept will be fitted or not
 * @param normalize
 *        boolean parameter to control if the data will be normalized or not;
 *        NB: the input is scaled by the column-wise biased sample standard deviation estimator.
 * @param epochs
 *        Maximum number of iterations that solver will run
 * @param loss
 *        enum to use different loss functions. Only linear regression loss functions is supported
 * right now
 * @param alpha
 *        L1 parameter
 * @param l1_ratio
 *        ratio of alpha will be used for L1. (1 - l1_ratio) * alpha will be used for L2
 * @param shuffle
 *        boolean parameter to control whether coordinates will be picked randomly or not
 * @param tol
 *        tolerance to stop the solver
 * @param sample_weight
 *        device pointer to sample weight vector of length n_rows (nullptr or uniform weights)
 *        This vector is modified during the computation
 */
void cdFit(raft::handle_t& handle,
           float* input,
           int n_rows,
           int n_cols,
           float* labels,
           float* coef,
           float* intercept,
           bool fit_intercept,
           bool normalize,
           int epochs,
           int loss,
           float alpha,
           float l1_ratio,
           bool shuffle,
           float tol,
           float* sample_weight = nullptr);

void cdFit(raft::handle_t& handle,
           double* input,
           int n_rows,
           int n_cols,
           double* labels,
           double* coef,
           double* intercept,
           bool fit_intercept,
           bool normalize,
           int epochs,
           int loss,
           double alpha,
           double l1_ratio,
           bool shuffle,
           double tol,
           double* sample_weight = nullptr);

void cdPredict(raft::handle_t& handle,
               const float* input,
               int n_rows,
               int n_cols,
               const float* coef,
               float intercept,
               float* preds,
               int loss);

void cdPredict(raft::handle_t& handle,
               const double* input,
               int n_rows,
               int n_cols,
               const double* coef,
               double intercept,
               double* preds,
               int loss);

};  // namespace Solver
};  // end namespace ML
