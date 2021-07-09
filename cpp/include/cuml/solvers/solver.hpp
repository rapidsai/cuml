/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
           float tol);

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
           double tol);

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
