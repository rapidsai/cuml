/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <cuml/cuml.hpp>
#include <cuml/linear_model/glm.hpp>
#include "glm/qn/qn.h"
#include "ols.h"
#include "ridge.h"

namespace ML {
namespace GLM {

using namespace MLCommon;

void olsFit(const cumlHandle &handle, float *input, int n_rows, int n_cols,
            float *labels, float *coef, float *intercept, bool fit_intercept,
            bool normalize, int algo) {
  olsFit(handle.getImpl(), input, n_rows, n_cols, labels, coef, intercept,
         fit_intercept, normalize, handle.getStream(), algo);
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void olsFit(const cumlHandle &handle, double *input, int n_rows, int n_cols,
            double *labels, double *coef, double *intercept, bool fit_intercept,
            bool normalize, int algo) {
  olsFit(handle.getImpl(), input, n_rows, n_cols, labels, coef, intercept,
         fit_intercept, normalize, handle.getStream(), algo);
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void olsPredict(const cumlHandle &handle, const float *input, int n_rows,
                int n_cols, const float *coef, float intercept, float *preds) {
  olsPredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds,
             handle.getStream());
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void olsPredict(const cumlHandle &handle, const double *input, int n_rows,
                int n_cols, const double *coef, double intercept,
                double *preds) {
  olsPredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds,
             handle.getStream());
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgeFit(const cumlHandle &handle, float *input, int n_rows, int n_cols,
              float *labels, float *alpha, int n_alpha, float *coef,
              float *intercept, bool fit_intercept, bool normalize, int algo) {
  ridgeFit(handle.getImpl(), input, n_rows, n_cols, labels, alpha, n_alpha,
           coef, intercept, fit_intercept, normalize, handle.getStream(), algo);
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgeFit(const cumlHandle &handle, double *input, int n_rows, int n_cols,
              double *labels, double *alpha, int n_alpha, double *coef,
              double *intercept, bool fit_intercept, bool normalize, int algo) {
  ridgeFit(handle.getImpl(), input, n_rows, n_cols, labels, alpha, n_alpha,
           coef, intercept, fit_intercept, normalize, handle.getStream(), algo);
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgePredict(const cumlHandle &handle, const float *input, int n_rows,
                  int n_cols, const float *coef, float intercept,
                  float *preds) {
  ridgePredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds,
               handle.getStream());
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgePredict(const cumlHandle &handle, const double *input, int n_rows,
                  int n_cols, const double *coef, double intercept,
                  double *preds) {
  ridgePredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds,
               handle.getStream());
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void qnFit(const cumlHandle &cuml_handle, float *X, float *y, int N, int D,
           int C, bool fit_intercept, float l1, float l2, int max_iter,
           float grad_tol, int linesearch_max_iter, int lbfgs_memory,
           int verbosity, float *w0, float *f, int *num_iters, bool X_col_major,
           int loss_type) {
  qnFit(cuml_handle.getImpl(), X, y, N, D, C, fit_intercept, l1, l2, max_iter,
        grad_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, X_col_major, loss_type, cuml_handle.getStream());
}

void qnFit(const cumlHandle &cuml_handle, double *X, double *y, int N, int D,
           int C, bool fit_intercept, double l1, double l2, int max_iter,
           double grad_tol, int linesearch_max_iter, int lbfgs_memory,
           int verbosity, double *w0, double *f, int *num_iters,
           bool X_col_major, int loss_type) {
  qnFit(cuml_handle.getImpl(), X, y, N, D, C, fit_intercept, l1, l2, max_iter,
        grad_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, X_col_major, loss_type, cuml_handle.getStream());
}

void qnDecisionFunction(const cumlHandle &cuml_handle, float *X, int N, int D,
                        int C, bool fit_intercept, float *params,
                        bool X_col_major, int loss_type, float *preds) {
  qnDecisionFunction(cuml_handle.getImpl(), X, N, D, C, fit_intercept, params,
                     X_col_major, loss_type, preds, cuml_handle.getStream());
}

void qnDecisionFunction(const cumlHandle &cuml_handle, double *X, int N, int D,
                        int C, bool fit_intercept, double *params,
                        bool X_col_major, int loss_type, double *scores) {
  qnDecisionFunction(cuml_handle.getImpl(), X, N, D, C, fit_intercept, params,
                     X_col_major, loss_type, scores, cuml_handle.getStream());
}

void qnPredict(const cumlHandle &cuml_handle, float *X, int N, int D, int C,
               bool fit_intercept, float *params, bool X_col_major,
               int loss_type, float *scores) {
  qnPredict(cuml_handle.getImpl(), X, N, D, C, fit_intercept, params,
            X_col_major, loss_type, scores, cuml_handle.getStream());
}

void qnPredict(const cumlHandle &cuml_handle, double *X, int N, int D, int C,
               bool fit_intercept, double *params, bool X_col_major,
               int loss_type, double *preds) {
  qnPredict(cuml_handle.getImpl(), X, N, D, C, fit_intercept, params,
            X_col_major, loss_type, preds, cuml_handle.getStream());
}

}  // namespace GLM
}  // namespace ML
