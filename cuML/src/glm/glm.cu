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
#include "ols.h"
#include "ridge.h"
#include "glm_c.h"
#include "glm/qn/qn.h"
#include "cuML.hpp"

namespace ML {
namespace GLM {

using namespace MLCommon;

void olsFit(float *input, int n_rows, int n_cols, float *labels, float *coef,
            float *intercept, bool fit_intercept, bool normalize, int algo) {
    cumlHandle handle;
    olsFit(handle.getImpl(), input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
           normalize, handle.getStream(), algo);
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void olsFit(double *input, int n_rows, int n_cols, double *labels, double *coef,
		double *intercept, bool fit_intercept, bool normalize, int algo) {
    cumlHandle handle;
    olsFit(handle.getImpl(), input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
           normalize, handle.getStream(), algo);
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void olsPredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds) {
    cumlHandle handle;
    olsPredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds,
               handle.getStream());
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void olsPredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds) {
    cumlHandle handle;
    olsPredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds,
               handle.getStream());
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgeFit(float *input, int n_rows, int n_cols, float *labels, float *alpha,
		int n_alpha, float *coef, float *intercept, bool fit_intercept,
		bool normalize, int algo) {
    cumlHandle handle;
    ridgeFit(handle.getImpl(), input, n_rows, n_cols, labels, alpha, n_alpha, coef, intercept,
             fit_intercept, normalize, handle.getStream(), algo);
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgeFit(double *input, int n_rows, int n_cols, double *labels,
		double *alpha, int n_alpha, double *coef, double *intercept,
		bool fit_intercept, bool normalize, int algo) {
    cumlHandle handle;
    ridgeFit(handle.getImpl(), input, n_rows, n_cols, labels, alpha, n_alpha, coef, intercept,
             fit_intercept, normalize, handle.getStream(), algo);
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgePredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds) {
    cumlHandle handle;
    ridgePredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds, handle.getStream());
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void ridgePredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds) {
    cumlHandle handle;
    ridgePredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds, handle.getStream());
    ///@todo this should go away after cumlHandle exposure in the interface
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void qnFit(float *X, float *y, int N, int D, int C, bool fit_intercept,
           float l1, float l2, int max_iter, float grad_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity, float *w0,
           float *f, int *num_iters, bool X_col_major, int loss_type) {

  // TODO handles will come from the cuml handle
  cudaStream_t stream = 0;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  qnFit(X, y, N, D, C, fit_intercept, l1, l2, max_iter, grad_tol,
        linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters,
        X_col_major, loss_type, cublas_handle, stream);
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void qnFit(double *X, double *y, int N, int D, int C, bool fit_intercept,
           double l1, double l2, int max_iter, double grad_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity, double *w0,
           double *f, int *num_iters, bool X_col_major, int loss_type) {

  // TODO handles will come from the cuml handle
  cudaStream_t stream = 0;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  qnFit(X, y, N, D, C, fit_intercept, l1, l2, max_iter, grad_tol,
        linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters,
        X_col_major, loss_type, cublas_handle, stream);
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

} // namespace GLM
} // namespace ML
