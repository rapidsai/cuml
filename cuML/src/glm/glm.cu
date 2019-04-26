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
#include "glm.hpp"
#include "glm/qn/qn.h"
#include "glm/glm_api.h"

namespace ML {
namespace GLM {

using namespace MLCommon;

void olsFit(float *input, int n_rows, int n_cols, float *labels, float *coef,
		float *intercept, bool fit_intercept, bool normalize, int algo) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	olsFit(input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
			normalize, cublas_handle, cusolver_handle, stream, algo);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

void olsFit(double *input, int n_rows, int n_cols, double *labels, double *coef,
		double *intercept, bool fit_intercept, bool normalize, int algo) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	olsFit(input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
			normalize, cublas_handle, cusolver_handle, stream, algo);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

void olsPredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	// cumlHandle_impl will set stream instead of creating
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	olsPredict(input, n_rows, n_cols, coef, intercept, preds, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void olsPredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	// cumlHandle_impl will set stream instead of creating
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	olsPredict(input, n_rows, n_cols, coef, intercept, preds, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void ridgeFit(float *input, int n_rows, int n_cols, float *labels, float *alpha,
		int n_alpha, float *coef, float *intercept, bool fit_intercept,
		bool normalize, int algo) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	ridgeFit(input, n_rows, n_cols, labels, alpha, n_alpha, coef, intercept,
			fit_intercept, normalize, cublas_handle, cusolver_handle, stream, algo);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void ridgeFit(double *input, int n_rows, int n_cols, double *labels,
		double *alpha, int n_alpha, double *coef, double *intercept,
		bool fit_intercept, bool normalize, int algo) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	ridgeFit(input, n_rows, n_cols, labels, alpha, n_alpha, coef, intercept,
			fit_intercept, normalize, cublas_handle, cusolver_handle, stream, algo);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void ridgePredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	// cumlHandle_impl will set stream instead of creating
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	ridgePredict(input, n_rows, n_cols, coef, intercept, preds, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void ridgePredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	// cumlHandle_impl will set stream instead of creating
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	ridgePredict(input, n_rows, n_cols, coef, intercept, preds, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

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

} // namespace GLM
} // namespace ML

extern "C" cumlError_t cumlSpQnFit(cumlHandle_t cuml_handle, float *X, float *y,
                                   int N, int D, int C, bool fit_intercept,
                                   float l1, float l2, int max_iter,
                                   float grad_tol, int linesearch_max_iter,
                                   int lbfgs_memory, int verbosity, float *w0,
                                   float *f, int *num_iters, bool X_col_major,
                                   int loss_type) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(cuml_handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::GLM::qnFit(*handle_ptr, X, y, N, D, C, fit_intercept, l1, l2,
                     max_iter, grad_tol, linesearch_max_iter, lbfgs_memory,
                     verbosity, w0, f, num_iters, X_col_major, loss_type);

    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

extern "C" cumlError_t
cumlDpQnFit(cumlHandle_t cuml_handle, double *X, double *y, int N, int D, int C,
            bool fit_intercept, double l1, double l2, int max_iter,
            double grad_tol, int linesearch_max_iter, int lbfgs_memory,
            int verbosity, double *w0, double *f, int *num_iters,
            bool X_col_major, int loss_type) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(cuml_handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::GLM::qnFit(*handle_ptr, X, y, N, D, C, fit_intercept, l1, l2,
                     max_iter, grad_tol, linesearch_max_iter, lbfgs_memory,
                     verbosity, w0, f, num_iters, X_col_major, loss_type);

    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
