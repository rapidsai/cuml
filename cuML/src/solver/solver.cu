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

#pragma once

#include "sgd.h"
#include "solver_c.h"
#include "ml_utils.h"
#include <linalg/cublas_wrappers.h>
#include <linalg/cusolver_wrappers.h>

namespace ML {
namespace Solver {

using namespace ML;


void sgdFit(float *input,
	        int n_rows,
	        int n_cols,
	        float *labels,
	        float *coef,
	        float *intercept,
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
	        int n_iter_no_change) {


	ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
	if (loss == 0) {
		loss_funct = ML::loss_funct::SQRD_LOSS;
	} else if (loss == 1) {
		loss_funct = ML::loss_funct::LOG;
	} else if (loss == 2) {
		loss_funct = ML::loss_funct::HINGE;
	} else {
		ASSERT(false,
					"glm.cu: other functions are not supported yet.");
	}

	MLCommon::Functions::penalty pen;

	if (penalty == 0) {
	    pen = MLCommon::Functions::penalty::NONE;
	} else if (penalty == 1) {
		pen = MLCommon::Functions::penalty::L1;
	} else if (penalty == 2) {
		pen = MLCommon::Functions::penalty::L2;
	} else if (penalty == 3) {
		pen = MLCommon::Functions::penalty::ELASTICNET;
	} else {
		ASSERT(false,
					"glm.cu: penalty is not supported yet.");
	}

	ML::lr_type learning_rate_type;
	if (lr_type == 0) {
		learning_rate_type = ML::lr_type::OPTIMAL;
	} else if (lr_type == 1) {
		learning_rate_type = ML::lr_type::CONSTANT;
	} else if (lr_type == 2) {
		learning_rate_type = ML::lr_type::INVSCALING;
	} else if (lr_type == 3) {
		learning_rate_type = ML::lr_type::ADAPTIVE;
	} else {
		ASSERT(false,
				    "glm.cu: this learning rate type is not supported.");
	}

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	sgdFit(input,
			       n_rows,
				   n_cols,
				   labels,
				   coef,
				   intercept,
				   fit_intercept,
				   batch_size,
				   epochs,
				   learning_rate_type,
				   eta0,
				   power_t,
				   loss_funct,
				   pen,
				   alpha,
				   l1_ratio,
				   shuffle,
				   tol,
				   n_iter_no_change,
				   cublas_handle,
				   cusolver_handle,
				   stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void sgdFit(double *input,
	        int n_rows,
	        int n_cols,
	        double *labels,
	        double *coef,
	        double *intercept,
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
	        int n_iter_no_change) {

	ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
	if (loss == 0) {
		loss_funct = ML::loss_funct::SQRD_LOSS;
	} else if (loss == 1) {
		loss_funct = ML::loss_funct::LOG;
	} else if (loss == 2) {
		loss_funct = ML::loss_funct::HINGE;
	} else {
		ASSERT(false,
				"glm.cu: other functions are not supported yet.");
	}

	MLCommon::Functions::penalty pen;
	if (penalty == 0) {
	    pen = MLCommon::Functions::penalty::NONE;
	} else if (penalty == 1) {
		pen = MLCommon::Functions::penalty::L1;
	} else if (penalty == 2) {
		pen = MLCommon::Functions::penalty::L2;
	} else if (penalty == 3) {
		pen = MLCommon::Functions::penalty::ELASTICNET;
	} else {
		ASSERT(false,
					"glm.cu: penalty is not supported yet.");
	}

	ML::lr_type learning_rate_type;
	if (lr_type == 0) {
		learning_rate_type = ML::lr_type::OPTIMAL;
	} else if (lr_type == 1) {
		learning_rate_type = ML::lr_type::CONSTANT;
	} else if (lr_type == 2) {
		learning_rate_type = ML::lr_type::INVSCALING;
	} else if (lr_type == 3) {
		learning_rate_type = ML::lr_type::ADAPTIVE;
	} else {
		ASSERT(false,
				    "glm.cu: this learning rate type is not supported.");
	}

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	sgdFit(input,
			       n_rows,
				   n_cols,
				   labels,
				   coef,
				   intercept,
				   fit_intercept,
				   batch_size,
				   epochs,
				   learning_rate_type,
				   eta0,
				   power_t,
				   loss_funct,
				   pen,
				   alpha,
				   l1_ratio,
				   shuffle,
				   tol,
				   n_iter_no_change,
				   cublas_handle,
				   cusolver_handle,
				   stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void sgdPredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds, int loss) {

	ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
	if (loss == 0) {
		loss_funct = ML::loss_funct::SQRD_LOSS;
	} else if (loss == 1) {
		loss_funct = ML::loss_funct::LOG;
	} else if (loss == 2) {
		loss_funct = ML::loss_funct::HINGE;
	} else {
		ASSERT(false,
			"glm.cu: other functions are not supported yet.");
	}

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	sgdPredict(input, n_rows, n_cols, coef, intercept, preds, loss_funct, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void sgdPredict(const double *input, int n_rows, int n_cols,
		const double *coef, double intercept, double *preds, int loss) {

	ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
	if (loss == 0) {
		loss_funct = ML::loss_funct::SQRD_LOSS;
	} else if (loss == 1) {
		loss_funct = ML::loss_funct::LOG;
	} else if (loss == 2) {
		loss_funct = ML::loss_funct::HINGE;
	} else {
		ASSERT(false,
			"glm.cu: other functions are not supported yet.");
	}

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamDestroy(stream));

	sgdPredict(input, n_rows, n_cols, coef, intercept, preds, loss_funct, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void sgdPredictBinaryClass(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds, int loss) {

	ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
	if (loss == 0) {
		loss_funct = ML::loss_funct::SQRD_LOSS;
	} else if (loss == 1) {
		loss_funct = ML::loss_funct::LOG;
	} else if (loss == 2) {
		loss_funct = ML::loss_funct::HINGE;
	} else {
		ASSERT(false,
			"glm.cu: other functions are not supported yet.");
	}

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	sgdPredictBinaryClass(input, n_rows, n_cols, coef, intercept, preds, loss_funct, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaStreamDestroy(stream));

}

void sgdPredictBinaryClass(const double *input, int n_rows, int n_cols,
		const double *coef, double intercept, double *preds, int loss) {

	ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
	if (loss == 0) {
		loss_funct = ML::loss_funct::SQRD_LOSS;
	} else if (loss == 1) {
		loss_funct = ML::loss_funct::LOG;
	} else if (loss == 2) {
		loss_funct = ML::loss_funct::HINGE;
	} else {
		ASSERT(false,
			"glm.cu: other functions are not supported yet.");
	}

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	sgdPredictBinaryClass(input, n_rows, n_cols, coef, intercept, preds, loss_funct, cublas_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));

	// should probably do a stream sync before destroy
	CUDA_CHECK(cudaStreamDestroy(stream));

}

}
}
