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


#include "svc.h"
#include "svm_c.h"
#include "ml_utils.h"
#include <linalg/cublas_wrappers.h>
#include <linalg/cusolver_wrappers.h>

namespace ML {
namespace SVM {

using namespace ML;


void svcFit(float *input,
	        int n_rows,
	        int n_cols,
	        float *labels,
	        float *coef,
	        float C,
	        float tol) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	svcFit(input,
			       n_rows,
				   n_cols,
				   labels,
				   coef,
				   C,
				   tol,
				   cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void svcFit(double *input,
	        int n_rows,
	        int n_cols,
	        double *labels,
	        double *coef,
	        double C,
	        double tol) {
	        
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	svcFit(input,
			       n_rows,
				   n_cols,
				   labels,
				   coef,
				   C,
				   tol,
				   cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}
/*
void svcPredict(const float *input, int n_rows, int n_cols, const float *coef,
		 float *preds) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	svcPredict(input, n_rows, n_cols, coef, intercept, preds, loss_funct, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));

}

void svcPredict(const double *input, int n_rows, int n_cols, const double *coef, 
         double *preds) {



	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	svcPredict(input, n_rows, n_cols, coef, intercept, preds, loss_funct, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));

}
*/
}
}

