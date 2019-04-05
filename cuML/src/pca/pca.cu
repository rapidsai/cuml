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


#include "pca.h"
#include "pca_c.h"


namespace ML {

using namespace MLCommon;


void pcaFit(float *input, float *components, float *explained_var,
		float *explained_var_ratio, float *singular_vals, float *mu,
		float *noise_vars, paramsPCA prms) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

	pcaFit(input, components, explained_var, explained_var_ratio, singular_vals,
			mu, noise_vars, prms, cublas_handle, cusolver_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void pcaFit(double *input, double *components, double *explained_var,
		double *explained_var_ratio, double *singular_vals, double *mu,
		double *noise_vars, paramsPCA prms) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

	pcaFit(input, components, explained_var, explained_var_ratio, singular_vals,
			mu, noise_vars, prms, cublas_handle, cusolver_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void pcaFitTransform(float *input, float *trans_input, float *components,
		float *explained_var, float *explained_var_ratio, float *singular_vals,
		float *mu, float *noise_vars, paramsPCA prms) {
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

	pcaFitTransform(input, trans_input, components, explained_var,
			explained_var_ratio, singular_vals, mu, noise_vars, prms,
			cublas_handle, cusolver_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));

}

void pcaFitTransform(double *input, double *trans_input, double *components,
		double *explained_var, double *explained_var_ratio,
		double *singular_vals, double *mu, double *noise_vars, paramsPCA prms) {
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

	pcaFitTransform(input, trans_input, components, explained_var,
			explained_var_ratio, singular_vals, mu, noise_vars, prms,
			cublas_handle, cusolver_handle, stream);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));

}

void pcaInverseTransform(float *trans_input, float *components,
		float *singular_vals, float *mu, float *input, paramsPCA prms) {
	cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

	pcaInverseTransform(trans_input, components, singular_vals, mu, input, prms,
			cublas_handle, stream);

  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void pcaInverseTransform(double *trans_input, double *components,
		double *singular_vals, double *mu, double *input, paramsPCA prms) {
	cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  
	pcaInverseTransform(trans_input, components, singular_vals, mu, input, prms,
			cublas_handle, stream);

  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}


void pcaTransform(float *input, float *components, float *trans_input,
		float *singular_vals, float *mu, paramsPCA prms) {
	cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

	pcaTransform(input, components, trans_input, singular_vals, mu, prms,
			cublas_handle, stream);

  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void pcaTransform(double *input, double *components, double *trans_input,
		double *singular_vals, double *mu, paramsPCA prms) {
	cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

	pcaTransform(input, components, trans_input, singular_vals, mu, prms,
			cublas_handle, stream);

  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/** @} */

}
;
// end namespace ML
