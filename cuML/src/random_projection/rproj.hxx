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

#include <vector>
#include <unordered_set>
#include <random>

#include "rproj_c.h"
#include "utils.hxx"
#include <linalg/cublas_wrappers.h>
#include <linalg/cusparse_wrappers.h>
#include <cuda_utils.h>
#include <common/cumlHandle.hpp>

namespace ML {

	using namespace MLCommon;
	using namespace MLCommon::LinAlg;

	/**
	 * @brief generates a gaussian random matrix
	 * @output param random_matrix: the random matrix to be allocated and generated
	 * @input param params: data structure that includes all the parameters of the model
	 * @input param stream: cuda stream
	 */
	template<typename math_t>
	void gaussian_random_matrix(rand_mat<math_t> *random_matrix, paramsRPROJ& params,
																		cudaStream_t stream)
	{
		int len = params.n_components * params.n_features;
		allocate(random_matrix->dense_data, len);
		auto rng = Random::Rng(params.random_state);
		math_t scale = 1.0 / sqrt(double(params.n_components));
		rng.normal(random_matrix->dense_data, len, math_t(0), scale, stream);
	}

	/**
	 * @brief generates a sparse random matrix
	 * @output param random_matrix: the random matrix to be allocated and generated
	 * @input param params: data structure that includes all the parameters of the model
	 * @input param stream: cuda stream
	 */
	template<typename math_t>
	void sparse_random_matrix(rand_mat<math_t> *random_matrix, paramsRPROJ& params,
																		cudaStream_t stream)
	{
		if (params.density == 1.0f)
		{
			int len = params.n_components * params.n_features;
			allocate(random_matrix->dense_data, len);
			auto rproj_rng = RPROJ_Rng(params.random_state);
			math_t scale = 1.0 / sqrt(math_t(params.n_components));
			rproj_rng.sparse_rand_gen(random_matrix->dense_data, len, scale, stream);
		}
		else
		{
			ML::cumlHandle h;
			auto alloc = h.getHostAllocator();

			double max_total_density = params.density * 1.2;
			size_t indices_alloc = (params.n_features * params.n_components * max_total_density) * sizeof(int);
			size_t indptr_alloc = (params.n_components + 1) * sizeof(int);
			int* indices = (int*)alloc->allocate(indices_alloc, stream);
			int* indptr = (int*)alloc->allocate(indptr_alloc, stream);

			size_t offset = 0;
			size_t indices_idx = 0;
			size_t indptr_idx = 0;

			for (size_t i = 0; i < params.n_components; i++)
			{
				int n_nonzero = binomial(params.n_features, params.density);
				sample_without_replacement(params.n_features, n_nonzero, indices, indices_idx);
				indptr[indptr_idx] = offset;
				indptr_idx++;
				offset += n_nonzero;
			}
			indptr[indptr_idx] = offset;

			size_t len = offset;
			allocate(random_matrix->indices, len);
			updateDevice(random_matrix->indices, indices, len, stream);
			alloc->deallocate(indices, indices_alloc, stream);

			len = indptr_idx+1;
			allocate(random_matrix->indptr, len);
			updateDevice(random_matrix->indptr, indptr, len, stream);
			alloc->deallocate(indptr, indptr_alloc, stream);

			len = offset;
			allocate(random_matrix->sparse_data, len);
			auto rproj_rng = RPROJ_Rng(params.random_state);
			math_t scale = sqrt(1.0 / params.density) / sqrt(params.n_components);
			rproj_rng.sparse_rand_gen(random_matrix->sparse_data, len, scale, stream);

			random_matrix->sparse_data_size = len;
		}
	}

	/**
	 * @brief fits the model by generating appropriate random matrix
	 * @output param random_matrix: the random matrix to be allocated and generated
	 * @input param params: data structure that includes all the parameters of the model
	 */
	template<typename math_t>
	void RPROJfit(rand_mat<math_t> *random_matrix, paramsRPROJ* params)
	{
		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		random_matrix->reset();

		build_parameters(*params);
		check_parameters(*params);

		if (params->gaussian_method)
		{
			gaussian_random_matrix<math_t>(random_matrix, *params, stream);
		}
		else
		{
			sparse_random_matrix<math_t>(random_matrix, *params, stream);
		}

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUDA_CHECK(cudaStreamDestroy(stream));
	}

	/**
	 * @brief transforms data according to generated random matrix
	 * @input param input: unprojected original dataset
	 * @input param random_matrix: the random matrix to be allocated and generated
	 * @output param output: projected dataset
	 * @input param params: data structure that includes all the parameters of the model
	 */
	template<typename math_t>
	void RPROJtransform(math_t *input, rand_mat<math_t> *random_matrix, math_t *output,
						paramsRPROJ* params)
	{
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		check_parameters(*params);

		if (random_matrix->dense_data)
		{
			cublasHandle_t cublas_handle;
			CUBLAS_CHECK(cublasCreate(&cublas_handle));

			const math_t alfa = 1;
			const math_t beta = 0;

			int& m = params->n_samples;
			int& n = params->n_components;
			int& k = params->n_features;

			int& lda = m;
			int& ldb = k;
			int& ldc = m;

			CUBLAS_CHECK(cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
				&alfa, input, lda, random_matrix->dense_data, ldb, &beta, output, ldc, stream));

			CUBLAS_CHECK(cublasDestroy(cublas_handle));
		}
		else if (random_matrix->sparse_data)
		{
			cusparseHandle_t cusparse_handle;
			CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
			CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, stream));

			const math_t alfa = 1;
			const math_t beta = 0;

			int& m = params->n_samples;
			int& n = params->n_components;
			int& k = params->n_features;
			size_t& nnz = random_matrix->sparse_data_size;

			int& lda = m;
			int& ldc = m;

			CUSPARSE_CHECK(cusparsegemmi(cusparse_handle, m, n, k, nnz, &alfa, input, lda,
							random_matrix->sparse_data, random_matrix->indptr,
							random_matrix->indices, &beta, output, ldc));

			CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
		}
		else
		{
			ASSERT(false,
					"Could not find a random matrix. Please perform a fit operation before applying transformation");
		}

		CUDA_CHECK(cudaStreamDestroy(stream));
	}

	/** @} */
};
// end namespace ML