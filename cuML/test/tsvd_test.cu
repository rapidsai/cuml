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

#include <gtest/gtest.h>
#include "random/rng.h"
#include "test_utils.h"
#include <cuda_utils.h>
#include "ml_utils.h"
#include "tsvd/tsvd.h"

namespace ML {

using namespace MLCommon;

template<typename T>
struct TsvdInputs {
	T tolerance;
	int len;
	int n_row;
	int n_col;
	int len2;
	int n_row2;
	int n_col2;
	unsigned long long int seed;
	int algo;
};

template<typename T>
::std::ostream& operator<<(::std::ostream& os, const TsvdInputs<T>& dims) {
	return os;
}

template<typename T>
class TsvdTest: public ::testing::TestWithParam<TsvdInputs<T> > {
protected:
	void basicTest() {
		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		params = ::testing::TestWithParam<TsvdInputs<T>>::GetParam();
		Random::Rng<T> r(params.seed);
		int len = params.len;

		allocate(data, len);

		T data_h[len] = { 1.0, 2.0, 4.0, 2.0, 4.0, 5.0, 5.0, 4.0, 2.0, 1.0, 6.0, 4.0 };
		updateDevice(data, data_h, len);

		int len_comp = params.n_col * params.n_col;
		allocate(components, len_comp);
		allocate(singular_vals, params.n_col);

		T components_ref_h[len_comp] = { -0.3951, 0.1532, 0.9058, -0.7111, -0.6752, -0.1959, -0.5816, 0.7215,
		                                 -0.3757 };

		allocate(components_ref, len_comp);
		updateDevice(components_ref, components_ref_h, len_comp);

		paramsTSVD prms;
		prms.n_cols = params.n_col;
		prms.n_rows = params.n_row;
		prms.n_components = params.n_col;
		if (params.algo == 0)
		    prms.algorithm = solver::COV_EIG_DQ;
		else
			prms.algorithm = solver::COV_EIG_JACOBI;

		tsvdFit(data, components, singular_vals, prms, cublas_handle, cusolver_handle);

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	}

	void advancedTest() {
		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		params = ::testing::TestWithParam<TsvdInputs<T>>::GetParam();
		Random::Rng<T> r(params.seed);
		int len = params.len2;

		paramsTSVD prms;
	    prms.n_cols = params.n_col2;
		prms.n_rows = params.n_row2;
		prms.n_components = params.n_col2;
		if (params.algo == 0)
		    prms.algorithm = solver::COV_EIG_DQ;
		else if (params.algo == 1)
			prms.algorithm = solver::COV_EIG_JACOBI;
		else if (params.algo == 2) {
			prms.algorithm = solver::RANDOMIZED;
			prms.n_components = params.n_col2 - 15;
		}


		allocate(data2, len);
		r.uniform(data2, len, T(-1.0), T(1.0));
		allocate(data2_trans, prms.n_rows * prms.n_components);

		int len_comp = params.n_col2 * prms.n_components;
		allocate(components2, len_comp);
		allocate(explained_vars2, prms.n_components);
		allocate(explained_var_ratio2, prms.n_components);
		allocate(singular_vals2, prms.n_components);

		tsvdFitTransform(data2, data2_trans, components2, explained_vars2, explained_var_ratio2,
				singular_vals2, prms, cublas_handle, cusolver_handle);

		allocate(data2_back, len);
		tsvdInverseTransform(data2_trans, components2, data2_back, prms, cublas_handle);

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
	}

	void SetUp() override {
		basicTest();
		advancedTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(data));
		CUDA_CHECK(cudaFree(components));
		CUDA_CHECK(cudaFree(singular_vals));
		CUDA_CHECK(cudaFree(components_ref));
		CUDA_CHECK(cudaFree(data2));
		CUDA_CHECK(cudaFree(data2_trans));
		CUDA_CHECK(cudaFree(data2_back));
		CUDA_CHECK(cudaFree(components2));
		CUDA_CHECK(cudaFree(explained_vars2));
		CUDA_CHECK(cudaFree(explained_var_ratio2));
		CUDA_CHECK(cudaFree(singular_vals2));
	}

protected:
	TsvdInputs<T> params;
	T *data, *components, *singular_vals,
			 *components_ref, *explained_vars_ref;

	T *data2, *data2_trans, *data2_back, *components2, *explained_vars2, *explained_var_ratio2,
			*singular_vals2;
};

const std::vector<TsvdInputs<float> > inputsf2 = {
		{ 0.01f, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 0 },
		{ 0.01f, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 1 },
		{ 0.05f, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2 },
		{ 0.05f, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2 }};


const std::vector<TsvdInputs<double> > inputsd2 = {
		{ 0.01, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 0 },
		{ 0.01, 4 * 3, 4, 3, 1024 * 128, 1024, 128, 1234ULL, 1 },
		{ 0.05, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2 },
		{ 0.05, 4 * 3, 4, 3, 512 * 64, 512, 64, 1234ULL, 2 }};


typedef TsvdTest<float> TsvdTestLeftVecF;
TEST_P(TsvdTestLeftVecF, Result) {
	ASSERT_TRUE(
			devArrMatch(components, components_ref,
					(params.n_col * params.n_col),
					CompareApproxAbs<float>(params.tolerance)));

}

typedef TsvdTest<double> TsvdTestLeftVecD;
TEST_P(TsvdTestLeftVecD, Result) {
	ASSERT_TRUE(
			devArrMatch(components, components_ref,
					(params.n_col * params.n_col),
					CompareApproxAbs<double>(params.tolerance)));
}

typedef TsvdTest<float> TsvdTestDataVecF;
TEST_P(TsvdTestDataVecF, Result) {
	ASSERT_TRUE(
			devArrMatch(data2, data2_back,
					(params.n_col2 * params.n_col2),
					CompareApproxAbs<float>(params.tolerance)));

}

typedef TsvdTest<double> TsvdTestDataVecD;
TEST_P(TsvdTestDataVecD, Result) {
	ASSERT_TRUE(
			devArrMatch(data2, data2_back,
					(params.n_col2 * params.n_col2),
					CompareApproxAbs<double>(params.tolerance)));
}


INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecD, ::testing::ValuesIn(inputsd2));

} // end namespace ML
