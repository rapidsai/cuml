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

#include "glm/ols.h"
#include <vector>
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include "ml_utils.h"

namespace ML {
namespace GLM {

using namespace MLCommon;

template<typename T>
struct OlsInputs {
	T tol;
	int n_row;
	int n_col;
	int n_row_2;
	int algo;
};

template<typename T>
class OlsTest: public ::testing::TestWithParam<OlsInputs<T> > {
protected:
	void basicTest() {
		params = ::testing::TestWithParam<OlsInputs<T>>::GetParam();
		int len = params.n_row * params.n_col;
		int len2 = params.n_row_2 * params.n_col;

		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		allocate(data, len);
		allocate(labels, params.n_row);
		allocate(coef, params.n_col);
		allocate(coef2, params.n_col);
		allocate(coef3, params.n_col);
		allocate(coef_ref, params.n_col);
		allocate(coef2_ref, params.n_col);
		allocate(coef3_ref, params.n_col);
		allocate(pred_data, len2);
		allocate(pred, params.n_row_2);
		allocate(pred_ref, params.n_row_2);
		allocate(pred2, params.n_row_2);
		allocate(pred2_ref, params.n_row_2);
		allocate(pred3, params.n_row_2);
		allocate(pred3_ref, params.n_row_2);

		std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0};
		data_h.resize(len);
		updateDevice(data, data_h.data(), len);

		std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0};
		labels_h.resize(params.n_row);
		updateDevice(labels, labels_h.data(), params.n_row);

		std::vector<T> coef_ref_h = {2.090908, 2.5454557};
		coef_ref_h.resize(params.n_col);
		updateDevice(coef_ref, coef_ref_h.data(), params.n_col);

		std::vector<T> coef2_ref_h = {1.000001 , 1.9999998};
		coef2_ref_h.resize(params.n_col);
		updateDevice(coef2_ref, coef2_ref_h.data(), params.n_col);

		std::vector<T> coef3_ref_h = {0.99999 , 2.00000};
		coef3_ref_h.resize(params.n_col);
		updateDevice(coef3_ref, coef3_ref_h.data(), params.n_col);

		std::vector<T> pred_data_h = {3.0, 2.0, 5.0, 5.0};
		pred_data_h.resize(len2);
		updateDevice(pred_data, pred_data_h.data(), len2);

		std::vector<T> pred_ref_h = {19.0, 16.9090};
		pred_ref_h.resize(params.n_row_2);
		updateDevice(pred_ref, pred_ref_h.data(), params.n_row_2);

		std::vector<T> pred2_ref_h = {16.0, 15.0};
		pred2_ref_h.resize(params.n_row_2);
		updateDevice(pred2_ref, pred2_ref_h.data(), params.n_row_2);

		std::vector<T> pred3_ref_h = {16.0, 15.0};
		pred3_ref_h.resize(params.n_row_2);
		updateDevice(pred3_ref, pred3_ref_h.data(), params.n_row_2);

		intercept = T(0);

		olsFit(data, params.n_row, params.n_col, labels, coef, &intercept,
				false, false, cublas_handle, cusolver_handle, params.algo);

		olsPredict(pred_data, params.n_row_2, params.n_col, coef, intercept,
				pred, cublas_handle);

		updateDevice(data, data_h.data(), len);
		updateDevice(labels, labels_h.data(), params.n_row);

		intercept2 = T(0);
		olsFit(data, params.n_row, params.n_col, labels, coef2, &intercept2,
				true, false, cublas_handle, cusolver_handle, params.algo);

		olsPredict(pred_data, params.n_row_2, params.n_col, coef2, intercept2,
				pred2, cublas_handle);

		updateDevice(data, data_h.data(), len);
		updateDevice(labels, labels_h.data(), params.n_row);

		intercept3 = T(0);
		olsFit(data, params.n_row, params.n_col, labels, coef3, &intercept3,
				true, true, cublas_handle, cusolver_handle, params.algo);

		olsPredict(pred_data, params.n_row_2, params.n_col, coef3, intercept3,
				pred3, cublas_handle);

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

	}

	void basicTest2() {
		params = ::testing::TestWithParam<OlsInputs<T>>::GetParam();
		int len = params.n_row * params.n_col;

		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		allocate(data_sc, len);
		allocate(labels_sc, len);
		allocate(coef_sc, 1);
		allocate(coef_sc_ref, 1);

		std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0};
		data_h.resize(len);
		updateDevice(data_sc, data_h.data(), len);

		std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0, -1.0, 2.0, -3.6, 3.3};
		labels_h.resize(len);
		updateDevice(labels_sc, labels_h.data(), len);

		std::vector<T> coef_sc_ref_h = {-0.29285714};
		coef_sc_ref_h.resize(1);
		updateDevice(coef_sc_ref, coef_sc_ref_h.data(), 1);

		T intercept_sc = T(0);

		olsFit(data_sc, len, 1, labels_sc, coef_sc, &intercept_sc,
				true, false, cublas_handle, cusolver_handle, params.algo);

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

	}

	void SetUp() override {
		basicTest();
		basicTest2();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(data));
		CUDA_CHECK(cudaFree(labels));
		CUDA_CHECK(cudaFree(coef));
		CUDA_CHECK(cudaFree(coef_ref));
		CUDA_CHECK(cudaFree(coef2));
		CUDA_CHECK(cudaFree(coef2_ref));
		CUDA_CHECK(cudaFree(coef3));
		CUDA_CHECK(cudaFree(coef3_ref));
		CUDA_CHECK(cudaFree(pred_data));
		CUDA_CHECK(cudaFree(pred));
		CUDA_CHECK(cudaFree(pred_ref));
		CUDA_CHECK(cudaFree(pred2));
		CUDA_CHECK(cudaFree(pred2_ref));
		CUDA_CHECK(cudaFree(pred3));
		CUDA_CHECK(cudaFree(pred3_ref));

		CUDA_CHECK(cudaFree(data_sc));
		CUDA_CHECK(cudaFree(labels_sc));
		CUDA_CHECK(cudaFree(coef_sc));
		CUDA_CHECK(cudaFree(coef_sc_ref));

	}

protected:
	OlsInputs<T> params;
	T *data, *labels, *coef, *coef_ref, *pred_data, *pred, *pred_ref;
	T *coef2, *coef2_ref, *pred2, *pred2_ref;
	T *coef3, *coef3_ref, *pred3, *pred3_ref;
	T *data_sc, *labels_sc, *coef_sc, *coef_sc_ref;
	T intercept, intercept2, intercept3;

};

const std::vector<OlsInputs<float> > inputsf2 = { { 0.001f, 4, 2, 2, 0 }, {
		0.001f, 4, 2, 2, 1 }, { 0.001f, 4, 2, 2, 2 } };

const std::vector<OlsInputs<double> > inputsd2 = { { 0.001, 4, 2, 2, 0 }, {
		0.001, 4, 2, 2, 1 }, { 0.001, 4, 2, 2, 2 } };

typedef OlsTest<float> OlsTestF;
TEST_P(OlsTestF, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef2_ref, coef2, params.n_col,
				    CompareApproxAbs<float>(params.tol)));


	ASSERT_TRUE(
			devArrMatch(coef3_ref, coef3, params.n_col,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred_ref, pred, params.n_row_2,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred2_ref, pred2, params.n_row_2,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred3_ref, pred3, params.n_row_2,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef_sc_ref, coef_sc, 1,
					CompareApproxAbs<float>(params.tol)));
}

typedef OlsTest<double> OlsTestD;
TEST_P(OlsTestD, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
				devArrMatch(coef2_ref, coef2, params.n_col,
					CompareApproxAbs<double>(params.tol)));


	ASSERT_TRUE(
				devArrMatch(coef3_ref, coef3, params.n_col,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred_ref, pred, params.n_row_2,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred2_ref, pred2, params.n_row_2,
					CompareApproxAbs<double>(params.tol)));


	ASSERT_TRUE(
			devArrMatch(pred3_ref, pred3, params.n_row_2,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef_sc_ref, coef_sc, 1,
					CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(OlsTests, OlsTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(OlsTests, OlsTestD, ::testing::ValuesIn(inputsd2));

}
} // end namespace ML
