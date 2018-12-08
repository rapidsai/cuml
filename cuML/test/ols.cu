#include "glm/ols.h"
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

		T data_h[len] = { 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0 };
		updateDevice(data, data_h, len);

		T labels_h[params.n_row] = { 6.0, 8.0, 9.0, 11.0 };
		updateDevice(labels, labels_h, params.n_row);

		T coef_ref_h[params.n_col] = { 2.090908, 2.5454557 };
		updateDevice(coef_ref, coef_ref_h, params.n_col);

		T coef2_ref_h[params.n_col] = { 1.000001 , 1.9999998 };
		updateDevice(coef2_ref, coef2_ref_h, params.n_col);

		T coef3_ref_h[params.n_col] = { 0.99999 , 2.00000 };
		updateDevice(coef3_ref, coef3_ref_h, params.n_col);

		T pred_data_h[len2] = { 3.0, 2.0, 5.0, 5.0 };
		updateDevice(pred_data, pred_data_h, len2);

		T pred_ref_h[params.n_row_2] = { 19.0, 16.9090 };
		updateDevice(pred_ref, pred_ref_h, params.n_row_2);

		T pred2_ref_h[params.n_row_2] = { 16.0, 15.0 };
		updateDevice(pred2_ref, pred2_ref_h, params.n_row_2);

		T pred3_ref_h[params.n_row_2] = { 16.0, 15.0 };
		updateDevice(pred3_ref, pred3_ref_h, params.n_row_2);

		intercept = T(0);

		olsFit(data, params.n_row, params.n_col, labels, coef,
				&intercept, false, false, cublas_handle,
				cusolver_handle, params.algo);

		olsPredict(pred_data, params.n_row_2, params.n_col, coef, intercept, pred,
				cublas_handle);

		updateDevice(data, data_h, len);
		updateDevice(labels, labels_h, params.n_row);

		intercept2 = T(0);
		olsFit(data, params.n_row, params.n_col, labels, coef2,
				&intercept2, true, false, cublas_handle,
						cusolver_handle, params.algo);

		olsPredict(pred_data, params.n_row_2, params.n_col, coef2, intercept2, pred2,
						cublas_handle);


		updateDevice(data, data_h, len);
		updateDevice(labels, labels_h, params.n_row);

		intercept3 = T(0);
		olsFit(data, params.n_row, params.n_col, labels, coef3,
				&intercept3, true, true, cublas_handle,
				cusolver_handle, params.algo);

		olsPredict(pred_data, params.n_row_2, params.n_col, coef3, intercept3, pred3,
				   cublas_handle);

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

	}

	void SetUp() override {
		basicTest();
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

	}

protected:
	OlsInputs<T> params;
	T *data, *labels, *coef, *coef_ref, *pred_data, *pred, *pred_ref;
	T *coef2, *coef2_ref, *pred2, *pred2_ref;
	T *coef3, *coef3_ref, *pred3, *pred3_ref;
	T intercept, intercept2, intercept3;

};

const std::vector<OlsInputs<float> > inputsf2 = {
		{ 0.001f, 4, 2, 2, 0 },
		{ 0.001f, 4, 2, 2, 1 },
		{ 0.001f, 4, 2, 2, 2 } };

const std::vector<OlsInputs<double> > inputsd2 = {
		{ 0.001, 4, 2, 2, 0 },
		{ 0.001, 4, 2, 2, 1 },
		{ 0.001, 4, 2, 2, 2 } };

typedef OlsTest<float> OlsTestF;
TEST_P(OlsTestF, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef2_ref, coef2, params.n_col,
				    CompareApproxAbs<float>(params.tol)));

	/*
	ASSERT_TRUE(
			devArrMatch(coef3_ref, coef3, params.n_col,
					CompareApproxAbs<float>(params.tol)));*/

	ASSERT_TRUE(
			devArrMatch(pred_ref, pred, params.n_row_2,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred2_ref, pred2, params.n_row_2,
					CompareApproxAbs<float>(params.tol)));

	/*
	ASSERT_TRUE(
			devArrMatch(pred3_ref, pred3, params.n_row_2,
					CompareApproxAbs<float>(params.tol)));*/
}

typedef OlsTest<double> OlsTestD;
TEST_P(OlsTestD, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
				devArrMatch(coef2_ref, coef2, params.n_col,
					CompareApproxAbs<double>(params.tol)));

	/*
	ASSERT_TRUE(
				devArrMatch(coef3_ref, coef3, params.n_col,
					CompareApproxAbs<double>(params.tol)));*/

	ASSERT_TRUE(
			devArrMatch(pred_ref, pred, params.n_row_2,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred2_ref, pred2, params.n_row_2,
					CompareApproxAbs<double>(params.tol)));

	/*
	ASSERT_TRUE(
			devArrMatch(pred3_ref, pred3, params.n_row_2,
					CompareApproxAbs<double>(params.tol)));*/
}

INSTANTIATE_TEST_CASE_P(OlsTests, OlsTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(OlsTests, OlsTestD, ::testing::ValuesIn(inputsd2));

}
} // end namespace ML
