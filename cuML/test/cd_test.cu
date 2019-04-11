#include "solver/cd.h"
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include "ml_utils.h"
#include <matrix/matrix.h>
#include <linalg/cusolver_wrappers.h>

namespace ML {
namespace Solver {

using namespace MLCommon;
using namespace MLCommon::LinAlg;

template<typename T>
struct CdInputs {
	T tol;
	int n_row;
	int n_col;
};

template<typename T>
class CdTest: public ::testing::TestWithParam<CdInputs<T> > {
protected:
	void lasso() {
		params = ::testing::TestWithParam<CdInputs<T>>::GetParam();
		int len = params.n_row * params.n_col;

		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		allocate(data, len);
		allocate(labels, params.n_row);
		allocate(coef, params.n_col, true);
		allocate(coef2, params.n_col, true);
		allocate(coef_ref, params.n_col, true);
		allocate(coef2_ref, params.n_col, true);

		T data_h[len] = { 1.0, 1.2, 2.0, 2.0, 4.5, 2.0, 2.0, 3.0 };
		updateDevice(data, data_h, len);

		T labels_h[params.n_row] = { 6.0, 8.3, 9.8, 11.2 };
		updateDevice(labels, labels_h, params.n_row);

		T coef_ref_h[params.n_col] = { 4.90832, 0.35031 };
		updateDevice(coef_ref, coef_ref_h, params.n_col);

		T coef2_ref_h[params.n_col] = { 2.53530, -0.36832 };
		updateDevice(coef2_ref, coef2_ref_h, params.n_col);

		bool fit_intercept = false;
		bool normalize = false;
		intercept = T(0);
		int epochs = 200;
		T alpha = T(0.2);
		T l1_ratio = T(0.0);
		bool shuffle = false;
		T tol = T(1e-10);
		ML::loss_funct loss = ML::loss_funct::SQRD_LOSS;
		MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::L1;
		int n_iter_no_change = 10;

		cdFit(data, params.n_row, params.n_col, labels, coef, &intercept,
			  fit_intercept, normalize, epochs, loss, pen, alpha, l1_ratio, shuffle,
			  tol, n_iter_no_change, stream, cublas_handle, cusolver_handle);


		fit_intercept = true;
		intercept2 = T(0);
		cdFit(data, params.n_row, params.n_col, labels, coef2, &intercept2,
					  fit_intercept, normalize, epochs, loss, pen, alpha, l1_ratio, shuffle,
					  tol, n_iter_no_change, stream, cublas_handle, cusolver_handle);


		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

	}

	void SetUp() override {
		CUDA_CHECK(cudaStreamCreate(&stream));
		lasso();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(data));
		CUDA_CHECK(cudaFree(labels));
		CUDA_CHECK(cudaFree(coef));
		CUDA_CHECK(cudaFree(coef_ref));
		CUDA_CHECK(cudaFree(coef2));
		CUDA_CHECK(cudaFree(coef2_ref));
		CUDA_CHECK(cudaStreamDestroy(stream));
	}

protected:
	CdInputs<T> params;
	T *data, *labels, *coef, *coef_ref;
	T *coef2, *coef2_ref;
	T intercept, intercept2;
	cudaStream_t stream;

};

const std::vector<CdInputs<float> > inputsf2 = { { 0.01f, 4, 2 } };

const std::vector<CdInputs<double> > inputsd2 = { { 0.01, 4, 2 } };

typedef CdTest<float> CdTestF;
TEST_P(CdTestF, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef2_ref, coef2, params.n_col,
					CompareApproxAbs<float>(params.tol)));

}

typedef CdTest<double> CdTestD;
TEST_P(CdTestD, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef2_ref, coef2, params.n_col,
					CompareApproxAbs<double>(params.tol)));

}

INSTANTIATE_TEST_CASE_P(CdTests, CdTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(CdTests, CdTestD, ::testing::ValuesIn(inputsd2));

}
} // end namespace ML
