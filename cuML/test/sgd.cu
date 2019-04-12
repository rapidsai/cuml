#include "solver/sgd.h"
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
struct SgdInputs {
	T tol;
	int n_row;
	int n_col;
	int n_row2;
	int n_col2;
	int batch_size;
};

template<typename T>
class SgdTest: public ::testing::TestWithParam<SgdInputs<T> > {
protected:
	void linearRegressionTest() {
		params = ::testing::TestWithParam<SgdInputs<T>>::GetParam();
		int len = params.n_row * params.n_col;

		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		allocate(data, len);
		allocate(labels, params.n_row);
		allocate(coef, params.n_col, true);
		allocate(coef2, params.n_col, true);
		allocate(coef_ref, params.n_col);
		allocate(coef2_ref, params.n_col);

		T data_h[len] = { 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0 };
		updateDevice(data, data_h, len);

		T labels_h[params.n_row] = { 6.0, 8.0, 9.0, 11.0 };
		updateDevice(labels, labels_h, params.n_row);

		T coef_ref_h[params.n_col] = { 2.087, 2.5454557 };
		updateDevice(coef_ref, coef_ref_h, params.n_col);

		T coef2_ref_h[params.n_col] = { 1.000001, 1.9999998 };
		updateDevice(coef2_ref, coef2_ref_h, params.n_col);

		bool fit_intercept = false;
		intercept = T(0);
		int epochs = 2000;
		T lr = T(0.01);
		ML::lr_type lr_type = ML::lr_type::ADAPTIVE;
		T power_t = T(0.5);
		T alpha = T(0.0001);
		T l1_ratio = T(0.15);
		bool shuffle = true;
		T tol = T(1e-10);
		ML::loss_funct loss = ML::loss_funct::SQRD_LOSS;
		MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::NONE;
		int n_iter_no_change = 10;

		sgdFit(data, params.n_row, params.n_col, labels, coef, &intercept,
				fit_intercept, params.batch_size, epochs, lr_type, lr, power_t, loss,
				pen, alpha, l1_ratio, shuffle, tol, n_iter_no_change,
				cublas_handle, cusolver_handle, stream);

		fit_intercept = true;
		intercept2 = T(0);
		sgdFit(data, params.n_row, params.n_col, labels, coef2, &intercept2,
				fit_intercept, params.batch_size, epochs, ML::lr_type::CONSTANT, lr,
				power_t, loss, pen, alpha, l1_ratio, shuffle, tol,
				n_iter_no_change, cublas_handle, cusolver_handle, stream);

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
		CUDA_CHECK(cudaStreamDestroy(stream));

	}

	void logisticRegressionTest() {
		params = ::testing::TestWithParam<SgdInputs<T>>::GetParam();
		int len = params.n_row2 * params.n_col2;

		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		T *coef_class;
		allocate(data_logreg, len);
		allocate(data_logreg_test, len);
		allocate(labels_logreg, params.n_row2);
		allocate(coef_class, params.n_col2, true);
		allocate(pred_log, params.n_row2);
		allocate(pred_log_ref, params.n_row2);

		T data_h[len] = { 0.1, -2.1, 5.4, 5.4, -1.5, -2.15, 2.65, 2.65, 3.25,
				-0.15, -7.35, -7.35 };
		updateDevice(data_logreg, data_h, len);

		T data_test_h[len] = { 0.3, 1.1, 2.1, -10.1, 0.5, 2.5, -3.55, -20.5,
				-1.3, 3.0, -5.0, 15.0 };
		updateDevice(data_logreg_test, data_test_h, len);

		T labels_logreg_h[params.n_row2] = { 0.0, 1.0, 1.0, 0.0 };
		updateDevice(labels_logreg, labels_logreg_h, params.n_row2);

		T pred_log_ref_h[params.n_row2] = { 1.0, 0.0, 1.0, 1.0 };
		updateDevice(pred_log_ref, pred_log_ref_h, params.n_row2);

		bool fit_intercept = true;
		T intercept_class = T(0);
		int epochs = 1000;
		T lr = T(0.05);
		ML::lr_type lr_type = ML::lr_type::CONSTANT;
		T power_t = T(0.5);
		T alpha = T(0.0);
		T l1_ratio = T(0.0);
		bool shuffle = false;
		T tol = T(0.0);
		ML::loss_funct loss = ML::loss_funct::LOG;
		MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::NONE;
		int n_iter_no_change = 10;

		sgdFit(data_logreg, params.n_row2, params.n_col2, labels_logreg,
				coef_class, &intercept_class, fit_intercept, params.batch_size, epochs,
				lr_type, lr, power_t, loss, pen, alpha, l1_ratio, shuffle, tol,
				n_iter_no_change, cublas_handle, cusolver_handle, stream);

		sgdPredictBinaryClass(data_logreg_test, params.n_row2, params.n_col2,
				coef_class, intercept_class, pred_log, loss, cublas_handle, stream);

		CUDA_CHECK(cudaFree(coef_class));

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
		CUDA_CHECK(cudaStreamDestroy(stream));

	}

	void svmTest() {
		params = ::testing::TestWithParam<SgdInputs<T>>::GetParam();
		int len = params.n_row2 * params.n_col2;

		cublasHandle_t cublas_handle;
		CUBLAS_CHECK(cublasCreate(&cublas_handle));

		cusolverDnHandle_t cusolver_handle = NULL;
		CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		T *coef_class;
		allocate(data_svmreg, len);
		allocate(data_svmreg_test, len);
		allocate(labels_svmreg, params.n_row2);
		allocate(coef_class, params.n_col2, true);
		allocate(pred_svm, params.n_row2);
		allocate(pred_svm_ref, params.n_row2);

		T data_h[len] = { 0.1, -2.1, 5.4, 5.4, -1.5, -2.15, 2.65, 2.65, 3.25,
				-0.15, -7.35, -7.35 };
		updateDevice(data_svmreg, data_h, len);

		T data_test_h[len] = { 0.3, 1.1, 2.1, -10.1, 0.5, 2.5, -3.55, -20.5,
				-1.3, 3.0, -5.0, 15.0 };
		updateDevice(data_svmreg_test, data_test_h, len);

		T labels_svmreg_h[params.n_row2] = { 0.0, 1.0, 1.0, 0.0 };
		updateDevice(labels_svmreg, labels_svmreg_h, params.n_row2);

		T pred_svm_ref_h[params.n_row2] = { 1.0, 0.0, 1.0, 1.0 };
		updateDevice(pred_svm_ref, pred_svm_ref_h, params.n_row2);

		bool fit_intercept = true;
		T intercept_class = T(0);
		int epochs = 1000;
		T lr = T(0.05);
		ML::lr_type lr_type = ML::lr_type::CONSTANT;
		T power_t = T(0.5);
		T alpha = T(1) / T(epochs);
		T l1_ratio = T(0.0);
		bool shuffle = false;
		T tol = T(0.0);
		ML::loss_funct loss = ML::loss_funct::HINGE;
		MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::L2;
		int n_iter_no_change = 10;

		sgdFit(data_svmreg, params.n_row2, params.n_col2, labels_svmreg,
				coef_class, &intercept_class, fit_intercept, params.batch_size, epochs,
				lr_type, lr, power_t, loss, pen, alpha, l1_ratio, shuffle, tol,
				n_iter_no_change, cublas_handle, cusolver_handle, stream);

		sgdPredictBinaryClass(data_svmreg_test, params.n_row2, params.n_col2,
				coef_class, intercept_class, pred_svm, loss, cublas_handle, stream);

		CUDA_CHECK(cudaFree(coef_class));

		CUBLAS_CHECK(cublasDestroy(cublas_handle));
		CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
		CUDA_CHECK(cudaStreamDestroy(stream));

	}

	void SetUp() override {
		linearRegressionTest();
		logisticRegressionTest();
		svmTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(data));
		CUDA_CHECK(cudaFree(labels));
		CUDA_CHECK(cudaFree(coef));
		CUDA_CHECK(cudaFree(coef_ref));
		CUDA_CHECK(cudaFree(coef2));
		CUDA_CHECK(cudaFree(coef2_ref));
		CUDA_CHECK(cudaFree(data_logreg));
		CUDA_CHECK(cudaFree(data_logreg_test));
		CUDA_CHECK(cudaFree(labels_logreg));
		CUDA_CHECK(cudaFree(data_svmreg));
		CUDA_CHECK(cudaFree(data_svmreg_test));
		CUDA_CHECK(cudaFree(labels_svmreg));
		CUDA_CHECK(cudaFree(pred_svm));
		CUDA_CHECK(cudaFree(pred_svm_ref));
		CUDA_CHECK(cudaFree(pred_log));
		CUDA_CHECK(cudaFree(pred_log_ref));
	}

protected:
	SgdInputs<T> params;
	T *data, *labels, *coef, *coef_ref;
	T *coef2, *coef2_ref;
	T *data_logreg, *data_logreg_test, *labels_logreg;
	T *data_svmreg, *data_svmreg_test, *labels_svmreg;
	T *pred_svm, *pred_svm_ref, *pred_log, *pred_log_ref;
	T intercept, intercept2;

};

const std::vector<SgdInputs<float> > inputsf2 = { { 0.01f, 4, 2, 4, 3, 2 } };

const std::vector<SgdInputs<double> > inputsd2 = { { 0.01, 4, 2, 4, 3, 2 } };

typedef SgdTest<float> SgdTestF;
TEST_P(SgdTestF, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef2_ref, coef2, params.n_col,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred_log_ref, pred_log, params.n_row,
					CompareApproxAbs<float>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred_svm_ref, pred_svm, params.n_row,
					CompareApproxAbs<float>(params.tol)));
}

typedef SgdTest<double> SgdTestD;
TEST_P(SgdTestD, Fit) {

	ASSERT_TRUE(
			devArrMatch(coef_ref, coef, params.n_col,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(coef2_ref, coef2, params.n_col,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred_log_ref, pred_log, params.n_row,
					CompareApproxAbs<double>(params.tol)));

	ASSERT_TRUE(
			devArrMatch(pred_svm_ref, pred_svm, params.n_row,
					CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(SgdTests, SgdTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SgdTests, SgdTestD, ::testing::ValuesIn(inputsd2));

}
} // end namespace ML
