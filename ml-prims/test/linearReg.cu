#include <gtest/gtest.h>
#include "functions/linearReg.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace Functions {

template <typename T>
struct LinRegLossInputs {
    T tolerance;
    T n_rows;
    T n_cols;
    int len;
};

template <typename T>
class LinRegLossTest: public ::testing::TestWithParam<LinRegLossInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<LinRegLossInputs<T>>::GetParam();
        int len = params.len;
        int n_rows = params.n_rows;
        int n_cols = params.n_cols;

        T *labels, *coef;

        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        allocate(in, len);
        allocate(out, 1);
        allocate(out_lasso, 1);
        allocate(out_ridge, 1);
        allocate(out_elasticnet, 1);
        allocate(out_grad, n_cols);
        allocate(out_lasso_grad, n_cols);
        allocate(out_ridge_grad, n_cols);
        allocate(out_elasticnet_grad, n_cols);
        allocate(out_ref, 1);
        allocate(out_lasso_ref, 1);
        allocate(out_ridge_ref, 1);
        allocate(out_elasticnet_ref, 1);
        allocate(out_grad_ref, n_cols);
        allocate(out_lasso_grad_ref, n_cols);
        allocate(out_ridge_grad_ref, n_cols);
        allocate(out_elasticnet_grad_ref, n_cols);

        allocate(labels, params.n_rows);
        allocate(coef, params.n_cols);

        T h_in[len] = {0.1, 0.35, -0.9, -1.4, 2.0, 3.1};
        updateDevice(in, h_in, len);

        T h_labels[n_rows] = {0.3, 2.0, -1.1};
        updateDevice(labels, h_labels, n_rows);

        T h_coef[n_cols] = {0.35, -0.24};
        updateDevice(coef, h_coef, n_cols);

        T h_out_ref[1] = {1.854842};
        updateDevice(out_ref, h_out_ref, 1);

        T h_out_lasso_ref[1] = {2.2088};
        updateDevice(out_lasso_ref, h_out_lasso_ref, 1);

        T h_out_ridge_ref[1] = {1.9629};
        updateDevice(out_ridge_ref, h_out_ridge_ref, 1);

        T h_out_elasticnet_ref[1] = {2.0858};
        updateDevice(out_elasticnet_ref, h_out_elasticnet_ref, 1);

        T h_out_grad_ref[n_cols] = {-0.56995, -3.12486};
        updateDevice(out_grad_ref, h_out_grad_ref, n_cols);

        T h_out_lasso_grad_ref[n_cols] = {0.03005, -3.724866};
        updateDevice(out_lasso_grad_ref, h_out_lasso_grad_ref, n_cols);

        T h_out_ridge_grad_ref[n_cols] = {-0.14995, -3.412866};
        updateDevice(out_ridge_grad_ref, h_out_ridge_grad_ref, n_cols);

        T h_out_elasticnet_grad_ref[n_cols] = {-0.05995, -3.568866};
        updateDevice(out_elasticnet_grad_ref, h_out_elasticnet_grad_ref, n_cols);

        T alpha = 0.6;
        T l1_ratio = 0.5;

        linearRegLoss(in, params.n_rows, params.n_cols, labels, coef, out, penalty::NONE,
                                      alpha, l1_ratio, cublas_handle, stream);

        updateDevice(in, h_in, len);

        linearRegLossGrads(in, params.n_rows, params.n_cols, labels, coef, out_grad, penalty::NONE,
                                      alpha, l1_ratio, cublas_handle, stream);

        updateDevice(in, h_in, len);

        linearRegLoss(in, params.n_rows, params.n_cols, labels, coef, out_lasso, penalty::L1,
                                      alpha, l1_ratio, cublas_handle, stream);

        updateDevice(in, h_in, len);

        linearRegLossGrads(in, params.n_rows, params.n_cols, labels, coef, out_lasso_grad, penalty::L1,
                                      alpha, l1_ratio, cublas_handle, stream);

        updateDevice(in, h_in, len);

        linearRegLoss(in, params.n_rows, params.n_cols, labels, coef, out_ridge, penalty::L2,
                                      alpha, l1_ratio, cublas_handle, stream);

        linearRegLossGrads(in, params.n_rows, params.n_cols, labels, coef, out_ridge_grad, penalty::L2,
                                      alpha, l1_ratio, cublas_handle, stream);

        updateDevice(in, h_in, len);

        linearRegLoss(in, params.n_rows, params.n_cols, labels, coef, out_elasticnet, penalty::ELASTICNET,
                                      alpha, l1_ratio, cublas_handle, stream);

        linearRegLossGrads(in, params.n_rows, params.n_cols, labels, coef, out_elasticnet_grad, penalty::ELASTICNET,
                                      alpha, l1_ratio, cublas_handle, stream);

        updateDevice(in, h_in, len);

        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(labels));
        CUDA_CHECK(cudaFree(coef));

    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out));
        CUDA_CHECK(cudaFree(out_lasso));
        CUDA_CHECK(cudaFree(out_ridge));
        CUDA_CHECK(cudaFree(out_elasticnet));
        CUDA_CHECK(cudaFree(out_grad));
        CUDA_CHECK(cudaFree(out_lasso_grad));
        CUDA_CHECK(cudaFree(out_ridge_grad));
        CUDA_CHECK(cudaFree(out_elasticnet_grad));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out_lasso_ref));
        CUDA_CHECK(cudaFree(out_ridge_ref));
        CUDA_CHECK(cudaFree(out_elasticnet_ref));
        CUDA_CHECK(cudaFree(out_grad_ref));
        CUDA_CHECK(cudaFree(out_lasso_grad_ref));
        CUDA_CHECK(cudaFree(out_ridge_grad_ref));
        CUDA_CHECK(cudaFree(out_elasticnet_grad_ref));
    }

protected:
    LinRegLossInputs<T> params;
    T *in;
    T *out, *out_lasso, *out_ridge, *out_elasticnet;
    T *out_ref, *out_lasso_ref, *out_ridge_ref, *out_elasticnet_ref;
    T *out_grad, *out_lasso_grad, *out_ridge_grad, *out_elasticnet_grad;
    T *out_grad_ref, *out_lasso_grad_ref, *out_ridge_grad_ref, *out_elasticnet_grad_ref;
};

const std::vector<LinRegLossInputs<float> > inputsf = {
    {0.01f, 3, 2, 6}
};

const std::vector<LinRegLossInputs<double> > inputsd = {
    {0.01, 3, 2, 6}
};

typedef LinRegLossTest<float> LinRegLossTestF;
TEST_P(LinRegLossTestF, Result) {

	ASSERT_TRUE(devArrMatch(out_ref, out, 1,
	                        CompareApprox<float>(params.tolerance)));

	ASSERT_TRUE(devArrMatch(out_lasso_ref, out_lasso, 1,
	                        CompareApprox<float>(params.tolerance)));

	ASSERT_TRUE(devArrMatch(out_ridge_ref, out_ridge, 1,
	                        CompareApprox<float>(params.tolerance)));

	ASSERT_TRUE(devArrMatch(out_elasticnet_ref, out_elasticnet, 1,
	                        CompareApprox<float>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_grad_ref, out_grad, params.n_cols,
	                        CompareApprox<float>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_lasso_grad_ref, out_lasso_grad, params.n_cols,
                            CompareApprox<float>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_ridge_grad_ref, out_ridge_grad, params.n_cols,
                            CompareApprox<float>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref, out_elasticnet_grad, params.n_cols,
                            CompareApprox<float>(params.tolerance)));

}

typedef LinRegLossTest<double> LinRegLossTestD;
TEST_P(LinRegLossTestD, Result){

	ASSERT_TRUE(devArrMatch(out_ref, out, 1,
	                        CompareApprox<double>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_lasso_ref, out_lasso, 1,
                            CompareApprox<double>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_ridge_ref, out_ridge, 1,
                            CompareApprox<double>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_elasticnet_ref, out_elasticnet, 1,
                            CompareApprox<double>(params.tolerance)));

	ASSERT_TRUE(devArrMatch(out_grad_ref, out_grad, params.n_cols,
	                         CompareApprox<double>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_lasso_grad_ref, out_lasso_grad, params.n_cols,
                            CompareApprox<double>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_ridge_grad_ref, out_ridge_grad, params.n_cols,
                            CompareApprox<double>(params.tolerance)));


    ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref, out_elasticnet_grad, params.n_cols,
                            CompareApprox<double>(params.tolerance)));

}

INSTANTIATE_TEST_CASE_P(LinRegLossTests, LinRegLossTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(LinRegLossTests, LinRegLossTestD, ::testing::ValuesIn(inputsd));

} // end namespace Functions
} // end namespace MLCommon
