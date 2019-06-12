#include <gtest/gtest.h>
#include "functions/logisticReg.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace Functions {

template <typename T>
struct LogRegLossInputs {
  T tolerance;
  T n_rows;
  T n_cols;
  int len;
};

template <typename T>
class LogRegLossTest : public ::testing::TestWithParam<LogRegLossInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<LogRegLossInputs<T>>::GetParam();
    int len = params.len;
    int n_rows = params.n_rows;
    int n_cols = params.n_cols;

    T *labels, *coef;

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocator.reset(new defaultDeviceAllocator);

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
    updateDevice(in, h_in, len, stream);

    T h_labels[n_rows] = {0.3, 2.0, -1.1};
    updateDevice(labels, h_labels, n_rows, stream);

    T h_coef[n_cols] = {0.35, -0.24};
    updateDevice(coef, h_coef, n_cols, stream);

    T h_out_ref[1] = {0.38752545};
    updateDevice(out_ref, h_out_ref, 1, stream);

    T h_out_lasso_ref[1] = {0.74152};
    updateDevice(out_lasso_ref, h_out_lasso_ref, 1, stream);

    T h_out_ridge_ref[1] = {0.4955854};
    updateDevice(out_ridge_ref, h_out_ridge_ref, 1, stream);

    T h_out_elasticnet_ref[1] = {0.618555};
    updateDevice(out_elasticnet_ref, h_out_elasticnet_ref, 1, stream);

    T h_out_grad_ref[n_cols] = {-0.58284, 0.207666};
    updateDevice(out_grad_ref, h_out_grad_ref, n_cols, stream);

    T h_out_lasso_grad_ref[n_cols] = {0.0171, -0.39233};
    updateDevice(out_lasso_grad_ref, h_out_lasso_grad_ref, n_cols, stream);

    T h_out_ridge_grad_ref[n_cols] = {-0.16284, -0.080333};
    updateDevice(out_ridge_grad_ref, h_out_ridge_grad_ref, n_cols, stream);

    T h_out_elasticnet_grad_ref[n_cols] = {-0.07284, -0.23633};
    updateDevice(out_elasticnet_grad_ref, h_out_elasticnet_grad_ref, n_cols,
                 stream);

    T alpha = 0.6;
    T l1_ratio = 0.5;

    logisticRegLoss(in, params.n_rows, params.n_cols, labels, coef, out,
                    penalty::NONE, alpha, l1_ratio, cublas_handle, allocator,
                    stream);

    updateDevice(in, h_in, len, stream);

    logisticRegLossGrads(in, params.n_rows, params.n_cols, labels, coef,
                         out_grad, penalty::NONE, alpha, l1_ratio,
                         cublas_handle, allocator, stream);

    updateDevice(in, h_in, len, stream);

    logisticRegLoss(in, params.n_rows, params.n_cols, labels, coef, out_lasso,
                    penalty::L1, alpha, l1_ratio, cublas_handle, allocator,
                    stream);

    updateDevice(in, h_in, len, stream);

    logisticRegLossGrads(in, params.n_rows, params.n_cols, labels, coef,
                         out_lasso_grad, penalty::L1, alpha, l1_ratio,
                         cublas_handle, allocator, stream);

    updateDevice(in, h_in, len, stream);

    logisticRegLoss(in, params.n_rows, params.n_cols, labels, coef, out_ridge,
                    penalty::L2, alpha, l1_ratio, cublas_handle, allocator,
                    stream);

    logisticRegLossGrads(in, params.n_rows, params.n_cols, labels, coef,
                         out_ridge_grad, penalty::L2, alpha, l1_ratio,
                         cublas_handle, allocator, stream);

    updateDevice(in, h_in, len, stream);

    logisticRegLoss(in, params.n_rows, params.n_cols, labels, coef,
                    out_elasticnet, penalty::ELASTICNET, alpha, l1_ratio,
                    cublas_handle, allocator, stream);

    logisticRegLossGrads(in, params.n_rows, params.n_cols, labels, coef,
                         out_elasticnet_grad, penalty::ELASTICNET, alpha,
                         l1_ratio, cublas_handle, allocator, stream);

    updateDevice(in, h_in, len, stream);

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
  LogRegLossInputs<T> params;
  T *in;
  T *out, *out_lasso, *out_ridge, *out_elasticnet;
  T *out_ref, *out_lasso_ref, *out_ridge_ref, *out_elasticnet_ref;
  T *out_grad, *out_lasso_grad, *out_ridge_grad, *out_elasticnet_grad;
  T *out_grad_ref, *out_lasso_grad_ref, *out_ridge_grad_ref,
    *out_elasticnet_grad_ref;
  std::shared_ptr<deviceAllocator> allocator;
};

const std::vector<LogRegLossInputs<float>> inputsf = {{0.01f, 3, 2, 6}};

const std::vector<LogRegLossInputs<double>> inputsd = {{0.01, 3, 2, 6}};

typedef LogRegLossTest<float> LogRegLossTestF;
TEST_P(LogRegLossTestF, Result) {
  ASSERT_TRUE(
    devArrMatch(out_ref, out, 1, CompareApprox<float>(params.tolerance)));

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

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref, out_elasticnet_grad,
                          params.n_cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef LogRegLossTest<double> LogRegLossTestD;
TEST_P(LogRegLossTestD, Result) {
  ASSERT_TRUE(
    devArrMatch(out_ref, out, 1, CompareApprox<double>(params.tolerance)));

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

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref, out_elasticnet_grad,
                          params.n_cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(LogRegLossTests, LogRegLossTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(LogRegLossTests, LogRegLossTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace Functions
}  // end namespace MLCommon
