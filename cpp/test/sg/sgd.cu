/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <raft/cudart_utils.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <test_utils.h>
#include <raft/matrix/matrix.hpp>
#include <solver/sgd.cuh>

namespace ML {
namespace Solver {

using namespace MLCommon;

template <typename T>
struct SgdInputs {
  T tol;
  int n_row;
  int n_col;
  int n_row2;
  int n_col2;
  int batch_size;
};

template <typename T>
class SgdTest : public ::testing::TestWithParam<SgdInputs<T>> {
 protected:
  void linearRegressionTest()
  {
    params  = ::testing::TestWithParam<SgdInputs<T>>::GetParam();
    int len = params.n_row * params.n_col;

    raft::allocate(data, len, stream);
    raft::allocate(labels, params.n_row, stream);
    raft::allocate(coef, params.n_col, stream, true);
    raft::allocate(coef2, params.n_col, stream, true);
    raft::allocate(coef_ref, params.n_col, stream);
    raft::allocate(coef2_ref, params.n_col, stream);

    T data_h[len] = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0};
    raft::update_device(data, data_h, len, stream);

    T labels_h[params.n_row] = {6.0, 8.0, 9.0, 11.0};
    raft::update_device(labels, labels_h, params.n_row, stream);

    T coef_ref_h[params.n_col] = {2.087, 2.5454557};
    raft::update_device(coef_ref, coef_ref_h, params.n_col, stream);

    T coef2_ref_h[params.n_col] = {1.000001, 1.9999998};
    raft::update_device(coef2_ref, coef2_ref_h, params.n_col, stream);

    bool fit_intercept               = false;
    intercept                        = T(0);
    int epochs                       = 2000;
    T lr                             = T(0.01);
    ML::lr_type lr_type              = ML::lr_type::ADAPTIVE;
    T power_t                        = T(0.5);
    T alpha                          = T(0.0001);
    T l1_ratio                       = T(0.15);
    bool shuffle                     = true;
    T tol                            = T(1e-10);
    ML::loss_funct loss              = ML::loss_funct::SQRD_LOSS;
    MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::NONE;
    int n_iter_no_change             = 10;

    sgdFit(handle,
           data,
           params.n_row,
           params.n_col,
           labels,
           coef,
           &intercept,
           fit_intercept,
           params.batch_size,
           epochs,
           lr_type,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);

    fit_intercept = true;
    intercept2    = T(0);
    sgdFit(handle,
           data,
           params.n_row,
           params.n_col,
           labels,
           coef2,
           &intercept2,
           fit_intercept,
           params.batch_size,
           epochs,
           ML::lr_type::CONSTANT,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);
  }

  void logisticRegressionTest()
  {
    params  = ::testing::TestWithParam<SgdInputs<T>>::GetParam();
    int len = params.n_row2 * params.n_col2;

    T* coef_class;
    raft::allocate(data_logreg, len, stream);
    raft::allocate(data_logreg_test, len, stream);
    raft::allocate(labels_logreg, params.n_row2, stream);
    raft::allocate(coef_class, params.n_col2, stream, true);
    raft::allocate(pred_log, params.n_row2, stream);
    raft::allocate(pred_log_ref, params.n_row2, stream);

    T data_h[len] = {0.1, -2.1, 5.4, 5.4, -1.5, -2.15, 2.65, 2.65, 3.25, -0.15, -7.35, -7.35};
    raft::update_device(data_logreg, data_h, len, stream);

    T data_test_h[len] = {0.3, 1.1, 2.1, -10.1, 0.5, 2.5, -3.55, -20.5, -1.3, 3.0, -5.0, 15.0};
    raft::update_device(data_logreg_test, data_test_h, len, stream);

    T labels_logreg_h[params.n_row2] = {0.0, 1.0, 1.0, 0.0};
    raft::update_device(labels_logreg, labels_logreg_h, params.n_row2, stream);

    T pred_log_ref_h[params.n_row2] = {1.0, 0.0, 1.0, 1.0};
    raft::update_device(pred_log_ref, pred_log_ref_h, params.n_row2, stream);

    bool fit_intercept               = true;
    T intercept_class                = T(0);
    int epochs                       = 1000;
    T lr                             = T(0.05);
    ML::lr_type lr_type              = ML::lr_type::CONSTANT;
    T power_t                        = T(0.5);
    T alpha                          = T(0.0);
    T l1_ratio                       = T(0.0);
    bool shuffle                     = false;
    T tol                            = T(0.0);
    ML::loss_funct loss              = ML::loss_funct::LOG;
    MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::NONE;
    int n_iter_no_change             = 10;

    sgdFit(handle,
           data_logreg,
           params.n_row2,
           params.n_col2,
           labels_logreg,
           coef_class,
           &intercept_class,
           fit_intercept,
           params.batch_size,
           epochs,
           lr_type,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);

    sgdPredictBinaryClass(handle,
                          data_logreg_test,
                          params.n_row2,
                          params.n_col2,
                          coef_class,
                          intercept_class,
                          pred_log,
                          loss,
                          stream);

    CUDA_CHECK(cudaFree(coef_class));
  }

  void svmTest()
  {
    params  = ::testing::TestWithParam<SgdInputs<T>>::GetParam();
    int len = params.n_row2 * params.n_col2;

    T* coef_class;
    raft::allocate(data_svmreg, len, stream);
    raft::allocate(data_svmreg_test, len, stream);
    raft::allocate(labels_svmreg, params.n_row2, stream);
    raft::allocate(coef_class, params.n_col2, stream, true);
    raft::allocate(pred_svm, params.n_row2, stream);
    raft::allocate(pred_svm_ref, params.n_row2, stream);

    T data_h[len] = {0.1, -2.1, 5.4, 5.4, -1.5, -2.15, 2.65, 2.65, 3.25, -0.15, -7.35, -7.35};
    raft::update_device(data_svmreg, data_h, len, stream);

    T data_test_h[len] = {0.3, 1.1, 2.1, -10.1, 0.5, 2.5, -3.55, -20.5, -1.3, 3.0, -5.0, 15.0};
    raft::update_device(data_svmreg_test, data_test_h, len, stream);

    T labels_svmreg_h[params.n_row2] = {0.0, 1.0, 1.0, 0.0};
    raft::update_device(labels_svmreg, labels_svmreg_h, params.n_row2, stream);

    T pred_svm_ref_h[params.n_row2] = {1.0, 0.0, 1.0, 1.0};
    raft::update_device(pred_svm_ref, pred_svm_ref_h, params.n_row2, stream);

    bool fit_intercept               = true;
    T intercept_class                = T(0);
    int epochs                       = 1000;
    T lr                             = T(0.05);
    ML::lr_type lr_type              = ML::lr_type::CONSTANT;
    T power_t                        = T(0.5);
    T alpha                          = T(1) / T(epochs);
    T l1_ratio                       = T(0.0);
    bool shuffle                     = false;
    T tol                            = T(0.0);
    ML::loss_funct loss              = ML::loss_funct::HINGE;
    MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::L2;
    int n_iter_no_change             = 10;

    sgdFit(handle,
           data_svmreg,
           params.n_row2,
           params.n_col2,
           labels_svmreg,
           coef_class,
           &intercept_class,
           fit_intercept,
           params.batch_size,
           epochs,
           lr_type,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);

    sgdPredictBinaryClass(handle,
                          data_svmreg_test,
                          params.n_row2,
                          params.n_col2,
                          coef_class,
                          intercept_class,
                          pred_svm,
                          loss,
                          stream);

    CUDA_CHECK(cudaFree(coef_class));
  }

  void SetUp() override
  {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    linearRegressionTest();
    logisticRegressionTest();
    svmTest();
  }

  void TearDown() override
  {
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
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  SgdInputs<T> params;
  T *data, *labels, *coef, *coef_ref;
  T *coef2, *coef2_ref;
  T *data_logreg, *data_logreg_test, *labels_logreg;
  T *data_svmreg, *data_svmreg_test, *labels_svmreg;
  T *pred_svm, *pred_svm_ref, *pred_log, *pred_log_ref;
  T intercept, intercept2;
  cudaStream_t stream = 0;
  raft::handle_t handle;
};

const std::vector<SgdInputs<float>> inputsf2 = {{0.01f, 4, 2, 4, 3, 2}};

const std::vector<SgdInputs<double>> inputsd2 = {{0.01, 4, 2, 4, 3, 2}};

typedef SgdTest<float> SgdTestF;
TEST_P(SgdTestF, Fit)
{
  ASSERT_TRUE(
    raft::devArrMatch(coef_ref, coef, params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(
    raft::devArrMatch(coef2_ref, coef2, params.n_col, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred_log_ref, pred_log, params.n_row, raft::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred_svm_ref, pred_svm, params.n_row, raft::CompareApproxAbs<float>(params.tol)));
}

typedef SgdTest<double> SgdTestD;
TEST_P(SgdTestD, Fit)
{
  ASSERT_TRUE(
    raft::devArrMatch(coef_ref, coef, params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(
    raft::devArrMatch(coef2_ref, coef2, params.n_col, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred_log_ref, pred_log, params.n_row, raft::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(raft::devArrMatch(
    pred_svm_ref, pred_svm, params.n_row, raft::CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(SgdTests, SgdTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SgdTests, SgdTestD, ::testing::ValuesIn(inputsd2));

}  // namespace Solver
}  // end namespace ML
