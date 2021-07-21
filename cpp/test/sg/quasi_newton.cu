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
#include <raft/linalg/transpose.h>
#include <test_utils.h>
#include <cuml/linear_model/glm.hpp>
#include <glm/qn/glm_linear.cuh>
#include <glm/qn/glm_logistic.cuh>
#include <glm/qn/glm_softmax.cuh>
#include <glm/qn/qn.cuh>
#include <raft/handle.hpp>
#include <raft/mr/device/allocator.hpp>
#include <vector>

namespace ML {
namespace GLM {

using namespace MLCommon;

struct QuasiNewtonTest : ::testing::Test {
  static constexpr int N = 10;
  static constexpr int D = 2;

  const static double* nobptr;
  const static double tol;
  const static double X[N][D];
  raft::handle_t cuml_handle;
  const raft::handle_t& handle;
  cudaStream_t stream;
  std::shared_ptr<SimpleMatOwning<double>> Xdev;
  std::shared_ptr<SimpleVecOwning<double>> ydev;

  std::shared_ptr<raft::mr::device::allocator> allocator;
  QuasiNewtonTest() : handle(cuml_handle) {}
  void SetUp()
  {
    stream = cuml_handle.get_stream();
    Xdev.reset(new SimpleMatOwning<double>(handle.get_device_allocator(), N, D, stream, ROW_MAJOR));
    raft::update_device(Xdev->data, &X[0][0], Xdev->len, stream);

    ydev.reset(new SimpleVecOwning<double>(handle.get_device_allocator(), N, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    allocator = handle.get_device_allocator();
  }
  void TearDown() {}
};

const double* QuasiNewtonTest::nobptr                                   = 0;
const double QuasiNewtonTest::tol                                       = 5e-6;
const double QuasiNewtonTest::X[QuasiNewtonTest::N][QuasiNewtonTest::D] = {
  {-0.2047076594847130, 0.4789433380575482},
  {-0.5194387150567381, -0.5557303043474900},
  {1.9657805725027142, 1.3934058329729904},
  {0.0929078767437177, 0.2817461528302025},
  {0.7690225676118387, 1.2464347363862822},
  {1.0071893575830049, -1.2962211091122635},
  {0.2749916334321240, 0.2289128789353159},
  {1.3529168351654497, 0.8864293405915888},
  {-2.0016373096603974, -0.3718425371402544},
  {1.6690253095248706, -0.4385697358355719}};

template <typename T, class Comp>
::testing::AssertionResult checkParamsEqual(const raft::handle_t& handle,
                                            const T* host_weights,
                                            const T* host_bias,
                                            const T* w,
                                            const GLMDims& dims,
                                            Comp& comp,
                                            cudaStream_t stream)
{
  int C              = dims.C;
  int D              = dims.D;
  bool fit_intercept = dims.fit_intercept;
  std::vector<T> w_ref_cm(C * D);
  int idx = 0;
  for (int d = 0; d < D; d++)
    for (int c = 0; c < C; c++) {
      w_ref_cm[idx++] = host_weights[c * D + d];
    }

  SimpleVecOwning<T> w_ref(handle.get_device_allocator(), dims.n_param, stream);
  raft::update_device(w_ref.data, &w_ref_cm[0], C * D, stream);
  if (fit_intercept) { raft::update_device(&w_ref.data[C * D], host_bias, C, stream); }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return raft::devArrMatch(w_ref.data, w, w_ref.len, comp);
}

template <typename T, class LossFunction>
T run(const raft::handle_t& handle,
      LossFunction& loss,
      const SimpleMat<T>& X,
      const SimpleVec<T>& y,
      T l1,
      T l2,
      T* w,
      SimpleDenseMat<T>& z,
      int verbosity,
      cudaStream_t stream)
{
  int max_iter            = 100;
  T grad_tol              = 1e-16;
  T change_tol            = 1e-16;
  int linesearch_max_iter = 50;
  int lbfgs_memory        = 5;
  int num_iters           = 0;

  T fx;

  qn_fit<T, LossFunction>(handle,
                          loss,
                          X,
                          y,
                          z,
                          l1,
                          l2,
                          max_iter,
                          grad_tol,
                          change_tol,
                          linesearch_max_iter,
                          lbfgs_memory,
                          verbosity,
                          w,
                          &fx,
                          &num_iters,
                          stream);

  return fx;
}

template <typename T>
T run_api(const raft::handle_t& cuml_handle,
          int loss_type,
          int C,
          bool fit_intercept,
          const SimpleMat<T>& X,
          const SimpleVec<T>& y,
          T l1,
          T l2,
          T* w,
          SimpleDenseMat<T>& z,
          int verbosity,
          cudaStream_t stream)
{
  int max_iter            = 100;
  T grad_tol              = 1e-8;
  T change_tol            = 1e-8;
  int linesearch_max_iter = 50;
  int lbfgs_memory        = 5;
  int num_iters           = 0;

  SimpleVec<T> w0(w, X.n + fit_intercept);
  w0.fill(T(0), stream);
  T fx;

  if (auto X_dense = dynamic_cast<const SimpleDenseMat<T>*>(&X)) {
    qnFit(cuml_handle,
          X_dense->data,
          X_dense->ord == COL_MAJOR,
          y.data,
          X_dense->m,
          X_dense->n,
          C,
          fit_intercept,
          l1,
          l2,
          max_iter,
          grad_tol,
          change_tol,
          linesearch_max_iter,
          lbfgs_memory,
          verbosity,
          w,
          &fx,
          &num_iters,
          loss_type);
  } else if (auto X_sparse = dynamic_cast<const SimpleSparseMat<T>*>(&X)) {
    qnFitSparse(cuml_handle,
                X_sparse->values,
                X_sparse->cols,
                X_sparse->row_ids,
                X_sparse->nnz,
                y.data,
                X_sparse->m,
                X_sparse->n,
                C,
                fit_intercept,
                l1,
                l2,
                max_iter,
                grad_tol,
                change_tol,
                linesearch_max_iter,
                lbfgs_memory,
                verbosity,
                w,
                &fx,
                &num_iters,
                loss_type);
  } else {
    ADD_FAILURE();
  }

  return fx;
}

TEST_F(QuasiNewtonTest, binary_logistic_vs_sklearn)
{
#if CUDART_VERSION >= 11020
  GTEST_SKIP();
#endif
  raft::CompareApprox<double> compApprox(tol);
  // Test case generated in python and solved with sklearn
  double y[N] = {1, 1, 1, 0, 1, 0, 1, 0, 1, 0};
  raft::update_device(ydev->data, &y[0], ydev->len, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double alpha = 0.01 * N;

  LogisticLoss<double> loss_b(handle, D, true);
  LogisticLoss<double> loss_no_b(handle, D, false);

  SimpleVecOwning<double> w0(allocator, D + 1, stream);
  SimpleMatOwning<double> z(allocator, 1, N, stream);

  double l1, l2, fx;

  double w_l1_b[2] = {-1.6899370396155091, 1.9021577534928300};
  double b_l1_b    = 0.8057670813749118;
  double obj_l1_b  = 0.44295941481024703;

  l1 = alpha;
  l2 = 0.0;
  fx = run(handle, loss_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));
  ASSERT_TRUE(checkParamsEqual(handle, &w_l1_b[0], &b_l1_b, w0.data, loss_b, compApprox, stream));

  fx =
    run_api(cuml_handle, 0, 2, loss_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  double w_l2_b[2] = {-1.5339880402781370, 1.6788639581350926};
  double b_l2_b    = 0.806087868102401;
  double obj_l2_b  = 0.4378085369889721;

  l1 = 0;
  l2 = alpha;
  fx = run(handle, loss_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);

  ASSERT_TRUE(compApprox(obj_l2_b, fx));
  ASSERT_TRUE(checkParamsEqual(handle, &w_l2_b[0], &b_l2_b, w0.data, loss_b, compApprox, stream));

  fx =
    run_api(cuml_handle, 0, 2, loss_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  double w_l1_no_b[2] = {-1.6215035298864591, 2.3650868394981086};
  double obj_l1_no_b  = 0.4769896009200278;

  l1 = alpha;
  l2 = 0.0;
  fx = run(handle, loss_no_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));
  ASSERT_TRUE(
    checkParamsEqual(handle, &w_l1_no_b[0], nobptr, w0.data, loss_no_b, compApprox, stream));

  fx = run_api(
    cuml_handle, 0, 2, loss_no_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  double w_l2_no_b[2] = {-1.3931049893764620, 2.0140103094119621};
  double obj_l2_no_b  = 0.47502098062114273;

  l1 = 0;
  l2 = alpha;
  fx = run(handle, loss_no_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
  ASSERT_TRUE(
    checkParamsEqual(handle, &w_l2_no_b[0], nobptr, w0.data, loss_no_b, compApprox, stream));

  fx = run_api(
    cuml_handle, 0, 2, loss_no_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
}

TEST_F(QuasiNewtonTest, multiclass_logistic_vs_sklearn)
{
#if CUDART_VERSION >= 11020
  GTEST_SKIP();
#endif
  // The data seems to small for the objective to be strongly convex
  // leaving out exact param checks

  raft::CompareApprox<double> compApprox(tol);
  double y[N] = {2, 2, 0, 3, 3, 0, 0, 0, 1, 0};
  raft::update_device(ydev->data, &y[0], ydev->len, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double fx, l1, l2;
  int C = 4;

  double alpha = 0.016 * N;

  SimpleMatOwning<double> z(allocator, C, N, stream);
  SimpleVecOwning<double> w0(allocator, C * (D + 1), stream);

  Softmax<double> loss_b(handle, D, C, true);
  Softmax<double> loss_no_b(handle, D, C, false);

  l1              = alpha;
  l2              = 0.0;
  double obj_l1_b = 0.5407911382311313;

  fx = run(handle, loss_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  fx =
    run_api(cuml_handle, 2, C, loss_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  l1              = 0.0;
  l2              = alpha;
  double obj_l2_b = 0.5721784062720949;

  fx = run(handle, loss_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  fx =
    run_api(cuml_handle, 2, C, loss_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  l1                 = alpha;
  l2                 = 0.0;
  double obj_l1_no_b = 0.6606929813245878;

  fx = run(handle, loss_no_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  fx = run_api(
    cuml_handle, 2, C, loss_no_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  l1 = 0.0;
  l2 = alpha;

  double obj_l2_no_b = 0.6597171282106854;

  fx = run(handle, loss_no_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));

  fx = run_api(
    cuml_handle, 2, C, loss_no_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
}

TEST_F(QuasiNewtonTest, linear_regression_vs_sklearn)
{
  raft::CompareApprox<double> compApprox(tol);
  double y[N] = {0.2675836026202781,
                 -0.0678277759663704,
                 -0.6334027174275105,
                 -0.1018336189077367,
                 0.0933815935886932,
                 -1.1058853496996381,
                 -0.1658298189619160,
                 -0.2954290675648911,
                 0.7966520536712608,
                 -1.0767450516284769};
  raft::update_device(ydev->data, &y[0], ydev->len, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double fx, l1, l2;
  double alpha = 0.01 * N;

  SimpleVecOwning<double> w0(allocator, D + 1, stream);
  SimpleMatOwning<double> z(allocator, 1, N, stream);
  SquaredLoss<double> loss_b(handle, D, true);
  SquaredLoss<double> loss_no_b(handle, D, false);

  l1               = alpha;
  l2               = 0.0;
  double w_l1_b[2] = {-0.4952397281519840, 0.3813315300180231};
  double b_l1_b    = -0.08140861819001188;
  double obj_l1_b  = 0.011136986298775138;
  fx               = run(handle, loss_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));
  ASSERT_TRUE(checkParamsEqual(handle, &w_l1_b[0], &b_l1_b, w0.data, loss_b, compApprox, stream));

  fx =
    run_api(cuml_handle, 1, 1, loss_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  l1               = 0.0;
  l2               = alpha;
  double w_l2_b[2] = {-0.5022384743587150, 0.3937352417485087};
  double b_l2_b    = -0.08062397391797513;
  double obj_l2_b  = 0.004268621967866347;

  fx = run(handle, loss_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));
  ASSERT_TRUE(checkParamsEqual(handle, &w_l2_b[0], &b_l2_b, w0.data, loss_b, compApprox, stream));

  fx =
    run_api(cuml_handle, 1, 1, loss_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  l1                  = alpha;
  l2                  = 0.0;
  double w_l1_no_b[2] = {-0.5175178128147135, 0.3720844589831813};
  double obj_l1_no_b  = 0.013981355746112447;

  fx = run(handle, loss_no_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));
  ASSERT_TRUE(
    checkParamsEqual(handle, &w_l1_no_b[0], nobptr, w0.data, loss_no_b, compApprox, stream));

  fx = run_api(
    cuml_handle, 1, 1, loss_no_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  l1                  = 0.0;
  l2                  = alpha;
  double w_l2_no_b[2] = {-0.5241651041233270, 0.3846317886627560};
  double obj_l2_no_b  = 0.007061261366969662;

  fx = run(handle, loss_no_b, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
  ASSERT_TRUE(
    checkParamsEqual(handle, &w_l2_no_b[0], nobptr, w0.data, loss_no_b, compApprox, stream));

  fx = run_api(
    cuml_handle, 1, 1, loss_no_b.fit_intercept, *Xdev, *ydev, l1, l2, w0.data, z, 0, stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
}

TEST_F(QuasiNewtonTest, predict)
{
  raft::CompareApprox<double> compApprox(1e-8);
  std::vector<double> w_host(D);
  w_host[0] = 1;
  std::vector<double> preds_host(N);
  SimpleVecOwning<double> w(allocator, D, stream);
  SimpleVecOwning<double> preds(allocator, N, stream);

  raft::update_device(w.data, &w_host[0], w.len, stream);

  qnPredict(handle, Xdev->data, false, N, D, 2, false, w.data, 0, preds.data, stream);
  raft::update_host(&preds_host[0], preds.data, preds.len, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int it = 0; it < N; it++) {
    ASSERT_TRUE(X[it][0] > 0 ? compApprox(preds_host[it], 1) : compApprox(preds_host[it], 0));
  }

  qnPredict(handle, Xdev->data, false, N, D, 1, false, w.data, 1, preds.data, stream);
  raft::update_host(&preds_host[0], preds.data, preds.len, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int it = 0; it < N; it++) {
    ASSERT_TRUE(compApprox(X[it][0], preds_host[it]));
  }
}

TEST_F(QuasiNewtonTest, predict_softmax)
{
  raft::CompareApprox<double> compApprox(1e-8);
  int C = 4;
  std::vector<double> w_host(C * D);
  w_host[0]         = 1;
  w_host[D * C - 1] = 1;

  std::vector<double> preds_host(N);
  SimpleVecOwning<double> w(allocator, w_host.size(), stream);
  SimpleVecOwning<double> preds(allocator, N, stream);

  raft::update_device(w.data, &w_host[0], w.len, stream);

  qnPredict(handle, Xdev->data, false, N, D, C, false, w.data, 2, preds.data, stream);
  raft::update_host(&preds_host[0], preds.data, preds.len, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int it = 0; it < N; it++) {
    if (X[it][0] < 0 && X[it][1] < 0) {
      ASSERT_TRUE(compApprox(1, preds_host[it]));
    } else if (X[it][0] > X[it][1]) {
      ASSERT_TRUE(compApprox(0, preds_host[it]));
    } else {
      ASSERT_TRUE(compApprox(C - 1, preds_host[it]));
    }
  }
}

TEST_F(QuasiNewtonTest, dense_vs_sparse_logistic)
{
#if CUDART_VERSION >= 11020
  GTEST_SKIP();
#endif
  // Prepare a sparse input matrix from the dense matrix X.
  // Yes, it's not sparse at all, yet the test does check whether the behaviour
  // of dense and sparse variants is the same.
  rmm::device_uvector<int> mem_X_cols(N * D, stream);
  rmm::device_uvector<int> mem_X_row_ids(N + 1, stream);
  int host_X_cols[N][D];
  int host_X_row_ids[N + 1];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      host_X_cols[i][j] = j;
    }
  }
  for (int i = 0; i < N + 1; i++) {
    host_X_row_ids[i] = i * D;
  }
  raft::update_device(mem_X_cols.data(), &host_X_cols[0][0], mem_X_cols.size(), stream);
  raft::update_device(mem_X_row_ids.data(), &host_X_row_ids[0], mem_X_row_ids.size(), stream);
  SimpleSparseMat<double> X_sparse(
    Xdev->data, mem_X_cols.data(), mem_X_row_ids.data(), N * D, N, D);

  raft::CompareApprox<double> compApprox(tol);
  double y[N] = {2, 2, 0, 3, 3, 0, 0, 0, 1, 0};
  raft::update_device(ydev->data, &y[0], ydev->len, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int C         = 4;
  int loss_type = 2;  // Softmax (loss_b, loss_no_b)
  double alpha  = 0.016 * N;
  Softmax<double> loss_b(handle, D, C, true);
  Softmax<double> loss_no_b(handle, D, C, false);

  SimpleMatOwning<double> z_dense(allocator, C, N, stream);
  SimpleMatOwning<double> z_sparse(allocator, C, N, stream);
  SimpleVecOwning<double> w0_dense(allocator, C * (D + 1), stream);
  SimpleVecOwning<double> w0_sparse(allocator, C * (D + 1), stream);

  std::vector<double> preds_dense_host(N);
  std::vector<double> preds_sparse_host(N);
  SimpleVecOwning<double> preds_dense(allocator, N, stream);
  SimpleVecOwning<double> preds_sparse(allocator, N, stream);

  auto test_run = [&](double l1, double l2, Softmax<double> loss) {
    double f_dense, f_sparse;
    f_dense  = run(handle, loss, *Xdev, *ydev, l1, l2, w0_dense.data, z_dense, 0, stream);
    f_sparse = run(handle, loss, X_sparse, *ydev, l1, l2, w0_sparse.data, z_sparse, 0, stream);
    ASSERT_TRUE(compApprox(f_dense, f_sparse));

    qnPredict(handle,
              Xdev->data,
              Xdev->ord == COL_MAJOR,
              N,
              D,
              C,
              loss.fit_intercept,
              w0_dense.data,
              loss_type,
              preds_dense.data,
              stream);
    qnPredictSparse(handle,
                    X_sparse.values,
                    X_sparse.cols,
                    X_sparse.row_ids,
                    X_sparse.nnz,
                    N,
                    D,
                    C,
                    loss.fit_intercept,
                    w0_sparse.data,
                    loss_type,
                    preds_sparse.data,
                    stream);

    raft::update_host(&preds_dense_host[0], preds_dense.data, preds_dense.len, stream);
    raft::update_host(&preds_sparse_host[0], preds_sparse.data, preds_sparse.len, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < N; i++) {
      ASSERT_TRUE(compApprox(preds_dense_host[i], preds_sparse_host[i]));
    }

    f_dense  = run_api(cuml_handle,
                      2,
                      C,
                      loss.fit_intercept,
                      *Xdev,
                      *ydev,
                      l1,
                      l2,
                      w0_dense.data,
                      z_dense,
                      0,
                      stream);
    f_sparse = run_api(cuml_handle,
                       2,
                       C,
                       loss.fit_intercept,
                       X_sparse,
                       *ydev,
                       l1,
                       l2,
                       w0_sparse.data,
                       z_sparse,
                       0,
                       stream);
    ASSERT_TRUE(compApprox(f_dense, f_sparse));
  };

  test_run(alpha, 0.0, loss_b);
  test_run(0.0, alpha, loss_b);
  test_run(alpha, 0.0, loss_no_b);
  test_run(0.0, alpha, loss_no_b);
}

}  // namespace GLM
}  // end namespace ML
