/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <iomanip>
#include <raft/handle.hpp>
#include <raft/random/rng.cuh>
#include <solver/lars_impl.cuh>
#include <sstream>
#include <vector>

namespace ML {
namespace Solver {
namespace Lars {

template <typename math_t>
class LarsTest : public ::testing::Test {
 protected:
  LarsTest()
    : allocator(handle.get_device_allocator()),
      cor(allocator, handle.get_stream(), n_cols),
      X(allocator, handle.get_stream(), n_cols * n_rows),
      G(allocator, handle.get_stream(), n_cols * n_cols),
      sign(allocator, handle.get_stream(), n_cols),
      ws(allocator, handle.get_stream(), n_cols),
      A(allocator, handle.get_stream(), 1) {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    raft::update_device(cor.data(), cor_host, n_cols, stream);
    raft::update_device(X.data(), X_host, n_cols * n_rows, stream);
    raft::update_device(G.data(), G_host, n_cols * n_cols, stream);
    raft::update_device(sign.data(), sign_host, n_cols, stream);
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }

  void testSelectMostCorrelated() {
    math_t cj;
    int idx;
    MLCommon::device_buffer<math_t> workspace(allocator, stream, n_cols);
    ML::Solver::Lars::selectMostCorrelated(n_active, n_cols, cor.data(), &cj,
                                           workspace, &idx, n_rows, indices, 1,
                                           stream);
    EXPECT_EQ(idx, 3);
    EXPECT_EQ(7, cj);
  }

  void testMoveToActive() {
    ML::Solver::Lars::moveToActive(
      handle.get_cublas_handle(), &n_active, 3, X.data(), n_rows, n_cols,
      n_rows, cor.data(), indices, G.data(), n_cols, sign.data(), stream);
    EXPECT_EQ(n_active, 3);

    EXPECT_TRUE(raft::devArrMatchHost(cor_exp, cor.data(), n_cols,
                                      raft::Compare<math_t>()));
    EXPECT_TRUE(raft::devArrMatchHost(G_exp, G.data(), n_cols * n_cols,
                                      raft::Compare<math_t>()));
    EXPECT_TRUE(raft::devArrMatch((math_t)1.0, sign.data() + n_active - 1, 1,
                                  raft::Compare<math_t>()));

    // Do it again with G == nullptr to test if X is properly changed
    n_active = 2;
    ML::Solver::Lars::moveToActive(handle.get_cublas_handle(), &n_active, 3,
                                   X.data(), n_rows, n_cols, n_rows, cor.data(),
                                   indices, (math_t*)nullptr, n_cols,
                                   sign.data(), stream);
    EXPECT_TRUE(raft::devArrMatchHost(X_exp, X.data(), n_rows * n_cols,
                                      raft::Compare<math_t>()));
  }

  void calcUExp(math_t* G, int n_cols, math_t* U_dev_exp) {
    auto allocator = handle.get_device_allocator();
    MLCommon::device_buffer<int> devInfo(allocator, stream, 1);
    MLCommon::device_buffer<math_t> workspace(allocator, stream);
    int n_work;
    const int ld_U = n_cols;
    CUSOLVER_CHECK(raft::linalg::cusolverDnpotrf_bufferSize(
      handle.get_cusolver_dn_handle(), CUBLAS_FILL_MODE_UPPER, n_cols,
      U_dev_exp, ld_U, &n_work));
    workspace.resize(n_work, stream);
    // Expected solution using Cholesky factorization from scratch
    raft::copy(U_dev_exp, G, n_cols * ld_U, stream);
    CUSOLVER_CHECK(raft::linalg::cusolverDnpotrf(
      handle.get_cusolver_dn_handle(), CUBLAS_FILL_MODE_UPPER, n_cols,
      U_dev_exp, ld_U, workspace.data(), n_work, devInfo.data(), stream));
  }

  // Initialize a mix of G and U matrices to test updateCholesky
  void initGU(math_t* GU, math_t* G, math_t* U, int n_active, bool copy_G) {
    const int ld_U = n_cols;
    // First we copy over all elements, because the factorization only replaces
    // the upper triangular part. This way it will be easier to compare to the
    // reference solution.
    raft::copy(GU, G, n_cols * n_cols, stream);
    if (!copy_G) {
      // zero the new colum of G
      CUDA_CHECK(cudaMemsetAsync(GU + (n_active - 1) * n_cols, 0,
                                 n_cols * sizeof(math_t), stream));
    }
    for (int i = 0; i < n_active - 1; i++) {
      raft::copy(GU + i * ld_U, U + i * ld_U, i + 1, stream);
    }
  }

  void testUpdateCholesky() {
    const int ld_X = n_rows;
    const int ld_G = n_cols;
    const int ld_U = ld_G;
    auto allocator = handle.get_device_allocator();
    MLCommon::device_buffer<math_t> workspace(allocator, stream);
    MLCommon::device_buffer<math_t> U_dev_exp(allocator, stream,
                                              n_cols * n_cols);
    calcUExp(G.data(), n_cols, U_dev_exp.data());

    MLCommon::device_buffer<math_t> U(allocator, stream, n_cols * n_cols);
    n_active = 4;
    math_t eps = -1;

    // First test with U already initialized
    initGU(U.data(), G.data(), U_dev_exp.data(), n_active, true);
    ML::Solver::Lars::updateCholesky(handle, n_active, X.data(), n_rows, n_cols,
                                     ld_X, U.data(), ld_U, U.data(), ld_G,
                                     workspace, eps, stream);
    EXPECT_TRUE(raft::devArrMatch(U_dev_exp.data(), U.data(), n_cols * n_cols,
                                  raft::CompareApprox<math_t>(1e-5)));

    // Next test where G and U are separate arrays
    initGU(U.data(), G.data(), U_dev_exp.data(), n_active, false);
    ML::Solver::Lars::updateCholesky(handle, n_active, X.data(), n_rows, n_cols,
                                     ld_X, U.data(), ld_U, G.data(), ld_G,
                                     workspace, eps, stream);
    EXPECT_TRUE(raft::devArrMatch(U_dev_exp.data(), U.data(), n_cols * n_cols,
                                  raft::CompareApprox<math_t>(1e-5)));

    // Third test without Gram matrix.
    initGU(U.data(), G.data(), U_dev_exp.data(), n_active, false);
    ML::Solver::Lars::updateCholesky(handle, n_active, X.data(), n_rows, n_cols,
                                     ld_X, U.data(), ld_U, (math_t*)nullptr, 0,
                                     workspace, eps, stream);
    EXPECT_TRUE(raft::devArrMatch(U_dev_exp.data(), U.data(), n_cols * n_cols,
                                  raft::CompareApprox<math_t>(1e-4)));
  }

  void testCalcW0() {
    n_active = 4;
    const int ld_U = n_cols;
    auto allocator = handle.get_device_allocator();
    MLCommon::device_buffer<math_t> ws(allocator, stream, n_active);
    MLCommon::device_buffer<math_t> U(allocator, stream, n_cols * ld_U);
    calcUExp(G.data(), n_cols, U.data());

    ML::Solver::Lars::calcW0(handle, n_active, n_cols, sign.data(), U.data(),
                             ld_U, ws.data(), stream);
    EXPECT_TRUE(raft::devArrMatchHost(ws0_exp, ws.data(), n_active,
                                      raft::CompareApprox<math_t>(1e-3)));
  }

  void testCalcA() {
    n_active = 4;
    MLCommon::device_buffer<math_t> ws(handle.get_device_allocator(), stream,
                                       n_active);
    raft::update_device(ws.data(), ws0_exp, n_active, stream);

    ML::Solver::Lars::calcA(handle, A.data(), n_active, sign.data(), ws.data(),
                            stream);
    EXPECT_TRUE(raft::devArrMatch((math_t)0.20070615686577709, A.data(), 1,
                                  raft::CompareApprox<math_t>(1e-6)));
  }

  void testEquiangular() {
    n_active = 4;
    auto allocator = handle.get_device_allocator();
    MLCommon::device_buffer<math_t> workspace(allocator, stream);
    MLCommon::device_buffer<math_t> u_eq(allocator, stream, n_rows);
    MLCommon::device_buffer<math_t> U(allocator, stream, n_cols * n_cols);
    calcUExp(G.data(), n_cols, U.data());
    initGU(G.data(), G.data(), U.data(), n_active, true);
    const int ld_X = n_rows;
    const int ld_U = n_cols;
    const int ld_G = n_cols;
    ML::Solver::Lars::calcEquiangularVec(
      handle, n_active, X.data(), n_rows, n_cols, ld_X, sign.data(), G.data(),
      ld_U, G.data(), ld_G, workspace, ws.data(), A.data(), u_eq.data(),
      (math_t)-1, stream);

    EXPECT_TRUE(raft::devArrMatchHost(ws_exp, ws.data(), n_active,
                                      raft::CompareApprox<math_t>(1e-3)));

    EXPECT_TRUE(raft::devArrMatch((math_t)0.20070615686577709, A.data(), 1,
                                  raft::CompareApprox<math_t>(1e-4)));

    // Now test without Gram matrix, u should be calculated in this case
    initGU(G.data(), G.data(), U.data(), n_active, false);
    ML::Solver::Lars::calcEquiangularVec(
      handle, n_active, X.data(), n_rows, n_cols, ld_X, sign.data(), G.data(),
      ld_U, (math_t*)nullptr, 0, workspace, ws.data(), A.data(), u_eq.data(),
      (math_t)-1, stream);

    EXPECT_TRUE(raft::devArrMatchHost(u_eq_exp, u_eq.data(), 1,
                                      raft::CompareApprox<math_t>(1e-3)));
  }

  void testCalcMaxStep() {
    n_active = 2;
    math_t A_host = 3.6534305290498055;
    math_t ws_host[2] = {0.25662594, -0.01708941};
    math_t u_host[4] = {0.10282127, -0.01595011, 0.07092104, -0.99204011};
    math_t cor_host[4] = {137, 42, 4.7, 13.2};
    const int ld_X = n_rows;
    const int ld_G = n_cols;
    MLCommon::device_buffer<math_t> u(handle.get_device_allocator(), stream,
                                      n_rows);
    MLCommon::device_buffer<math_t> ws(handle.get_device_allocator(), stream,
                                       n_active);
    MLCommon::device_buffer<math_t> gamma(handle.get_device_allocator(), stream,
                                          1);
    MLCommon::device_buffer<math_t> U(handle.get_device_allocator(), stream,
                                      n_cols * n_cols);
    MLCommon::device_buffer<math_t> a_vec(handle.get_device_allocator(), stream,
                                          n_cols - n_active);
    raft::update_device(A.data(), &A_host, 1, stream);
    raft::update_device(ws.data(), ws_host, n_active, stream);
    raft::update_device(u.data(), u_host, n_rows, stream);
    raft::update_device(cor.data(), cor_host, n_cols, stream);

    const int max_iter = n_cols;
    math_t cj = 42;
    ML::Solver::Lars::calcMaxStep(handle, max_iter, n_rows, n_cols, n_active,
                                  cj, A.data(), cor.data(), G.data(), ld_G,
                                  X.data(), ld_X, (math_t*)nullptr, ws.data(),
                                  gamma.data(), a_vec.data(), stream);
    math_t gamma_exp = 0.20095407186830386;
    EXPECT_TRUE(raft::devArrMatch(gamma_exp, gamma.data(), 1,
                                  raft::CompareApprox<math_t>(1e-6)));
    math_t a_vec_exp[2] = {24.69447886, -139.66289908};
    EXPECT_TRUE(raft::devArrMatchHost(a_vec_exp, a_vec.data(), a_vec.size(),
                                      raft::CompareApprox<math_t>(1e-4)));

    // test without G matrix, we use U as input in this case
    CUDA_CHECK(cudaMemsetAsync(gamma.data(), 0, sizeof(math_t), stream));
    CUDA_CHECK(
      cudaMemsetAsync(a_vec.data(), 0, a_vec.size() * sizeof(math_t), stream));
    ML::Solver::Lars::calcMaxStep(handle, max_iter, n_rows, n_cols, n_active,
                                  cj, A.data(), cor.data(), (math_t*)nullptr, 0,
                                  X.data(), ld_X, u.data(), ws.data(),
                                  gamma.data(), a_vec.data(), stream);
    EXPECT_TRUE(raft::devArrMatch(gamma_exp, gamma.data(), 1,
                                  raft::CompareApprox<math_t>(1e-6)));
    EXPECT_TRUE(raft::devArrMatchHost(a_vec_exp, a_vec.data(), a_vec.size(),
                                      raft::CompareApprox<math_t>(1e-4)));

    // Last iteration
    n_active = max_iter;
    CUDA_CHECK(cudaMemsetAsync(gamma.data(), 0, sizeof(math_t), stream));
    ML::Solver::Lars::calcMaxStep(handle, max_iter, n_rows, n_cols, n_active,
                                  cj, A.data(), cor.data(), (math_t*)nullptr, 0,
                                  X.data(), ld_X, u.data(), ws.data(),
                                  gamma.data(), a_vec.data(), stream);
    gamma_exp = 11.496044516528272;
    EXPECT_TRUE(raft::devArrMatch(gamma_exp, gamma.data(), 1,
                                  raft::CompareApprox<math_t>(1e-6)));
  }

  raft::handle_t handle;
  cudaStream_t stream;
  std::shared_ptr<deviceAllocator> allocator;

  const int n_rows = 4;
  const int n_cols = 4;
  int n_active = 2;

  math_t cor_host[4] = {0, 137, 4, 7};
  math_t cor_exp[4] = {0, 137, 7, 4};
  // clang-format off
  // Keep in mind that we actually define column major matrices, so a row here
  // corresponds to a column of the matrix.
  math_t X_host[16] = { 1.,   4.,   9.,  -3.,
                        9.,  61., 131.,  13.,
                        3.,  22., 111., -17.,
                        0.,  40.,  40., 143.};
  math_t X_exp[16] =  { 1.,   4.,   9.,  -3.,
                        9.,  61., 131.,  13.,
                        0.,  40.,  40., 143.,
                        3.,  22., 111., -17.};
  math_t G_host[16] = { 107.,  1393.,  1141.,    91.,
                       1393., 21132., 15689.,  9539.,
                       1141., 15689., 13103.,  2889.,
                         91.,  9539.,  2889., 23649.};
  math_t G_exp[16] = {  107.,  1393.,    91.,  1141.,
                       1393., 21132.,  9539., 15689.,
                         91.,  9539., 23649.,  2889.,
                       1141., 15689.,  2889., 13103.};
  // clang-format on
  int indices[4] = {3, 2, 1, 0};
  int indices_exp[4] = {3, 4, 0, 1};
  math_t sign_host[4] = {1, -1, 1, -1};
  math_t ws0_exp[4] = {22.98636271, -2.15225918, 0.41474128, 0.72897179};
  math_t ws_exp[4] = {4.61350452, -0.43197167, 0.08324113, 0.14630913};
  math_t u_eq_exp[4] = {0.97548288, -0.21258388, 0.02538227, 0.05096055};

  MLCommon::device_buffer<math_t> cor;
  MLCommon::device_buffer<math_t> X;
  MLCommon::device_buffer<math_t> G;
  MLCommon::device_buffer<math_t> sign;
  MLCommon::device_buffer<math_t> ws;
  MLCommon::device_buffer<math_t> A;
};

typedef ::testing::Types<float, double> FloatTypes;

TYPED_TEST_CASE(LarsTest, FloatTypes);

TYPED_TEST(LarsTest, select) { this->testSelectMostCorrelated(); }
TYPED_TEST(LarsTest, moveToActive) { this->testMoveToActive(); }
TYPED_TEST(LarsTest, updateCholesky) { this->testUpdateCholesky(); }
TYPED_TEST(LarsTest, calcW0) { this->testCalcW0(); }
TYPED_TEST(LarsTest, calcA) { this->testCalcA(); }
TYPED_TEST(LarsTest, equiangular) { this->testEquiangular(); }
TYPED_TEST(LarsTest, maxStep) { this->testCalcMaxStep(); }

template <typename math_t>
class LarsTestFitPredict : public ::testing::Test {
 protected:
  LarsTestFitPredict()
    : allocator(handle.get_device_allocator()),
      X(allocator, handle.get_stream(), n_cols * n_rows),
      y(allocator, handle.get_stream(), n_rows),
      G(allocator, handle.get_stream(), n_cols * n_cols),
      beta(allocator, handle.get_stream(), n_cols),
      coef_path(allocator, handle.get_stream(), (n_cols + 1) * n_cols),
      alphas(allocator, handle.get_stream(), n_cols + 1),
      active_idx(allocator, handle.get_stream(), n_cols) {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    raft::update_device(X.data(), X_host, n_cols * n_rows, stream);
    raft::update_device(y.data(), y_host, n_rows, stream);
    raft::update_device(G.data(), G_host, n_cols * n_cols, stream);
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }

  void testFitGram() {
    int max_iter = 10;
    int verbosity = 0;
    int n_active;
    ML::Solver::Lars::larsFit(handle, X.data(), n_rows, n_cols, y.data(),
                              beta.data(), active_idx.data(), alphas.data(),
                              &n_active, G.data(), max_iter,
                              (math_t*)nullptr,  //coef_path.data(),
                              verbosity, n_rows, n_cols, (math_t)-1);
    EXPECT_EQ(n_cols, n_active);
    EXPECT_TRUE(raft::devArrMatchHost(beta_exp, beta.data(), n_cols,
                                      raft::CompareApprox<math_t>(1e-5)));
    EXPECT_TRUE(raft::devArrMatchHost(alphas_exp, alphas.data(), n_cols + 1,
                                      raft::CompareApprox<math_t>(1e-4)));
    EXPECT_TRUE(raft::devArrMatchHost(indices_exp, active_idx.data(), n_cols,
                                      raft::Compare<int>()));
  }

  void testFitX() {
    int max_iter = 10;
    int verbosity = 0;
    int n_active;
    ML::Solver::Lars::larsFit(handle, X.data(), n_rows, n_cols, y.data(),
                              beta.data(), active_idx.data(), alphas.data(),
                              &n_active, (math_t*)nullptr, max_iter,
                              (math_t*)nullptr,  //coef_path.data(),
                              verbosity, n_rows, n_cols, (math_t)-1);
    EXPECT_EQ(n_cols, n_active);
    EXPECT_TRUE(raft::devArrMatchHost(beta_exp, beta.data(), n_cols,
                                      raft::CompareApprox<math_t>(1e-5)));
    EXPECT_TRUE(raft::devArrMatchHost(alphas_exp, alphas.data(), n_cols + 1,
                                      raft::CompareApprox<math_t>(1e-4)));
    EXPECT_TRUE(raft::devArrMatchHost(indices_exp, active_idx.data(), n_cols,
                                      raft::Compare<int>()));
  }

  void testPredictV1() {
    int ld_X = n_rows;
    int n_active = n_cols;
    raft::update_device(beta.data(), beta_exp, n_active, stream);
    raft::update_device(active_idx.data(), indices_exp, n_active, stream);
    CUDA_CHECK(cudaMemsetAsync(y.data(), 0, n_rows * sizeof(math_t), stream));
    math_t intercept = 0;
    ML::Solver::Lars::larsPredict(handle, X.data(), n_rows, n_cols, ld_X,
                                  beta.data(), n_active, active_idx.data(),
                                  intercept, y.data());
    EXPECT_TRUE(raft::devArrMatchHost(pred_exp, y.data(), n_rows,
                                      raft::CompareApprox<math_t>(1e-5)));
  }

  void testPredictV2() {
    int ld_X = n_rows;
    int n_active = n_cols;

    // We set n_cols > n_active to trigger prediction path where columns of X
    // are copied.
    int n_cols_loc = n_cols + 1;
    raft::update_device(beta.data(), beta_exp, n_active, stream);
    raft::update_device(active_idx.data(), indices_exp, n_active, stream);
    CUDA_CHECK(cudaMemsetAsync(y.data(), 0, n_rows * sizeof(math_t), stream));
    math_t intercept = 0;
    ML::Solver::Lars::larsPredict(handle, X.data(), n_rows, n_cols_loc, ld_X,
                                  beta.data(), n_active, active_idx.data(),
                                  intercept, y.data());
    EXPECT_TRUE(raft::devArrMatchHost(pred_exp, y.data(), n_rows,
                                      raft::CompareApprox<math_t>(1e-5)));
  }

  void testFitLarge() {
    int n_rows = 65536;
    int n_cols = 10;
    int max_iter = n_cols;
    int verbosity = 0;
    int n_active;
    MLCommon::device_buffer<math_t> X(allocator, stream, n_rows * n_cols);
    MLCommon::device_buffer<math_t> y(allocator, stream, n_rows);
    beta.resize(max_iter, stream);
    active_idx.resize(max_iter, stream);
    alphas.resize(max_iter + 1, stream);
    raft::random::Rng r(1234);
    r.uniform(X.data(), n_rows * n_cols, math_t(-1.0), math_t(1.0), stream);
    r.uniform(y.data(), n_rows, math_t(-1.0), math_t(1.0), stream);

    ML::Solver::Lars::larsFit(
      handle, X.data(), n_rows, n_cols, y.data(), beta.data(),
      active_idx.data(), alphas.data(), &n_active, (math_t*)nullptr, max_iter,
      (math_t*)nullptr, verbosity, n_rows, n_cols, (math_t)-1);

    EXPECT_EQ(n_cols, n_active);
  }

  raft::handle_t handle;
  cudaStream_t stream;
  std::shared_ptr<deviceAllocator> allocator;

  const int n_rows = 10;
  const int n_cols = 5;

  math_t cor_host[4] = {0, 137, 4, 7};
  math_t cor_exp[4] = {0, 137, 7, 4};
  // clang-format off
  // We actually define column major matrices, so a row here corresponds to a
  // column of the matrix.
  math_t X_host[50] = {
      -1.59595376,  1.02675861,  0.45079426,  0.32621407,  0.29018821,
      -1.30640121,  0.67025452,  0.30196285,  1.28636261, -1.45018015,
      -1.39544855,  0.90533337, -0.36980987,  0.23706301,  1.33296593,
      -0.524911  , -0.86187751,  0.30764958, -1.24415885,  1.61319389,
      -0.01500442, -2.25985187, -0.11147508,  1.08410381,  0.59451579,
       0.62568849,  0.99811378, -1.09709453, -0.51940485,  0.70040887,
      -1.81995734, -0.24101756,  1.21308053,  0.87517302, -0.19806613,
       1.50733111,  0.06332581, -0.65824129,  0.45640974, -1.19803788,
       0.13838875, -1.01018604, -0.15828873, -1.26652781,  0.41229797,
      -0.00953721, -0.10602222, -0.51746536, -0.10397987,  2.62132051};
  math_t G_host[25] = {
       10.        , -0.28482905, -3.98401069,  3.63094793, -5.77295066,
       -0.28482905, 10.        , -0.68437245, -1.73251284,  3.49545153,
       -3.98401069, -0.68437245, 10.        ,  1.92006934,  3.51643227,
        3.63094793, -1.73251284,  1.92006934, 10.        , -4.25887055,
       -5.77295066,  3.49545153,  3.51643227, -4.25887055, 10.         };
  math_t y_host[10] = {
      -121.34354343, -170.25131089,   19.34173641,  89.75429795,  99.97210232,
        83.67110463,   40.65749808, -109.1490306 , -72.97243308, 140.31957861};
  // clang-format on
  math_t beta_exp[10] = {7.48589389e+01, 3.90513025e+01, 3.81912823e+01,
                         2.69095277e+01, -4.74545001e-02};
  math_t alphas_exp[6] = {8.90008255e+01, 4.00677648e+01, 2.46147690e+01,
                          2.06052321e+01, 3.70155968e-02, 0.0740366429090};
  math_t pred_exp[10] = {
    -121.32409183, -170.25278892, 19.26177047,   89.73931476,  100.07545046,
    83.71217894,   40.59397899,   -109.19137223, -72.89633962, 140.28189898};
  int indices_exp[5] = {2, 1, 3, 4, 0};

  MLCommon::device_buffer<math_t> X;
  MLCommon::device_buffer<math_t> G;
  MLCommon::device_buffer<math_t> y;
  MLCommon::device_buffer<math_t> beta;
  MLCommon::device_buffer<math_t> alphas;
  MLCommon::device_buffer<math_t> coef_path;
  MLCommon::device_buffer<int> active_idx;
};

TYPED_TEST_CASE(LarsTestFitPredict, FloatTypes);

TYPED_TEST(LarsTestFitPredict, fitGram) { this->testFitGram(); }
TYPED_TEST(LarsTestFitPredict, fitX) { this->testFitX(); }
TYPED_TEST(LarsTestFitPredict, fitLarge) { this->testFitLarge(); }
TYPED_TEST(LarsTestFitPredict, predictV1) { this->testPredictV1(); }
TYPED_TEST(LarsTestFitPredict, predictV2) { this->testPredictV2(); }
};  // namespace Lars
};  // namespace Solver
};  // namespace ML
