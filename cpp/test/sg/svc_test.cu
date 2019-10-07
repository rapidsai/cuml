/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <cub/cub.cuh>
#include <iostream>
#include <string>
#include <vector>
#include "common/cumlHandle.hpp"
#include "linalg/binary_op.h"
#include "linalg/map_then_reduce.h"
#include "linalg/transpose.h"
#include "matrix/grammatrix.h"
#include "matrix/kernelmatrices.h"
#include "random/make_blobs.h"
#include "random/rng.h"
#include "svm/smoblocksolve.h"
#include "svm/smosolver.h"
#include "svm/svc.hpp"
#include "svm/svm_model.h"
#include "svm/svm_parameter.h"
#include "svm/workingset.h"
#include "test_utils.h"

namespace ML {
namespace SVM {
using namespace MLCommon;
using namespace Matrix;

template <typename math_t>
class WorkingSetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
    allocate(f_dev, 10);
    allocate(y_dev, 10);
    allocate(alpha_dev, 10);
    updateDevice(f_dev, f_host, 10, stream);
    updateDevice(y_dev, y_host, 10, stream);
    updateDevice(alpha_dev, alpha_host, 10, stream);
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }
  cumlHandle handle;
  cudaStream_t stream;
  WorkingSet<math_t> *ws;

  math_t f_host[10] = {1, 3, 10, 4, 2, 8, 6, 5, 9, 7};
  math_t *f_dev;

  math_t y_host[10] = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  math_t *y_dev;

  math_t C = 1.5;

  math_t alpha_host[10] = {0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5};
  math_t *alpha_dev;  //   l  l  l/u  l/u    u  u  l/u  l/u  l    l

  int expected_idx[4] = {4, 3, 8, 2};
  int expected_idx2[4] = {8, 2, 4, 9};
};

typedef ::testing::Types<float, double> FloatTypes;

TYPED_TEST_CASE(WorkingSetTest, FloatTypes);

TYPED_TEST(WorkingSetTest, Init) {
  this->ws = new WorkingSet<TypeParam>(this->handle.getImpl(),
                                       this->handle.getStream(), 10);
  EXPECT_EQ(this->ws->GetSize(), 10);
  delete this->ws;

  this->ws =
    new WorkingSet<TypeParam>(this->handle.getImpl(), this->stream, 100000);
  EXPECT_EQ(this->ws->GetSize(), 1024);
  delete this->ws;
}

TYPED_TEST(WorkingSetTest, Select) {
  this->ws =
    new WorkingSet<TypeParam>(this->handle.getImpl(), this->stream, 10, 4);
  EXPECT_EQ(this->ws->GetSize(), 4);
  this->ws->SimpleSelect(this->f_dev, this->alpha_dev, this->y_dev, this->C);
  ASSERT_TRUE(devArrMatchHost(this->expected_idx, this->ws->GetIndices(),
                              this->ws->GetSize(), Compare<int>()));

  this->ws->Select(this->f_dev, this->alpha_dev, this->y_dev, this->C);
  ASSERT_TRUE(devArrMatchHost(this->expected_idx, this->ws->GetIndices(),
                              this->ws->GetSize(), Compare<int>()));
  this->ws->Select(this->f_dev, this->alpha_dev, this->y_dev, this->C);

  ASSERT_TRUE(devArrMatchHost(this->expected_idx2, this->ws->GetIndices(),
                              this->ws->GetSize(), Compare<int>()));
  delete this->ws;
}

//TYPED_TEST(WorkingSetTest, Priority) {
// See Issue #946
//}

template <typename math_t>
class KernelCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
    cublas_handle = handle.getImpl().getCublasHandle();
    allocate(x_dev, n_rows * n_cols);
    updateDevice(x_dev, x_host, n_rows * n_cols, stream);

    allocate(ws_idx_dev, n_ws);
    updateDevice(ws_idx_dev, ws_idx_host, n_ws, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(ws_idx_dev));
  }

  // Naive host side kernel implementation used for comparison
  void ApplyNonlin(Matrix::KernelParams params) {
    switch (params.kernel) {
      case Matrix::LINEAR:
        break;
      case Matrix::POLYNOMIAL:
        for (int z = 0; z < n_rows * n_ws; z++) {
          math_t val = params.gamma * tile_host_expected[z] + params.coef0;
          tile_host_expected[z] = pow(val, params.degree);
        }
        break;
      case Matrix::TANH:
        for (int z = 0; z < n_rows * n_ws; z++) {
          math_t val = params.gamma * tile_host_expected[z] + params.coef0;
          tile_host_expected[z] = tanh(val);
        }
        break;
      case Matrix::RBF:
        for (int i = 0; i < n_ws; i++) {
          for (int j = 0; j < n_rows; j++) {
            math_t d = 0;
            for (int k = 0; k < n_cols; k++) {
              int idx_i = ws_idx_host[i];
              math_t diff = x_host[idx_i + k * n_rows] - x_host[j + k * n_rows];
              d += diff * diff;
            }
            tile_host_expected[i * n_rows + j] = exp(-params.gamma * d);
          }
        }
        break;
    }
  }
  cumlHandle handle;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;

  int n_rows = 4;
  int n_cols = 2;
  int n_ws = 3;

  math_t *x_dev;
  int *ws_idx_dev;

  math_t x_host[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int ws_idx_host[4] = {0, 1, 3};
  math_t tile_host_expected[12] = {26, 32, 38, 44, 32, 40,
                                   48, 56, 44, 56, 68, 80};
};

TYPED_TEST_CASE_P(KernelCacheTest);

TYPED_TEST_P(KernelCacheTest, EvalTest) {
  std::vector<Matrix::KernelParams> param_vec{
    Matrix::KernelParams{Matrix::LINEAR, 3, 1, 0},
    Matrix::KernelParams{Matrix::POLYNOMIAL, 2, 1.3, 1},
    Matrix::KernelParams{Matrix::TANH, 2, 0.5, 2.4},
    Matrix::KernelParams{Matrix::RBF, 2, 0.5, 0}};
  for (auto params : param_vec) {
    Matrix::GramMatrixBase<TypeParam> *kernel =
      Matrix::KernelFactory<TypeParam>::create(
        params, this->handle.getImpl().getCublasHandle());
    KernelCache<TypeParam> cache(this->handle.getImpl(), this->x_dev,
                                 this->n_rows, this->n_cols, this->n_ws,
                                 kernel);
    TypeParam *tile_dev = cache.GetTile(this->ws_idx_dev);
    // apply nonlinearity on tile_host_expected
    this->ApplyNonlin(params);
    ASSERT_TRUE(devArrMatchHost(this->tile_host_expected, tile_dev,
                                this->n_rows * this->n_ws,
                                CompareApprox<TypeParam>(1e-6f)));
    delete kernel;
  }
}

REGISTER_TYPED_TEST_CASE_P(KernelCacheTest, EvalTest);
INSTANTIATE_TYPED_TEST_CASE_P(My, KernelCacheTest, FloatTypes);

template <typename math_t>
class GetResultsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }

  void TestResults() {
    auto allocator = handle.getImpl().getDeviceAllocator();
    device_buffer<math_t> x_dev(allocator, stream, n_rows * n_cols);
    updateDevice(x_dev.data(), x_host, n_rows * n_cols, stream);
    device_buffer<math_t> f_dev(allocator, stream, n_rows);
    updateDevice(f_dev.data(), f_host, n_rows, stream);
    device_buffer<math_t> y_dev(allocator, stream, n_rows);
    updateDevice(y_dev.data(), y_host, n_rows, stream);
    device_buffer<math_t> alpha_dev(allocator, stream, n_rows);
    updateDevice(alpha_dev.data(), alpha_host, n_rows, stream);

    Results<math_t> res(handle.getImpl(), x_dev.data(), y_dev.data(), n_rows,
                        n_cols, C);
    res.Get(alpha_dev.data(), f_dev.data(), &dual_coefs, &n_coefs, &idx,
            &x_support, &b);

    ASSERT_EQ(n_coefs, 7);

    math_t dual_coefs_exp[] = {-0.1, -0.2, -1.5, 0.2, 0.4, 1.5, 1.5};
    EXPECT_TRUE(devArrMatchHost(dual_coefs_exp, dual_coefs, n_coefs,
                                CompareApprox<math_t>(1e-6f)));

    int idx_exp[] = {2, 3, 4, 6, 7, 8, 9};
    EXPECT_TRUE(devArrMatchHost(idx_exp, idx, n_coefs, Compare<int>()));

    math_t x_support_exp[] = {3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20};
    EXPECT_TRUE(devArrMatchHost(x_support_exp, x_support, n_coefs * n_cols,
                                CompareApprox<math_t>(1e-6f)));

    EXPECT_FLOAT_EQ(b, -6.25f);

    if (n_coefs > 0) {
      allocator->deallocate(dual_coefs, n_coefs * sizeof(math_t), stream);
      allocator->deallocate(idx, n_coefs * sizeof(int), stream);
      allocator->deallocate(x_support, n_coefs * n_cols * sizeof(math_t),
                            stream);
    }

    // Modify the test by setting all SVs bound, then b is calculated differently
    math_t alpha_host2[10] = {0, 0, 1.5, 1.5, 1.5, 0, 1.5, 1.5, 1.5, 1.5};
    updateDevice(alpha_dev.data(), alpha_host2, n_rows, stream);
    res.Get(alpha_dev.data(), f_dev.data(), &dual_coefs, &n_coefs, &idx,
            &x_support, &b);
    EXPECT_FLOAT_EQ(b, -5.5f);
  }
  int n_rows = 10;
  int n_cols = 2;
  math_t x_host[20] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  math_t f_host[10] = {1, 3, 10, 4, 2, 8, 6, 5, 9, 7};
  math_t y_host[10] = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  math_t alpha_host[10] = {0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5};
  //                      l  l  l/u  l/u    u  u  l/u  l/u  l    l
  math_t C = 1.5;

  math_t *dual_coefs;
  int n_coefs;
  int *idx;
  math_t *x_support;
  math_t b;

  cumlHandle handle;
  cudaStream_t stream;
};

TYPED_TEST_CASE(GetResultsTest, FloatTypes);

TYPED_TEST(GetResultsTest, Results) { this->TestResults(); }

template <typename math_t>
class SmoUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    stream = handle.getImpl().getInternalStream(0);
    cublasHandle_t cublas_handle = handle.getImpl().getCublasHandle();
    allocate(f_dev, n_rows, true);
    allocate(kernel_dev, n_rows * n_ws);
    updateDevice(kernel_dev, kernel_host, n_ws * n_rows, stream);
    allocate(delta_alpha_dev, n_ws);
    updateDevice(delta_alpha_dev, delta_alpha_host, n_ws, stream);
  }
  void RunTest() {
    SmoSolver<float> smo(handle.getImpl(), 1, 0.001, nullptr);
    smo.UpdateF(f_dev, n_rows, delta_alpha_dev, n_ws, kernel_dev);

    float f_host_expected[] = {0.1f, 7.4505806e-9f, 0.3f, 0.2f, 0.5f, 0.4f};
    devArrMatchHost(f_host_expected, f_dev, n_rows,
                    CompareApprox<math_t>(1e-6));
  }
  void TearDown() override {
    CUDA_CHECK(cudaFree(delta_alpha_dev));
    CUDA_CHECK(cudaFree(kernel_dev));
    CUDA_CHECK(cudaFree(f_dev));
  }
  cumlHandle handle;
  cudaStream_t stream;
  int n_rows = 6;
  int n_ws = 2;
  float *kernel_dev;
  float *f_dev;
  float *delta_alpha_dev;
  float kernel_host[12] = {3, 5, 4, 6, 5, 7, 4, 5, 7, 8, 10, 11};
  float delta_alpha_host[2] = {-0.1f, 0.1f};
};

TYPED_TEST_CASE(SmoUpdateTest, FloatTypes);
TYPED_TEST(SmoUpdateTest, Update) { this->RunTest(); }

template <typename math_t>
class SmoBlockSolverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
    cublas_handle = handle.getImpl().getCublasHandle();
    allocate(ws_idx_dev, n_ws);
    allocate(y_dev, n_rows);
    allocate(f_dev, n_rows);
    allocate(alpha_dev, n_rows, true);
    allocate(delta_alpha_dev, n_ws, true);
    allocate(kernel_dev, n_ws * n_rows);
    allocate(return_buff_dev, 2);

    updateDevice(ws_idx_dev, ws_idx_host, n_ws, stream);
    updateDevice(y_dev, y_host, n_rows, stream);
    updateDevice(f_dev, f_host, n_rows, stream);
    updateDevice(kernel_dev, kernel_host, n_ws * n_rows, stream);
  }

 public:  // because of the device lambda
  void testBlockSolve() {
    SmoBlockSolve<math_t, 1024><<<1, n_ws, 0, stream>>>(
      y_dev, n_rows, alpha_dev, n_ws, delta_alpha_dev, f_dev, kernel_dev,
      ws_idx_dev, 1.5f, 1e-3f, return_buff_dev, 1);
    CUDA_CHECK(cudaPeekAtLastError());

    math_t return_buff_exp[2] = {0.2, 1};
    devArrMatchHost(return_buff_exp, return_buff_dev, 2,
                    CompareApprox<math_t>(1e-6));

    math_t *delta_alpha_calc;
    allocate(delta_alpha_calc, n_rows);
    LinAlg::binaryOp(
      delta_alpha_calc, y_dev, alpha_dev, n_rows,
      [] __device__(math_t a, math_t b) { return a * b; }, stream);
    devArrMatch(delta_alpha_dev, delta_alpha_calc, n_rows,
                CompareApprox<math_t>(1e-6));
    CUDA_CHECK(cudaFree(delta_alpha_calc));
    math_t alpha_expected[] = {0, 0.1f, 0.1f, 0};
    devArrMatch(alpha_expected, alpha_dev, n_rows, CompareApprox<math_t>(1e-6));
  }

 protected:
  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(y_dev));
    CUDA_CHECK(cudaFree(f_dev));
    CUDA_CHECK(cudaFree(ws_idx_dev));
    CUDA_CHECK(cudaFree(alpha_dev));
    CUDA_CHECK(cudaFree(delta_alpha_dev));
    CUDA_CHECK(cudaFree(kernel_dev));
    CUDA_CHECK(cudaFree(return_buff_dev));
  }

  cumlHandle handle;
  cudaStream_t stream;
  cublasHandle_t cublas_handle;

  int n_rows = 4;
  int n_cols = 2;
  int n_ws = 4;

  int *ws_idx_dev;
  math_t *y_dev;
  math_t *f_dev;
  math_t *alpha_dev;
  math_t *delta_alpha_dev;
  math_t *kernel_dev;
  math_t *return_buff_dev;

  int ws_idx_host[4] = {0, 1, 2, 3};
  math_t y_host[4] = {1, 1, -1, -1};
  math_t f_host[4] = {0.4, 0.3, 0.5, 0.1};
  math_t kernel_host[16] = {26, 32, 38, 44, 32, 40, 48, 56,
                            38, 48, 58, 68, 44, 56, 68, 80};
};

TYPED_TEST_CASE(SmoBlockSolverTest, FloatTypes);

// test a single iteration of the block solver
TYPED_TEST(SmoBlockSolverTest, SolveSingleTest) { this->testBlockSolve(); }

template <typename math_t>
struct smoInput {
  math_t C;
  math_t tol;
  KernelParams kernel_params;
  int max_iter;
  int max_inner_iter;
};

template <typename math_t>
struct svcInput {
  math_t C;
  math_t tol;
  KernelParams kernel_params;
  int n_rows;
  int n_cols;
  math_t *x_dev;
  math_t *y_dev;
  bool predict;
};

template <typename math_t>
struct smoOutput {
  int n_support;
  std::vector<math_t> dual_coefs;
  math_t b;
  std::vector<math_t> w;
  std::vector<math_t> x_support;
  std::vector<int> idx;
};

template <typename math_t>
struct svmTol {
  math_t b;
  math_t cs;
  int n_sv;
};

template <typename math_t>
class SmoSolverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
    allocate(x_dev, n_rows * n_cols);
    allocate(ws_idx_dev, n_ws);
    allocate(y_dev, n_rows);
    allocate(y_pred, n_rows);
    allocate(f_dev, n_rows);
    allocate(alpha_dev, n_rows, true);
    allocate(delta_alpha_dev, n_ws, true);
    allocate(kernel_dev, n_ws * n_rows);
    allocate(return_buff_dev, 2);

    cublas_handle = handle.getImpl().getCublasHandle();

    updateDevice(x_dev, x_host, n_rows * n_cols, stream);
    updateDevice(ws_idx_dev, ws_idx_host, n_ws, stream);
    updateDevice(y_dev, y_host, n_rows, stream);
    updateDevice(f_dev, f_host, n_rows, stream);
    updateDevice(kernel_dev, kernel_host, n_ws * n_rows, stream);

    kernel = new Matrix::GramMatrixBase<math_t>(cublas_handle);
  }

  void FreeResultBuffers() {
    if (dual_coefs_d) CUDA_CHECK(cudaFree(dual_coefs_d));
    if (idx_d) CUDA_CHECK(cudaFree(idx_d));
    if (x_support_d) CUDA_CHECK(cudaFree(x_support_d));
    dual_coefs_d = nullptr;
    idx_d = nullptr;
    x_support_d = nullptr;
  }
  void TearDown() override {
    delete kernel;
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(y_dev));
    CUDA_CHECK(cudaFree(y_pred));
    CUDA_CHECK(cudaFree(f_dev));
    CUDA_CHECK(cudaFree(ws_idx_dev));
    CUDA_CHECK(cudaFree(alpha_dev));
    CUDA_CHECK(cudaFree(delta_alpha_dev));
    CUDA_CHECK(cudaFree(kernel_dev));
    CUDA_CHECK(cudaFree(return_buff_dev));
    FreeResultBuffers();
  }

 public:
  void blockSolveTest() {
    SmoBlockSolve<math_t, 1024><<<1, n_ws, 0, stream>>>(
      y_dev, n_rows, alpha_dev, n_ws, delta_alpha_dev, f_dev, kernel_dev,
      ws_idx_dev, 1.0, 1e-3, return_buff_dev);
    CUDA_CHECK(cudaPeekAtLastError());

    math_t return_buff[2];
    updateHost(return_buff, return_buff_dev, 2, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_FLOAT_EQ(return_buff[0], 2.0f) << return_buff[0];
    EXPECT_LT(return_buff[1], 100) << return_buff[1];

    // check results won't work, because it expects that GetResults was called
    math_t *delta_alpha_calc;
    allocate(delta_alpha_calc, n_rows);
    LinAlg::binaryOp(
      delta_alpha_calc, y_dev, alpha_dev, n_rows,
      [] __device__(math_t a, math_t b) { return a * b; }, stream);
    devArrMatch(delta_alpha_dev, delta_alpha_calc, n_rows,
                CompareApprox<math_t>(1e-6));
    CUDA_CHECK(cudaFree(delta_alpha_calc));

    math_t alpha_expected[] = {0.6f, 0, 1, 1, 0, 0.6f};
    //for C=10: {0.25f, 0, 2.25f, 3.75f, 0, 1.75f};
    devArrMatch(alpha_expected, alpha_dev, n_rows, CompareApprox<math_t>(1e-6));

    math_t host_alpha[6];
    updateHost(host_alpha, alpha_dev, n_rows, stream);

    math_t w[] = {0, 0};
    math_t ay = 0;
    for (int i = 0; i < n_rows; i++) {
      EXPECT_FLOAT_EQ(host_alpha[i], alpha_expected[i]) << "alpha " << i;
      w[0] += x_host[i] * host_alpha[i] * y_host[i];
      w[1] += x_host[i + n_rows] * host_alpha[i] * y_host[i];
      ay += host_alpha[i] * y_host[i];
    }
    EXPECT_FLOAT_EQ(ay, 0.0);
    EXPECT_FLOAT_EQ(w[0], -0.4);
    EXPECT_FLOAT_EQ(w[1], 1.2);
    // for C=10
    //EXPECT_FLOAT_EQ(w[0], -2.0);
    //EXPECT_FLOAT_EQ(w[1],  2.0);
  }

 protected:
  void checkResults(int n_coefs_exp, math_t *dual_coefs_exp, math_t b_exp,
                    math_t *w_exp, math_t *x_support_exp, int *idx_exp,
                    int n_coefs, int n_cols, math_t *dual_coefs_d = nullptr,
                    math_t b = 0, math_t *x_support_d = nullptr,
                    int *idx_d = nullptr, math_t b_tol = 0.001,
                    math_t cs_tol = 0.99999, int n_sv_diff = -1,
                    math_t ay_tol = 1e-5f) {
    if (n_sv_diff == -1) {
      n_sv_diff = n_coefs_exp * 0.01;
      if (n_coefs_exp > 10 && n_sv_diff < 3) n_sv_diff = 3;
    }
    EXPECT_LE(abs(n_coefs - n_coefs_exp), n_sv_diff);
    if (dual_coefs_exp) {
      EXPECT_TRUE(devArrMatchHost(dual_coefs_exp, dual_coefs_d, n_coefs,
                                  CompareApprox<math_t>(1e-3f)));
    }
    math_t *dual_coefs_host = new math_t[n_coefs];
    updateHost(dual_coefs_host, dual_coefs_d, n_coefs, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    math_t ay = 0;
    for (int i = 0; i < n_coefs; i++) {
      ay += dual_coefs_host[i];
    }
    // Test if \sum \alpha_i y_i = 0
    EXPECT_LT(abs(ay), ay_tol);

    if (x_support_exp) {
      EXPECT_TRUE(devArrMatchHost(x_support_exp, x_support_d, n_coefs * n_cols,
                                  CompareApprox<math_t>(1e-6f)));
    }

    if (idx_exp) {
      EXPECT_TRUE(devArrMatchHost(idx_exp, idx_d, n_coefs, Compare<int>()));
    }

    math_t *x_support_host = new math_t[n_coefs * n_cols];
    updateHost(x_support_host, x_support_d, n_coefs * n_cols, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (w_exp) {
      std::vector<math_t> w(n_cols, 0);
      for (int i = 0; i < n_coefs; i++) {
        for (int j = 0; j < n_cols; j++)
          w[j] += x_support_host[i + n_coefs * j] * dual_coefs_host[i];
      }
      // Calculate the cosine similarity between w and w_exp
      math_t abs_w = 0;
      math_t abs_w_exp = 0;
      math_t cs = 0;
      for (int i = 0; i < n_cols; i++) {
        abs_w += w[i] * w[i];
        abs_w_exp += w_exp[i] * w_exp[i];
        cs += w[i] * w_exp[i];
      }
      cs /= sqrt(abs_w * abs_w_exp);
      EXPECT_GT(cs, cs_tol);
    }

    EXPECT_LT(abs(b - b_exp), b_tol);

    delete[] dual_coefs_host;
    delete[] x_support_host;
  }

  void checkResults(svmModel<math_t> model, smoOutput<math_t> expected,
                    svmTol<math_t> tol = svmTol<math_t>{0.001, 0.99999, -1}) {
    math_t *dcoef_exp =
      expected.dual_coefs.size() > 0 ? expected.dual_coefs.data() : nullptr;
    math_t *w_exp = expected.w.size() > 0 ? expected.w.data() : nullptr;
    math_t *x_sv_exp =
      expected.x_support.size() > 0 ? expected.x_support.data() : nullptr;
    int *idx_exp = expected.idx.size() > 0 ? expected.idx.data() : nullptr;

    checkResults(expected.n_support, dcoef_exp, expected.b, w_exp, x_sv_exp,
                 idx_exp, model.n_support, model.n_cols, model.dual_coefs,
                 model.b, model.x_support, model.support_idx, tol.b, tol.cs,
                 tol.n_sv);
  }
  cumlHandle handle;
  cudaStream_t stream;
  Matrix::GramMatrixBase<math_t> *kernel;
  int n_rows = 6;
  const int n_cols = 2;
  int n_ws = 6;

  math_t *x_dev;
  int *ws_idx_dev;
  math_t *y_dev;
  math_t *y_pred;
  math_t *f_dev;
  math_t *alpha_dev;
  math_t *delta_alpha_dev;
  math_t *kernel_dev;
  math_t *return_buff_dev;

  math_t x_host[12] = {1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3};
  int ws_idx_host[6] = {0, 1, 2, 3, 4, 5};
  math_t y_host[6] = {-1, -1, 1, -1, 1, 1};

  math_t f_host[6] = {1, 1, -1, 1, -1, -1};

  math_t kernel_host[36] = {2, 3, 3, 4, 4,  5,  3, 5, 4, 6,  5,  7,
                            3, 4, 5, 6, 7,  8,  4, 6, 6, 8,  8,  10,
                            4, 5, 7, 8, 10, 11, 5, 7, 8, 10, 11, 13};
  cublasHandle_t cublas_handle;

  math_t *dual_coefs_d = nullptr;
  int n_coefs;
  int *idx_d = nullptr;
  math_t *x_support_d = nullptr;
  math_t b;
};

TYPED_TEST_CASE(SmoSolverTest, FloatTypes);

TYPED_TEST(SmoSolverTest, BlockSolveTest) { this->blockSolveTest(); }

std::string kernelName(KernelParams k) {
  std::vector<std::string> names{"linear", "poly", "rbf", "tanh"};
  return names[k.kernel];
}

template <typename math_t>
std::ostream &operator<<(std::ostream &os, const smoInput<math_t> &b) {
  os << kernelName(b.kernel_params) << ", C=" << b.C << ", tol=" << b.tol;
  return os;
}

TYPED_TEST(SmoSolverTest, SmoSolveTest) {
  std::vector<std::pair<smoInput<TypeParam>, smoOutput<TypeParam>>> data{
    {smoInput<TypeParam>{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 1},
     smoOutput<TypeParam>{4,                         // n_sv
                          {-0.6, 1, -1, 0.6},        // dual_coefs
                          -1.8,                      // b
                          {-0.4, 1.2},               // w
                          {1, 1, 2, 2, 1, 2, 2, 3},  // x_support
                          {0, 2, 3, 5}}},            // support idx
    {smoInput<TypeParam>{10, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 1},
     smoOutput<TypeParam>{3, {-2, 4, -2, 0, 0}, -1.0, {-2, 2}, {}, {}}},
    {smoInput<TypeParam>{1, 1e-6, KernelParams{POLYNOMIAL, 3, 1, 1}, 100, 1},
     smoOutput<TypeParam>{3,
                          {-0.02556136, 0.03979708, -0.01423571},
                          -1.07739149,
                          {},
                          {1, 1, 2, 1, 2, 2},
                          {0, 2, 3}}}};

  for (auto d : data) {
    auto p = d.first;
    auto exp = d.second;
    SCOPED_TRACE(p);
    GramMatrixBase<TypeParam> *kernel = KernelFactory<TypeParam>::create(
      p.kernel_params, this->handle.getImpl().getCublasHandle());
    SmoSolver<TypeParam> smo(this->handle.getImpl(), p.C, p.tol, kernel);
    svmModel<TypeParam> model{0,       this->n_cols, 0, nullptr,
                              nullptr, nullptr,      0, nullptr};
    smo.Solve(this->x_dev, this->n_rows, this->n_cols, this->y_dev,
              &model.dual_coefs, &model.n_support, &model.x_support,
              &model.support_idx, &model.b, p.max_iter, p.max_inner_iter);
    this->checkResults(model, exp);
    svmFreeBuffers(this->handle, model);
  }
}

TYPED_TEST(SmoSolverTest, SvcTest) {
  std::vector<std::pair<svcInput<TypeParam>, smoOutput<TypeParam>>> data{
    {svcInput<TypeParam>{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, this->n_rows,
                         this->n_cols, this->x_dev, this->y_dev, true},
     smoOutput<TypeParam>{4,
                          {-0.6, 1, -1, 0.6},
                          -1.8f,
                          {-0.4, 1.2},
                          {1, 1, 2, 2, 1, 2, 2, 3},
                          {0, 2, 3, 5}}},
    {svcInput<TypeParam>{1, 1e-6, KernelParams{POLYNOMIAL, 3, 1, 0},
                         this->n_rows, this->n_cols, this->x_dev, this->y_dev,
                         true},
     smoOutput<TypeParam>{3,
                          {-0.03900895, 0.05904058, -0.02003163},
                          -0.99999959,
                          {},
                          {1, 1, 2, 1, 2, 2},
                          {0, 2, 3}}},
    {svcInput<TypeParam>{10, 1e-6, KernelParams{TANH, 3, 0.3, 1.0},
                         this->n_rows, this->n_cols, this->x_dev, this->y_dev,
                         false},
     smoOutput<TypeParam>{6,
                          {-10., -10., 10., -10., 10., 10.},
                          -0.3927505,
                          {},
                          {1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3},
                          {0, 1, 2, 3, 4, 5}}},
    {svcInput<TypeParam>{1, 1.0e-6, KernelParams{RBF, 0, 0.15, 0}, this->n_rows,
                         this->n_cols, this->x_dev, this->y_dev, true},
     smoOutput<TypeParam>{6,
                          {-1., -1, 1., -1., 1, 1.},
                          0,
                          {},
                          {1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3},
                          {0, 1, 2, 3, 4, 5}}}};

  for (auto d : data) {
    auto p = d.first;
    auto exp = d.second;
    SCOPED_TRACE(kernelName(p.kernel_params));
    SVC<TypeParam> svc(this->handle, p.C, p.tol, p.kernel_params);
    svc.fit(p.x_dev, p.n_rows, p.n_cols, p.y_dev);
    this->checkResults(svc.model, exp);
    device_buffer<TypeParam> y_pred(this->handle.getDeviceAllocator(),
                                    this->stream, p.n_rows);
    if (p.predict) {
      svc.predict(p.x_dev, p.n_rows, p.n_cols, y_pred.data());
      EXPECT_TRUE(devArrMatch(this->y_dev, y_pred.data(), p.n_rows,
                              CompareApprox<TypeParam>(1e-6f)));
    }
  }
}

struct blobInput {
  double C;
  double tol;
  KernelParams kernel_params;
  int n_rows;
  int n_cols;
};

std::ostream &operator<<(std::ostream &os, const blobInput &b) {
  os << kernelName(b.kernel_params) << " " << b.n_rows << "x" << b.n_cols;
  return os;
}

// until there is progress with Issue #935
template <typename inType, typename outType>
__global__ void cast(outType *out, int n, inType *in) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) out[tid] = in[tid];
}

// To have the same input data for both single and double precision,
// we generate the blobs in single precision only, and cast to dp if needed.
template <typename math_t>
void make_blobs(math_t *x, math_t *y, int n_rows, int n_cols, int n_cluster,
                std::shared_ptr<MLCommon::deviceAllocator> &allocator,
                cublasHandle_t cublas_h, cudaStream_t stream,
                float *centers = nullptr) {
  device_buffer<float> x_float(allocator, stream, n_rows * n_cols);
  device_buffer<int> y_int(allocator, stream, n_rows);

  Random::make_blobs(x_float.data(), y_int.data(), n_rows, n_cols, n_cluster,
                     allocator, stream, centers, (float *)nullptr, 1.0f, true,
                     -2.0f, 2.0f, 0);
  int TPB = 256;
  if (std::is_same<float, math_t>::value) {
    LinAlg::transpose(x_float.data(), (float *)x, n_cols, n_rows, cublas_h,
                      stream);
  } else {
    device_buffer<math_t> x2(allocator, stream, n_rows * n_cols);
    cast<<<MLCommon::ceildiv(n_rows * n_cols, TPB), TPB, 0, stream>>>(
      x2.data(), n_rows * n_cols, x_float.data());
    LinAlg::transpose(x2.data(), x, n_cols, n_rows, cublas_h, stream);
    CUDA_CHECK(cudaPeekAtLastError());
  }
  cast<<<MLCommon::ceildiv(n_rows, TPB), TPB, 0, stream>>>(y, n_rows,
                                                           y_int.data());
  CUDA_CHECK(cudaPeekAtLastError());
}

TYPED_TEST(SmoSolverTest, Blobs) {
  std::vector<std::pair<blobInput, smoOutput<TypeParam>>> data{
    {blobInput{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 1},
     smoOutput<TypeParam>{94, {}, 0.5694, {-0.59181675}, {}, {}}},
    {blobInput{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 2, 100},
     smoOutput<TypeParam>{2, {0.00377128, -0.00377128}, 0.0392874, {}, {}, {}}},
    {blobInput{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 100},
     smoOutput<TypeParam>{
       21,
       {},
       -0.027839,
       {-0.01694475, -0.0058431,  -0.00540597, 0.03296302,  0.00389359,
        0.02106245,  0.0039061,   -0.01332385, 0.00660107,  0.02471894,
        -0.00449132, 0.0036424,   0.01026903,  0.02074523,  -0.01541529,
        0.00959903,  -0.01748434, 0.00014968,  0.00110548,  -0.00965647,
        0.00798169,  0.00621375,  0.01358159,  0.00528406,  0.01153438,
        -0.01368673, -0.03216547, -0.00200592, 0.0120133,   0.00176928,
        0.00642773,  0.01619464,  -0.00064005, 0.01400618,  0.01727131,
        -0.02331973, 0.00197682,  -0.00378402, 0.00612855,  -0.02184831,
        -0.00363246, -0.0127956,  0.02767534,  -0.00553812, -0.02225143,
        0.02767332,  0.00623573,  -0.02094788, -0.02012747, 0.01102353,
        0.00080433,  -0.0311776,  -0.01562613, -0.02279347, 0.02422357,
        -0.00057321, -0.02881205, 0.00038414,  0.00919522,  0.00644822,
        -0.02633642, -0.00157511, -0.00627405, -0.00604624, 0.0104084,
        0.00264318,  0.00155375,  -0.00997641, 0.00110094,  0.01547085,
        -0.00342524, -0.00329727, 0.00836803,  -0.00882695, -0.0022986,
        0.0046427,   -0.0214064,  -0.01128186, 0.00787328,  -0.02030728,
        -0.00835369, 0.0141688,   -0.01291866, 0.0155667,   -0.01748242,
        -0.01181509, -0.00194211, -0.01517817, 0.01449254,  -0.00437902,
        0.00218727,  0.01684742,  -0.00274239, 0.01743845,  0.02874734,
        0.00412968,  0.00070507,  -0.00334403, 0.02057458,  -0.01662802},
       {},
       {}}},
    {blobInput{1, 1e-3, KernelParams{LINEAR, 3, 1, 0}, 1000, 100},
     smoOutput<TypeParam>{
       34,
       {},
       0.0681913,
       {-2.74771248e-02, -1.93489987e-03, -5.10868053e-03, 3.62352833e-02,
        -4.95548876e-03, 2.12629716e-02,  7.24245748e-03,  -9.81239809e-03,
        1.04116603e-02,  1.98374788e-02,  -8.85246492e-03, 3.87150029e-03,
        1.33185355e-02,  2.48162036e-02,  -2.19338463e-02, -5.84521038e-03,
        -1.52917105e-02, -2.02794426e-03, 1.02024677e-02,  -9.34250325e-03,
        5.39213720e-03,  1.04247828e-02,  2.11718472e-02,  5.84453748e-05,
        9.20193459e-03,  -2.68006157e-02, -3.37565162e-02, 7.36789225e-03,
        4.61224812e-03,  -1.16150969e-03, 5.51388041e-03,  2.28353876e-02,
        1.01278434e-02,  8.88320327e-03,  2.84587444e-02,  -2.69580067e-02,
        1.28087948e-02,  3.91011504e-03,  5.37012676e-03,  -3.16363924e-02,
        -3.35284170e-03, -6.50114513e-03, 3.20651614e-02,  -1.16159910e-02,
        -1.38094305e-02, 3.07976442e-02,  1.63332093e-02,  -2.74646081e-02,
        -3.29649271e-02, 1.47660371e-02,  6.09094255e-03,  -3.94666490e-02,
        -1.23632624e-02, -3.69395768e-02, 2.86938112e-02,  -6.44606846e-03,
        -2.28087983e-02, 7.52382321e-03,  1.09257772e-02,  1.74754686e-03,
        -2.18251777e-02, 2.26055336e-03,  -7.57926813e-03, -8.66892829e-03,
        5.56296376e-03,  1.87841417e-03,  -1.16321532e-02, -3.61208310e-03,
        1.23176737e-02,  2.37739726e-02,  -1.27088417e-02, -5.08065174e-03,
        2.75792244e-03,  -6.36667336e-03, -4.09090428e-03, 5.91545951e-03,
        -1.50748294e-02, -1.32456566e-02, 5.02961559e-03,  -2.07146521e-02,
        1.53120474e-03,  1.16797660e-02,  -1.32767277e-02, 1.50207104e-02,
        -2.39799601e-02, -7.01546552e-03, 8.88361434e-04,  -1.60787453e-02,
        1.89898754e-02,  -4.73620907e-03, -4.03131155e-03, 2.93305320e-02,
        1.33827967e-03,  1.94703514e-02,  2.87304765e-02,  9.69693810e-03,
        5.91821346e-05,  -9.06411618e-03, 3.03503309e-02,  -1.82080580e-02},
       {},
       {}}},
    {blobInput{1, 1e-3, KernelParams{LINEAR, 3, 0.001, 0}, 100, 1000},
     smoOutput<TypeParam>{41, {}, -0.0077539393, {}, {}, {}}}};

  std::vector<svmTol<TypeParam>> tolerance{
    svmTol<TypeParam>{1e-3, 0.99999, -1}, svmTol<TypeParam>{1e-3, 0.99999, -1},
    svmTol<TypeParam>{1e-3, 0.99999, -1}, svmTol<TypeParam>{0.05, 0.95, 10},
    svmTol<TypeParam>{0.05, 0.95, 8}};
  int i = 0;
  auto allocator = this->handle.getDeviceAllocator();
  for (auto d : data) {
    auto p = d.first;
    SCOPED_TRACE(p);
    device_buffer<TypeParam> x(allocator, this->stream, p.n_rows * p.n_cols);
    device_buffer<TypeParam> y(allocator, this->stream, p.n_rows);
    make_blobs(x.data(), y.data(), p.n_rows, p.n_cols, 2, allocator,
               this->handle.getImpl().getCublasHandle(), this->stream);
    SVC<TypeParam> svc(this->handle, p.C, p.tol, p.kernel_params);
    svc.fit(x.data(), p.n_rows, p.n_cols, y.data());
    //std::cout << p << ": " << svc.model.n_support << " " << svc.model.b << "\n";
    auto exp = d.second;
    svmTol<TypeParam> tol = tolerance[i];
    i++;
    this->checkResults(svc.model, exp, tol);
    device_buffer<TypeParam> y_pred(this->handle.getDeviceAllocator(),
                                    this->stream, p.n_rows);
    svc.predict(x.data(), p.n_rows, p.n_cols, y_pred.data());
  }
}

struct is_same_functor {
  template <typename Tuple>
  __host__ __device__ int operator()(Tuple t) {
    return thrust::get<0>(t) == thrust::get<1>(t);
  }
};

TYPED_TEST(SmoSolverTest, BlobPredict) {
  // Pair.second is the expected accuracy. It might change if the Rng changes.
  std::vector<std::pair<blobInput, TypeParam>> data{
    {blobInput{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 200, 10}, 98},
    {blobInput{1, 0.001, KernelParams{POLYNOMIAL, 3, 1, 0}, 200, 10}, 98},
    {blobInput{1, 0.001, KernelParams{RBF, 3, 1, 0}, 200, 2}, 98},
    {blobInput{1, 0.009, KernelParams{TANH, 3, 0.1, 0}, 200, 10}, 98}};

  // This should be larger then N_PRED_BATCH in svcPredict
  const int n_pred = 5000;

  auto allocator = this->handle.getDeviceAllocator();

  for (auto d : data) {
    auto p = d.first;
    SCOPED_TRACE(p);
    // explicit centers for the blobs
    device_buffer<float> centers(allocator, this->stream, 2 * p.n_cols);
    thrust::device_ptr<float> thrust_ptr(centers.data());
    thrust::fill(thrust::cuda::par.on(this->stream), thrust_ptr,
                 thrust_ptr + p.n_cols, -5.0f);
    thrust::fill(thrust::cuda::par.on(this->stream), thrust_ptr + p.n_cols,
                 thrust_ptr + 2 * p.n_cols, +5.0f);

    device_buffer<TypeParam> x(allocator, this->stream, p.n_rows * p.n_cols);
    device_buffer<TypeParam> y(allocator, this->stream, p.n_rows);
    device_buffer<TypeParam> x_pred(allocator, this->stream, n_pred * p.n_cols);
    device_buffer<TypeParam> y_pred(allocator, this->stream, n_pred);

    make_blobs(x.data(), y.data(), p.n_rows, p.n_cols, 2, allocator,
               this->handle.getImpl().getCublasHandle(), this->stream,
               centers.data());
    SVC<TypeParam> svc(this->handle, p.C, p.tol, p.kernel_params);
    svc.fit(x.data(), p.n_rows, p.n_cols, y.data());

    // Create a different dataset for prediction
    make_blobs(x_pred.data(), y_pred.data(), n_pred, p.n_cols, 2, allocator,
               this->handle.getImpl().getCublasHandle(), this->stream,
               centers.data());
    device_buffer<TypeParam> y_pred2(this->handle.getDeviceAllocator(),
                                     this->stream, n_pred);
    svc.predict(x_pred.data(), n_pred, p.n_cols, y_pred2.data());

    // Count the number of correct predictions
    device_buffer<int> is_correct(this->handle.getDeviceAllocator(),
                                  this->stream, n_pred);
    thrust::device_ptr<TypeParam> ptr1(y_pred.data());
    thrust::device_ptr<TypeParam> ptr2(y_pred2.data());
    thrust::device_ptr<int> ptr3(is_correct.data());
    auto first = thrust::make_zip_iterator(thrust::make_tuple(ptr1, ptr2));
    auto last = thrust::make_zip_iterator(
      thrust::make_tuple(ptr1 + n_pred, ptr2 + n_pred));
    thrust::transform(thrust::cuda::par.on(this->stream), first, last, ptr3,
                      is_same_functor());
    int n_correct =
      thrust::reduce(thrust::cuda::par.on(this->stream), ptr3, ptr3 + n_pred);

    TypeParam accuracy = 100 * n_correct / n_pred;
    TypeParam accuracy_exp = d.second;
    EXPECT_GE(accuracy, accuracy_exp);
  }
}

TYPED_TEST(SmoSolverTest, MemoryLeak) {
  std::vector<std::pair<blobInput, smoOutput<TypeParam>>> data{
    {blobInput{1, 0.001, KernelParams{LINEAR, 3, 0.01, 0}, 1000, 1000},
     smoOutput<TypeParam>{34, {}, 0.0681913, {}, {}, {}}}};
  size_t free1, total, free2;
  CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  auto allocator = this->handle.getDeviceAllocator();
  for (auto d : data) {
    auto p = d.first;
    SCOPED_TRACE(p);

    device_buffer<TypeParam> x(allocator, this->stream, p.n_rows * p.n_cols);
    device_buffer<TypeParam> y(allocator, this->stream, p.n_rows);
    make_blobs(x.data(), y.data(), p.n_rows, p.n_cols, 2, allocator,
               this->handle.getImpl().getCublasHandle(), this->stream);

    SVC<TypeParam> svc(this->handle, p.C, p.tol, p.kernel_params);
    svc.fit(x.data(), p.n_rows, p.n_cols, y.data());
    device_buffer<TypeParam> y_pred(this->handle.getDeviceAllocator(),
                                    this->stream, p.n_rows);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));
    CUDA_CHECK(cudaMemGetInfo(&free2, &total));
    float delta = (free1 - free2);
    EXPECT_GT(delta, p.n_rows * p.n_cols * 4);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));
    svc.predict(x.data(), p.n_rows, p.n_cols, y_pred.data());
  }
  CUDA_CHECK(cudaMemGetInfo(&free2, &total));
  float delta = (free1 - free2);
  EXPECT_EQ(delta, 0);
}

};  // namespace SVM
};  // end namespace ML
