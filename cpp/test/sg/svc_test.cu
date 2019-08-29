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
#include <cub/cub.cuh>
#include <iostream>
#include <string>
#include <vector>
#include "common/cumlHandle.hpp"
#include "gram/grammatrix.h"
#include "gram/kernelmatrices.h"
#include "linalg/binary_op.h"
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
using namespace GramMatrix;

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
  void ApplyNonlin(GramMatrix::KernelParams params) {
    switch (params.kernel) {
      case GramMatrix::LINEAR:
        break;
      case GramMatrix::POLYNOMIAL:
        for (int z = 0; z < n_rows * n_ws; z++) {
          math_t val = params.gamma * tile_host_expected[z] + params.coef0;
          tile_host_expected[z] = pow(val, params.degree);
        }
        break;
      case GramMatrix::TANH:
        for (int z = 0; z < n_rows * n_ws; z++) {
          math_t val = params.gamma * tile_host_expected[z] + params.coef0;
          tile_host_expected[z] = tanh(val);
        }
        break;
      case GramMatrix::RBF:
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
  std::vector<GramMatrix::KernelParams> param_vec{
    GramMatrix::KernelParams{GramMatrix::LINEAR, 3, 1, 0},
    GramMatrix::KernelParams{GramMatrix::POLYNOMIAL, 2, 1.3, 1},
    GramMatrix::KernelParams{GramMatrix::TANH, 2, 0.5, 2.4},
    GramMatrix::KernelParams{GramMatrix::RBF, 2, 0.5, 0}};
  for (auto params : param_vec) {
    GramMatrix::GramMatrixBase<TypeParam> *kernel =
      GramMatrix::KernelFactory<TypeParam>::create(
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

    kernel = new GramMatrix::GramMatrixBase<math_t>(cublas_handle);
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
                    int *idx_d = nullptr, math_t epsilon = 0.001) {
    ASSERT_LE(n_coefs, n_coefs_exp);
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
    EXPECT_LT(abs(ay), 1.0e-6f);

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
      for (int i = 0; i < n_cols; i++)
        EXPECT_LT(abs(w[i] - w_exp[i]), epsilon) << "@" << i;
    }

    EXPECT_LT(abs(b - b_exp), epsilon);

    delete[] dual_coefs_host;
    delete[] x_support_host;
  }

  void checkResults(svmModel<math_t> model, smoOutput<math_t> expected) {
    math_t *dcoef_exp =
      expected.dual_coefs.size() > 0 ? expected.dual_coefs.data() : nullptr;
    math_t *w_exp = expected.w.size() > 0 ? expected.w.data() : nullptr;
    math_t *x_sv_exp =
      expected.x_support.size() > 0 ? expected.x_support.data() : nullptr;
    int *idx_exp = expected.idx.size() > 0 ? expected.idx.data() : nullptr;

    checkResults(expected.n_support, dcoef_exp, expected.b, w_exp, x_sv_exp,
                 idx_exp, model.n_support, model.n_cols, model.dual_coefs,
                 model.b, model.x_support, model.support_idx);
  }
  cumlHandle handle;
  cudaStream_t stream;
  GramMatrix::GramMatrixBase<math_t> *kernel;
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

TYPED_TEST(SmoSolverTest, SmoSolveTest) {
  std::vector<smoInput<TypeParam>> param_vec{
    {1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 1},
    {10, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 1},
    {1, 1e-6, KernelParams{POLYNOMIAL, 3, 1, 1}, 100, 1},

  };

  std::vector<smoOutput<TypeParam>> out_vec{
    {4,
     {-0.6, 1, -1, 0.6},        // dual_coefs
     -1.8,                      // b
     {-0.4, 1.2},               // w
     {1, 1, 2, 2, 1, 2, 2, 3},  //x_support
     {0, 2, 3, 5}},             //idx
    {5, {-2, 4, -2, 0, 0}, -1.0, {-2, 2}, {}, {}},
    {3,
     {-0.02556136, 0.03979708, -0.01423571},
     -1.07739149,
     {},
     {1, 1, 2, 1, 2, 2},
     {0, 2, 3}}

  };

  for (int i = 0; i < param_vec.size(); i++) {
    auto p = param_vec[i];
    auto exp = out_vec[i];
    SCOPED_TRACE(kernelName(p.kernel_params));
    GramMatrixBase<TypeParam> *kernel = KernelFactory<TypeParam>::create(
      p.kernel_params, this->handle.getImpl().getCublasHandle());
    SmoSolver<TypeParam> smo(this->handle.getImpl(), p.C, p.tol, kernel);
    svmModel<TypeParam> model{0, 0, 0, nullptr, nullptr, nullptr, 0, nullptr};
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

TYPED_TEST(SmoSolverTest, Blobs) {
  std::vector<std::pair<blobInput, smoOutput<TypeParam>>> data{
    {blobInput{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 1},
     smoOutput<TypeParam>{98, {}, 5.33624, {}, {}, {}}},
    {blobInput{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 2, 100},
     smoOutput<TypeParam>{2, {}, 0.941554, {}, {}, {}}}};
  //  {blobInput{1, 0.001, KernelParams{LINEAR, 3, 1, 0}, 100, 100},
  //   smoOutput<TypeParam>{68, {}, 3.571, {}, {}, {}}}};
  //{blobInput{1, 0.001, KernelParams{LINEAR, 3, 0.01, 0}, 1000, 100},
  // smoOutput<TypeParam>{844, {}, -11.7999, {}, {}, {}}},
  //{blobInput{1, 0.001, KernelParams{LINEAR, 3, 0.001, 0}, 100, 10000},
  // smoOutput<TypeParam>{100, {}, 1.27648, {}, {}, {}}}};
  // The last three (disabled) tests are sensitive to the precision that we use.
  // TODO: confirm that this is correct behavior, and enable these tests

  auto allocator = this->handle.getDeviceAllocator();
  for (auto d : data) {
    auto p = d.first;
    SCOPED_TRACE(p);
    device_buffer<float> x_float(allocator, this->stream, p.n_rows * p.n_cols);
    device_buffer<TypeParam> x2(allocator, this->stream);
    device_buffer<int> y_int(allocator, this->stream, p.n_rows);
    device_buffer<TypeParam> y(allocator, this->stream, p.n_rows);

    // To have the same input data for both single and double precision,
    // we generate the blobs in single precision only, and later cast to dp
    // if needed.
    Random::make_blobs(x_float.data(), y_int.data(), p.n_rows, p.n_cols, 2,
                       allocator, this->stream);
    int TPB = 256;
    TypeParam *x;
    if (std::is_same<float, TypeParam>::value) {
      x = (TypeParam *)x_float.data();
    } else {
      x2.resize(p.n_rows * p.n_cols, this->stream);
      cast<<<MLCommon::ceildiv(p.n_rows, TPB), TPB, 0, this->stream>>>(
        x2.data(), p.n_rows * p.n_cols, x_float.data());
      CUDA_CHECK(cudaPeekAtLastError());
      x = x2.data();
    }
    cast<<<MLCommon::ceildiv(p.n_rows, TPB), TPB, 0, this->stream>>>(
      y.data(), p.n_rows, y_int.data());
    CUDA_CHECK(cudaPeekAtLastError());

    SVC<TypeParam> svc(this->handle, p.C, p.tol, p.kernel_params, 200, 100,
                       false);
    svc.fit(x, p.n_rows, p.n_cols, y.data());
    std::cout << p << ": " << svc.model.n_support << " " << svc.model.b << "\n";
    auto exp = d.second;
    this->checkResults(svc.model, exp);
    device_buffer<TypeParam> y_pred(this->handle.getDeviceAllocator(),
                                    this->stream, p.n_rows);

    svc.predict(x, p.n_rows, p.n_cols, y_pred.data());
  }
}

};  // namespace SVM
};  // end namespace ML
