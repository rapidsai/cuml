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
#include "common/cumlHandle.hpp"
#include "gram/grammatrix.h"
#include "gram/kernelmatrices.h"
#include "svm/smosolver.h"
#include "svm/svc.h"
#include "svm/workingset.h"
#include "test_utils.h"

namespace ML {
namespace SVM {
using namespace MLCommon;
using namespace GramMatrix;

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
  WorkingSet<float> *ws;

  float f_host[10] = {1, 3, 10, 4, 2, 8, 6, 5, 9, 7};
  float *f_dev;

  float y_host[10] = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  float *y_dev;

  float C = 1.5;

  float alpha_host[10] = {0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5};
  float *alpha_dev;  //   l  l  l/u  l/u    u  u  l/u  l/u  l    l

  int expected_idx[4] = {4, 3, 8, 2};
  int expected_idx2[4] = {8, 2, 4, 9};
};

TEST_F(WorkingSetTest, Init) {
  ws = new WorkingSet<float>(handle.getImpl(), handle.getStream(), 10);
  EXPECT_EQ(ws->GetSize(), 10);
  delete ws;

  ws = new WorkingSet<float>(handle.getImpl(), stream, 100000);
  EXPECT_EQ(ws->GetSize(), 1024);
  delete ws;
}

TEST_F(WorkingSetTest, Select) {
  ws = new WorkingSet<float>(handle.getImpl(), stream, 10, 4);
  EXPECT_EQ(ws->GetSize(), 4);
  ws->SimpleSelect(f_dev, alpha_dev, y_dev, C);
  ASSERT_TRUE(devArrMatchHost(expected_idx, ws->GetIndices(), ws->GetSize(),
                              Compare<int>()));

  ws->Select(f_dev, alpha_dev, y_dev, C);
  ASSERT_TRUE(devArrMatchHost(expected_idx, ws->GetIndices(), ws->GetSize(),
                              Compare<int>()));
  ws->Select(f_dev, alpha_dev, y_dev, C);

  ASSERT_TRUE(devArrMatchHost(expected_idx2, ws->GetIndices(), ws->GetSize(),
                              Compare<int>()));
  delete ws;
}
/*
TEST_F(WorkingSetTest, Priority) {
  ws = new WorkingSet<float>(handle.getImpl(), stream, 10, 8);
  ASSERT_EQ(ws->GetSize(), 8);
  ws->FIFO_strategy = false;
  ws->Select(f_dev, alpha_dev, y_dev, C);
  int expected_idx[] = {4, 3, 7, 6, 2, 8, 9, 1};
  //ws->Select(f_dev, alpha_dev, y_dev, C);
  ASSERT_TRUE(
    devArrMatchHost(expected_idx, ws->GetIndices(), ws->GetSize(), Compare<int>())
  );
  delete ws;
}
*/

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

  cumlHandle handle;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;

  int n_rows = 4;
  int n_cols = 2;
  int n_ws = 4;

  float *x_dev;
  int *ws_idx_dev;

  float x_host[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int ws_idx_host[4] = {0, 1, 2, 3};
  float tile_host_expected[16] = {26, 32, 38, 44, 32, 40, 48, 56,
                                  38, 48, 58, 68, 44, 56, 68, 80};
};

TEST_F(KernelCacheTest, EvalTestLin) {
  auto lin = new GramMatrix::GramMatrixBase<float>(cublas_handle);
  KernelCache<float> cache(handle.getImpl(), x_dev, n_rows, n_cols, n_ws, lin);
  float *tile_dev = cache.GetTile(ws_idx_dev);
  ASSERT_TRUE(devArrMatchHost(tile_host_expected, tile_dev, n_ws * n_ws,
                              CompareApprox<float>(1e-6f)));
}

TEST_F(KernelCacheTest, EvalTestPoly) {
  float offset = 1.3;
  float gain = 1.0;
  auto nonlin = new GramMatrix::PolynomialKernel<float, int>(2, gain, offset,
                                                             cublas_handle);
  for (int z = 0; z < 16; z++) {
    float val = tile_host_expected[z] + offset;
    tile_host_expected[z] = val * val;
  }
  KernelCache<float> cache(handle.getImpl(), x_dev, n_rows, n_cols, n_ws,
                           nonlin);
  float *tile_dev = cache.GetTile(ws_idx_dev);
  ASSERT_TRUE(devArrMatchHost(tile_host_expected, tile_dev, n_ws * n_ws,
                              CompareApprox<float>(1e-6f)));
}

TEST_F(KernelCacheTest, EvalTestTanh) {
  auto nonlin = new GramMatrix::TanhKernel<float>(0.5, 2.4, cublas_handle);
  for (int z = 0; z < 16; z++)
    tile_host_expected[z] = tanh(tile_host_expected[z] * 0.5 + 2.4);
  KernelCache<float> cache(handle.getImpl(), x_dev, n_rows, n_cols, n_ws,
                           nonlin);
  float *tile_dev = cache.GetTile(ws_idx_dev);
  ASSERT_TRUE(devArrMatchHost(tile_host_expected, tile_dev, n_ws * n_ws,
                              CompareApprox<float>(1e-6f)));
}

TEST_F(KernelCacheTest, EvalTestRBF) {
  auto nonlin = new GramMatrix::RBFKernel<float>(0.5);
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_rows; j++) {
      float d = 0;
      for (int k = 0; k < n_cols; k++) {
        float diff = x_host[i + k * n_rows] - x_host[j + k * n_rows];
        d += diff * diff;
      }
      tile_host_expected[i * n_rows + j] = exp(-0.5 * d);
    }
  }
  KernelCache<float> cache(handle.getImpl(), x_dev, n_rows, n_cols, n_ws,
                           nonlin);
  float *tile_dev = cache.GetTile(ws_idx_dev);
  ASSERT_TRUE(devArrMatchHost(tile_host_expected, tile_dev, n_ws * n_ws,
                              CompareApprox<float>(1e-6f)));
}

TEST_F(KernelCacheTest, EvalTestRBF_Rectangular) {
  float gamma = 0.7;
  auto nonlin = new GramMatrix::RBFKernel<float>(gamma);
  // The exp function is disabled, we just calculate the L2 distance.
  //
  // Instead of a 5x5 tile, we want to calculate a 5x3 tile here.
  // The inputs to the distance function are the vectors x1 and x2.
  //
  // x1 = [ [1, 6],
  //        [2, 7],
  //        [3, 8],
  //        [4, 9],
  //        [5, 10] ];
  // The vectors are stored in column major format, so actually
  float x1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  n_rows = 5;
  //
  n_ws = 3;
  // We set n_ws = 3, this way the working set (x2 input for distance)
  // will be the first three vectors.
  // x2 = [ [1, 6],
  //        [2, 7],
  //        [3, 8] ];
  // KernelCache.CollectRows collects the data to a contiguous memory space
  // (column major), therefore:
  // x2[] = {1, 2, 3, 6, 7, 8};
  //
  // The GetTile function should calculate the 5x3 output matrix. Here is the
  // distance matrix
  //  K(x1,x2)  = [ [ 0,  2, 8],
  //                [ 2,  0, 2],
  //                [ 8,  2, 0],
  //                [18,  8, 2],
  //                [32, 18, 8] ];
  //
  // It is also stored in colum major format, therefore:
  float K[] = {0, 2, 8, 18, 32, 2, 0, 2, 8, 18, 8, 2, 0, 2, 8};

  // The RBF kernel calculates exp for the distance matrix
  for (int i = 0; i < n_rows * n_ws; i++) {
    K[i] = exp(-gamma * K[i]);
  }

  float *x1_dev;
  allocate(x1_dev, n_rows * n_cols);
  updateDevice(x1_dev, x1, n_rows * n_cols, stream);

  KernelCache<float> cache(handle.getImpl(), x1_dev, n_rows, n_cols, n_ws,
                           nonlin, 0);
  float *tile_dev = cache.GetTile(ws_idx_dev);
  //myPrintHostVector("Exp:  ", K, n_rows * n_ws);
  //myPrintDevVector("Actual:", tile_dev, n_rows * n_ws);
  ASSERT_TRUE(
    devArrMatchHost(K, tile_dev, n_rows * n_ws, CompareApprox<float>(1e-6f)));
}

TEST_F(KernelCacheTest, EvalWsOnly) {
  // now check with selecting a subset of the rows
  n_ws = 2;
  ws_idx_host[1] = 3;  // i.e. ws_idx_host[] = {0,3}
  auto lin = new GramMatrix::GramMatrixBase<float>(cublas_handle);
  KernelCache<float> cache(handle.getImpl(), x_dev, n_rows, n_cols, n_ws, lin);
  float *tile_dev = cache.GetTile(ws_idx_dev);
  float tile_host_expected[] = {26, 32, 38, 44, 44, 56, 68, 80};
  ASSERT_TRUE(devArrMatchHost(tile_host_expected, tile_dev, n_ws * n_ws,
                              CompareApprox<float>(1e-6f)));
}

__global__ void init_training_vectors(float *x, int n_rows, int n_cols,
                                      int *ws_idx, int n_ws) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows * n_cols) {
    int i = tid % n_rows;
    int k = tid / n_rows;
    x[tid] = tid;
    if (k == 0) {
      ws_idx[i] = i;
    }
  }
}

TEST(SmoSolverTest, KernelCacheLargeTest) {
  // This was used earlier to cache memory errors
  // since no test is active, it should probably be removed
  cumlHandle handle;
  cudaStream_t stream;
  stream = handle.getImpl().getInternalStream(0);
  cublasHandle_t cublas_handle = handle.getImpl().getCublasHandle();
  int n_rows = 10;
  int n_cols = 700;
  int n_ws = n_rows;

  float *x_dev;
  allocate(x_dev, n_rows * n_cols);
  int *ws_idx_dev;
  allocate(ws_idx_dev, n_ws);

  int TPB = 256;
  init_training_vectors<<<ceildiv(n_rows * n_cols, TPB), TPB, 0, stream>>>(
    x_dev, n_rows, n_cols, ws_idx_dev, n_ws);
  CUDA_CHECK(cudaPeekAtLastError());

  auto lin = new GramMatrix::GramMatrixBase<float>(cublas_handle);
  KernelCache<float> *cache =
    new KernelCache<float>(handle.getImpl(), x_dev, n_rows, n_cols, n_ws, lin);
  float *tile_dev = cache->GetTile(ws_idx_dev);
  float *tile_host = new float[n_rows * n_cols];
  //updateHost(tile_host, tile_dev, n_ws*n_rows, stream);
  //CUDA_CHECK(cudaStreamSynchronize(stream));

  //for (int i=0; i<n_ws*n_ws; i++) {
  //  EXPECT_EQ(tile_host[i], tile_host_expected[i])<< "First tile " << i;
  //}

  delete cache;
  delete[] tile_host;
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(ws_idx_dev));
}

class SmoBlockSolverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
    cublas_handle = handle.getImpl().getCublasHandle();
    auto lin = new GramMatrix::GramMatrixBase<float>(cublas_handle);

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

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(y_dev));
    CUDA_CHECK(cudaFree(f_dev));
    CUDA_CHECK(cudaFree(ws_idx_dev));
    CUDA_CHECK(cudaFree(alpha_dev));
    CUDA_CHECK(cudaFree(delta_alpha_dev));
    CUDA_CHECK(cudaFree(kernel_dev));
    CUDA_CHECK(cudaFree(return_buff_dev));
    delete kernel;
  }

  cumlHandle handle;
  cudaStream_t stream;
  cublasHandle_t cublas_handle;

  GramMatrix::GramMatrixBase<float> *kernel;
  int n_rows = 4;
  int n_cols = 2;
  int n_ws = 4;

  int *ws_idx_dev;
  float *y_dev;
  float *f_dev;
  float *alpha_dev;
  float *delta_alpha_dev;
  float *kernel_dev;
  float *return_buff_dev;

  int ws_idx_host[4] = {0, 1, 2, 3};
  float y_host[4] = {1, 1, -1, -1};
  float f_host[4] = {0.4, 0.3, 0.5, 0.1};
  float kernel_host[16] = {26, 32, 38, 44, 32, 40, 48, 56,
                           38, 48, 58, 68, 44, 56, 68, 80};
};

// test a single iteration of the block solver
TEST_F(SmoBlockSolverTest, SolveSingleTest) {
  SmoBlockSolve<float, 1024><<<1, n_ws, 0, stream>>>(
    y_dev, n_rows, alpha_dev, n_ws, delta_alpha_dev, f_dev, kernel_dev,
    ws_idx_dev, 1.5f, 1e-3f, return_buff_dev, 1);
  CUDA_CHECK(cudaPeekAtLastError());

  float return_buff[2];
  updateHost(return_buff, return_buff_dev, 2, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  EXPECT_FLOAT_EQ(return_buff[0], 0.2f) << return_buff[0];
  EXPECT_EQ(return_buff[1], 1) << "Number of iterations ";

  float host_alpha[4], host_dalpha[4];
  updateHost(host_alpha, alpha_dev, n_rows, stream);
  updateHost(host_dalpha, delta_alpha_dev, n_ws, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < n_ws; i++) {
    EXPECT_FLOAT_EQ(y_host[i] * host_alpha[i], host_dalpha[i])
      << "alpha and delta alpha " << i;
  }
  float alpha_expected[] = {0, 0.1f, 0.1f, 0};
  for (int i = 0; i < n_rows; i++) {
    EXPECT_FLOAT_EQ(host_alpha[i], alpha_expected[i]) << "alpha " << i;
  }

  // now check if updateF works
  SmoSolver<float> smo(handle.getImpl(), 1, 0.001, kernel);

  smo.UpdateF(f_dev, n_rows, delta_alpha_dev, n_ws, kernel_dev);
  updateHost(f_host, f_dev, n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float f_host_expected[] = {-0.2, -0.5, -0.5, -1.1};
  for (int i = 0; i < n_rows; i++) {
    EXPECT_FLOAT_EQ(f_host[i], f_host_expected[i]) << "UpdateF " << i;
  }
}

class SmoSolverTestF : public ::testing::Test {
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

    kernel = new GramMatrix::GramMatrixBase<float>(cublas_handle);
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
    if (dual_coefs_d) CUDA_CHECK(cudaFree(dual_coefs_d));
    if (idx_d) CUDA_CHECK(cudaFree(idx_d));
    if (x_support_d) CUDA_CHECK(cudaFree(x_support_d));
  }

  void checkResults(int n_coefs_exp, float *dual_coefs_exp, float b_exp,
                    float *w_exp, float *x_support_exp = nullptr,
                    int *idx_exp = nullptr, float *dual_coefs_d = nullptr,
                    float *x_support_d = nullptr, int *idx_d = nullptr,
                    float epsilon = 0.001) {
    if (dual_coefs_d == nullptr) dual_coefs_d = this->dual_coefs_d;
    if (x_support_d == nullptr) x_support_d = this->x_support_d;
    if (idx_d == nullptr) idx_d = this->idx_d;

    ASSERT_LE(n_coefs, n_coefs_exp);
    EXPECT_TRUE(devArrMatchHost(dual_coefs_exp, dual_coefs_d, n_coefs,
                                CompareApprox<float>(1e-3f)));
    float *dual_coefs_host = new float[n_coefs];
    updateHost(dual_coefs_host, dual_coefs_d, n_coefs, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ay = 0;
    for (int i = 0; i < n_coefs; i++) {
      ay += dual_coefs_host[i];
    }
    // Test if \sum \alpha_i y_i = 0
    EXPECT_LT(abs(ay), 1.0e-6f);

    if (x_support_exp) {
      EXPECT_TRUE(devArrMatchHost(x_support_exp, x_support_d, n_coefs * n_cols,
                                  CompareApprox<float>(1e-6f)));
    }

    if (idx_exp) {
      EXPECT_TRUE(devArrMatchHost(idx_exp, idx_d, n_coefs, Compare<int>()));
    }

    float *x_support_host = new float[n_coefs * n_cols];
    updateHost(x_support_host, x_support_d, n_coefs * n_cols, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (w_exp) {
      for (int i = 0; i < n_cols; i++) w[i] = 0;

      for (int i = 0; i < n_coefs; i++) {
        w[0] += x_support_host[i] * dual_coefs_host[i];
        w[1] += x_support_host[i + n_coefs] * dual_coefs_host[i];
      }

      for (int i = 0; i < n_cols; i++)
        EXPECT_LT(abs(w[i] - w_exp[i]), epsilon) << "@" << i;
    }

    EXPECT_LT(abs(b - b_exp), epsilon);

    delete[] dual_coefs_host;
    delete[] x_support_host;
  }

  cumlHandle handle;
  cudaStream_t stream;
  GramMatrix::GramMatrixBase<float> *kernel;
  int n_rows = 6;
  const int n_cols = 2;
  int n_ws = 6;

  float *x_dev;
  int *ws_idx_dev;
  float *y_dev;
  float *y_pred;
  float *f_dev;
  float *alpha_dev;
  float *delta_alpha_dev;
  float *kernel_dev;
  float *return_buff_dev;

  float x_host[12] = {1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3};
  int ws_idx_host[6] = {0, 1, 2, 3, 4, 5};
  float y_host[6] = {-1, -1, 1, -1, 1, 1};

  float f_host[6] = {1, 1, -1, 1, -1, -1};

  float kernel_host[36] = {2, 3, 3, 4, 4,  5,  3, 5, 4, 6,  5,  7,
                           3, 4, 5, 6, 7,  8,  4, 6, 6, 8,  8,  10,
                           4, 5, 7, 8, 10, 11, 5, 7, 8, 10, 11, 13};
  cublasHandle_t cublas_handle;

  float *dual_coefs_d = nullptr;
  int n_coefs;
  int *idx_d = nullptr;
  float *x_support_d = nullptr;
  float b;
  float w[2];
};

TEST_F(SmoSolverTestF, BlockSolveTest) {
  SmoBlockSolve<float, 1024><<<1, n_ws, 0, stream>>>(
    y_dev, n_rows, alpha_dev, n_ws, delta_alpha_dev, f_dev, kernel_dev,
    ws_idx_dev, 1.0f, 1e-3f, return_buff_dev);

  CUDA_CHECK(cudaPeekAtLastError());
  float return_buff[2];
  updateHost(return_buff, return_buff_dev, 2, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  EXPECT_FLOAT_EQ(return_buff[0], 2.0f) << return_buff[0];
  EXPECT_LT(return_buff[1], 100) << return_buff[1];

  float host_alpha[6], host_dalpha[6];
  updateHost(host_alpha, alpha_dev, n_rows, stream);
  updateHost(host_dalpha, delta_alpha_dev, n_ws, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < n_ws; i++) {
    EXPECT_FLOAT_EQ(y_host[i] * host_alpha[i], host_dalpha[i])
      << "alpha and delta alpha " << i;
  }
  float w[] = {0, 0};

  //float alpha_expected[] = {0.6f, 0, 1, 1, 0, 0.6f};
  //for C=10: {0.25f, 0, 2.25f, 3.75f, 0, 1.75f};
  float ay = 0;
  for (int i = 0; i < n_rows; i++) {
    //   EXPECT_FLOAT_EQ(host_alpha[i], alpha_expected[i]) << "alpha " << i;
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

TEST(SmoSolverTest, GetResultsTest) {
  cumlHandle handle;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  handle.setStream(stream);
  auto allocator = handle.getImpl().getDeviceAllocator();

  int n_rows = 6;
  int n_cols = 2;

  device_buffer<float> x_dev(allocator, stream, n_rows * n_cols);
  float x_host[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  updateDevice(x_dev.data(), x_host, n_rows * n_cols, stream);

  device_buffer<float> y_dev(allocator, stream, n_rows);
  float y_host[] = {1, 1, 1, -1, -1, -1};
  updateDevice(y_dev.data(), y_host, n_rows, stream);

  device_buffer<float> alpha_dev(allocator, stream, n_rows);
  float alpha_host[] = {0.0, 0.5, 0.5, 0, 1.0, 0, 0};
  updateDevice(alpha_dev.data(), alpha_host, n_rows, stream);

  float *dual_coefs;
  int n_coefs;
  int *idx;
  float *x_support;
  float b;
  float C = 1;
  GramMatrix::GramMatrixBase<float> kernel(handle.getImpl().getCublasHandle());
  Results<float> res(handle.getImpl(), x_dev.data(), y_dev.data(), n_rows,
                     n_cols, C, &kernel);
  res.Get(alpha_dev.data(), &dual_coefs, &n_coefs, &idx, &x_support, &b);

  ASSERT_EQ(n_coefs, 3);

  float dual_coefs_exp[] = {0.5, 0.5, -1.0};
  EXPECT_TRUE(devArrMatchHost(dual_coefs_exp, dual_coefs, n_coefs,
                              CompareApprox<float>(1e-6f)));

  int idx_exp[] = {1, 2, 4};
  EXPECT_TRUE(devArrMatchHost(idx_exp, idx, n_coefs, Compare<int>()));

  float x_support_exp[] = {2, 3, 5, 8, 9, 11};
  EXPECT_TRUE(devArrMatchHost(x_support_exp, x_support, n_coefs * n_cols,
                              CompareApprox<float>(1e-6f)));

  EXPECT_FLOAT_EQ(b, 26.0f);

  if (n_coefs > 0) {
    allocator->deallocate(dual_coefs, n_coefs * sizeof(float), stream);
    allocator->deallocate(idx, n_coefs * sizeof(int), stream);
    allocator->deallocate(x_support, n_coefs * n_cols * sizeof(float), stream);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceSynchronize());
}

TEST(SmoSolverTest, SmoUpdateFTest) {
  cumlHandle handle;
  cudaStream_t stream;
  stream = handle.getImpl().getInternalStream(0);
  cublasHandle_t cublas_handle = handle.getImpl().getCublasHandle();

  int n_rows = 6;
  int n_cols = 2;
  int n_ws = 2;

  float *kernel_dev;
  allocate(kernel_dev, n_rows * n_ws);

  float *f_dev;
  allocate(f_dev, n_rows, true);

  float *delta_alpha_dev;
  allocate(delta_alpha_dev, n_ws);

  float kernel_host[] = {3, 5, 4, 6, 5, 7, 4, 5, 7, 8, 10, 11};
  updateDevice(kernel_dev, kernel_host, n_ws * n_rows, stream);

  float delta_alpha_host[] = {-0.1f, 0.1f};
  updateDevice(delta_alpha_dev, delta_alpha_host, n_ws, stream);

  GramMatrix::GramMatrixBase<float> kernel(handle.getImpl().getCublasHandle());
  SmoSolver<float> smo(handle.getImpl(), 1, 0.001, &kernel);

  smo.UpdateF(f_dev, n_rows, delta_alpha_dev, n_ws, kernel_dev);

  float f_host[6];
  updateHost(f_host, f_dev, n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float f_host_expected[] = {0.1f, 7.4505806e-9f, 0.3f, 0.2f, 0.5f, 0.4f};
  for (int i = 0; i < n_rows; i++) {
    EXPECT_FLOAT_EQ(f_host[i], f_host_expected[i]) << "UpdateF " << i;
  }

  CUDA_CHECK(cudaFree(delta_alpha_dev));
  CUDA_CHECK(cudaFree(kernel_dev));
  CUDA_CHECK(cudaFree(f_dev));
}

TEST_F(SmoSolverTestF, SmoSolveTest) {
  SmoSolver<float> smo(handle.getImpl(), 1, 0.001, kernel);
  smo.Solve(x_dev, n_rows, n_cols, y_dev, &dual_coefs_d, &n_coefs, &x_support_d,
            &idx_d, &b, 100, 1);

  float dual_coefs_exp[] = {-0.6, 1, -1, 0.6};
  float w_exp[] = {-0.4, 1.2};
  float x_support_exp[] = {1, 1, 2, 2, 1, 2, 2, 3};
  int idx_exp[] = {0, 2, 3, 5};
  SCOPED_TRACE("SmoSolveTest");
  checkResults(4, dual_coefs_exp, -1.8f, w_exp, x_support_exp, idx_exp);
}

TEST_F(SmoSolverTestF, SmoSolveTestLargeC) {
  float epsilon = 0.001;
  SmoSolver<float> smo(handle.getImpl(), 100, epsilon, kernel);

  smo.Solve(x_dev, n_rows, n_cols, y_dev, &dual_coefs_d, &n_coefs, &x_support_d,
            &idx_d, &b, 100, 1);

  float dual_coefs_exp[] = {-2, 4, -2, 0, 0};
  float w_exp[] = {-2, 2};

  int *idx_exp = nullptr;  //{ 0, 2, 3 };
  //float x_support_exp[] = { 1, 1, 2,  1, 2, 2, 0,0};
  float *x_support_exp = nullptr;

  SCOPED_TRACE("SmoSolveTestLargeC");
  checkResults(4, dual_coefs_exp, -1.0f, w_exp, x_support_exp, idx_exp);
}

TEST_F(SmoSolverTestF, SmoSolvePolynomial) {
  auto nonlin =
    new GramMatrix::PolynomialKernel<float, int>(3, 1.0, 1.0, cublas_handle);
  SmoSolver<float> smo(handle.getImpl(), 1, 1e-6f, nonlin);
  smo.Solve(x_dev, n_rows, n_cols, y_dev, &dual_coefs_d, &n_coefs, &x_support_d,
            &idx_d, &b, 100, 1);

  float dual_coefs_exp[] = {-0.02556136, 0.03979708, -0.01423571};
  float *w_exp = nullptr;
  float x_support_exp[] = {1, 1, 2, 1, 2, 2};
  int idx_exp[] = {0, 2, 3};
  SCOPED_TRACE("SmoSolveTestPolynomial");
  checkResults(3, dual_coefs_exp, -1.07739149f, w_exp, x_support_exp, idx_exp);
}

TEST_F(SmoSolverTestF, SvcTest) {
  float epsilon = 0.001;
  SVC<float> svc(handle, 1.0f, epsilon, KernelParams(LINEAR));
  svc.fit(x_dev, n_rows, n_cols, y_dev);
  n_coefs = svc.n_support;
  b = svc.b;
  float dual_coefs_exp[] = {-0.6, 1, -1, 0.6};
  float w_exp[] = {-0.4, 1.2};
  float x_support_exp[] = {1, 1, 2, 2, 1, 2, 2, 3};
  int idx_exp[] = {0, 2, 3, 5};
  SCOPED_TRACE("SvcTest");
  checkResults(4, dual_coefs_exp, -1.8f, w_exp, x_support_exp, idx_exp,
               svc.dual_coefs, svc.x_support, svc.support_idx);
  //  float dual_coefs_exp[] = { -2, 4, -2, 0, 0 };
  //  float x_support_exp[] = { 1, 1, 2,  1, 2, 2, 0,0};
  // allocate a prediction buffer, then we can compare pred buffer to y_dev
  for (int i = 0; i < 3; i++) {
    svc.predict(x_dev, n_rows, n_cols, y_pred);
    EXPECT_TRUE(devArrMatch(y_dev, y_pred, n_rows, CompareApprox<float>(1e-6f)))
      << i << "th prediction";
  }
}

TEST_F(SmoSolverTestF, SvcTestPoly) {
  float epsilon = 0.001;
  SVC<float> svc(handle, 1.0f, epsilon,
                 GramMatrix::KernelParams(GramMatrix::POLYNOMIAL));
  svc.fit(x_dev, n_rows, n_cols, y_dev);
  n_coefs = svc.n_support;
  b = svc.b;
  float dual_coefs_exp[] = {-0.6, 1, -1, 0.6};
  float w_exp[] = {-0.4, 1.2};
  float x_support_exp[] = {1, 1, 2, 2, 1, 2, 2, 3};
  int idx_exp[] = {0, 2, 3, 5};
  SCOPED_TRACE("SvcTestPoly");
}

__global__ void init_training_vectors(float *x, int n_rows, int n_cols,
                                      float *y) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows * n_cols) {
    int i = tid % n_rows;
    int k = tid / n_rows;
    x[tid] = tid;
    if (k == 0) {
      y[i] = (i % 2) * 2 - 1;
    }
  }
}

TEST(SvcSolverTest, SvcTestLargeNonlin) {
  cumlHandle handle;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  handle.setStream(stream);

  int n_rows = 1000;
  int n_cols = 780;
  int n_ws = n_rows;
  float *x_dev;
  allocate(x_dev, n_rows * n_cols);
  float *y_dev;
  allocate(y_dev, n_rows);

  int TPB = 256;
  init_training_vectors<<<ceildiv(n_rows * n_cols, TPB), TPB>>>(x_dev, n_rows,
                                                                n_cols, y_dev);
  CUDA_CHECK(cudaPeekAtLastError());

  float epsilon = 0.001;

  SVC<float> svc(handle, 1.0f, epsilon, KernelParams(RBF), 200, 1);
  svc.fit(x_dev, n_rows, n_cols, y_dev);

  ASSERT_LE(svc.n_support, n_rows);

  float *dual_coefs_host = new float[n_rows];
  updateHost(dual_coefs_host, svc.dual_coefs, svc.n_support, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ay = 0;
  for (int i = 0; i < svc.n_support; i++) {
    ay += dual_coefs_host[i];
  }
  // \sum \alpha_i y_i = 0
  EXPECT_LT(abs(ay), 1.0e-5f);

  //EXPECT_FLOAT_EQ(svc.b, -1.50995291e+09f);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
  delete[] dual_coefs_host;
}
TEST(SvcSolverTest, SvcTestLarge) {
  cumlHandle handle;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  handle.setStream(stream);

  int n_rows = 1000;
  int n_cols = 780;
  int n_ws = n_rows;
  float *x_dev;
  allocate(x_dev, n_rows * n_cols);
  float *y_dev;
  allocate(y_dev, n_rows);

  int TPB = 256;
  init_training_vectors<<<ceildiv(n_rows * n_cols, TPB), TPB>>>(x_dev, n_rows,
                                                                n_cols, y_dev);
  CUDA_CHECK(cudaPeekAtLastError());

  float epsilon = 0.001;

  SVC<float> svc(handle, 1.0f, epsilon, KernelParams(), 200, 200);
  svc.fit(x_dev, n_rows, n_cols, y_dev);

  ASSERT_LE(svc.n_support, n_rows);

  float *dual_coefs_host = new float[n_rows];
  updateHost(dual_coefs_host, svc.dual_coefs, svc.n_support, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ay = 0;
  for (int i = 0; i < svc.n_support; i++) {
    ay += dual_coefs_host[i];
  }
  // \sum \alpha_i y_i = 0
  EXPECT_LT(abs(ay), 1.0e-5f);

  float *x_support_host = new float[n_rows * n_cols];

  updateHost(x_support_host, svc.x_support, svc.n_support * n_cols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float *w = new float[n_cols];
  memset(w, 0, sizeof(float) * n_cols);
  for (int i = 0; i < svc.n_support; i++) {
    for (int k = 0; k < n_cols; k++) {
      w[k] += x_support_host[i + k * svc.n_support] * dual_coefs_host[i];
    }
  }

  // for linear problems it should be unique
  for (int k = 0; k < n_cols; k++) {
    //  EXPECT_LT(abs(w[k] - 5.00001139), epsilon) << k;
  }

  //EXPECT_FLOAT_EQ(svc.b, -1.50995291e+09f);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
  delete[] dual_coefs_host;
  delete[] x_support_host;
  delete[] w;
}
};  // end namespace SVM
};  // end namespace ML
