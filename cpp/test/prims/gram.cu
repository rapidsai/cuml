/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cuml/matrix/kernelparams.h>
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <cuml/common/device_buffer.hpp>
#include <cuml/common/host_buffer.hpp>
#include <iostream>
#include <matrix/grammatrix.cuh>
#include <matrix/kernelfactory.cuh>
#include <memory>
#include <raft/cuda_utils.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace Matrix {

class GramMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    allocator = std::make_shared<raft::mr::device::default_allocator>();
    host_allocator = std::make_shared<raft::mr::host::default_allocator>();
    raft::allocate(x_dev, n1 * n_cols);
    raft::update_device(x_dev, x_host, n1 * n_cols, stream);

    raft::allocate(gram_dev, n1 * n1);
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(gram_dev));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
  }

  void naiveRBFKernel(float *x1_dev, int n1, int n_cols, float *x2_dev, int n2,
                      float gamma) {
    host_buffer<float> x1_host(host_allocator, stream, n1 * n_cols);
    raft::update_host(x1_host.data(), x1_dev, n1 * n_cols, stream);
    host_buffer<float> x2_host(host_allocator, stream, n2 * n_cols);
    raft::update_host(x2_host.data(), x2_dev, n2 * n_cols, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n2; j++) {
        float d = 0;
        for (int k = 0; k < n_cols; k++) {
          float diff = x1_host[i + k * n1] - x2_host[j + k * n2];
          d += diff * diff;
        }
        gram_host_expected[i + j * n2] = exp(-gamma * d);
      }
    }
  }
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  std::shared_ptr<deviceAllocator> allocator;
  std::shared_ptr<hostAllocator> host_allocator;
  int n1 = 4;
  int n_cols = 2;
  int n2 = 4;

  float *x_dev;
  float *gram_dev;
  float x_host[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float gram_host_expected[16] = {26, 32, 38, 44, 32, 40, 48, 56,
                                  38, 48, 58, 68, 44, 56, 68, 80};
};

TEST_F(GramMatrixTest, Base) {
  GramMatrixBase<float> kernel(cublas_handle);
  kernel(x_dev, n1, n_cols, x_dev, n1, gram_dev, stream);
  ASSERT_TRUE(raft::devArrMatchHost(gram_host_expected, gram_dev, n1 * n1,
                                    raft::CompareApprox<float>(1e-6f)));
}
TEST_F(GramMatrixTest, Poly) {
  float offset = 2.4;
  float gain = 0.5;
  // naive kernel
  for (int z = 0; z < n1 * n1; z++) {
    float val = gain * gram_host_expected[z] + offset;
    gram_host_expected[z] = val * val;
  }

  PolynomialKernel<float, int> kernel(2, gain, offset, cublas_handle);
  kernel(x_dev, n1, n_cols, x_dev, n1, gram_dev, stream);
  ASSERT_TRUE(raft::devArrMatchHost(gram_host_expected, gram_dev, n1 * n1,
                                    raft::CompareApprox<float>(1e-6f)));
}

TEST_F(GramMatrixTest, Tanh) {
  float offset = 2.4;
  float gain = 0.5;
  // naive kernel
  for (int z = 0; z < n1 * n1; z++) {
    gram_host_expected[z] = tanh(gain * gram_host_expected[z] + offset);
  }
  TanhKernel<float> kernel(gain, offset, cublas_handle);
  kernel(x_dev, n1, n_cols, x_dev, n1, gram_dev, stream);
  ASSERT_TRUE(raft::devArrMatchHost(gram_host_expected, gram_dev, n1 * n1,
                                    raft::CompareApprox<float>(1e-6f)));
}

TEST_F(GramMatrixTest, RBF) {
  float gamma = 0.5;
  naiveRBFKernel(x_dev, n1, n_cols, x_dev, n1, gamma);
  RBFKernel<float> kernel(gamma);
  kernel(x_dev, n1, n_cols, x_dev, n1, gram_dev, stream);
  ASSERT_TRUE(raft::devArrMatchHost(gram_host_expected, gram_dev, n1 * n1,
                                    raft::CompareApprox<float>(3e-6f)));
}

TEST_F(GramMatrixTest, RBF_Rectangular) {
  float gamma = 0.7;
  RBFKernel<float> kernel(gamma);
  // Instead of a 5x5 Gram matrix, we want to calculate a 5x3 matrix here.
  // The inputs to the distance function are the vector sets x1 and x2.
  //
  // x1 = [ [1, 6],
  //        [2, 7],
  //        [3, 8],
  //        [4, 9],
  //        [5, 10] ];
  // The vectors are stored in column major format, so actually
  float x1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int n1 = 5;

  // x2 = [ [1, 6],
  //        [2, 7],
  //        [3, 8] ];
  // In column major format:
  float x2[] = {1, 2, 3, 6, 7, 8};
  int n2 = 3;
  //
  // The output is a 5x3 matrix. Here is the distance matrix (without exp)
  //  K(x1,x2)  = [ [ 0,  2, 8],
  //                [ 2,  0, 2],
  //                [ 8,  2, 0],
  //                [18,  8, 2],
  //                [32, 18, 8] ];
  //
  // It is also stored in colum major format, therefore:
  float K[] = {0, 2, 8, 18, 32, 2, 0, 2, 8, 18, 8, 2, 0, 2, 8};

  // The RBF kernel calculates exp for the distance matrix
  for (int i = 0; i < n1 * n2; i++) {
    K[i] = exp(-gamma * K[i]);
  }

  device_buffer<float> x1_dev(allocator, stream, n1 * n_cols);
  raft::update_device(x1_dev.data(), x1, n1 * n_cols, stream);
  device_buffer<float> x2_dev(allocator, stream, n2 * n_cols);
  raft::update_device(x2_dev.data(), x2, n2 * n_cols, stream);

  kernel(x1_dev.data(), n1, n_cols, x2_dev.data(), n2, gram_dev, stream);
  ASSERT_TRUE(raft::devArrMatchHost(K, gram_dev, n1 * n2,
                                    raft::CompareApprox<float>(1e-6f)));
}
};  // end namespace Matrix
};  // end namespace MLCommon
