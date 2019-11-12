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
#include <iostream>
#include <vector>
#include "label/classlabels.h"
#include "random/make_regression.h"
#include "selection/knn.h"

namespace MLCommon {
namespace Selection {

struct KNNRegressionInputs {
  int rows;
  int cols;
  int n_labels;
  float cluster_std;
  int k;
};

class KNNRegressionTest : public ::testing::TestWithParam<KNNRegressionInputs> {
 protected:
  void basicTest() {
    std::shared_ptr<MLCommon::deviceAllocator> alloc(
      new defaultDeviceAllocator);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t cusolverDn_handle;
    cusolverDnCreate(&cusolverDn_handle);

    params = ::testing::TestWithParam<KNNRegressionInputs>::GetParam();

    allocate(train_samples, params.rows * params.cols);
    allocate(train_labels, params.rows);

    allocate(pred_labels, params.rows);

    allocate(knn_indices, params.rows * params.k);
    allocate(knn_dists, params.rows * params.k);

    MLCommon::Random::make_regression(train_samples, train_labels, params.rows,
                                      params.cols, params.cols, cublas_handle,
                                      cusolverDn_handle, alloc, stream);

    float **ptrs = new float *[1];
    int *sizes = new int[1];
    ptrs[0] = train_samples;
    sizes[0] = params.rows;

    brute_force_knn(ptrs, sizes, 1, params.cols, train_samples, params.rows,
                    knn_indices, knn_dists, params.k, alloc, stream);

    std::vector<float *> y;
    y.push_back(train_labels);

    knn_regress(pred_labels, knn_indices, y, params.rows, params.k, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    cudaFree(train_samples);
    cudaFree(train_labels);

    cudaFree(pred_labels);

    cudaFree(knn_indices);
    cudaFree(knn_dists);
  }

 protected:
  KNNRegressionInputs params;

  float *train_samples;
  float *train_labels;

  float *pred_labels;

  int64_t *knn_indices;
  float *knn_dists;
};

typedef KNNRegressionTest KNNRegressionTestF;
TEST_P(KNNRegressionTestF, Fit) {
  ASSERT_TRUE(devArrMatch(train_labels, pred_labels, params.rows,
                          CompareApprox<float>(1)));
}

const std::vector<KNNRegressionInputs> inputsf = {
  {100, 10, 2, 0.01f, 2},  {1000, 10, 5, 0.01f, 2},  {10000, 10, 5, 0.01f, 2},
  {100, 10, 2, 0.01f, 10}, {1000, 10, 5, 0.01f, 10}, {10000, 10, 5, 0.01f, 10},
  {100, 10, 2, 0.01f, 50}, {1000, 10, 5, 0.01f, 50}, {10000, 10, 5, 0.01f, 50}};

INSTANTIATE_TEST_CASE_P(KNNRegressionTest, KNNRegressionTestF,
                        ::testing::ValuesIn(inputsf));

};  // end namespace Selection
};  // namespace MLCommon
