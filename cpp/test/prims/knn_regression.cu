/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <cuda_utils.cuh>
#include <iostream>
#include <label/classlabels.cuh>
#include <linalg/reduce.cuh>
#include <random/rng.cuh>
#include <selection/knn.cuh>
#include <vector>
#include "test_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace MLCommon {
namespace Selection {

struct KNNRegressionInputs {
  int rows;
  int cols;
  int n_labels;
  float cluster_std;
  int k;
};

void generate_data(float *out_samples, float *out_labels, int n_rows,
                   int n_cols, cudaStream_t stream) {
  Random::Rng r(0ULL, MLCommon::Random::GenTaps);

  r.uniform(out_samples, n_rows * n_cols, 0.0f, 1.0f, stream);

  MLCommon::LinAlg::unaryOp<float>(
    out_samples, out_samples, n_rows,
    [=] __device__(float input) { return 2 * input - 1; }, stream);

  MLCommon::LinAlg::reduce(
    out_labels, out_samples, n_cols, n_rows, 0.0f, true, true, stream, false,
    [=] __device__(float in, int n) { return in * in; }, Sum<float>(),
    [=] __device__(float in) { return sqrt(in); });

  thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(out_labels);
  float max =
    *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr, d_ptr + n_rows));

  MLCommon::LinAlg::unaryOp<float>(
    out_labels, out_labels, n_rows,
    [=] __device__(float input) { return input / max; }, stream);
}

class KNNRegressionTest : public ::testing::TestWithParam<KNNRegressionInputs> {
 protected:
  void basicTest() {
    std::shared_ptr<MLCommon::deviceAllocator> alloc(
      new raft::mr::device::default_allocator);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    cusolverDnHandle_t cusolverDn_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverDn_handle));

    params = ::testing::TestWithParam<KNNRegressionInputs>::GetParam();

    allocate(train_samples, params.rows * params.cols);
    allocate(train_labels, params.rows);

    allocate(pred_labels, params.rows);

    allocate(knn_indices, params.rows * params.k);
    allocate(knn_dists, params.rows * params.k);

    generate_data(train_samples, train_labels, params.rows, params.cols,
                  stream);

    std::vector<float *> ptrs(1);
    std::vector<int> sizes(1);
    ptrs[0] = train_samples;
    sizes[0] = params.rows;

    brute_force_knn(ptrs, sizes, params.cols, train_samples, params.rows,
                    knn_indices, knn_dists, params.k, alloc, stream);

    std::vector<float *> y;
    y.push_back(train_labels);

    knn_regress(pred_labels, knn_indices, y, params.rows, params.rows, params.k,
                stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(train_samples));
    CUDA_CHECK(cudaFree(train_labels));

    CUDA_CHECK(cudaFree(pred_labels));

    CUDA_CHECK(cudaFree(knn_indices));
    CUDA_CHECK(cudaFree(knn_dists));
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
                          CompareApprox<float>(0.3)));
}

const std::vector<KNNRegressionInputs> inputsf = {
  {100, 10, 2, 0.01f, 2},  {1000, 10, 5, 0.01f, 2},  {10000, 10, 5, 0.01f, 2},
  {100, 10, 2, 0.01f, 10}, {1000, 10, 5, 0.01f, 10}, {10000, 10, 5, 0.01f, 10},
  {100, 10, 2, 0.01f, 15}, {1000, 10, 5, 0.01f, 15}, {10000, 10, 5, 0.01f, 15}};

INSTANTIATE_TEST_CASE_P(KNNRegressionTest, KNNRegressionTestF,
                        ::testing::ValuesIn(inputsf));

};  // end namespace Selection
};  // namespace MLCommon
