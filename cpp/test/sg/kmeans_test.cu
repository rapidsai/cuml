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
#include <vector>

#include <cuml/cluster/kmeans.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>

#include "common/device_buffer.hpp"

namespace ML {

using namespace MLCommon;
using namespace Datasets;
using namespace Metrics;

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
};

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  void basicTest() {
    cumlHandle handle;
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples = testparams.n_row;
    int n_features = testparams.n_col;
    params.n_clusters = testparams.n_clusters;
    params.tol = testparams.tol;
    params.n_init = 5;
    params.seed = 1;
    params.oversampling_factor = 0;

    device_buffer<T> X(handle.getDeviceAllocator(), handle.getStream(),
                       n_samples * n_features);

    device_buffer<int> labels(handle.getDeviceAllocator(), handle.getStream(),
                              n_samples);

    make_blobs(handle, X.data(), labels.data(), n_samples, n_features,
               params.n_clusters, nullptr, nullptr, 1.0, false, -10.0f, 10.0f,
               1234ULL);

    allocate(d_labels, n_samples);
    allocate(d_labels_ref, n_samples);
    allocate(d_centroids, params.n_clusters * n_features);

    MLCommon::copy(d_labels_ref, labels.data(), n_samples, handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    T inertia = 0;
    int n_iter = 0;
    kmeans::fit_predict(handle, params, X.data(), n_samples, n_features,
                        d_centroids, d_labels, inertia, n_iter);

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    score = adjustedRandIndex(handle, d_labels_ref, d_labels, n_samples, 0,
                              params.n_clusters - 1);

    if (score < 1.0) {
      std::cout << "Expected: "
                << arr2Str(d_labels_ref, 25, "d_labels_ref", handle.getStream())
                << std::endl;
      std::cout << "Actual: "
                << arr2Str(d_labels, 25, "d_labels", handle.getStream())
                << std::endl;

      std::cout << "Score = " << score << std::endl;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_labels_ref));
  }

 protected:
  KmeansInputs<T> testparams;
  int *d_labels, *d_labels_ref;
  T *d_centroids;
  double score;
  ML::kmeans::KMeansParams params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001},
                                                   {1000, 100, 20, 0.0001},
                                                   {10000, 32, 10, 0.0001},
                                                   {10000, 100, 50, 0.0001},
                                                   {10000, 1000, 200, 0.0001}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001},
                                                    {1000, 100, 20, 0.0001},
                                                    {10000, 32, 10, 0.0001},
                                                    {10000, 100, 50, 0.0001},
                                                    {10000, 1000, 200, 0.0001}};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace ML
