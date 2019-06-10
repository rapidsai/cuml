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
#include "kmeans/kmeans.cu"

namespace ML {

using namespace MLCommon;

template <typename T>
struct KmeansInputs {
  int n_clusters;
  T tol;
  int n_row;
  int n_col;
};

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
    int m = params.n_row;
    int n = params.n_col;
    int k = params.n_clusters;

    // make space for outputs : d_centroids, d_labels
    // and reference output : d_labels_ref
    allocate(d_srcdata, n * m);
    allocate(d_labels, m);
    allocate(d_labels_ref, m);
    allocate(d_centroids, k * n);
    allocate(d_centroids_ref, k * n);

    // make testdata on host
    std::vector<T> h_srcdata = {1.0, 1.0, 3.0, 4.0, 1.0, 2.0, 2.0, 3.0};
    h_srcdata.resize(n * m);
    updateDevice(d_srcdata, h_srcdata.data(), m * n, stream);

    // make and assign reference output
    std::vector<int> h_labels_ref = {0, 1, 0, 1};
    h_labels_ref.resize(m);
    updateDevice(d_labels_ref, h_labels_ref.data(), m, stream);

    std::vector<T> h_centroids_ref = {1.0, 1.5, 2.5, 3.5};
    h_centroids_ref.resize(k * n);
    updateDevice(d_centroids_ref, h_centroids_ref.data(), k * n, stream);

    cumlHandle handle;
    handle.setStream(stream);

    // The actual kmeans api calls
    // fit
    kmeans::fit_predict(handle, k, metric, init, max_iterations, params.tol,
                        seed, d_srcdata, m, n, d_centroids, d_labels);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    basicTest();
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(d_srcdata));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_labels_ref));
    CUDA_CHECK(cudaFree(d_centroids_ref));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  KmeansInputs<T> params;
  T *d_srcdata;
  int *d_labels, *d_labels_ref;
  T *d_centroids, *d_centroids_ref;
  int verbose = 0;
  int seed = 0;
  int max_iterations = 300;
  kmeans::InitMethod init = kmeans::InitMethod::Random;
  int metric = 1;
  cudaStream_t stream;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{2, 0.05f, 4, 2}};

const std::vector<KmeansInputs<double>> inputsd2 = {{2, 0.05, 4, 2}};

// FIXME: These tests are disabled due to being too sensitive to RNG:
// https://github.com/rapidsai/cuml/issues/71
typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) {
  ASSERT_TRUE(devArrMatch(d_labels_ref, d_labels, params.n_row,
                          CompareApproxAbs<float>(params.tol)));
  ASSERT_TRUE(devArrMatch(d_centroids_ref, d_centroids,
                          params.n_clusters * params.n_col,
                          CompareApproxAbs<float>(params.tol)));
}

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) {
  ASSERT_TRUE(devArrMatch(d_labels_ref, d_labels, params.n_row,
                          CompareApproxAbs<double>(params.tol)));
  ASSERT_TRUE(devArrMatch(d_centroids_ref, d_centroids,
                          params.n_clusters * params.n_col,
                          CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace ML
