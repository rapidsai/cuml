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
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
    int n_samples = testparams.n_row;
    int n_features = testparams.n_col;
    params.n_clusters = testparams.n_clusters;
    params.metric = 1;
    params.init = ML::kmeans::KMeansParams::Random;

    // make space for outputs : d_centroids, d_labels
    // and reference output : d_labels_ref
    allocate(d_srcdata, n_samples * n_features);
    allocate(d_labels, n_samples);
    allocate(d_labels_ref, n_samples);
    allocate(d_centroids, params.n_clusters * n_features);
    allocate(d_centroids_ref, params.n_clusters * n_features);

    // make testdata on host
    std::vector<T> h_srcdata = {1.0, 1.0, 3.0, 4.0, 1.0, 2.0, 2.0, 3.0};
    h_srcdata.resize(n_features * n_samples);
    updateDevice(d_srcdata, h_srcdata.data(), n_samples * n_features, stream);

    // make and assign reference output
    std::vector<int> h_labels_ref = {0, 1, 0, 1};
    h_labels_ref.resize(n_samples);
    updateDevice(d_labels_ref, h_labels_ref.data(), n_samples, stream);

    std::vector<T> h_centroids_ref = {1.0, 1.5, 2.5, 3.5};
    h_centroids_ref.resize(params.n_clusters * n_features);
    updateDevice(d_centroids_ref, h_centroids_ref.data(),
                 params.n_clusters * n_features, stream);

    cumlHandle handle;
    handle.setStream(stream);

    T inertia = 0;
    int n_iter = 0;
    kmeans::fit_predict(handle, params, d_srcdata, n_samples, n_features,
                        d_centroids, d_labels, inertia, n_iter);
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
  KmeansInputs<T> testparams;
  T *d_srcdata;
  int *d_labels, *d_labels_ref;
  T *d_centroids, *d_centroids_ref;
  ML::kmeans::KMeansParams params;
  cudaStream_t stream;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{2, 0.05f, 4, 2}};

const std::vector<KmeansInputs<double>> inputsd2 = {{2, 0.05, 4, 2}};

// FIXME: These tests are disabled due to being too sensitive to RNG:
// https://github.com/rapidsai/cuml/issues/71
typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) {
  ASSERT_TRUE(devArrMatch(d_labels_ref, d_labels, testparams.n_row,
                          CompareApproxAbs<float>(testparams.tol)));
  ASSERT_TRUE(devArrMatch(d_centroids_ref, d_centroids,
                          testparams.n_clusters * testparams.n_col,
                          CompareApproxAbs<float>(testparams.tol)));
}

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) {
  ASSERT_TRUE(devArrMatch(d_labels_ref, d_labels, testparams.n_row,
                          CompareApproxAbs<double>(testparams.tol)));
  ASSERT_TRUE(devArrMatch(d_centroids_ref, d_centroids,
                          testparams.n_clusters * testparams.n_col,
                          CompareApproxAbs<double>(testparams.tol)));
}

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace ML
